//! Wire-format tests: each provider against a mock HTTP server.

use std::collections::BTreeMap;
use std::time::Duration;

use hca_core::{AgentRole, BudgetError, TokenBudget};
use hca_llm::{
    Conversation, Llm, LlmConfig, LlmError, Message, ProviderConfig, RetryConfig, RoleConfig, StopReason,
    ToolCall, ToolSpec,
};
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const ROLE: AgentRole = AgentRole::Coder;

fn config(provider: ProviderConfig, model: &str) -> LlmConfig {
    LlmConfig {
        providers: BTreeMap::from([("p".to_string(), provider)]),
        roles: BTreeMap::from([(
            ROLE,
            RoleConfig {
                provider: "p".into(),
                model: model.into(),
                max_tokens: 1000,
                temperature: Some(0.0),
            },
        )]),
        timeout_secs: 10,
        retry: RetryConfig { max_retries: 2, base_delay_ms: 1 },
    }
}

fn env(name: &str) -> Option<String> {
    match name {
        "XAI_API_KEY" => Some("xai-test".into()),
        "ANTHROPIC_API_KEY" => Some("sk-ant-test".into()),
        "OPENAI_API_KEY" => Some("sk-openai-test".into()),
        _ => None,
    }
}

/// A conversation that exercises every message kind: a tool call with two
/// results, then a follow-up user message.
fn conversation() -> Conversation {
    let call_a = ToolCall { id: "c1".into(), name: "read_file".into(), arguments: json!({"path": "a.py"}) };
    let call_b = ToolCall { id: "c2".into(), name: "read_file".into(), arguments: json!({"path": "b.py"}) };
    Conversation {
        system: Some("You are the coder.".into()),
        messages: vec![
            Message::user("Fix the bug."),
            Message::Assistant {
                content: "Reading files.".into(),
                tool_calls: vec![call_a.clone(), call_b.clone()],
            },
            Message::tool_result(&call_a, "print(1)"),
            Message::tool_error(&call_b, "no such file"),
            Message::user("Go on."),
        ],
        tools: vec![ToolSpec {
            name: "read_file".into(),
            description: "Read a file".into(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
        }],
    }
}

fn budget() -> TokenBudget {
    TokenBudget::new(1_000_000)
}

async fn sent_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    serde_json::from_slice(&requests.last().unwrap().body).unwrap()
}

fn openai_reply() -> Value {
    json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"c.py\"}"}},
                    {"id": "t2", "type": "function", "function": {"name": "read_file", "arguments": "{not json"}}
                ]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    })
}

#[tokio::test]
async fn grok_speaks_openai_chat_completions_with_the_xai_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer xai-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_reply()))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        ProviderConfig::Grok { base_url: format!("{}/v1", server.uri()), api_key_env: "XAI_API_KEY".into() };
    let llm = config(provider, "grok-4").build_with_env(env).unwrap();
    let response = llm.chat(ROLE, &conversation(), &budget()).await.unwrap();

    assert_eq!(response.stop, StopReason::ToolUse);
    assert_eq!(response.usage.total(), 150);
    assert_eq!(response.tool_calls[0].arguments, json!({"path": "c.py"}));
    assert_eq!(
        response.tool_calls[1].arguments,
        json!("{not json"),
        "bad JSON is kept for the agent to report"
    );

    let body = sent_body(&server).await;
    assert_eq!(body["model"], "grok-4");
    assert_eq!(body["max_tokens"], 1000);
    assert_eq!(body["temperature"], 0.0, "a zero temperature is sent, not dropped");
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0], json!({"role": "system", "content": "You are the coder."}));
    assert_eq!(messages[2]["tool_calls"][1]["function"]["arguments"], "{\"path\":\"b.py\"}");
    assert_eq!(messages[3], json!({"role": "tool", "tool_call_id": "c1", "content": "print(1)"}));
    assert_eq!(messages[4]["content"], "ERROR: no such file");
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
}

#[tokio::test]
async fn openai_uses_max_completion_tokens() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-openai-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_reply()))
        .mount(&server)
        .await;

    let provider = ProviderConfig::Openai {
        base_url: format!("{}/v1", server.uri()),
        api_key_env: "OPENAI_API_KEY".into(),
    };
    let llm = config(provider, "gpt-test").build_with_env(env).unwrap();
    llm.chat(ROLE, &conversation(), &budget()).await.unwrap();

    let body = sent_body(&server).await;
    assert_eq!(body["max_completion_tokens"], 1000);
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn anthropic_merges_tool_results_into_one_user_turn() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Writing the fix."},
                {"type": "tool_use", "id": "tu1", "name": "write_file", "input": {"path": "a.py", "content": "x"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 200, "output_tokens": 50}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = ProviderConfig::Anthropic {
        base_url: format!("{}/v1", server.uri()),
        api_key_env: "ANTHROPIC_API_KEY".into(),
    };
    let llm = config(provider, "claude-test").build_with_env(env).unwrap();
    let response = llm.chat(ROLE, &conversation(), &budget()).await.unwrap();

    assert_eq!(response.content, "Writing the fix.");
    assert_eq!(response.tool_calls[0].name, "write_file");
    assert_eq!(response.stop, StopReason::ToolUse);
    assert_eq!(response.usage.total(), 250);

    let body = sent_body(&server).await;
    assert_eq!(body["system"], "You are the coder.");
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    let messages = body["messages"].as_array().unwrap();
    let roles: Vec<&str> = messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "assistant", "user"], "turns must alternate");
    let last = messages[2]["content"].as_array().unwrap();
    assert_eq!(last.len(), 3, "two tool results and the follow-up text share one turn");
    assert_eq!(last[0]["type"], "tool_result");
    assert_eq!(last[1]["is_error"], true);
    assert_eq!(last[2], json!({"type": "text", "text": "Go on."}));
    assert_eq!(messages[1]["content"][1]["input"], json!({"path": "a.py"}));
}

#[tokio::test]
async fn ollama_sends_options_and_gets_unique_call_ids() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "x.py"}}},
                    {"function": {"name": "read_file", "arguments": {"path": "y.py"}}}
                ]
            },
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 80,
            "eval_count": 20
        })))
        .mount(&server)
        .await;

    let provider =
        ProviderConfig::Ollama { base_url: server.uri(), num_ctx: Some(16_384), think: Some(true) };
    let llm = config(provider, "qwen3:8b").build_with_env(env).unwrap();
    let response = llm.chat(ROLE, &conversation(), &budget()).await.unwrap();

    assert_eq!(response.stop, StopReason::ToolUse);
    assert_ne!(response.tool_calls[0].id, response.tool_calls[1].id);
    assert_eq!(response.usage.total(), 100);

    let body = sent_body(&server).await;
    assert_eq!(body["stream"], false);
    assert_eq!(body["think"], true);
    assert_eq!(body["options"], json!({"num_predict": 1000, "temperature": 0.0, "num_ctx": 16_384}));
    assert_eq!(body["messages"][3], json!({"role": "tool", "tool_name": "read_file", "content": "print(1)"}));
}

#[tokio::test]
async fn server_errors_are_retried_then_succeed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_reply()))
        .mount(&server)
        .await;

    let provider =
        ProviderConfig::Grok { base_url: format!("{}/v1", server.uri()), api_key_env: "XAI_API_KEY".into() };
    let llm = config(provider, "grok-4").build_with_env(env).unwrap();
    llm.chat(ROLE, &conversation(), &budget()).await.unwrap();
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn timeouts_are_not_retried_because_they_may_have_been_billed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(openai_reply()).set_delay(Duration::from_secs(3)),
        )
        .mount(&server)
        .await;
    let provider =
        ProviderConfig::Grok { base_url: format!("{}/v1", server.uri()), api_key_env: "XAI_API_KEY".into() };
    let mut cfg = config(provider, "grok-4");
    cfg.timeout_secs = 1;
    let err = cfg.build_with_env(env).unwrap().chat(ROLE, &conversation(), &budget()).await.unwrap_err();
    assert!(err.is_retryable() && !err.is_unbilled(), "{err:?}");
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        1,
        "a retry would hide the first attempt's cost"
    );

    let refused = LlmError::Status { provider: "p".into(), status: 503, body: String::new() };
    assert!(refused.is_unbilled(), "error statuses aren't billed, so they are retried");
}

#[tokio::test]
async fn client_errors_are_not_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad model"))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        ProviderConfig::Grok { base_url: format!("{}/v1", server.uri()), api_key_env: "XAI_API_KEY".into() };
    let llm = config(provider, "nope").build_with_env(env).unwrap();
    let err = llm.chat(ROLE, &conversation(), &budget()).await.unwrap_err();
    assert!(matches!(&err, LlmError::Status { status: 400, body, .. } if body == "bad model"), "{err:?}");
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn calls_over_budget_are_refused_before_sending() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(openai_reply()))
        .expect(0)
        .mount(&server)
        .await;

    let provider =
        ProviderConfig::Grok { base_url: format!("{}/v1", server.uri()), api_key_env: "XAI_API_KEY".into() };
    let llm = config(provider, "grok-4").build_with_env(env).unwrap();

    let nearly_spent = TokenBudget::with_used(10_000, 9_500);
    let err = llm.chat(ROLE, &conversation(), &nearly_spent).await.unwrap_err();
    assert!(matches!(err, LlmError::Budget(BudgetError::Insufficient { .. })), "{err:?}");

    let spent = TokenBudget::with_used(10_000, 10_000);
    let err = llm.chat(ROLE, &conversation(), &spent).await.unwrap_err();
    assert!(matches!(err, LlmError::Budget(BudgetError::Exhausted { .. })), "{err:?}");
}

#[test]
fn config_errors_are_specific() {
    let grok = ProviderConfig::Grok { base_url: "http://x".into(), api_key_env: "XAI_API_KEY".into() };

    let err = config(grok.clone(), "m").build_with_env(|_| None).unwrap_err();
    assert!(err.to_string().contains("XAI_API_KEY"), "{err}");

    let mut unknown = config(grok, "m");
    unknown.roles.get_mut(&ROLE).unwrap().provider = "missing".into();
    let err = unknown.build_with_env(env).unwrap_err();
    assert!(err.to_string().contains("\"missing\""), "{err}");
}

#[test]
fn unused_providers_need_no_key() {
    let mut cfg = config(
        ProviderConfig::Ollama { base_url: "http://localhost:11434".into(), num_ctx: None, think: None },
        "m",
    );
    cfg.providers.insert(
        "unused".into(),
        ProviderConfig::Anthropic { base_url: "http://x".into(), api_key_env: "UNSET_KEY".into() },
    );
    let llm = cfg.build_with_env(|_| None).unwrap();
    assert!(llm.role(ROLE).is_some());
    assert!(llm.role(AgentRole::Pm).is_none());
}

#[test]
fn config_reads_from_a_file_format() {
    let cfg: LlmConfig = serde_json::from_value(json!({
        "providers": {
            "local": {"kind": "ollama", "num_ctx": 16384},
            "grok": {"kind": "grok"},
            "claude": {"kind": "anthropic"}
        },
        "roles": {
            "pm": {"provider": "claude", "model": "claude-opus-5-5"},
            "coder": {"provider": "grok", "model": "grok-4", "max_tokens": 8192}
        }
    }))
    .unwrap();
    assert_eq!(cfg.timeout_secs, 600);
    assert_eq!(cfg.roles[&AgentRole::Coder].max_tokens, 8192);
    assert_eq!(
        cfg.providers["grok"],
        ProviderConfig::Grok { base_url: "https://api.x.ai/v1".into(), api_key_env: "XAI_API_KEY".into() }
    );
    let llm = cfg.build_with_env(env).unwrap();
    assert_eq!(llm.role(AgentRole::Pm).unwrap().provider.name(), "anthropic");
}

#[tokio::test]
async fn unconfigured_roles_are_refused() {
    let llm = Llm::new(std::collections::HashMap::new());
    let err = llm.chat(AgentRole::Critic, &Conversation::default(), &budget()).await.unwrap_err();
    assert!(matches!(err, LlmError::NoModelForRole(AgentRole::Critic)));
}

#[test]
fn research_and_spec_fall_back_to_the_coder_model() {
    let ollama = ProviderConfig::Ollama { base_url: "http://x".into(), num_ctx: Some(32_768), think: None };
    let llm = config(ollama, "qwen2.5-coder").build_with_env(env).unwrap();
    assert_eq!(llm.role(AgentRole::Research).unwrap().model, "qwen2.5-coder");
    assert_eq!(llm.role(AgentRole::Spec).unwrap().model, "qwen2.5-coder");
    assert!(llm.role(AgentRole::Critic).is_none(), "only work roles fall back");
    assert_eq!(llm.context_window(AgentRole::Research), Some(32_768), "num_ctx is the window");
}

#[test]
fn cloud_providers_report_a_context_window() {
    let claude =
        ProviderConfig::Anthropic { base_url: "http://x".into(), api_key_env: "ANTHROPIC_API_KEY".into() };
    let llm = config(claude, "claude-opus-5-5").build_with_env(env).unwrap();
    assert_eq!(llm.context_window(ROLE), Some(200_000));
    let compatible = ProviderConfig::OpenaiCompatible {
        base_url: "http://x/v1".into(),
        api_key_env: None,
        max_tokens_field: hca_llm::MaxTokensField::default(),
        context_tokens: None,
    };
    let llm = config(compatible, "m").build_with_env(env).unwrap();
    assert_eq!(
        llm.context_window(ROLE),
        Some(u64::from(hca_llm::ASSUMED_CONTEXT_TOKENS)),
        "assumed, so conversations are still trimmed"
    );
}

#[tokio::test]
async fn ollama_always_gets_a_context_size() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {"role": "assistant", "content": "hi"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1
        })))
        .mount(&server)
        .await;
    let provider = ProviderConfig::Ollama { base_url: server.uri(), num_ctx: None, think: None };
    let llm = config(provider, "qwen3:8b").build_with_env(env).unwrap();
    llm.chat(ROLE, &conversation(), &budget()).await.unwrap();

    let sent = sent_body(&server).await["options"]["num_ctx"].as_u64();
    assert_eq!(sent, Some(u64::from(hca_llm::DEFAULT_OLLAMA_NUM_CTX)), "never Ollama's own default");
    assert_eq!(llm.context_window(ROLE), sent, "the window HCA trims to is the one Ollama loads");
}
