# Prompt Cookbook

## Overview

Each agent in HCA is driven by a system prompt stored in `src/hca/prompts/{role}.txt`. These prompts are loaded at startup and can be hot-reloaded without restarting the application (via the API).

The LLM interaction pattern is:

```
system: <system prompt from prompts/{role}.txt>
user:   <conversation history (per-project)>
user:   <current prompt from process_message()>
assistant: <LLM response>
```

---

## Prompt Structure

Every agent prompt follows this structure:

1. **Role definition** — who the agent is and its purpose
2. **Responsibilities** — what the agent does
3. **Rules** — behavioral constraints and guidelines
4. **Tool instructions** — how and when to use tools
5. **Output format** — expected structure of responses
6. **Few-shot examples** — concrete input/output pairs

### Example Layout

```
You are the <Role> agent in an autonomous AI development team.

YOUR ROLE:
<concise role description>

YOUR RESPONSIBILITIES:
1. ...
2. ...

RULES:
- <rule 1>
- <rule 2>

TOOL USE:
<how to use tools>

FEW-SHOT EXAMPLE:
<example input>
<example output>
```

---

## Per-Agent Prompt Patterns

### PM Agent

**File**: `prompts/pm.txt`
**Goal**: Decompose ideas into tasks

Key patterns:
- Use the `create_task` tool for EVERY task — one call per task
- Tasks should follow the Research → Spec → Coder pipeline
- The Critic reviews automatically — no need to create Critic tasks
- Use `depends_on_titles` for dependency ordering
- Task titles should be descriptive, unique, and consistent

**Temperature**: 0.7 (default) — balances creativity with consistency
**Few-shot**: Demonstrates breaking down a "todo app with auth" into research and coding tasks

### Research Agent

**File**: `prompts/research.txt`
**Goal**: Investigate technologies and produce actionable reports

Key patterns:
- Structure reports with clear sections: Technology Analysis → Architecture → Data Model → Challenges → Best Practices → Recommendations
- Always give specific recommendations with reasoning
- Include version numbers and compatibility notes
- Favor well-documented, widely-used technologies

**Temperature**: 0.6 — moderate creativity for analysis
**Few-shot**: Shows a JWT auth research report with specific library versions and data model

### Specification Agent

**File**: `prompts/spec.txt`
**Goal**: Write detailed technical specs

Key patterns:
- Every spec must include: Overview, Architecture (ASCII diagram), Data Models (typed fields), API Specification (exact endpoints with examples), File Structure, Implementation Notes, Testing Strategy
- Use code blocks for schemas, API definitions, and file structures
- Do NOT write implementation code — only the specification
- Every field in a data model must have a type and description

**Temperature**: 0.5 — lower for more deterministic output
**Few-shot**: Shows a complete todo API spec with SQLAlchemy models, endpoint docs, and file tree

### Coder Agent

**File**: `prompts/coder.txt`
**Goal**: Implement working code from specifications

Key patterns:
- Use the `write_file` tool for EVERY file — one call per file
- Write complete, production-ready code (no TODOs, no placeholders)
- Include type hints, docstrings, error handling, logging
- Generate tests alongside implementation
- Follow the specification exactly — don't add unrequested features

**Temperature**: 0.4 — low for precise implementation
**Tool**: `write_file` — called once per file with path, content, and artifact_type
**Few-shot**: Shows a FastAPI app main.py and lists the files to create

### Critic Agent

**File**: `prompts/critic.txt`
**Goal**: Review deliverables for quality and correctness

Key patterns:
- Use the `submit_review` tool for structured verdicts
- Grade issues: 🔴 Critical (blocking) / 🟡 Major (should fix) / 🔵 Minor (nice to have)
- Provide specific, actionable feedback with line-level references
- Be constructive — don't just criticize, suggest fixes

**Temperature**: 0.3 — low for consistent, deterministic reviews
**Few-shot**: Shows a review rejection with a critical issue (missing password hashing) and major issues

---

## Temperature and Top_P Tuning

| Parameter | Effect | Typical Range |
|---|---|---|
| `temperature` | Randomness. Higher = more creative, lower = more deterministic | 0.0–1.0 |
| `top_p` | Nucleus sampling. Lower = more focused | 0.0–1.0 |

### Recommended Settings by Task

| Task Type | Temperature | Top_P | Rationale |
|---|---|---|---|
| Task decomposition (PM) | 0.7 | 0.9 | Needs creativity to break down ideas |
| Research | 0.6 | 0.9 | Analytical but needs some exploration |
| Spec writing | 0.5 | 0.8 | Precise but requires some creative structuring |
| Code generation | 0.4 | 0.8 | Low variance for consistent code |
| Code fixes | 0.3 | 0.7 | Very low variance — just apply the fix |
| Code review | 0.3 | 0.7 | Consistent, deterministic |
| Questions / Answers | 0.5 | 0.9 | Moderate — needs explanation |

### Per-Agent Config

Each agent's temperature and top_p can be overridden in `.env`:

```ini
OLLAMA_RESEARCH_TEMPERATURE=0.6
OLLAMA_CODER_TEMPERATURE=0.4
OLLAMA_CRITIC_TEMPERATURE=0.3
OLLAMA_RESEARCH_TOP_P=0.9
OLLAMA_CODER_TOP_P=0.8
```

Setting a value to `0.0` means "use the default" (`OLLAMA_DEFAULT_TEMPERATURE` / `OLLAMA_DEFAULT_TOP_P`).

---

## Tool Calling Format

Three tools are defined in OpenAI function-calling format in `src/hca/core/tools.py`. All tools are available to all agents, but each agent is trained to use specific tools.

### `create_task` (PM Agent only)

```json
{
  "name": "create_task",
  "parameters": {
    "title": "Implement user registration",
    "description": "Create POST /api/auth/register with email and password...",
    "assigned_to": "coder",
    "priority": "high",
    "depends_on_titles": ["Design auth data model"]
  }
}
```

### `write_file` (Coder Agent only)

```json
{
  "name": "write_file",
  "parameters": {
    "path": "src/auth/routes.py",
    "content": "from fastapi import APIRouter\n\nrouter = APIRouter()\n...",
    "artifact_type": "code"
  }
}
```

### `submit_review` (Critic Agent only)

```json
{
  "name": "submit_review",
  "parameters": {
    "verdict": "needs_revision",
    "summary": "The implementation has a security vulnerability...",
    "issues": [
      {
        "severity": "critical",
        "description": "Passwords stored in plaintext",
        "suggestion": "Use passlib with bcrypt hashing"
      }
    ],
    "recommendations": "Add password hashing before merging"
  }
}
```

---

## Tool Call Validation

Every tool call is automatically validated against its JSON schema. If a call has missing required fields, wrong types, or invalid enum values, the agent retries the LLM with a descriptive error message.

### Validation Rules

| Rule | Example Error |
|---|---|
| Required fields must be present | `missing required field 'title'` |
| Enum values must be valid | `field 'assigned_to' has invalid value 'dev'. Allowed: research, spec, coder` |
| Field types must match | `field 'depends_on_titles' must be an array, got string` |
| Array items must be correct type | `field 'depends_on_titles'[0] must be a string, got number` |

### Retry Flow

1. Agent calls `think_with_tools()` with tool definitions
2. LLM returns tool calls
3. `validate_and_log()` checks all calls against their schemas
4. If invalid → LLM receives formatted errors and is asked to retry
5. If still invalid after retry → agent falls back to regex/string parsing

This means the LLM is given a chance to self-correct before the system resorts to fallback parsing.

### When to use few-shot vs plain instructions

- **Few-shot** is best for output format guidance (reports, specs, reviews)
- **Plain instructions** are better for behavioral rules (safety, constraints)
- Use 1–2 examples per prompt — more can confuse the model

### Best practices

- Examples should be realistic and complete, not truncated
- Input → Output pairs should clearly show the expected transformation
- Include both the structure AND the level of detail expected
- For tool-using agents, show the tool call invocation pattern
- Keep examples concise — long examples waste context window

---

## Conversation History

Agents maintain per-project conversation history. The history includes:
- The user's original prompt
- The agent's response
- All subsequent interactions in that project

### History Management

- Maximum 40 turns per project (`MAX_HISTORY_PER_PROJECT`)
- Auto-pruning: keeps the first 2 turns (early context) + most recent 38
- History is loaded before each `think()` call
- `auto_trim=True` on the Ollama client drops older history entries if the context window is exceeded

### Context Window

Default: 8192 tokens (`OLLAMA_NUM_CTX`). The client estimates token counts and trims messages to fit, keeping the system prompt and the latest user message.

---

## Common Pitfalls

### The model ignores tool calling instructions

- Ensure the model supports tool/function calling
- Tool descriptions should be clear about required parameters
- Use few-shot examples showing exact tool call syntax

### Responses are too verbose

- Lower temperature and top_p
- Add explicit length constraints in the prompt ("Keep responses under 500 words")
- Check that `max_tokens` is set appropriately

### The agent can't stay on task

- The system prompt may be too vague — add more behavioral rules
- The conversation history may be confusing — try `clear_history()`
- Increase temperature may cause drift — lower it

### Tool calls are malformed

- Verify the tool definitions are correct (matching OpenAI function-calling schema)
- Add validation fallbacks (regex parsing in PM, file markers in Coder)
- The Critic has a fallback for string-based approval detection

### Context window overflow

- Reduce `OLLAMA_NUM_CTX` (uses less memory)
- Reduce conversation history (`MAX_HISTORY_PER_PROJECT`)
- The auto-trim feature should handle this automatically

---

## Hot-Reloading Prompts

System prompts can be reloaded without restarting the server:

```python
agent.reload_prompt()  # Reads the .txt file from disk again
```

This is useful during development for iterating on prompt changes. Simply edit the `.txt` file and call `reload_prompt()` on the agent via the API or dashboard.

---

## Prompt Testing Checklist

When modifying a prompt, verify:

- [ ] Does the output format match what downstream agents expect?
- [ ] Are tool calling instructions clear and unambiguous?
- [ ] Do few-shot examples match real-world usage?
- [ ] Are behavioral rules specific enough to prevent off-task behavior?
- [ ] Does the temperature/top_p setting match the task type?
- [ ] Will the context window accommodate the prompt + history + response?
- [ ] Have you updated any agent code that depended on the old output format?
