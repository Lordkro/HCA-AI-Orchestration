"""Prometheus metrics for HCA Orchestration.

All metrics are module-level singletons registered with the default
Prometheus registry.  Import this module to ensure metrics exist,
then call the record_* helpers from service classes.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ============================================================
# Ollama / LLM metrics
# ============================================================

ollama_requests_total = Counter(
    "hca_ollama_requests_total",
    "Total LLM requests by model and status",
    ["model", "status"],
)

ollama_request_duration_seconds = Histogram(
    "hca_ollama_request_duration_seconds",
    "LLM request latency by model",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

ollama_tokens_total = Counter(
    "hca_ollama_tokens_total",
    "Total tokens processed by model and type (prompt/completion)",
    ["model", "type"],
)

ollama_up = Gauge(
    "hca_ollama_up",
    "1 if Ollama server is reachable, 0 otherwise",
)

ollama_concurrent_requests = Gauge(
    "hca_ollama_concurrent_requests",
    "Current number of in-flight LLM requests",
)

ollama_circuit_breaker_state = Gauge(
    "hca_ollama_circuit_breaker_state",
    "Circuit breaker state: 0=closed, 1=half-open, 2=open",
)

ollama_circuit_breaker_tripped_total = Counter(
    "hca_ollama_circuit_breaker_tripped_total",
    "Total times the circuit breaker has opened",
)

# ============================================================
# Message Bus metrics
# ============================================================

bus_messages_published_total = Counter(
    "hca_bus_messages_published_total",
    "Messages published by type",
    ["message_type"],
)

bus_messages_consumed_total = Counter(
    "hca_bus_messages_consumed_total",
    "Messages consumed from streams",
)

bus_messages_dead_lettered_total = Counter(
    "hca_bus_messages_dead_lettered_total",
    "Messages moved to dead-letter by reason",
    ["reason"],
)

bus_errors_total = Counter(
    "hca_bus_errors_total",
    "Bus errors by type",
    ["error_type"],
)

bus_reconnections_total = Counter(
    "hca_bus_reconnections_total",
    "Total Redis reconnection events",
)

bus_connected = Gauge(
    "hca_bus_connected",
    "1 if the message bus is connected to Redis, 0 otherwise",
)

# ============================================================
# Database metrics
# ============================================================

db_queries_total = Counter(
    "hca_db_queries_total",
    "Total database queries",
    ["operation"],
)

db_errors_total = Counter(
    "hca_db_errors_total",
    "Total database errors",
    ["operation"],
)

db_size_bytes = Gauge(
    "hca_db_size_bytes",
    "Database file size in bytes",
)

db_project_count = Gauge(
    "hca_db_project_count",
    "Number of projects by status",
    ["status"],
)

db_task_count = Gauge(
    "hca_db_task_count",
    "Number of tasks by state",
    ["state"],
)

db_connected = Gauge(
    "hca_db_connected",
    "1 if the database connection is open, 0 otherwise",
)

# ============================================================
# Agent metrics
# ============================================================

agent_messages_received_total = Counter(
    "hca_agent_messages_received_total",
    "Messages received by agent",
    ["agent"],
)

agent_messages_sent_total = Counter(
    "hca_agent_messages_sent_total",
    "Messages sent by agent",
    ["agent"],
)

agent_messages_failed_total = Counter(
    "hca_agent_messages_failed_total",
    "Message processing failures by agent",
    ["agent"],
)

agent_messages_dead_lettered_total = Counter(
    "hca_agent_messages_dead_lettered_total",
    "Dead-lettered messages by agent",
    ["agent"],
)

agent_llm_calls_total = Counter(
    "hca_agent_llm_calls_total",
    "LLM calls by agent",
    ["agent"],
)

agent_llm_errors_total = Counter(
    "hca_agent_llm_errors_total",
    "LLM errors by agent",
    ["agent"],
)

agent_llm_duration_seconds = Histogram(
    "hca_agent_llm_duration_seconds",
    "LLM call latency by agent",
    ["agent"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

agent_status = Gauge(
    "hca_agent_status",
    "Agent status: 0=idle, 1=busy, 2=error",
    ["agent"],
)

agent_uptime_seconds = Gauge(
    "hca_agent_uptime_seconds",
    "Seconds since agent started",
    ["agent"],
)

# ============================================================
# API metrics
# ============================================================

api_requests_total = Counter(
    "hca_api_requests_total",
    "Total API requests by method, path, and status",
    ["method", "path", "status"],
)

api_request_duration_seconds = Histogram(
    "hca_api_request_duration_seconds",
    "API request latency by method and path",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

api_requests_in_flight = Gauge(
    "hca_api_requests_in_flight",
    "Current number of in-flight API requests",
    ["method"],
)

# ============================================================
# Record helpers (called by service classes)
# ============================================================


def record_ollama_request(
    model: str, status: str, duration_s: float, prompt_tokens: int, completion_tokens: int
) -> None:
    ollama_requests_total.labels(model=model, status=status).inc()
    ollama_request_duration_seconds.labels(model=model).observe(duration_s)
    if prompt_tokens:
        ollama_tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
    if completion_tokens:
        ollama_tokens_total.labels(model=model, type="completion").inc(completion_tokens)


def record_bus_publish(message_type: str) -> None:
    bus_messages_published_total.labels(message_type=message_type).inc()


def record_bus_consume() -> None:
    bus_messages_consumed_total.inc()


def record_bus_dead_letter(reason: str) -> None:
    bus_messages_dead_lettered_total.labels(reason=reason).inc()


def record_bus_error(error_type: str) -> None:
    bus_errors_total.labels(error_type=error_type).inc()


def record_bus_reconnect() -> None:
    bus_reconnections_total.inc()


def record_db_query(operation: str) -> None:
    db_queries_total.labels(operation=operation).inc()


def record_db_error(operation: str) -> None:
    db_errors_total.labels(operation=operation).inc()


def record_agent_message_received(agent: str) -> None:
    agent_messages_received_total.labels(agent=agent).inc()


def record_agent_message_sent(agent: str) -> None:
    agent_messages_sent_total.labels(agent=agent).inc()


def record_agent_message_failed(agent: str) -> None:
    agent_messages_failed_total.labels(agent=agent).inc()


def record_agent_dead_lettered(agent: str) -> None:
    agent_messages_dead_lettered_total.labels(agent=agent).inc()


def record_agent_llm_call(agent: str) -> None:
    agent_llm_calls_total.labels(agent=agent).inc()


def record_agent_llm_error(agent: str) -> None:
    agent_llm_errors_total.labels(agent=agent).inc()


def record_agent_llm_duration(agent: str, duration_s: float) -> None:
    agent_llm_duration_seconds.labels(agent=agent).observe(duration_s)
