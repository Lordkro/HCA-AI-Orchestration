"""Tests for Prometheus metrics recording.

Verifies that each record_* helper increments/decrements the correct
metric with the correct labels.
"""

from __future__ import annotations

from prometheus_client import REGISTRY

from hca.core import metrics


def _counter_value(name: str, labels: dict[str, str] | None = None) -> float:
    """Get the current value of a Counter metric."""
    sample = REGISTRY.get_sample_value(name, labels) if labels else REGISTRY.get_sample_value(name)
    return sample or 0.0


class TestOllamaMetrics:
    def test_record_ollama_request(self) -> None:
        metrics.record_ollama_request("qwen3:14b", "ok", 1.5, 100, 50)
        assert _counter_value("hca_ollama_requests_total", {"model": "qwen3:14b", "status": "ok"}) == 1.0

    def test_record_ollama_request_multiple(self) -> None:
        before = _counter_value("hca_ollama_requests_total", {"model": "test", "status": "error"})
        metrics.record_ollama_request("test", "error", 0.5, 0, 0)
        metrics.record_ollama_request("test", "error", 0.5, 0, 0)
        assert _counter_value("hca_ollama_requests_total", {"model": "test", "status": "error"}) >= before + 2


class TestBusMetrics:
    def test_record_bus_publish(self) -> None:
        before = _counter_value("hca_bus_messages_published_total", {"message_type": "test_type"})
        metrics.record_bus_publish("test_type")
        assert _counter_value("hca_bus_messages_published_total", {"message_type": "test_type"}) == before + 1

    def test_record_bus_consume(self) -> None:
        before = _counter_value("hca_bus_messages_consumed_total")
        metrics.record_bus_consume()
        assert _counter_value("hca_bus_messages_consumed_total") == before + 1

    def test_record_bus_dead_letter(self) -> None:
        before = _counter_value("hca_bus_messages_dead_lettered_total", {"reason": "expired"})
        metrics.record_bus_dead_letter("expired")
        assert _counter_value("hca_bus_messages_dead_lettered_total", {"reason": "expired"}) == before + 1

    def test_record_bus_error(self) -> None:
        before = _counter_value("hca_bus_errors_total", {"error_type": "timeout"})
        metrics.record_bus_error("timeout")
        assert _counter_value("hca_bus_errors_total", {"error_type": "timeout"}) == before + 1

    def test_record_bus_reconnect(self) -> None:
        before = _counter_value("hca_bus_reconnections_total")
        metrics.record_bus_reconnect()
        assert _counter_value("hca_bus_reconnections_total") == before + 1


class TestDbMetrics:
    def test_record_db_query(self) -> None:
        before = _counter_value("hca_db_queries_total", {"operation": "select"})
        metrics.record_db_query("select")
        assert _counter_value("hca_db_queries_total", {"operation": "select"}) == before + 1

    def test_record_db_error(self) -> None:
        before = _counter_value("hca_db_errors_total", {"operation": "insert"})
        metrics.record_db_error("insert")
        assert _counter_value("hca_db_errors_total", {"operation": "insert"}) == before + 1


class TestAgentMetrics:
    def test_record_agent_message_received(self) -> None:
        before = _counter_value("hca_agent_messages_received_total", {"agent": "pm"})
        metrics.record_agent_message_received("pm")
        assert _counter_value("hca_agent_messages_received_total", {"agent": "pm"}) == before + 1

    def test_record_agent_message_sent(self) -> None:
        before = _counter_value("hca_agent_messages_sent_total", {"agent": "coder"})
        metrics.record_agent_message_sent("coder")
        assert _counter_value("hca_agent_messages_sent_total", {"agent": "coder"}) == before + 1

    def test_record_agent_message_failed(self) -> None:
        before = _counter_value("hca_agent_messages_failed_total", {"agent": "critic"})
        metrics.record_agent_message_failed("critic")
        assert _counter_value("hca_agent_messages_failed_total", {"agent": "critic"}) == before + 1

    def test_record_agent_dead_lettered(self) -> None:
        before = _counter_value("hca_agent_messages_dead_lettered_total", {"agent": "pm"})
        metrics.record_agent_dead_lettered("pm")
        assert _counter_value("hca_agent_messages_dead_lettered_total", {"agent": "pm"}) == before + 1

    def test_record_agent_llm_call(self) -> None:
        before = _counter_value("hca_agent_llm_calls_total", {"agent": "spec"})
        metrics.record_agent_llm_call("spec")
        assert _counter_value("hca_agent_llm_calls_total", {"agent": "spec"}) == before + 1

    def test_record_agent_llm_error(self) -> None:
        before = _counter_value("hca_agent_llm_errors_total", {"agent": "research"})
        metrics.record_agent_llm_error("research")
        assert _counter_value("hca_agent_llm_errors_total", {"agent": "research"}) == before + 1

    def test_record_agent_llm_duration(self) -> None:
        metrics.record_agent_llm_duration("pm", 2.5)
