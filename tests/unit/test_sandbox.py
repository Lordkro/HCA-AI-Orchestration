"""Tests for the sandboxed code execution module."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from hca.agents.coder_agent import CoderAgent
from hca.agents.critic_agent import CriticAgent
from hca.core.database import Database
from hca.core.models import AgentRole, MessageType, TaskState
from hca.orchestrator.sandbox import SandboxExecutor, SandboxResult
from hca.orchestrator.task_manager import TaskManager
from tests.conftest import MockMessageBus, MockOllamaClient, make_message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _patch_workspace(tmp_path: Path, project_id: str) -> Path:
    """Temporarily override settings.workspace_dir and create the project dir."""
    ws_dir = tmp_path / "workspaces" / project_id
    ws_dir.mkdir(parents=True, exist_ok=True)
    with patch("hca.orchestrator.sandbox.settings.workspace_dir", str(tmp_path / "workspaces")):
        yield ws_dir


async def _setup_project(db: Database) -> object:
    """Create a minimal project for agent tests."""
    from hca.core.models import Project

    project = Project(
        name="Test Project",
        description="A test project",
        idea="Build a test project",
    )
    await db.create_project(project)
    return project


# ---------------------------------------------------------------------------
# Basic unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_missing_workspace() -> None:
    """validate_project returns error for nonexistent projects."""
    executor = SandboxExecutor()
    result = await executor.validate_project("nonexistent")
    assert result.error == "workspace_not_found"
    assert not result.passed


@pytest.mark.asyncio
async def test_sandbox_graceful_degradation_no_docker(tmp_path: Path) -> None:
    """When Docker is unavailable, validate_project passes gracefully."""
    executor = SandboxExecutor()
    executor._docker_available = False
    with _patch_workspace(tmp_path, "test-degradation") as ws:
        (ws / "main.py").write_text("print('hello')\n")
        result = await executor.validate_project("test-degradation")
        assert result.passed
        assert result.error in ("docker_unavailable",)


@pytest.mark.asyncio
async def test_sandbox_detects_python_project(tmp_path: Path) -> None:
    """Python project detection works even when Docker is unavailable."""
    executor = SandboxExecutor()
    executor._docker_available = False
    with _patch_workspace(tmp_path, "py-project") as ws:
        (ws / "main.py").write_text("print('hello')\n")
        (ws / "utils.py").write_text("def foo(): return 42\n")
        result = await executor.validate_project("py-project")
        assert result.passed


@pytest.mark.asyncio
async def test_sandbox_result_to_dict() -> None:
    """SandboxResult.to_dict returns expected keys."""
    r = SandboxResult()
    r.passed = True
    r.syntax_check = "OK"
    d = r.to_dict()
    assert d["passed"] is True
    assert d["syntax_check"] == "OK"
    assert d["import_check"] == ""
    assert "error" in d


@pytest.mark.asyncio
async def test_sandbox_detects_entrypoints(tmp_path: Path) -> None:
    """Entrypoint detection finds main.py as a candidate."""
    executor = SandboxExecutor()
    executor._docker_available = False
    with _patch_workspace(tmp_path, "entrypoints") as ws:
        (ws / "main.py").write_text("if __name__ == '__main__':\n    pass\n")
        (ws / "module.py").write_text("SOME_CONST = 42\n")
        result = await executor.validate_project("entrypoints")
        assert result.passed


@pytest.mark.asyncio
async def test_sandbox_unknown_language(tmp_path: Path) -> None:
    """Projects with no recognised files return unknown language error."""
    executor = SandboxExecutor()
    executor._docker_available = True
    with _patch_workspace(tmp_path, "unknown-lang") as ws:
        (ws / "readme.md").write_text("# Project\n")
        result = await executor.validate_project("unknown-lang")
        assert not result.passed
        assert "unsupported_language" in result.error


# ---------------------------------------------------------------------------
# CoderAgent integration: sandbox results in deliverable metadata
# ---------------------------------------------------------------------------


class TestCoderSandboxIntegration:
    """SandboxExecutor is called after CoderAgent writes code and results
    are attached to the deliverable metadata."""

    async def test_coder_sandbox_metadata_present(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient, tmp_path: Path
    ) -> None:
        """Deliverable from Coder includes sandbox_* metadata keys."""
        ws_dir = str(tmp_path / "ws")
        with (
            patch("hca.agents.coder_agent.settings.workspace_dir", ws_dir),
            patch("hca.orchestrator.sandbox.settings.workspace_dir", ws_dir),
        ):
            project = await _setup_project(db)
            tm = TaskManager(db=db, bus=mock_bus)
            task = await tm.create_task(
                project_id=project.id,
                title="Code task",
                description="Write code",
                assigned_to=AgentRole.CODER,
            )
            await tm.transition(task.id, TaskState.ASSIGNED)

            mock_ollama.default_response = (
                '=== FILE: main.py ===\n```python\nprint("hello")\n```'
            )

            agent = CoderAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)
            msg = make_message(
                sender=AgentRole.PM,
                recipient=AgentRole.CODER,
                msg_type=MessageType.TASK_ASSIGNMENT,
                project_id=project.id,
                task_id=task.id,
                content="Implement the backend",
            )
            result = await agent.process_message(msg)

            assert result is not None
            meta = result.payload.metadata
            assert "sandbox_passed" in meta
            assert "sandbox_error" in meta

    async def test_coder_sandbox_skipped_no_artifacts(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """Sandbox is skipped when no artifacts were produced."""
        project = await _setup_project(db)
        tm = TaskManager(db=db, bus=mock_bus)
        task = await tm.create_task(
            project_id=project.id,
            title="Code task",
            description="Write code",
            assigned_to=AgentRole.CODER,
        )
        await tm.transition(task.id, TaskState.ASSIGNED)

        mock_ollama.default_response = "I cannot write code for this task."

        agent = CoderAgent(bus=mock_bus, ollama=mock_ollama, db=db, task_manager=tm)
        msg = make_message(
            sender=AgentRole.PM,
            recipient=AgentRole.CODER,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id=project.id,
            task_id=task.id,
            content="Write something",
        )
        result = await agent.process_message(msg)

        assert result is not None
        meta = result.payload.metadata
        assert "sandbox_passed" not in meta


# ---------------------------------------------------------------------------
# CriticAgent integration: sandbox results in review context
# ---------------------------------------------------------------------------


class TestCriticSandboxIntegration:
    """SandboxExecutor is called before Critic reviews code artifacts."""

    async def test_critic_includes_sandbox_for_code(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """Critic prompt includes sandbox context when reviewing code."""
        mock_ollama.default_response = "**APPROVED**\n\nLooks good."

        agent = CriticAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        msg = make_message(
            sender=AgentRole.CODER,
            recipient=AgentRole.CRITIC,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-code-review",
            content="Some code output",
            metadata={"artifact_type": "code"},
        )
        result = await agent.process_message(msg)

        assert result is not None
        meta = result.payload.metadata
        # Critic should have sandbox metadata for code reviews
        assert meta.get("review_result") in ("approved", "needs_revision")
        assert meta.get("artifact_type") == "code"

    async def test_critic_skips_sandbox_for_spec(
        self, db: Database, mock_bus: MockMessageBus, mock_ollama: MockOllamaClient
    ) -> None:
        """Critic does not run sandbox when reviewing non-code artifacts."""
        mock_ollama.default_response = "**APPROVED**\n\nSpec looks good."

        agent = CriticAgent(bus=mock_bus, ollama=mock_ollama, db=db)
        msg = make_message(
            sender=AgentRole.SPEC,
            recipient=AgentRole.CRITIC,
            msg_type=MessageType.TASK_ASSIGNMENT,
            project_id="proj-spec",
            content="A specification",
            metadata={"artifact_type": "specification"},
        )
        result = await agent.process_message(msg)

        assert result is not None
        meta = result.payload.metadata
        # No sandbox metadata for non-code artifacts
        assert "sandbox_passed" not in meta
