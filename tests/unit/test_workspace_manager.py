"""Tests for WorkspaceManager — workspace lifecycle and cleanup.

Uses temp directories to simulate workspace directories without touching
real project data.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from hca.orchestrator.workspace_manager import WorkspaceManager, _get_dir_size

_HAVE_GIT: bool = (
    __import__("shutil").which("git") is not None
)


class TestGetDirSize:
    def test_empty_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert _get_dir_size(d) == 0

    def test_single_file(self, tmp_path: Path) -> None:
        d = tmp_path / "single"
        d.mkdir()
        (d / "file.txt").write_text("hello")
        assert _get_dir_size(d) > 0

    def test_nested_directories(self, tmp_path: Path) -> None:
        d = tmp_path / "nested"
        d.mkdir()
        sub = d / "sub"
        sub.mkdir()
        (sub / "data.bin").write_text("x" * 1000)
        assert _get_dir_size(d) >= 1000

    def test_non_existent_path(self, tmp_path: Path) -> None:
        assert _get_dir_size(tmp_path / "nonexistent") == 0


class TestCleanupOldWorkspaces:
    async def _create_workspace(self, root: Path, name: str, age_hours: float = 0) -> Path:
        ws = root / name
        ws.mkdir(parents=True)
        (ws / "main.py").write_text("print('hello')")
        # Set mtime to simulate age
        old_mtime = time.time() - (age_hours * 3600)
        os.utime(ws, (old_mtime, old_mtime))
        return ws

    @pytest.mark.asyncio
    async def test_no_workspace_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", "/nonexistent/path")
        result = await WorkspaceManager.cleanup_old_workspaces()
        assert result == {"cleaned": 0, "size_freed_mb": 0}

    @pytest.mark.asyncio
    async def test_empty_workspace_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        result = await WorkspaceManager.cleanup_old_workspaces()
        assert result == {"cleaned": 0, "size_freed_mb": 0}

    @pytest.mark.asyncio
    async def test_cleans_old_workspaces(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_retention_days", 1)
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_max_count", 100)

        await self._create_workspace(tmp_path, "old-project", age_hours=48)
        await self._create_workspace(tmp_path, "new-project", age_hours=1)

        result = await WorkspaceManager.cleanup_old_workspaces()
        assert result["cleaned"] == 1
        assert not (tmp_path / "old-project").exists()
        assert (tmp_path / "new-project").exists()

    @pytest.mark.asyncio
    async def test_cleans_beyond_max_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_retention_days", 30)
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_max_count", 2)

        _ = await self._create_workspace(tmp_path, "ws-1", age_hours=1)
        _ = await self._create_workspace(tmp_path, "ws-2", age_hours=2)
        _ = await self._create_workspace(tmp_path, "ws-3", age_hours=3)

        result = await WorkspaceManager.cleanup_old_workspaces()
        assert result["cleaned"] == 1  # ws-3 is oldest, should be removed
        assert result["remaining"] == 2

    @pytest.mark.asyncio
    async def test_keeps_all_when_within_limits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_retention_days", 7)
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_max_count", 100)

        await self._create_workspace(tmp_path, "ws-1", age_hours=1)
        await self._create_workspace(tmp_path, "ws-2", age_hours=2)

        result = await WorkspaceManager.cleanup_old_workspaces()
        assert result["cleaned"] == 0
        assert result["remaining"] == 2


class TestGetWorkspaceStats:
    async def _create_workspace(self, root: Path, name: str, age_hours: float = 0) -> None:
        ws = root / name
        ws.mkdir(parents=True)
        (ws / "main.py").write_text("content")

    @pytest.mark.asyncio
    async def test_no_workspace_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", "/nonexistent")
        stats = await WorkspaceManager.get_workspace_stats()
        assert stats["total_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        stats = await WorkspaceManager.get_workspace_stats()
        assert stats["total_count"] == 0

    @pytest.mark.asyncio
    async def test_with_workspaces(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await self._create_workspace(tmp_path, "ws-1", age_hours=1)
        await self._create_workspace(tmp_path, "ws-2", age_hours=48)

        stats = await WorkspaceManager.get_workspace_stats()
        assert stats["total_count"] == 2
        assert stats["total_size_mb"] >= 0
        assert stats["avg_size_mb"] >= 0


# ============================================================
# Git Integration Tests
# ============================================================


pytestmark = pytest.mark.skipif(not _HAVE_GIT, reason="git not available")


class TestGitInit:
    """Tests for init_project_repo."""

    @pytest.mark.asyncio
    async def test_init_new_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        result = await WorkspaceManager.init_project_repo("proj-1")
        assert result is True
        assert (tmp_path / "proj-1" / ".git").exists()

    @pytest.mark.asyncio
    async def test_init_twice_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        assert await WorkspaceManager.init_project_repo("proj-1") is True
        assert await WorkspaceManager.init_project_repo("proj-1") is False

    @pytest.mark.asyncio
    async def test_init_creates_gitignore(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        gitignore = ws / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()
        assert "__pycache__" in content
        assert ".env" in content


class TestGitCommit:
    """Tests for commit_workspace."""

    @pytest.mark.asyncio
    async def test_commit_with_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "test.txt").write_text("hello")
        result = await WorkspaceManager.commit_workspace("proj-1", "initial commit")
        assert result is True

    @pytest.mark.asyncio
    async def test_commit_no_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "placeholder").write_text("init")
        await WorkspaceManager.commit_workspace("proj-1", "init")  # first commit
        result = await WorkspaceManager.commit_workspace("proj-1", "no-op")
        assert result is False

    @pytest.mark.asyncio
    async def test_commit_with_tag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "app.py").write_text("print('hello')")
        result = await WorkspaceManager.commit_workspace("proj-1", "with tag", tag="task-123")
        assert result is True
        # Verify tag exists
        proc = await asyncio.create_subprocess_exec(
            "git", "tag", "-l", "task-123",
            stdout=asyncio.subprocess.PIPE,
            cwd=str(ws),
        )
        stdout, _ = await proc.communicate()
        assert stdout.decode().strip() == "task-123"

    @pytest.mark.asyncio
    async def test_commit_auto_inits_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        ws = tmp_path / "proj-1"
        ws.mkdir(parents=True)
        (ws / "data.txt").write_text("content")
        # No prior init — commit should auto-init
        result = await WorkspaceManager.commit_workspace("proj-1", "auto init commit")
        assert result is True
        assert (ws / ".git").exists()


class TestGitLog:
    """Tests for get_workspace_log."""

    @pytest.mark.asyncio
    async def test_log_empty_no_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        log = await WorkspaceManager.get_workspace_log("nonexistent")
        assert log == []

    @pytest.mark.asyncio
    async def test_log_no_commits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        log = await WorkspaceManager.get_workspace_log("proj-1")
        assert log == []

    @pytest.mark.asyncio
    async def test_log_after_commits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "a.txt").write_text("a")
        await WorkspaceManager.commit_workspace("proj-1", "first")
        (ws / "b.txt").write_text("b")
        await WorkspaceManager.commit_workspace("proj-1", "second")
        log = await WorkspaceManager.get_workspace_log("proj-1", n=5)
        assert len(log) == 2
        assert log[0]["message"] == "second"
        assert log[1]["message"] == "first"
        assert len(log[0]["hash"]) == 12

    @pytest.mark.asyncio
    async def test_log_includes_tags(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "main.py").write_text("code")
        await WorkspaceManager.commit_workspace("proj-1", "tagged", tag="t-42")
        log = await WorkspaceManager.get_workspace_log("proj-1")
        assert len(log) == 1
        assert "t-42" in log[0]["tags"]


class TestGitPush:
    """Tests for push_to_github."""

    @pytest.mark.asyncio
    async def test_push_no_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        result = await WorkspaceManager.push_to_github(
            "nonexistent", "https://github.com/owner/repo", token="fake"  # noqa: S106
        )
        assert result["success"] is False
        assert "No git repository" in result["message"]

    @pytest.mark.asyncio
    async def test_push_no_token(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.github_token", "")
        ws = tmp_path / "proj-1"
        ws.mkdir(parents=True)
        await WorkspaceManager.init_project_repo("proj-1")
        (ws / "f.txt").write_text("data")
        await WorkspaceManager.commit_workspace("proj-1", "init")
        result = await WorkspaceManager.push_to_github(
            "proj-1", "https://github.com/owner/repo", token=None
        )
        assert result["success"] is False
        assert "No GitHub token" in result["message"]

    @pytest.mark.asyncio
    async def test_push_sets_remote_and_pushes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use a local bare repo as a push target to verify end-to-end."""
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.github_token", "test-token")

        ws = tmp_path / "proj-1"
        ws.mkdir(parents=True)
        await WorkspaceManager.init_project_repo("proj-1")
        (ws / "f.txt").write_text("data")
        await WorkspaceManager.commit_workspace("proj-1", "init")

        # Create local bare repo as push target
        bare = tmp_path / "target.git"
        proc = await asyncio.create_subprocess_exec(
            "git", "init", "--bare", str(bare),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

        result = await WorkspaceManager.push_to_github(
            "proj-1", str(bare), token="irrelevant",  # noqa: S106
        )
        assert result["success"] is True, result["message"]
        # Verify the commit is reachable from the bare repo
        log_proc = await asyncio.create_subprocess_exec(
            "git", "log", "--oneline", "-1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(bare),
        )
        stdout, _ = await log_proc.communicate()
        assert stdout.decode().strip(), "Bare repo should have at least one commit"


class TestGitDiff:
    """Tests for get_workspace_diff."""

    @pytest.mark.asyncio
    async def test_diff_no_repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        diff = await WorkspaceManager.get_workspace_diff("nonexistent")
        assert diff == ""

    @pytest.mark.asyncio
    async def test_diff_no_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "f.txt").write_text("content")
        await WorkspaceManager.commit_workspace("proj-1", "init")
        diff = await WorkspaceManager.get_workspace_diff("proj-1")
        assert diff == ""

    @pytest.mark.asyncio
    async def test_diff_with_uncommitted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        await WorkspaceManager.init_project_repo("proj-1")
        ws = tmp_path / "proj-1"
        (ws / "f.txt").write_text("content")
        await WorkspaceManager.commit_workspace("proj-1", "init")
        (ws / "f.txt").write_text("modified")
        diff = await WorkspaceManager.get_workspace_diff("proj-1")
        assert "modified" in diff or "+modified" in diff


class TestGitFileList:
    """Tests for get_workspace_file_list."""

    @pytest.mark.asyncio
    async def test_file_list_no_workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        files = await WorkspaceManager.get_workspace_file_list("nonexistent")
        assert files == []

    @pytest.mark.asyncio
    async def test_file_list_with_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("hca.orchestrator.workspace_manager.settings.workspace_dir", str(tmp_path))
        ws = tmp_path / "proj-1"
        ws.mkdir(parents=True)
        (ws / "main.py").write_text("code")
        (ws / "README.md").write_text("docs")
        sub = ws / "lib"
        sub.mkdir()
        (sub / "util.py").write_text("helper")
        files = await WorkspaceManager.get_workspace_file_list("proj-1")
        paths = [f["path"] for f in files]
        assert "main.py" in paths
        assert "README.md" in paths
        assert "lib/util.py" in paths
        assert all(f["size"] > 0 for f in files)
