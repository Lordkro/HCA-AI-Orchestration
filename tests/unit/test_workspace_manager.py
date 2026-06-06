"""Tests for WorkspaceManager — workspace lifecycle and cleanup.

Uses temp directories to simulate workspace directories without touching
real project data.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hca.orchestrator.workspace_manager import WorkspaceManager, _get_dir_size


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
