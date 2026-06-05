"""Workspace lifecycle and cleanup management.

Handles:
- Retention policies (max age, max count)
- Cleanup of old/orphaned workspaces
- Metrics on workspace usage
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from hca.core.config import settings

logger = structlog.get_logger()


class WorkspaceManager:
    """Manages workspace lifecycle and cleanup."""

    @staticmethod
    async def cleanup_old_workspaces() -> dict[str, int]:
        """Remove workspaces exceeding retention policy.

        Returns metrics on cleanup (count, size_freed_mb).
        """
        workspace_root = Path(settings.workspace_dir)
        if not workspace_root.exists():
            return {"cleaned": 0, "size_freed_mb": 0}

        # Collect workspace metadata
        workspaces: list[tuple[Path, datetime]] = []
        for ws_dir in workspace_root.iterdir():
            if not ws_dir.is_dir():
                continue
            stat_result = ws_dir.stat()
            mtime = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)
            workspaces.append((ws_dir, mtime))

        if not workspaces:
            return {"cleaned": 0, "size_freed_mb": 0}

        # Sort by modification time (newest first)
        workspaces.sort(key=lambda x: x[1], reverse=True)

        # Apply retention policies
        now = datetime.now(UTC)
        max_age = timedelta(days=settings.workspace_retention_days)
        to_remove: list[Path] = []

        for i, (ws_dir, mtime) in enumerate(workspaces):
            age = now - mtime
            is_too_old = age > max_age
            # Keep the first workspace_max_count newest workspaces, remove the rest
            is_beyond_limit = i >= settings.workspace_max_count

            if is_too_old or is_beyond_limit:
                to_remove.append(ws_dir)

        # Execute cleanup
        size_freed_mb = 0
        for ws_dir in to_remove:
            try:
                size_before = _get_dir_size(ws_dir)
                shutil.rmtree(ws_dir)
                size_freed_mb += size_before / (1024 * 1024)
                logger.info(
                    "workspace_removed",
                    workspace_id=ws_dir.name,
                    size_mb=size_before / (1024 * 1024),
                )
            except Exception as e:
                logger.error(
                    "workspace_cleanup_failed",
                    workspace_id=ws_dir.name,
                    error=str(e),
                )

        result = {
            "cleaned": len(to_remove),
            "size_freed_mb": int(size_freed_mb),
            "total_workspaces": len(workspaces),
            "remaining": len(workspaces) - len(to_remove),
        }
        logger.info("workspace_cleanup_complete", **result)
        return result

    @staticmethod
    async def get_workspace_stats() -> dict:
        """Get current workspace usage statistics."""
        workspace_root = Path(settings.workspace_dir)
        if not workspace_root.exists():
            return {
                "total_count": 0,
                "total_size_mb": 0,
                "oldest_age_days": 0,
                "newest_age_days": 0,
            }

        workspaces = [d for d in workspace_root.iterdir() if d.is_dir()]
        if not workspaces:
            return {
                "total_count": 0,
                "total_size_mb": 0,
                "oldest_age_days": 0,
                "newest_age_days": 0,
            }

        now = datetime.now(UTC)
        mtimes = [
            datetime.fromtimestamp(ws.stat().st_mtime, tz=UTC)
            for ws in workspaces
        ]
        sizes = [_get_dir_size(ws) for ws in workspaces]

        oldest_age = (now - min(mtimes)).days
        newest_age = (now - max(mtimes)).days

        return {
            "total_count": len(workspaces),
            "total_size_mb": int(sum(sizes) / (1024 * 1024)),
            "avg_size_mb": int(sum(sizes) / len(sizes) / (1024 * 1024)),
            "oldest_age_days": oldest_age,
            "newest_age_days": newest_age,
        }


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total
