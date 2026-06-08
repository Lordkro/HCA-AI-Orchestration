"""Workspace lifecycle and cleanup management.

Handles:
- Retention policies (max age, max count)
- Cleanup of old/orphaned workspaces
- Metrics on workspace usage
- Git integration per project workspace (init, commit, log, diff, tag, .gitignore)
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from hca.core.config import settings

logger = structlog.get_logger()


class WorkspaceManager:
    """Manages workspace lifecycle and cleanup."""

    @staticmethod
    def _workspace_path(project_id: str) -> Path:
        """Get the filesystem path for a project workspace."""
        root = Path(settings.workspace_dir)
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        return (root / project_id).resolve()

    @staticmethod
    async def init_project_repo(project_id: str) -> bool:
        """Initialise a git repository in the project workspace.

        Safe to call multiple times — skips if .git already exists.
        Writes a sensible .gitignore to exclude caches and env files.
        Returns True if a new repo was initialised.
        """
        ws = WorkspaceManager._workspace_path(project_id)
        if not ws.exists():
            ws.mkdir(parents=True, exist_ok=True)

        git_dir = ws / ".git"
        if git_dir.exists():
            return False  # Already a repo

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "init",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            rc = await proc.wait()
            if rc != 0:
                logger.warning("git_init_failed", project_id=project_id, returncode=rc)
                return False

            # Set local git config for commit authorship
            await asyncio.create_subprocess_exec(
                "git", "config", "user.name", "HCA Agent",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            await asyncio.create_subprocess_exec(
                "git", "config", "user.email", "agent@hca.local",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )

            # Write sensible .gitignore for generated/cache files
            gitignore = ws / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "# HCA workspace — generated files to exclude\n"
                    "__pycache__/\n"
                    "*.pyc\n"
                    "*.pyo\n"
                    "*.egg-info/\n"
                    ".env\n"
                    ".venv/\n"
                    "venv/\n"
                    "node_modules/\n"
                    ".DS_Store\n"
                    "*.log\n"
                    ".data/\n"
                )

            logger.info("git_repo_initialised", project_id=project_id)
            return True
        except FileNotFoundError:
            logger.warning("git_not_available", project_id=project_id)
            return False

    @staticmethod
    async def commit_workspace(project_id: str, message: str, tag: str = "") -> bool:
        """Stage all changes and commit to the project git repo.

        Optionally tags the commit with a lightweight tag (e.g. a task ID).

        Returns True if a commit was made (has changes), False otherwise.
        """
        ws = WorkspaceManager._workspace_path(project_id)
        if not (ws / ".git").exists():
            inited = await WorkspaceManager.init_project_repo(project_id)
            if not inited:
                return False

        try:
            # git add -A
            add_proc = await asyncio.create_subprocess_exec(
                "git", "add", "-A",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            rc_add = await add_proc.wait()
            if rc_add != 0:
                logger.warning("git_add_failed", project_id=project_id, returncode=rc_add)
                return False

            # Check if anything changed
            diff_proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--cached", "--quiet",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            rc_diff = await diff_proc.wait()
            if rc_diff == 0:
                return False

            # git commit
            commit_proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", message,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            rc_commit = await commit_proc.wait()
            if rc_commit != 0:
                logger.warning("git_commit_failed", project_id=project_id, returncode=rc_commit)
                return False

            # Optional lightweight tag
            if tag:
                tag_proc = await asyncio.create_subprocess_exec(
                    "git", "tag", "-f", tag,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(ws),
                )
                await tag_proc.wait()

            logger.info("git_commit_created", project_id=project_id, message=message, tag=tag or None)
            return True
        except FileNotFoundError:
            logger.warning("git_not_available", project_id=project_id)
            return False

    @staticmethod
    async def get_workspace_diff(project_id: str) -> str:
        """Get the git diff of uncommitted changes in the project workspace.

        Returns empty string if no diff or git is unavailable.
        """
        ws = WorkspaceManager._workspace_path(project_id)
        if not (ws / ".git").exists():
            return ""

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            stdout, _ = await proc.communicate()
            return stdout.decode("utf-8", errors="replace").strip()
        except FileNotFoundError:
            return ""

    @staticmethod
    async def get_workspace_log(project_id: str, n: int = 10) -> list[dict[str, str]]:
        """Get the recent commit log for a project workspace.

        Returns a list of dicts with ``hash``, ``author``, ``date``, ``message``, ``tags``.
        Empty list if git is unavailable or the repo has no commits.
        """
        ws = WorkspaceManager._workspace_path(project_id)
        if not (ws / ".git").exists():
            return []

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "log", f"-{n}",
                "--format=%x00%H%n%an%n%aI%n%s%n%D",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(ws),
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return []

            raw = stdout.decode("utf-8", errors="replace").lstrip("\x00")
            if not raw.strip():
                return []

            entries: list[dict[str, str]] = []
            for block in raw.split("\x00"):
                block = block.strip()
                if not block:
                    continue
                lines = block.splitlines()
                # Expect 4-5 lines: hash, author, date, msg [, tags]
                if len(lines) < 4:
                    continue
                tags = lines[4].strip() if len(lines) > 4 else ""
                entries.append({
                    "hash": lines[0][:12],
                    "author": lines[1],
                    "date": lines[2],
                    "message": lines[3],
                    "tags": tags,
                })
            return entries
        except FileNotFoundError:
            return []

    @staticmethod
    async def get_workspace_file_list(project_id: str) -> list[dict[str, object]]:
        """List files in the workspace with size and modification time.

        Returns a list of dicts with ``path``, ``size``, ``mtime``.
        Empty list if workspace does not exist.
        """
        ws = WorkspaceManager._workspace_path(project_id)
        if not ws.exists():
            return []

        files: list[dict[str, object]] = []
        for entry in sorted(ws.rglob("*"), key=lambda p: str(p)):
            if not entry.is_file() or entry.name.startswith("."):
                continue
            stat = entry.stat()
            files.append({
                "path": str(entry.relative_to(ws)),
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            })
        return files

    # ---------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------

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
        size_freed_mb = 0.0
        for ws_dir in to_remove:
            try:
                size_before = await asyncio.to_thread(_get_dir_size, ws_dir)
                await asyncio.to_thread(shutil.rmtree, ws_dir)
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
        sizes = [await asyncio.to_thread(_get_dir_size, ws) for ws in workspaces]

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
