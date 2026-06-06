"""Sandboxed code execution — validates generated code in isolated containers.

Uses Docker to spin up ephemeral containers that execute basic smoke tests
on generated projects (import checks, syntax validation, dependency checks).

If Docker is unavailable, all checks gracefully degrade to no-ops.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from hca.core.config import settings

logger = structlog.get_logger()

# Languages we know how to test
SUPPORTED_LANGUAGES = {"python", "javascript", "typescript"}

# File extensions we consider "runnable" entrypoints
_RUNNABLE_EXTENSIONS = {".py", ".js", ".ts", ".sh"}


class SandboxResult:
    """Result of a sandboxed code execution check."""

    def __init__(self) -> None:
        self.passed: bool = False
        self.import_check: str = ""
        self.syntax_check: str = ""
        self.smoke_test: str = ""
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "import_check": self.import_check,
            "syntax_check": self.syntax_check,
            "smoke_test": self.smoke_test,
            "error": self.error,
        }


class SandboxExecutor:
    """Executes generated code in isolated Docker containers.

    Usage::

        executor = SandboxExecutor()
        result = await executor.validate_project("project-id")

    """

    def __init__(self) -> None:
        self._docker_available: bool | None = None

    async def _check_docker(self) -> bool:
        """Check if Docker is available on the host."""
        if self._docker_available is not None:
            return self._docker_available
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()
            self._docker_available = rc == 0
        except FileNotFoundError:
            self._docker_available = False
        return self._docker_available

    async def validate_project(self, project_id: str) -> SandboxResult:
        """Run all validation checks on a project workspace.

        Steps:
        1. Detect project language
        2. Syntax check
        3. Import check (try importing modules)
        4. Smoke test (run entrypoint briefly)

        Returns a SandboxResult with details of each check.
        """
        result = SandboxResult()
        ws = Path(settings.workspace_dir) / project_id
        if not ws.exists():
            result.error = "workspace_not_found"
            return result

        language, entrypoints = self._detect_project_language(ws)

        docker_ok = await self._check_docker()
        if not docker_ok:
            result.passed = True  # Graceful degradation
            result.error = "docker_unavailable"
            logger.info("sandbox_docker_unavailable", project_id=project_id)
            return result

        try:
            if language == "python":
                await self._validate_python(result, ws, entrypoints)
            else:
                result.error = f"unsupported_language:{language}"
                return result
        except Exception as e:
            result.error = str(e)
            logger.error("sandbox_error", project_id=project_id, error=str(e))

        return result

    @staticmethod
    def _detect_project_language(ws: Path) -> tuple[str, list[str]]:
        """Detect the primary language of a project workspace.

        Returns (language, list_of_entrypoint_paths).
        """
        files = list(ws.rglob("*"))
        py_files = [f for f in files if f.suffix == ".py"]
        js_files = [f for f in files if f.suffix == ".js"]
        ts_files = [f for f in files if f.suffix == ".ts"]

        entrypoints: list[str] = []

        if py_files:
            # Look for main.py, app.py, cli.py, or any file with an if __name__ block
            candidates = ["main.py", "app.py", "cli.py", "run.py", "server.py"]
            for f in py_files:
                if f.name in candidates:
                    rel = f.relative_to(ws)
                    entrypoints.append(str(rel))
            if not entrypoints:
                # Just pick the first .py file with a shebang or __name__ guard
                for f in py_files[:3]:
                    try:
                        content = f.read_text(encoding="utf-8")
                        if 'if __name__' in content or content.startswith("#!"):
                            rel = f.relative_to(ws)
                            entrypoints.append(str(rel))
                    except Exception as exc:
                        logger.debug("sandbox_detect_error", error=str(exc))
            return ("python", entrypoints)

        if js_files:
            candidates = ["index.js", "app.js", "server.js", "main.js"]
            for f in js_files:
                if f.name in candidates:
                    rel = f.relative_to(ws)
                    entrypoints.append(str(rel))
            return ("javascript", entrypoints)

        if ts_files:
            return ("typescript", entrypoints)

        return ("unknown", [])

    async def _validate_python(
        self, result: SandboxResult, ws: Path, entrypoints: list[str]
    ) -> None:
        """Run Python-specific validation inside a Docker container."""
        project_id = ws.name

        # Build a Docker command that runs inside a Python container
        # Mount the workspace, run syntax check, imports, and optional smoke test
        workspace_host = str(ws.resolve())
        workspace_container = "/workspace"

        # Step 1: Syntax check
        syntax_cmd = (
            f"python -m py_compile {workspace_container}/**/*.py 2>&1 || true"
        )

        # Step 2: Import check on entrypoints
        import_cmd_parts: list[str] = []
        for ep in entrypoints[:2]:
            module_path = ep.replace("/", ".").replace(".py", "")
            import_cmd_parts.append(
                f"python -c 'import {module_path}' 2>&1 || echo 'Import failed: {module_path}'"
            )
        import_cmd = " && ".join(import_cmd_parts) if import_cmd_parts else "echo 'No entrypoints'"

        # Step 3: Quick smoke test (run for 5 seconds max)
        smoke_cmd_parts: list[str] = []
        for ep in entrypoints[:1]:
            smoke_cmd_parts.append(
                f"timeout 5 python {workspace_container}/{ep} 2>&1 || echo 'Smoke exit code: $?'"
            )
        smoke_cmd = " && ".join(smoke_cmd_parts) if smoke_cmd_parts else "echo 'No runnable entrypoint'"

        combined_cmd = (
            f"echo '=== SYNTAX CHECK ===' && {syntax_cmd} && "
            f"echo '=== IMPORT CHECK ===' && {import_cmd} && "
            f"echo '=== SMOKE TEST ===' && {smoke_cmd}"
        )

        docker_args = [
            "docker", "run", "--rm",
            "--network", "none",  # No network access
            "--read-only",        # Read-only filesystem
            "-v", f"{workspace_host}:{workspace_container}:ro",
            "python:3.11-slim",
            "sh", "-c", combined_cmd,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except TimeoutError:
                proc.kill()
                result.error = "timeout"
                logger.warning("sandbox_timeout", project_id=project_id)
                return

            output = stdout.decode("utf-8", errors="replace")

            # Parse sections
            sections = output.split("=== ")
            for section in sections:
                if section.startswith("SYNTAX CHECK"):
                    result.syntax_check = section[len("SYNTAX CHECK"):].strip()[:500]
                elif section.startswith("IMPORT CHECK"):
                    result.import_check = section[len("IMPORT CHECK"):].strip()[:500]
                elif section.startswith("SMOKE TEST"):
                    result.smoke_test = section[len("SMOKE TEST"):].strip()[:500]

            # Determine overall pass/fail
            if stderr and stderr.decode():
                logger.warning(
                    "sandbox_stderr",
                    project_id=project_id,
                    stderr=stderr.decode(errors="replace")[:300],
                )

            # Pass if syntax is valid
            result.passed = (
                "SyntaxError" not in result.syntax_check
                and "Error" not in result.syntax_check
            )

        except FileNotFoundError:
            result.passed = True  # Graceful degradation
            result.error = "docker_not_found"
            logger.info("sandbox_docker_not_found", project_id=project_id)


async def validate_project(project_id: str) -> dict:
    """Convenience wrapper for one-shot project validation.

    Returns the SandboxResult as a dict.
    """
    executor = SandboxExecutor()
    result = await executor.validate_project(project_id)
    return result.to_dict()
