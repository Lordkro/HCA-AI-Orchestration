"""Coder Agent — writes implementation code from specifications."""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from hca.agents.base_agent import BaseAgent
from hca.core.config import settings
from hca.core.database import Database
from hca.core.message_bus import MessageBus
from hca.core.models import (
    AgentMessage,
    AgentRole,
    Artifact,
    MessageType,
    TaskState,
)
from hca.core.ollama_client import OllamaClient
from hca.core.tools import WRITE_FILE_TOOL, format_validation_errors, validate_and_log
from hca.orchestrator.workspace_manager import WorkspaceManager

logger = structlog.get_logger()


class WorkspaceWriteError(ValueError):
    """Raised when an artifact path cannot be written safely."""


class CoderAgent(BaseAgent):
    """The Coder agent.

    Responsibilities:
    - Write implementation code based on specifications
    - Generate tests alongside code
    - Fix issues based on Critic feedback
    - Manage file creation in the workspace
    """

    def __init__(
        self,
        *,
        bus: MessageBus,
        ollama: OllamaClient,
        db: Database,
        task_manager: object | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.CODER, bus=bus, ollama=ollama, db=db, task_manager=task_manager
        )

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages."""
        match message.type:
            case MessageType.TASK_ASSIGNMENT:
                return await self._handle_coding_task(message)
            case MessageType.FEEDBACK:
                return await self._handle_feedback(message)
            case MessageType.QUESTION:
                return await self._handle_question(message)
            case _:
                logger.debug("coder_skipping_message", type=message.type)
                return None

    async def _handle_coding_task(self, message: AgentMessage) -> AgentMessage | None:
        """Generate code based on the specification."""
        await self._transition_task(message.task_id, TaskState.IN_PROGRESS)
        self._set_activity("Writing implementation code")

        prompt = f"""You have been assigned a coding task. Implement the following based on the specification provided.

SPECIFICATION / INSTRUCTIONS:
{message.payload.content}

RULES:
1. Write complete, production-ready code. No placeholders, no TODOs, no "implement here" comments.
2. Include proper error handling, input validation, and logging.
3. Follow the project structure defined in the specification.
4. Write clean, well-documented code with docstrings.
5. Include type hints for all functions.
6. Generate unit tests for critical functionality.
7. Use the `write_file` tool for EVERY file you create. Call it once per file.

Create ALL necessary files for a working implementation. Include:
- Source code files
- Configuration files (if needed)
- Test files
- Requirements/dependency files (if needed)
- README or usage notes"""

        tool_defs = [WRITE_FILE_TOOL]
        response, tool_calls = await self.think_with_tools(
            prompt, tool_defs, project_id=message.project_id,
            task_id=message.task_id, temperature=0.4,
        )

        # Validate tool calls and retry if needed
        valid_calls, errors = validate_and_log(
            tool_calls, tool_defs, agent_name=self.role.value
        )
        if errors:
            logger.warning(
                "coder_invalid_tool_calls",
                task_id=message.task_id,
                error_count=len(errors),
            )
            fix_prompt = f"""{format_validation_errors(errors)}

Original task:
{message.payload.content[:500]}

Please call write_file again with corrected arguments."""
            response, tool_calls = await self.think_with_tools(
                fix_prompt, tool_defs, project_id=message.project_id,
                task_id=message.task_id, temperature=0.3,
            )
            valid_calls, errors = validate_and_log(
                tool_calls, tool_defs, agent_name=self.role.value
            )

        # Process validated tool calls
        artifacts = await self._process_write_file_calls(
            valid_calls, message.project_id, message.task_id
        )

        # Init git repo and commit
        if artifacts:
            await WorkspaceManager.init_project_repo(message.project_id)
            await WorkspaceManager.commit_workspace(
                message.project_id,
                f"Coding iteration for task {message.task_id}",
            )

        # Fall back to regex parsing if no tool calls were made
        if not artifacts and response:
            artifacts = self._parse_file_outputs(response, message.project_id, message.task_id)
            for artifact in artifacts:
                await self.db.create_artifact(artifact)
                await self._write_to_workspace(artifact, message.project_id)

        artifact_names = [a.filename for a in artifacts]

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response or f"Created {len(artifacts)} files",
            artifacts=artifact_names,
            metadata={"artifact_type": "code", "file_count": str(len(artifacts))},
        )

    async def _process_write_file_calls(
        self, tool_calls: list[dict], project_id: str, task_id: str
    ) -> list[Artifact]:
        """Process write_file tool calls into artifacts and write to workspace."""
        artifacts: list[Artifact] = []
        for call in tool_calls:
            if call["name"] != "write_file":
                continue
            args = call["arguments"]
            path = args.get("path", "").strip()
            content = args.get("content", "")
            artifact_type = args.get("artifact_type", "code")

            if not path or not content:
                logger.warning("coder_skipping_empty_write_file", args=args)
                continue

            artifact = Artifact(
                project_id=project_id,
                task_id=task_id,
                agent=AgentRole.CODER,
                filename=path,
                content=content,
                artifact_type=artifact_type,
            )
            await self.db.create_artifact(artifact)
            await self._write_to_workspace(artifact, project_id)
            artifacts.append(artifact)

        return artifacts

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Fix code based on Critic feedback."""
        self._set_activity("Fixing code based on review feedback")

        # Get previous diff to provide context
        diff_context = ""
        try:
            prev_diff = await WorkspaceManager.get_workspace_diff(message.project_id)
            if prev_diff:
                diff_context = f"\n\nCURRENT CHANGES SINCE LAST COMMIT:\n{prev_diff}\n"
        except Exception as exc:
            logger.debug("diff_context_unavailable", error=str(exc))

        prompt = f"""Your code received feedback from the Critic. Please fix the issues.

FEEDBACK:
{message.payload.content}{diff_context}

Address ALL issues mentioned in the feedback. Use the `write_file` tool for each file you need to correct. Only output files that have changed."""

        tool_defs = [WRITE_FILE_TOOL]
        response, tool_calls = await self.think_with_tools(
            prompt, tool_defs, project_id=message.project_id,
            task_id=message.task_id, temperature=0.3,
        )

        # Validate tool calls and retry if needed
        valid_calls, errors = validate_and_log(
            tool_calls, tool_defs, agent_name=self.role.value
        )
        if errors:
            logger.warning(
                "coder_feedback_invalid_tool_calls",
                task_id=message.task_id,
                error_count=len(errors),
            )
            fix_prompt = f"""{format_validation_errors(errors)}

Original feedback:
{message.payload.content[:500]}

Please call write_file with corrected arguments."""
            response, tool_calls = await self.think_with_tools(
                fix_prompt, tool_defs, project_id=message.project_id,
                task_id=message.task_id, temperature=0.3,
            )
            valid_calls, errors = validate_and_log(
                tool_calls, tool_defs, agent_name=self.role.value
            )

        artifacts = await self._process_write_file_calls(
            valid_calls, message.project_id, message.task_id
        )

        # Commit changes
        if artifacts:
            await WorkspaceManager.commit_workspace(
                message.project_id,
                f"Revision for task {message.task_id}",
            )

        if not artifacts and response:
            artifacts = self._parse_file_outputs(response, message.project_id, message.task_id)
            for artifact in artifacts:
                await self.db.create_artifact(artifact)
                await self._write_to_workspace(artifact, message.project_id)

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response or f"Fixed {len(artifacts)} files",
            artifacts=[a.filename for a in artifacts],
            metadata={"artifact_type": "code", "revision": "true"},
        )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer questions about the implementation."""
        self._set_activity(f"Answering question from {message.sender.value}")
        prompt = f"""The {message.sender.value} agent has a question about the code:

{message.payload.content}

Provide a clear answer with code examples if needed."""

        response = await self.think(prompt, project_id=message.project_id, task_id=message.task_id)

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    # --------------------------------------------------------
    # File Parsing
    # --------------------------------------------------------

    # Regex patterns for extracting file blocks from LLM output.
    # Pattern 1 (preferred):  === FILE: path/to/file ===
    # Pattern 2 (fallback):   **path/to/file**  or  `path/to/file`
    _FILE_MARKER_RE = re.compile(
        r"^={2,}\s*FILE:\s*(.+?)\s*={2,}\s*$",
        re.IGNORECASE,
    )
    _FALLBACK_MARKER_RE = re.compile(r"^(?:\*\*|`)([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)(?:\*\*|`)\s*$")

    def _parse_file_outputs(self, response: str, project_id: str, task_id: str) -> list[Artifact]:
        """Parse the LLM response to extract file artifacts.

        Supports the canonical ``=== FILE: path ===`` format and falls
        back to ``**path**`` or `` `path` `` markers.  Logs warnings when
        no artifacts are found so issues are visible.
        """
        artifacts: list[Artifact] = []
        lines = response.split("\n")
        current_file: str | None = None
        current_content: list[str] = []
        in_code_block = False

        def _save_current() -> None:
            """Flush the accumulated content into an Artifact."""
            nonlocal current_file, current_content, in_code_block
            if current_file and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    artifacts.append(
                        Artifact(
                            project_id=project_id,
                            task_id=task_id,
                            agent=AgentRole.CODER,
                            filename=current_file,
                            content=content,
                            artifact_type=self._detect_artifact_type(current_file),
                        )
                    )
                else:
                    logger.warning(
                        "coder_empty_file_content",
                        filename=current_file,
                    )
            current_file = None
            current_content = []
            in_code_block = False

        for line in lines:
            stripped = line.strip()

            # Try canonical marker first
            m = self._FILE_MARKER_RE.match(stripped)
            if m:
                _save_current()
                current_file = m.group(1).strip()
                continue

            # Try fallback marker
            m2 = self._FALLBACK_MARKER_RE.match(stripped)
            if m2 and not in_code_block:
                _save_current()
                current_file = m2.group(1).strip()
                continue

            # Handle code fences
            if stripped.startswith("```") and not in_code_block:
                in_code_block = True
                continue
            if stripped == "```" and in_code_block:
                in_code_block = False
                continue

            # Accumulate content when inside a file block
            if current_file is not None:
                current_content.append(line)

        # Flush the last file
        _save_current()

        if not artifacts:
            logger.warning(
                "coder_no_artifacts_parsed",
                response_length=len(response),
                hint="LLM output did not contain recognised file markers",
            )

        return artifacts

    @staticmethod
    def _detect_artifact_type(filename: str) -> str:
        """Detect the artifact type from the filename."""
        if "test" in filename.lower():
            return "test"
        if filename.endswith((".md", ".txt", ".rst")):
            return "doc"
        if filename.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".env")):
            return "config"
        return "code"

    async def _write_to_workspace(self, artifact: Artifact, project_id: str) -> None:
        """Write an artifact to the workspace filesystem."""
        # Validate project_id for path traversal
        if not project_id or "/" in project_id or "\\" in project_id or ".." in project_id:
            raise WorkspaceWriteError(
                f"Invalid project_id (path traversal blocked): {project_id}"
            )

        workspace_root = Path(settings.workspace_dir)
        try:
            workspace_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            fallback_root = Path.cwd() / "workspace"
            logger.warning(
                "workspace_root_unavailable",
                configured_path=str(workspace_root),
                fallback_path=str(fallback_root),
                error=str(exc),
            )
            workspace_root = fallback_root
            workspace_root.mkdir(parents=True, exist_ok=True)

        workspace = (workspace_root / project_id).resolve()
        file_path = (workspace / artifact.filename).resolve()

        try:
            file_path.relative_to(workspace)
        except ValueError as exc:
            raise WorkspaceWriteError(
                f"Artifact path escapes project workspace: {artifact.filename}"
            ) from exc

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(artifact.content, encoding="utf-8")
        logger.info("file_written", path=str(file_path))
