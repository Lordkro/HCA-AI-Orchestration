"""Coder Agent — writes implementation code from specifications."""

from __future__ import annotations

import re
from pathlib import Path

import structlog

from src.agents.base_agent import BaseAgent
from src.core.config import settings
from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import (
    AgentMessage,
    AgentRole,
    Artifact,
    MessageType,
)
from src.core.ollama_client import OllamaClient

logger = structlog.get_logger()


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
    ) -> None:
        super().__init__(role=AgentRole.CODER, bus=bus, ollama=ollama, db=db)

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

OUTPUT FORMAT:
For each file you create, use this exact format:

=== FILE: path/to/filename.ext ===
```language
(file contents here)
```

=== FILE: path/to/another_file.ext ===
```language
(file contents here)
```

Create ALL necessary files for a working implementation. Include:
- Source code files
- Configuration files (if needed)
- Test files
- Requirements/dependency files (if needed)
- README or usage notes"""

        response = await self.think(prompt, project_id=message.project_id, temperature=0.4)

        # Parse and save artifacts
        artifacts = self._parse_file_outputs(response, message.project_id, message.task_id)
        for artifact in artifacts:
            await self.db.create_artifact(artifact)
            # Also write to the workspace filesystem
            await self._write_to_workspace(artifact, message.project_id)

        artifact_names = [a.filename for a in artifacts]

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            artifacts=artifact_names,
            metadata={"artifact_type": "code", "file_count": str(len(artifacts))},
        )

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Fix code based on Critic feedback."""
        prompt = f"""Your code received feedback from the Critic. Please fix the issues.

FEEDBACK:
{message.payload.content}

Address ALL issues mentioned in the feedback. Output the complete corrected files using the same format:

=== FILE: path/to/filename.ext ===
```language
(corrected file contents)
```

Only output files that have changed."""

        response = await self.think(prompt, project_id=message.project_id, temperature=0.3)

        artifacts = self._parse_file_outputs(response, message.project_id, message.task_id)
        for artifact in artifacts:
            await self.db.create_artifact(artifact)
            await self._write_to_workspace(artifact, message.project_id)

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            artifacts=[a.filename for a in artifacts],
            metadata={"artifact_type": "code", "revision": "true"},
        )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer questions about the implementation."""
        prompt = f"""The {message.sender.value} agent has a question about the code:

{message.payload.content}

Provide a clear answer with code examples if needed."""

        response = await self.think(prompt, project_id=message.project_id)

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
    _FALLBACK_MARKER_RE = re.compile(
        r"^(?:\*\*|`)([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)(?:\*\*|`)\s*$"
    )

    def _parse_file_outputs(
        self, response: str, project_id: str, task_id: str
    ) -> list[Artifact]:
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
        workspace = Path(settings.workspace_dir) / project_id
        file_path = workspace / artifact.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(artifact.content, encoding="utf-8")
        logger.info("file_written", path=str(file_path))
