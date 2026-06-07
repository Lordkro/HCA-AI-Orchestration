"""Critic Agent — reviews all outputs for quality and correctness."""

from __future__ import annotations

import structlog

from hca.agents.base_agent import BaseAgent
from hca.core.database import Database
from hca.core.message_bus import MessageBus
from hca.core.models import (
    AgentMessage,
    AgentRole,
    MessageType,
)
from hca.core.ollama_client import OllamaClient
from hca.core.tools import SUBMIT_REVIEW_TOOL, format_validation_errors, validate_and_log
from hca.orchestrator.sandbox import SandboxExecutor, SandboxResult

logger = structlog.get_logger()


class CriticAgent(BaseAgent):
    """The Critic agent.

    Responsibilities:
    - Review code for bugs, security issues, and best practices
    - Validate specifications for completeness and consistency
    - Provide actionable feedback with specific suggestions
    - Approve or reject deliverables with clear reasoning
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
            role=AgentRole.CRITIC, bus=bus, ollama=ollama, db=db, task_manager=task_manager
        )

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages."""
        match message.type:
            case MessageType.TASK_ASSIGNMENT:
                return await self._handle_review_task(message)
            case MessageType.QUESTION:
                return await self._handle_question(message)
            case _:
                logger.debug("critic_skipping_message", type=message.type)
                return None

    async def _handle_review_task(self, message: AgentMessage) -> AgentMessage | None:
        """Review a deliverable (code, spec, or research)."""
        original_artifact_type = message.payload.metadata.get("artifact_type", "unknown")
        self._set_activity(f"Reviewing {original_artifact_type} from {message.sender.value}")

        # Run sandbox validation for code artifacts
        sandbox_context = ""
        sb: SandboxResult | None = None
        if original_artifact_type == "code":
            sandbox = SandboxExecutor()
            sb = await sandbox.validate_project(message.project_id)
            if sb.error not in ("docker_unavailable", "docker_not_found", ""):
                sandbox_context = f"""
SANDBOX VALIDATION RESULTS:
  Passed: {sb.passed}
  Syntax check: {sb.syntax_check[:300] if sb.syntax_check else 'N/A'}
  Import check: {sb.import_check[:300] if sb.import_check else 'N/A'}
  Smoke test: {sb.smoke_test[:300] if sb.smoke_test else 'N/A'}
  Error: {sb.error}
"""
                if sb.error:
                    sandbox_context += "\nNOTE: Sandbox validation encountered an error but the review should still proceed."

        prompt = f"""You are reviewing a deliverable from the {message.sender.value} agent.

ARTIFACT TYPE: {original_artifact_type}
DELIVERABLE:
{message.payload.content}
{sandbox_context}
Perform a thorough review. Check for:

**If reviewing CODE:**
- Correctness: Does it work as intended? Any logic errors?
- Security: SQL injection, XSS, path traversal, hardcoded secrets, etc.
- Error handling: Are errors caught and handled appropriately?
- Code quality: Clean code, proper naming, DRY principle, SOLID principles
- Type safety: Are type hints present and correct?
- Testing: Are tests included and do they cover critical paths?
- Documentation: Are docstrings and comments adequate?
- Dependencies: Are all imports and dependencies accounted for?
- Edge cases: Are edge cases handled?

**If reviewing a SPECIFICATION:**
- Completeness: Are all features specified?
- Consistency: Do different parts of the spec agree with each other?
- Feasibility: Can this actually be built as specified?
- Clarity: Is it clear enough for the Coder agent to implement?
- Data models: Are all entities and relationships defined?
- API design: Are endpoints well-designed and RESTful?

**If reviewing RESEARCH:**
- Accuracy: Are the claims accurate?
- Relevance: Is the research relevant to the project?
- Actionability: Can the team act on the recommendations?

Use the `submit_review` tool to provide your structured verdict. Be constructive but thorough.
Do not approve work that has critical or major issues."""

        tool_defs = [SUBMIT_REVIEW_TOOL]
        response, tool_calls = await self.think_with_tools(
            prompt, tool_defs,
            project_id=message.project_id, task_id=message.task_id,
            temperature=0.3,
        )

        # Validate tool calls and retry if needed
        valid_calls, errors = validate_and_log(
            tool_calls, tool_defs, agent_name=self.role.value
        )
        if errors:
            logger.warning(
                "critic_invalid_tool_calls",
                task_id=message.task_id,
                error_count=len(errors),
            )
            fix_prompt = f"""{format_validation_errors(errors)}

Please submit your review again with corrected arguments."""
            response, tool_calls = await self.think_with_tools(
                fix_prompt, tool_defs, project_id=message.project_id,
                task_id=message.task_id, temperature=0.3,
            )
            valid_calls, errors = validate_and_log(
                tool_calls, tool_defs, agent_name=self.role.value
            )

        # Extract verdict from validated tool calls
        verdict = ""
        review_content = response
        for call in valid_calls:
            if call["name"] == "submit_review":
                args = call["arguments"]
                verdict = args.get("verdict", "")
                summary = args.get("summary", "")
                issues = args.get("issues", [])
                recommendations = args.get("recommendations", "")
                # Build a structured review text from the tool arguments
                parts = [f"# Review: {verdict.upper()}", "", summary]
                if issues:
                    parts.append("")
                    parts.append("## Issues Found")
                    for i, issue in enumerate(issues, 1):
                        sev = issue.get("severity", "unknown")
                        desc = issue.get("description", "")
                        sug = issue.get("suggestion", "")
                        parts.append(f"{i}. [{sev}] {desc}")
                        if sug:
                            parts.append(f"   Suggestion: {sug}")
                if recommendations:
                    parts.append("")
                    parts.append(f"## Recommendations\n{recommendations}")
                review_content = "\n".join(parts)
                break

        # Fallback: string matching if no tool call was made
        if not verdict:
            first_lines = "\n".join((response or "").split("\n")[:5]).upper()
            is_approved = "**APPROVED**" in first_lines or first_lines.lstrip().startswith("APPROVED")
            verdict = "approved" if is_approved else "needs_revision"

        critic_metadata: dict[str, str] = {
            "review_result": verdict,
            "artifact_type": original_artifact_type,
        }
        if sandbox_context and sb is not None:
            critic_metadata["sandbox_passed"] = str(sb.passed)
            if sb.error and sb.error not in ("docker_unavailable", "docker_not_found"):
                critic_metadata["sandbox_error"] = sb.error

        if verdict == "approved":
            return self.create_message(
                recipient=AgentRole.PM,
                msg_type=MessageType.DELIVERABLE,
                project_id=message.project_id,
                task_id=message.task_id,
                content=review_content,
                metadata=critic_metadata,
            )
        else:
            return self.create_message(
                recipient=AgentRole.PM,
                msg_type=MessageType.FEEDBACK,
                project_id=message.project_id,
                task_id=message.task_id,
                content=review_content,
                metadata=critic_metadata,
            )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer questions about review feedback."""
        self._set_activity(f"Clarifying review for {message.sender.value}")
        prompt = f"""The {message.sender.value} agent has a question about your review:

{message.payload.content}

Clarify your feedback with specific examples and suggestions."""

        response = await self.think(
            prompt, project_id=message.project_id, task_id=message.task_id, temperature=0.4
        )

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )
