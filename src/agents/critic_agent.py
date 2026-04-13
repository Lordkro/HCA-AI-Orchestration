"""Critic Agent — reviews all outputs for quality and correctness."""

from __future__ import annotations

import structlog

from src.agents.base_agent import BaseAgent
from src.core.database import Database
from src.core.message_bus import MessageBus
from src.core.models import (
    AgentMessage,
    AgentRole,
    MessageType,
)
from src.core.ollama_client import OllamaClient

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
    ) -> None:
        super().__init__(role=AgentRole.CRITIC, bus=bus, ollama=ollama, db=db)

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
        artifact_type = message.payload.metadata.get("artifact_type", "unknown")

        prompt = f"""You are reviewing a deliverable from the {message.sender.value} agent.

ARTIFACT TYPE: {artifact_type}
DELIVERABLE:
{message.payload.content}

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

OUTPUT FORMAT:
Start with a verdict: **APPROVED** or **NEEDS REVISION**

Then provide:
1. **Summary**: One-paragraph overall assessment
2. **Issues Found**: Numbered list of specific issues (if any)
   - For each issue: severity (critical/major/minor), description, and suggested fix
3. **Strengths**: What was done well
4. **Recommendations**: Specific improvements (even if approved)

Be constructive but thorough. Do not approve work that has critical or major issues."""

        response = await self.think(prompt, temperature=0.3)

        # Determine if approved or needs revision
        is_approved = "**APPROVED**" in response.upper() or "APPROVED" in response.split("\n")[0].upper()

        if is_approved:
            return self.create_message(
                recipient=AgentRole.PM,
                msg_type=MessageType.DELIVERABLE,
                project_id=message.project_id,
                task_id=message.task_id,
                content=response,
                metadata={"review_result": "approved", "artifact_type": "review"},
            )
        else:
            return self.create_message(
                recipient=AgentRole.PM,
                msg_type=MessageType.FEEDBACK,
                project_id=message.project_id,
                task_id=message.task_id,
                content=response,
                metadata={"review_result": "needs_revision", "artifact_type": "review"},
            )

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer questions about review feedback."""
        prompt = f"""The {message.sender.value} agent has a question about your review:

{message.payload.content}

Clarify your feedback with specific examples and suggestions."""

        response = await self.think(prompt, temperature=0.4)

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )
