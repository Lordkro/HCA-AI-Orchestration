"""Research Agent — investigates technologies and provides context."""

from __future__ import annotations

import structlog

from hca.agents.base_agent import BaseAgent
from hca.core.database import Database
from hca.core.message_bus import MessageBus
from hca.core.models import (
    AgentMessage,
    AgentRole,
    MessageType,
    TaskState,
)
from hca.core.ollama_client import OllamaClient
from hca.core.tools import (
    FETCH_PAGE_TOOL,
    WEB_SEARCH_TOOL,
    format_validation_errors,
    validate_and_log,
)

logger = structlog.get_logger()


class ResearchAgent(BaseAgent):
    """The Research agent.

    Responsibilities:
    - Investigate technologies, libraries, and patterns
    - Analyze feasibility of requested features
    - Provide context and recommendations
    - Synthesize findings into actionable reports
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
            role=AgentRole.RESEARCH, bus=bus, ollama=ollama, db=db, task_manager=task_manager
        )

    async def process_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle incoming messages."""
        match message.type:
            case MessageType.TASK_ASSIGNMENT:
                return await self._handle_research_task(message)
            case MessageType.QUESTION:
                return await self._handle_question(message)
            case MessageType.FEEDBACK:
                return await self._handle_feedback(message)
            case _:
                logger.debug("research_skipping_message", type=message.type)
                return None

    async def _handle_research_task(self, message: AgentMessage) -> AgentMessage | None:
        """Conduct research based on the assigned task."""
        await self._transition_task(message.task_id, TaskState.IN_PROGRESS)
        self._set_activity("Researching technologies and patterns")

        prompt = f"""You have been assigned a research task by the Project Manager.

TASK/CONTEXT:
{message.payload.content}

You have access to web search and page fetching tools. Use them to:
1. Search for relevant technologies, libraries, and frameworks
2. Fetch documentation or articles for deeper understanding
3. Gather real-world information about the topics

After researching, provide a detailed report covering:

1. **Technology Analysis**: What technologies, frameworks, and libraries are best suited for this project? Explain why.
2. **Architecture Recommendations**: What architecture patterns should be used? (e.g., monolith, microservices, event-driven)
3. **Data Model Considerations**: What are the key data entities and relationships?
4. **Potential Challenges**: What are the likely technical challenges and how to address them?
5. **Best Practices**: What best practices should the team follow?
6. **Estimated Complexity**: How complex is this project? What are the main risk areas?

Be specific and actionable. The Specification Agent will use your report to write detailed technical specs.
Format your output clearly with sections and bullet points."""

        tool_defs = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]
        research_text, tool_calls = await self.think_with_tools(
            prompt, tool_defs, project_id=message.project_id,
            task_id=message.task_id, temperature=0.6,
        )

        # Validate tool calls
        valid_calls, errors = validate_and_log(
            tool_calls, tool_defs, agent_name=self.role.value
        )
        if errors:
            logger.warning(
                "research_invalid_tool_calls",
                task_id=message.task_id,
                error_count=len(errors),
            )
            fix_prompt = f"""{format_validation_errors(errors)}

Original task:
{message.payload.content[:500]}

Please call web_search or fetch_page with corrected arguments."""
            research_text, tool_calls = await self.think_with_tools(
                fix_prompt, tool_defs, project_id=message.project_id,
                task_id=message.task_id, temperature=0.5,
            )
            valid_calls, errors = validate_and_log(
                tool_calls, tool_defs, agent_name=self.role.value
            )

        # Execute tool calls
        if valid_calls:
            tool_results = await self._execute_research_tools(valid_calls)
            research_context = self._format_tool_results(tool_results)

            synthesis_prompt = f"""You previously gathered the following research data:

{research_context}

Now synthesize this information into a comprehensive research report covering:
1. Technology Analysis
2. Architecture Recommendations
3. Data Model Considerations
4. Potential Challenges
5. Best Practices
6. Estimated Complexity

Be specific and actionable. The Specification Agent will use your report to write detailed technical specs."""

            research_text = await self.think(
                synthesis_prompt, project_id=message.project_id,
                task_id=message.task_id, temperature=0.6,
            )

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=research_text or "Research completed.",
            metadata={"artifact_type": "research_report"},
        )

    async def _execute_research_tools(self, tool_calls: list[dict]) -> list[tuple[str, str]]:
        """Execute research tool calls and return (tool_name, result) pairs."""
        results: list[tuple[str, str]] = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if name == "web_search":
                query = args.get("query", "")
                result = await self._web_search(query)
                results.append((f"web_search({query!r})", result))
            elif name == "fetch_page":
                url = args.get("url", "")
                result = await self._fetch_page(url)
                results.append((f"fetch_page({url})", result))
            else:
                logger.warning("research_unknown_tool", tool=name)
        return results

    @staticmethod
    def _format_tool_results(results: list[tuple[str, str]]) -> str:
        """Format tool execution results into a readable context block."""
        parts: list[str] = []
        for tool_label, result in results:
            parts.append(f"=== {tool_label} ===\n{result}\n")
        return "\n".join(parts)

    @staticmethod
    async def _web_search(query: str) -> str:
        """Perform a web search using DuckDuckGo lite HTML endpoint."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
                resp = await http.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; HCA-Research/1.0; "
                            "+https://github.com/Lordkro/HCA-AI-Orchestration)"
                        ),
                    },
                )
                resp.raise_for_status()
                text = resp.text

            # Extract result snippets from DuckDuckGo HTML response
            import re
            snippets: list[str] = []
            for result_block in re.findall(
                r'<a rel="nofollow" class="result__a" href="(.*?)".*?>(.*?)</a>'
                r'.*?<a class="result__snippet".*?>(.*?)</a>',
                text,
                re.DOTALL,
            ):
                url, title, snippet = result_block
                import html
                snippets.append(
                    f"- {html.unescape(re.sub(r'<[^>]+>', '', title)).strip()}\n"
                    f"  {html.unescape(re.sub(r'<[^>]+>', '', snippet)).strip()}\n"
                    f"  {url}"
                )

            if snippets:
                return "Web Search Results:\n" + "\n\n".join(snippets[:8])
            return f"No structured results found for query: {query}"
        except Exception as exc:
            logger.warning("web_search_failed", query=query, error=str(exc))
            return f"Web search unavailable for '{query}': {exc}"

    @staticmethod
    async def _fetch_page(url: str) -> str:
        """Fetch the text content of a webpage."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http:
                resp = await http.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; HCA-Research/1.0; "
                            "+https://github.com/Lordkro/HCA-AI-Orchestration)"
                        ),
                    },
                )
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "text" not in content_type and "json" not in content_type:
                    return f"Cannot display {url}: content-type is {content_type}"

                import re
                text = resp.text
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:8000] + ("..." if len(text) > 8000 else "")
        except Exception as exc:
            logger.warning("fetch_page_failed", url=url, error=str(exc))
            return f"Failed to fetch {url}: {exc}"

    async def _handle_question(self, message: AgentMessage) -> AgentMessage | None:
        """Answer a specific research question."""
        self._set_activity(f"Answering question from {message.sender.value}")
        prompt = f"""Another agent has a research question:

FROM: {message.sender.value}
QUESTION: {message.payload.content}

Provide a thorough, well-reasoned answer with specific recommendations."""

        response = await self.think(
            prompt, project_id=message.project_id, task_id=message.task_id, temperature=0.5
        )

        return self.create_message(
            recipient=message.sender,
            msg_type=MessageType.ANSWER,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
        )

    async def _handle_feedback(self, message: AgentMessage) -> AgentMessage | None:
        """Revise research based on feedback."""
        self._set_activity("Revising research based on feedback")
        prompt = f"""Your previous research report received feedback:

FEEDBACK:
{message.payload.content}

Please revise and improve your research based on this feedback. Address all points raised."""

        response = await self.think(
            prompt, project_id=message.project_id, task_id=message.task_id, temperature=0.6
        )

        return self.create_message(
            recipient=AgentRole.PM,
            msg_type=MessageType.DELIVERABLE,
            project_id=message.project_id,
            task_id=message.task_id,
            content=response,
            metadata={"artifact_type": "research_report", "revision": "true"},
        )
