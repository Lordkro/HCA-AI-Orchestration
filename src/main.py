"""HCA Orchestration - Application entrypoint."""

import asyncio
import signal

import structlog

from src.core.config import settings
from src.core.database import Database
from src.core.logger import setup_logging
from src.core.message_bus import MessageBus
from src.core.ollama_client import OllamaClient
from src.agents.pm_agent import PMAgent
from src.agents.research_agent import ResearchAgent
from src.agents.spec_agent import SpecAgent
from src.agents.coder_agent import CoderAgent
from src.agents.critic_agent import CriticAgent
from src.orchestrator.task_manager import TaskManager
from src.orchestrator.pipeline import Pipeline
from src.api.app import create_app

logger = structlog.get_logger()


async def main() -> None:
    """Start the HCA Orchestration system."""
    setup_logging()
    logger.info("Starting HCA Orchestration", version="0.1.0")

    # Initialize core services
    db = Database(settings.database_url)
    await db.initialize()

    bus = MessageBus(settings.redis_url)
    await bus.connect()

    ollama = OllamaClient(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
        timeout=settings.ollama_timeout,
        num_ctx=settings.ollama_num_ctx,
        max_retries=settings.ollama_max_retries,
        retry_base_delay=settings.ollama_retry_base_delay,
    )

    # Validate Ollama connection and preload default model
    if not await ollama.health_check():
        logger.error("Ollama is not reachable", url=settings.ollama_base_url)
        return
    logger.info("Ollama connected, preloading default model...")
    await ollama.preload_model()

    # Initialize agents
    agents = [
        PMAgent(bus=bus, ollama=ollama, db=db),
        ResearchAgent(bus=bus, ollama=ollama, db=db),
        SpecAgent(bus=bus, ollama=ollama, db=db),
        CoderAgent(bus=bus, ollama=ollama, db=db),
        CriticAgent(bus=bus, ollama=ollama, db=db),
    ]

    # Initialize orchestration
    task_manager = TaskManager(db=db, bus=bus)
    pipeline = Pipeline(task_manager=task_manager, bus=bus)

    # Start the web API
    app = create_app(db=db, bus=bus, task_manager=task_manager, agents=agents)

    # Start all agents
    agent_tasks = [asyncio.create_task(agent.start()) for agent in agents]
    pipeline_task = asyncio.create_task(pipeline.start())

    # Start the web server
    import uvicorn
    config = uvicorn.Config(app, host=settings.web_host, port=settings.web_port, log_level="info")
    server = uvicorn.Server(config)

    # Handle graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info(
        "HCA Orchestration started",
        web_url=f"http://{settings.web_host}:{settings.web_port}",
        ollama_url=settings.ollama_base_url,
        model=settings.ollama_default_model,
    )

    # Run server and wait for shutdown
    server_task = asyncio.create_task(server.serve())

    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    server.should_exit = True
    for agent in agents:
        await agent.stop()
    pipeline.stop()
    await bus.disconnect()
    await db.close()

    # Cancel remaining tasks
    for task in agent_tasks:
        task.cancel()
    pipeline_task.cancel()
    await server_task

    logger.info("HCA Orchestration stopped")


if __name__ == "__main__":
    asyncio.run(main())
