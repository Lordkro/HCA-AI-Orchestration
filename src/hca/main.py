"""HCA Orchestration - Application entrypoint."""

import asyncio
import atexit
import contextlib
import fcntl
import os
import signal
import sys
import tempfile
from pathlib import Path

import structlog
import uvicorn

from hca.agents.coder_agent import CoderAgent
from hca.agents.critic_agent import CriticAgent
from hca.agents.pm_agent import PMAgent
from hca.agents.research_agent import ResearchAgent
from hca.agents.spec_agent import SpecAgent
from hca.api.app import create_app
from hca.core.config import settings
from hca.core.database import Database
from hca.core.logger import setup_logging
from hca.core.message_bus import MessageBus
from hca.core.models import AgentRole
from hca.core.ollama_client import OllamaClient
from hca.orchestrator.pipeline import Pipeline
from hca.orchestrator.task_manager import TaskManager

logger = structlog.get_logger()


async def main() -> None:
    """Start the HCA Orchestration system."""
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger.info("Starting HCA Orchestration", version="0.1.0")

    # Acquire instance lock BEFORE initializing any resources
    lock_dir = Path(tempfile.gettempdir()) / "HCA-AI-Orchestration"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "run.lock"
    lock_file = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        with contextlib.suppress(Exception):
            existing = lock_path.read_text().strip()
            logger.error("Another instance is already running; aborting startup.", pid=existing)
        lock_file.close()
        return

    # Write PID to lock file so stale locks can be diagnosed
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    def _release_lock() -> None:
        try:
            lock_file.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(lock_path)
            logger.info("Lock file removed, instance shutdown cleanly.")
        except Exception as e:
            logger.error("Failed to remove lock file", error=str(e))

    atexit.register(_release_lock)

    # Initialize core services
    db = Database(settings.database_url)
    await db.initialize()

    bus = MessageBus(settings.redis_url)
    await bus.connect()

    agent_roles = [
        AgentRole.PM,
        AgentRole.RESEARCH,
        AgentRole.SPEC,
        AgentRole.CODER,
        AgentRole.CRITIC,
    ]
    await bus.setup_agent_streams(agent_roles)
    logger.info("Message bus ready with consumer groups")

    ollama = OllamaClient(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_default_model,
        timeout=settings.ollama_timeout,
        num_ctx=settings.ollama_num_ctx,
        max_retries=settings.ollama_max_retries,
        retry_base_delay=settings.ollama_retry_base_delay,
        max_concurrent=settings.ollama_max_concurrent,
    )

    # Validate Ollama connection (model loads lazily on first request)
    if not await ollama.health_check():
        logger.error("Ollama is not reachable", url=settings.ollama_base_url)
        await ollama.close()
        await bus.disconnect()
        await db.close()
        return
    logger.info("Ollama connected, models will load on first use")

    # Initialize orchestration
    task_manager = TaskManager(db=db, bus=bus)
    pipeline = Pipeline(task_manager=task_manager, bus=bus)

    # Initialize agents (with TaskManager injected)
    agents = [
        PMAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        ResearchAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        SpecAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        CoderAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
        CriticAgent(bus=bus, ollama=ollama, db=db, task_manager=task_manager),
    ]

    # Start the web API
    app = create_app(db=db, bus=bus, task_manager=task_manager, agents=agents)

    # Start all agents
    agent_tasks = [asyncio.create_task(agent.start()) for agent in agents]
    pipeline_task = asyncio.create_task(pipeline.start())

    config = uvicorn.Config(
        app,
        host=settings.web_host,
        port=settings.web_port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    # Handle graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()

    # Unix supports loop.add_signal_handler; Windows does not
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)
    else:
        # On Windows, use signal.signal for SIGINT (Ctrl+C).
        # SIGTERM is not reliably supported on Windows.
        def _win_handler(signum: int, frame: object) -> None:
            _signal_handler()

        signal.signal(signal.SIGINT, _win_handler)

    logger.info(
        "HCA Orchestration started",
        web_url=f"http://{settings.web_host}:{settings.web_port}",
        ollama_url=settings.ollama_base_url,
        model=settings.ollama_default_model,
    )

    # Run server and wait for shutdown
    server_task = asyncio.create_task(server.serve())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    done, pending = await asyncio.wait(
        {server_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

    # Graceful shutdown
    logger.info("Shutting down...")
    server.should_exit = True
    await asyncio.gather(*[agent.stop() for agent in agents], return_exceptions=True)
    pipeline.stop()
    await ollama.close()
    await bus.disconnect()
    await db.close()

    # Cancel remaining tasks
    for task in agent_tasks:
        task.cancel()
    pipeline_task.cancel()
    await asyncio.gather(*agent_tasks, pipeline_task, return_exceptions=True)
    await server_task

    logger.info("HCA Orchestration stopped")


if __name__ == "__main__":
    asyncio.run(main())
