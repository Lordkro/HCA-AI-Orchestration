"""Pipeline definitions — standard agent workflow orchestration."""

from __future__ import annotations

import asyncio

import structlog

from src.core.message_bus import MessageBus
from src.orchestrator.task_manager import TaskManager

logger = structlog.get_logger()

# How often to run maintenance tasks (seconds)
HEALTH_CHECK_INTERVAL = 30
STREAM_TRIM_INTERVAL = 300  # 5 minutes


class Pipeline:
    """Manages the overall agent workflow pipeline.

    Standard flow: PM → Research → Spec → Code → Critic → (iterate) → Done

    Also handles periodic maintenance:
    - Stream trimming to prevent unbounded memory growth
    - Health checks for stuck tasks and deadlocks
    """

    def __init__(self, *, task_manager: TaskManager, bus: MessageBus) -> None:
        self.task_manager = task_manager
        self.bus = bus
        self._running = False
        self._tick_count = 0

    async def start(self) -> None:
        """Start the pipeline monitor."""
        self._running = True
        logger.info("pipeline_started")

        while self._running:
            try:
                self._tick_count += 1

                # Health check every tick
                await self._check_health()

                # Stream maintenance less frequently
                if self._tick_count % (STREAM_TRIM_INTERVAL // HEALTH_CHECK_INTERVAL) == 0:
                    await self.bus.trim_streams()

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("pipeline_error", error=str(e))
                await asyncio.sleep(10)

        logger.info("pipeline_stopped")

    def stop(self) -> None:
        """Stop the pipeline monitor."""
        self._running = False

    async def _check_health(self) -> None:
        """Check for stuck tasks, deadlocks, and timeouts."""
        # TODO: Implement health checks in Phase 3
        # - Check for tasks stuck in a state for too long
        # - Detect circular dependencies
        # - Auto-retry failed tasks
        pass
