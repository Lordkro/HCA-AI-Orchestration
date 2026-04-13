"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.core.database import Database
    from src.core.message_bus import MessageBus
    from src.orchestrator.task_manager import TaskManager


def create_app(
    *,
    db: Database,
    bus: MessageBus,
    task_manager: TaskManager,
    agents: list[BaseAgent],
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="HCA Orchestration",
        description="Hybrid Cognitive Architecture — Autonomous AI Development Team",
        version="0.1.0",
    )

    # Store dependencies on app state for access in routes
    app.state.db = db
    app.state.bus = bus
    app.state.task_manager = task_manager
    app.state.agents = agents

    # Register route modules
    from src.api.routes.projects import router as projects_router
    from src.api.routes.agents import router as agents_router
    from src.api.routes.tasks import router as tasks_router
    from src.api.routes.websocket import router as ws_router

    app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
    app.include_router(agents_router, prefix="/api/agents", tags=["agents"])
    app.include_router(tasks_router, prefix="/api/tasks", tags=["tasks"])
    app.include_router(ws_router, tags=["websocket"])

    # Health / diagnostics endpoint
    @app.get("/api/health", tags=["system"])
    async def health_check() -> dict:
        """System health and statistics."""
        return {
            "status": "ok",
            "bus": bus.get_stats(),
            "ollama": agents[0].ollama.get_stats() if agents else {},
        }

    # Serve static frontend files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
