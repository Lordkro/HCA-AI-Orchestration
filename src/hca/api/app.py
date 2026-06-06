"""FastAPI application factory."""

from __future__ import annotations

import secrets
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from hca.core.config import settings as app_settings
from hca.core.metrics import (
    api_request_duration_seconds,
    api_requests_in_flight,
    api_requests_total,
)

if TYPE_CHECKING:
    from hca.agents.base_agent import BaseAgent
    from hca.core.database import Database
    from hca.core.message_bus import MessageBus
    from hca.orchestrator.task_manager import TaskManager


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

    # --------------------------------------------------------
    # CORS middleware — allow configurable origins
    # --------------------------------------------------------

    cors_origins = app_settings.cors_origins.split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------
    # API key authentication middleware
    # --------------------------------------------------------

    api_key = app_settings.hca_api_key

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not api_key:
            return await call_next(request)

        public_paths = {"/metrics", "/api/health/live", "/api/health/ready", "/docs", "/openapi.json"}
        if request.url.path in public_paths:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.removeprefix("Bearer ")
        else:
            token = ""

        if not token or not secrets.compare_digest(token, api_key):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})

        return await call_next(request)

    # --------------------------------------------------------
    # Request timing middleware
    # --------------------------------------------------------

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        method = request.method
        path = request.url.path

        api_requests_in_flight.labels(method=method).inc()
        start = time.monotonic()

        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.monotonic() - start
            api_requests_in_flight.labels(method=method).dec()
            api_requests_total.labels(method=method, path=path, status=status).inc()
            api_request_duration_seconds.labels(method=method, path=path).observe(elapsed)

    # --------------------------------------------------------
    # Route modules
    # --------------------------------------------------------

    from hca.api.routes.agents import router as agents_router
    from hca.api.routes.projects import router as projects_router
    from hca.api.routes.tasks import router as tasks_router
    from hca.api.routes.websocket import router as ws_router

    app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
    app.include_router(agents_router, prefix="/api/agents", tags=["agents"])
    app.include_router(tasks_router, prefix="/api/tasks", tags=["tasks"])
    app.include_router(ws_router, tags=["websocket"])

    # --------------------------------------------------------
    # Prometheus /metrics endpoint
    # --------------------------------------------------------

    @app.get("/metrics", tags=["system"])
    async def metrics_endpoint() -> PlainTextResponse:
        """Prometheus metrics scrape endpoint."""
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # --------------------------------------------------------
    # Health probes
    # --------------------------------------------------------

    @app.get("/api/health", tags=["system"])
    async def health_check() -> dict:
        """System health and statistics."""
        return {
            "status": "ok",
            "bus": bus.get_stats(),
            "ollama": agents[0].ollama.get_stats() if agents else {},
        }

    @app.get("/api/health/live", tags=["system"])
    async def liveness() -> dict:
        """Liveness probe — returns 200 if the process is alive."""
        return {"status": "alive"}

    @app.get("/api/health/ready", tags=["system"])
    async def readiness() -> dict:
        """Readiness probe — returns 200 only when all subsystems are healthy."""
        issues = []

        ollama = agents[0].ollama if agents else None
        if ollama and not await ollama.health_check():
            issues.append("ollama_unreachable")

        ready = len(issues) == 0
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={"status": "ready" if ready else "not_ready", "issues": issues},
            status_code=200 if ready else 503,
        )

    # Serve static frontend files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
