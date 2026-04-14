"""Project API endpoints."""

from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request

from src.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    Project,
)

router = APIRouter()


class CreateProjectRequest(BaseModel):
    """Request body for creating a new project."""
    idea: str
    name: str = ""


@router.post("/", response_model=dict)
async def create_project(req: CreateProjectRequest, request: Request) -> dict:
    """Submit a new product idea to the agent team."""
    db = request.app.state.db
    bus = request.app.state.bus

    project = Project(
        name=req.name or "New Project",
        description=req.idea,
        idea=req.idea,
    )
    await db.create_project(project)

    # Send the idea to the PM agent to kick off the pipeline
    msg = AgentMessage(
        sender=AgentRole.USER,
        recipient=AgentRole.PM,
        type=MessageType.SYSTEM,
        project_id=project.id,
        payload=MessagePayload(content=req.idea),
    )
    await bus.publish(msg)

    return {"project_id": project.id, "status": "created", "message": "Project submitted to PM agent"}


@router.get("/", response_model=list[dict])
async def list_projects(request: Request) -> list[dict]:
    """List all projects."""
    db = request.app.state.db
    projects = await db.list_projects()
    return [p.model_dump(mode="json") for p in projects]


@router.get("/{project_id}")
async def get_project(project_id: str, request: Request) -> dict:
    """Get a project by ID."""
    db = request.app.state.db
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    progress = await request.app.state.task_manager.get_project_progress(project_id)
    messages = await db.get_project_messages(project_id)
    artifacts = await db.list_artifacts(project_id)

    return {
        "project": project.model_dump(mode="json"),
        "progress": progress,
        "message_count": len(messages),
        "artifact_count": len(artifacts),
    }


@router.get("/{project_id}/messages")
async def get_project_messages(project_id: str, request: Request) -> list[dict]:
    """Get all messages for a project."""
    db = request.app.state.db
    return await db.get_project_messages(project_id)


@router.get("/{project_id}/artifacts")
async def get_project_artifacts(project_id: str, request: Request) -> list[dict]:
    """Get all artifacts for a project."""
    db = request.app.state.db
    artifacts = await db.list_artifacts(project_id)
    return [a.model_dump(mode="json") for a in artifacts]


@router.post("/{project_id}/pause")
async def pause_project(project_id: str, request: Request) -> dict:
    """Pause an active project.  Agents will skip messages for paused projects."""
    db = request.app.state.db
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != "active":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause project in '{project.status}' state",
        )
    await db.update_project(project_id, status="paused")
    return {"project_id": project_id, "status": "paused"}


@router.post("/{project_id}/resume")
async def resume_project(project_id: str, request: Request) -> dict:
    """Resume a paused project."""
    db = request.app.state.db
    project = await db.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.status != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume project in '{project.status}' state",
        )
    await db.update_project(project_id, status="active")
    return {"project_id": project_id, "status": "active"}
