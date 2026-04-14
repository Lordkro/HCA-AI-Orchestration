"""Task API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.core.models import TaskState

router = APIRouter()


@router.get("/{project_id}")
async def list_tasks(project_id: str, request: Request, state: str | None = None) -> list[dict]:
    """List all tasks for a project, optionally filtered by state."""
    db = request.app.state.db
    task_state = TaskState(state) if state else None
    tasks = await db.list_tasks(project_id, task_state)
    return [t.model_dump(mode="json") for t in tasks]


@router.get("/detail/{task_id}")
async def get_task(task_id: str, request: Request) -> dict:
    """Get a specific task by ID."""
    db = request.app.state.db
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.model_dump(mode="json")


@router.post("/detail/{task_id}/retry")
async def retry_task(task_id: str, request: Request) -> dict:
    """Retry a failed task by transitioning it back to PENDING.

    Only tasks in the FAILED state can be retried.
    """
    task_manager = request.app.state.task_manager
    db = request.app.state.db

    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.state != TaskState.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Only FAILED tasks can be retried (current state: {task.state.value})",
        )

    updated = await task_manager.transition(task_id, TaskState.PENDING)
    return {
        "task_id": task_id,
        "state": updated.state.value,
        "message": "Task reset to pending and will be picked up by the PM",
    }
