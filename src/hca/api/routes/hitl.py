"""Human-in-the-loop API endpoints.

Provides REST endpoints for humans to:
- Inject feedback into any task
- Approve or reject tasks (overriding the Critic)
- Steer tasks (reassign, reprioritize)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from hca.core.models import (
    AgentMessage,
    AgentRole,
    MessagePayload,
    MessageType,
    Priority,
    TaskState,
)

router = APIRouter()


@router.post("/projects/{project_id}/feedback")
async def inject_feedback(project_id: str, request: Request) -> dict:
    """Inject human feedback for a specific task.

    The feedback is published as a FEEDBACK message to the task's current
    or specified agent.  If no ``agent_role`` is given the PM will route it.
    """
    body = await request.json()
    task_id: str = body.get("task_id", "")
    content: str = body.get("content", "")
    agent_role_str: str | None = body.get("agent_role", None)

    if not task_id or not content:
        raise HTTPException(status_code=400, detail="task_id and content are required")

    db = request.app.state.db
    bus = request.app.state.bus

    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    agent_role: AgentRole
    if agent_role_str:
        try:
            agent_role = AgentRole(agent_role_str)
        except ValueError:
            valid = ", ".join(r.value for r in AgentRole)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_role '{agent_role_str}'. Valid: {valid}",
            ) from None
    else:
        agent_role = task.assigned_to or AgentRole.CODER

    msg = AgentMessage(
        sender=AgentRole.USER,
        recipient=agent_role,
        type=MessageType.FEEDBACK,
        project_id=project_id,
        task_id=task_id,
        payload=MessagePayload(
            content=content,
            metadata={"source": "human"},
        ),
    )
    await bus.publish(msg)
    await db.save_message(msg.model_dump(mode="json"))

    return {
        "status": "ok",
        "message": f"Feedback sent to {agent_role.value} for task {task_id}",
        "task_id": task_id,
    }


@router.post("/projects/{project_id}/review")
async def submit_human_review(project_id: str, request: Request) -> dict:
    """Submit a human review decision for a task.

    Acts like the Critic — accepts ``approved`` or ``needs_revision`` verdicts
    with an optional summary.  The message is sent to the PM who will route it
    through the standard pipeline (approve → next task, reject → revision).
    """
    body = await request.json()
    task_id: str = body.get("task_id", "")
    verdict: str = body.get("verdict", "")
    summary: str = body.get("summary", "")

    if not task_id or verdict not in ("approved", "needs_revision"):
        raise HTTPException(
            status_code=400,
            detail="task_id and verdict ('approved' | 'needs_revision') are required",
        )

    db = request.app.state.db
    bus = request.app.state.bus

    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.state != TaskState.REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Task is in '{task.state.value}' state. Only REVIEW tasks can be reviewed",
        )

    msg = AgentMessage(
        sender=AgentRole.USER,
        recipient=AgentRole.PM,
        type=MessageType.DELIVERABLE,
        project_id=project_id,
        task_id=task_id,
        payload=MessagePayload(
            content=summary or f"Human review: {verdict}",
            metadata={
                "review_result": verdict,
                "source": "human",
            },
        ),
    )
    await bus.publish(msg)
    await db.save_message(msg.model_dump(mode="json"))

    return {
        "status": "ok",
        "verdict": verdict,
        "task_id": task_id,
        "message": f"Human review submitted: {verdict}",
    }


@router.patch("/tasks/{task_id}/steer")
async def steer_task(task_id: str, request: Request) -> dict:
    """Modify task properties mid-flight (steering).

    Allows changing priority, assigned_to, title, and description
    of a task that hasn't been completed yet.
    """
    body = await request.json()
    db = request.app.state.db

    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.state == TaskState.DONE:
        raise HTTPException(status_code=400, detail="Cannot steer a completed task")

    if "priority" in body:
        valid_priorities = ("low", "normal", "high", "critical")
        if body["priority"] not in valid_priorities:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority. Valid: {', '.join(valid_priorities)}",
            )
        task.priority = Priority(body["priority"])

    if "assigned_to" in body:
        try:
            agent_role = AgentRole(body["assigned_to"])
            if agent_role in (AgentRole.SYSTEM, AgentRole.USER):
                raise ValueError
        except ValueError:
            valid = ", ".join(r.value for r in AgentRole if r not in (AgentRole.SYSTEM, AgentRole.USER))
            raise HTTPException(
                status_code=400,
                detail=f"Invalid assigned_to. Valid agent roles: {valid}",
            ) from None
        task.assigned_to = agent_role

    if "title" in body:
        task.title = body["title"]

    if "description" in body:
        task.description = body["description"]

    await db.update_task(task)

    return {
        "status": "ok",
        "task_id": task_id,
        "task": task.model_dump(mode="json"),
    }
