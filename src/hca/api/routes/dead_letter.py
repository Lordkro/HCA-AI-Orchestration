"""Dead-letter queue API endpoints for inspecting and replaying failed messages."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from hca.core.message_bus import MessageBus
from hca.core.models import AgentMessage

router = APIRouter()


@router.get("")
async def list_dead_letter(request: Request, count: int = 50) -> list[dict]:
    """List the most recent dead-lettered messages, newest first."""
    bus: MessageBus = request.app.state.bus
    return await bus.list_dead_letter_messages(count=count)


@router.post("/{entry_id}/replay")
async def replay_dead_letter(entry_id: str, request: Request) -> dict:
    """Replay a dead-lettered message by re-publishing it to its original stream.

    The dead-letter entry is deleted after successful replay.
    """
    bus: MessageBus = request.app.state.bus

    entries = await bus.list_dead_letter_messages(count=100)
    target = None
    for entry in entries:
        if entry["id"] == entry_id:
            target = entry
            break

    if not target:
        raise HTTPException(status_code=404, detail="Dead-letter entry not found")

    try:
        msg = AgentMessage(**target["data"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid message data: {e}") from e

    await bus.publish(msg)
    await bus.delete_dead_letter(entry_id)

    return {
        "entry_id": entry_id,
        "original_stream": target["original_stream"],
        "message_id": msg.id,
        "status": "replayed",
    }
