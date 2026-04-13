"""Agent status API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def list_agents(request: Request) -> list[dict]:
    """Get the status of all agents."""
    agents = request.app.state.agents
    return [
        {
            "role": agent.role.value,
            "status": agent.status.value,
            "model": agent._model,
        }
        for agent in agents
    ]


@router.get("/{role}")
async def get_agent(role: str, request: Request) -> dict:
    """Get the status of a specific agent."""
    agents = request.app.state.agents
    for agent in agents:
        if agent.role.value == role:
            return {
                "role": agent.role.value,
                "status": agent.status.value,
                "model": agent._model,
                "history_length": len(agent._conversation_history),
            }
    return {"error": f"Agent '{role}' not found"}
