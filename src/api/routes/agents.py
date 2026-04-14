"""Agent status API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/")
async def list_agents(request: Request) -> list[dict]:
    """Get the status of all agents."""
    agents = request.app.state.agents
    return [agent.get_info() for agent in agents]


@router.get("/stats")
async def get_ollama_stats(request: Request) -> dict:
    """Get Ollama client statistics (token usage, performance)."""
    agents = request.app.state.agents
    if agents:
        return agents[0].ollama.get_stats()
    return {}


@router.get("/{role}")
async def get_agent(role: str, request: Request) -> dict:
    """Get the status of a specific agent."""
    agents = request.app.state.agents
    for agent in agents:
        if agent.role.value == role:
            return agent.get_info()
    raise HTTPException(status_code=404, detail=f"Agent '{role}' not found")
