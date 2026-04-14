"""Pydantic data models for HCA Orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================
# Enums
# ============================================================


class TaskState(str, Enum):
    """Lifecycle states for a task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    REVISION = "revision"
    APPROVED = "approved"
    DONE = "done"
    FAILED = "failed"


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK_ASSIGNMENT = "task_assignment"
    DELIVERABLE = "deliverable"
    FEEDBACK = "feedback"
    STATUS_UPDATE = "status_update"
    QUESTION = "question"
    ANSWER = "answer"
    SYSTEM = "system"


class Priority(str, Enum):
    """Message / task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    """Roles of the agents in the system."""
    PM = "pm"
    RESEARCH = "research"
    SPEC = "spec"
    CODER = "coder"
    CRITIC = "critic"
    SYSTEM = "system"
    USER = "user"


class AgentStatus(str, Enum):
    """Operational status of an agent."""
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    STOPPED = "stopped"


# ============================================================
# Core Models
# ============================================================


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class MessagePayload(BaseModel):
    """The content portion of an agent message."""
    content: str
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class AgentMessage(BaseModel):
    """A message passed between agents via the message bus."""
    id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_utc_now)
    sender: AgentRole
    recipient: AgentRole | str  # AgentRole or "*" for broadcast
    type: MessageType
    project_id: str
    task_id: str = ""
    payload: MessagePayload
    priority: Priority = Priority.NORMAL


class Project(BaseModel):
    """A project being built by the agent team."""
    id: str = Field(default_factory=_new_id)
    name: str
    description: str
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    status: str = "active"  # active, paused, completed, failed
    idea: str  # The original product idea from the user


class Task(BaseModel):
    """A unit of work within a project."""
    id: str = Field(default_factory=_new_id)
    project_id: str
    title: str
    description: str
    state: TaskState = TaskState.PENDING
    assigned_to: AgentRole | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    iteration: int = 0
    max_iterations: int = 5
    parent_task_id: str = ""
    deliverable: str = ""
    feedback: str = ""
    priority: Priority = Priority.NORMAL


class Artifact(BaseModel):
    """A file or document produced by an agent."""
    id: str = Field(default_factory=_new_id)
    project_id: str
    task_id: str
    agent: AgentRole
    filename: str
    content: str
    artifact_type: str  # "code", "spec", "research", "doc", "test"
    created_at: datetime = Field(default_factory=_utc_now)
    version: int = 1


class ConversationEntry(BaseModel):
    """A single turn in an agent's conversation history."""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = Field(default_factory=_utc_now)
