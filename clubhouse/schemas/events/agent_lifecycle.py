"""
Agent Lifecycle Event Schema.

This module defines the event schemas for agent lifecycle events such as
agent creation, updates, and deletion.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List
from uuid import UUID

from pydantic import Field, field_validator

from clubhouse.schemas.events.base import EventBase


class AgentLifecycleEventType(str, Enum):
    """Types of agent lifecycle events."""
    
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    CAPABILITY_ADDED = "capability_added"
    CAPABILITY_REMOVED = "capability_removed"
    PERMISSION_CHANGED = "permission_changed"


class AgentLifecycleEvent(EventBase):
    """
    Events related to agent lifecycle changes.
    
    These events are published when an agent is created, updated, deleted,
    or undergoes significant state changes.
    """
    
    kafka_topic: ClassVar[str] = "agent.lifecycle"
    
    agent_id: str = Field(..., description="ID of the agent")
    lifecycle_type: AgentLifecycleEventType = Field(
        ..., description="Type of lifecycle event"
    )
    previous_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Previous state of the agent (for updates)"
    )
    current_state: Dict[str, Any] = Field(
        ..., description="Current state of the agent"
    )
    initiated_by: str = Field(
        ..., description="ID of user or system component that initiated the change"
    )
    reason: Optional[str] = Field(
        default=None, description="Reason for the lifecycle change"
    )
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type based on the lifecycle_type."""
        return "agent_lifecycle"
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "event_id": "123e4567-e89b-12d3-a456-426614174000",
                    "event_type": "agent_lifecycle",
                    "agent_id": "agent-123",
                    "lifecycle_type": "created",
                    "current_state": {
                        "name": "Research Assistant", 
                        "description": "Helps with research tasks",
                        "capabilities": ["search", "summarize"]
                    },
                    "initiated_by": "user-456",
                    "producer_id": "agent_service",
                    "timestamp": "2025-03-16T12:00:00Z"
                }
            ]
        }
