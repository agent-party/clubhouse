"""
Command Event Schema.

This module defines the event schemas for command events that represent
requests to agents and systems within the Clubhouse platform.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List, Union
from uuid import UUID

from pydantic import Field, field_validator, ConfigDict, validator

from clubhouse.schemas.events.base import EventBase


class CommandPriority(str, Enum):
    """Priority levels for commands."""
    
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class CommandEvent(EventBase):
    """
    Events representing commands or requests to agents and systems.
    
    These events encapsulate requests from users or other agents to
    perform specific actions or execute capabilities.
    """
    
    kafka_topic: ClassVar[str] = "agent.commands"
    
    # Core command fields
    command_id: UUID = Field(..., description="Unique identifier for this command")
    session_id: UUID = Field(..., description="Session ID for tracking related commands")
    sender_id: str = Field(..., description="ID of the sender (user or agent)")
    recipient_id: str = Field(..., description="ID of the recipient (agent or service)")
    
    # Command details
    capability: str = Field(..., description="Capability to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the capability"
    )
    
    # Execution control
    priority: CommandPriority = Field(default=CommandPriority.NORMAL, description="Priority of this command")
    timeout_seconds: Optional[int] = Field(
        default=None, description="Timeout in seconds for command execution"
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Expiration time for this command"
    )
    
    # Command flow control
    is_retry: bool = Field(default=False, description="Whether this is a retry of a previous command")
    original_command_id: Optional[UUID] = Field(
        default=None, description="ID of the original command if this is a retry"
    )
    continuation_of: Optional[UUID] = Field(
        default=None, description="ID of the command this continues"
    )
    
    # Context information
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Contextual information for command execution"
    )
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type for command events."""
        return "command"
    
    @validator("expires_at", pre=True, always=True)
    def set_expires_at(cls, v, values):
        """Set expires_at based on timeout_seconds if not provided."""
        if v is not None:
            return v
        timeout = values.get("timeout_seconds")
        if timeout is not None:
            # Get timestamp from values or use current time
            timestamp = values.get("timestamp", datetime.now())
            return timestamp + timedelta(seconds=timeout)
        return None
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "123e4567-e89b-12d3-a456-426614174000",
                    "event_type": "command",
                    "command_id": "567h8901-e89b-12d3-a456-426614174000",
                    "session_id": "789i0123-e89b-12d3-a456-426614174000",
                    "sender_id": "user-789",
                    "recipient_id": "agent-123",
                    "capability": "search",
                    "parameters": {
                        "query": "machine learning techniques",
                        "limit": 10
                    },
                    "priority": "normal",
                    "timeout_seconds": 30,
                    "context": {
                        "conversation_id": "conv-456"
                    }
                }
            ]
        }
    )


# Create a Command alias for backward compatibility and testing
# This is used by the test files and allows simpler schema usage
Command = CommandEvent
