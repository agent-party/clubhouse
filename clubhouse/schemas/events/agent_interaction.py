"""
Agent Interaction Event Schema.

This module defines the event schemas for agent interactions, including
token usage tracking, performance metrics, and error reporting.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List
from uuid import UUID

from pydantic import Field, field_validator, ConfigDict

from clubhouse.schemas.events.base import EventBase


class InteractionType(str, Enum):
    """Types of agent interactions."""
    
    GENERATE_RESPONSE = "generate_response"
    EXECUTE_CAPABILITY = "execute_capability"
    PROCESS_USER_INPUT = "process_user_input"
    COLLABORATION = "collaboration"
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    OTHER = "other"


class AgentInteractionEvent(EventBase):
    """
    Events tracking agent interactions and performance metrics.
    
    These events provide observability into agent operations, including
    token usage accounting, performance metrics, and error tracking.
    """
    
    kafka_topic: ClassVar[str] = "agent.interactions"
    
    # Core interaction fields
    agent_id: str = Field(..., description="ID of the agent")
    interaction_id: UUID = Field(..., description="Unique identifier for this interaction")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    
    # Performance metrics
    prompt_tokens: int = Field(..., description="Number of prompt tokens used")
    completion_tokens: int = Field(..., description="Number of completion tokens generated")
    total_tokens: int = Field(..., description="Total tokens used (prompt + completion)")
    duration_ms: int = Field(..., description="Duration of the interaction in milliseconds")
    
    # Model information
    model_name: str = Field(..., description="Name of the model used")
    model_provider: str = Field(..., description="Provider of the model (e.g., OpenAI, Anthropic)")
    
    # Context
    conversation_id: Optional[str] = Field(None, description="ID of the conversation, if applicable")
    message_id: Optional[str] = Field(None, description="ID of the message, if applicable")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="When this event was created")
    
    # Optional fields for rich context
    input_text: Optional[str] = Field(None, description="Input text for the interaction")
    output_text: Optional[str] = Field(None, description="Output text from the interaction")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Cost analysis
    estimated_cost_usd: Optional[float] = Field(
        None, description="Estimated cost in USD for this interaction"
    )
    
    # Error tracking
    error: Optional[str] = Field(None, description="Error message, if an error occurred")
    error_type: Optional[str] = Field(None, description="Type of error, if an error occurred")
    stack_trace: Optional[str] = Field(None, description="Stack trace, if an error occurred")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "agent_id": "agent-123",
                    "interaction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "interaction_type": "generate_response",
                    "prompt_tokens": 150,
                    "completion_tokens": 50,
                    "total_tokens": 200,
                    "duration_ms": 1200,
                    "model_name": "gpt-4",
                    "model_provider": "OpenAI",
                    "conversation_id": "conv-123",
                    "created_at": "2023-01-01T00:00:00Z",
                    "input_text": "Tell me about machine learning",
                    "output_text": "Machine learning is a branch of AI...",
                    "metadata": {
                        "user_id": "user-123",
                        "session_id": "session-123"
                    },
                    "estimated_cost_usd": 0.02
                }
            ]
        }
    )
    
    @field_validator("error_type")
    def validate_error_type(cls, error_type: Optional[str], info: Any) -> Optional[str]:
        """Validate that error is set if error_type is set."""
        if error_type is not None:
            error = info.data.get("error")
            if not error:
                raise ValueError("Error must be set if error_type is set")
        return error_type
    
    # Set event_type to a consistent value
    @field_validator("event_type", mode="before")
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type to a consistent value."""
        return "agent_interaction"


class ConversationCreatedEvent(EventBase):
    """
    Event emitted when a new conversation is created.
    
    This event provides information about the creation of a new conversation,
    including the conversation ID, title, and metadata.
    """
    
    kafka_topic: ClassVar[str] = "conversation.created"
    
    # Core conversation fields
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    title: str = Field(..., description="Title of the conversation")
    
    # Context and metadata
    created_at: datetime = Field(default_factory=datetime.now, description="When this conversation was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Optional agent information
    agent_id: Optional[str] = Field(None, description="ID of the associated agent, if applicable")
    user_id: Optional[str] = Field(None, description="ID of the user who created the conversation, if applicable")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "conversation_id": "conv-123",
                    "title": "Discussion about AI",
                    "created_at": "2023-01-01T00:00:00Z",
                    "metadata": {
                        "tags": ["ai", "discussion"],
                        "importance": "high"
                    },
                    "agent_id": "agent-123",
                    "user_id": "user-456"
                }
            ]
        }
    )
    
    # Set event_type to a consistent value
    @field_validator("event_type", mode="before")
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type to a consistent value."""
        return "conversation_created"


class MessageAddedEvent(EventBase):
    """
    Event emitted when a message is added to a conversation.
    
    This event provides information about a message being added to a conversation,
    including the message content, sender, and relevant IDs.
    """
    
    kafka_topic: ClassVar[str] = "conversation.message.added"
    
    # Core message fields
    conversation_id: str = Field(..., description="ID of the conversation the message belongs to")
    message_id: str = Field(..., description="Unique identifier for the message")
    content: str = Field(..., description="Content of the message")
    sender: str = Field(..., description="Sender of the message (user ID or agent ID)")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now, description="When this message was created")
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "conversation_id": "conv-123",
                    "message_id": "msg-456",
                    "content": "Hello, how can I help you today?",
                    "sender": "agent-123",
                    "created_at": "2023-01-01T00:00:00Z",
                    "metadata": {
                        "sentiment": "positive",
                        "tokens": 8
                    }
                }
            ]
        }
    )
    
    # Set event_type to a consistent value
    @field_validator("event_type", mode="before")
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type to a consistent value."""
        return "message_added"


class ConversationDeletedEvent(EventBase):
    """
    Event emitted when a conversation is deleted.
    
    This event provides information about a conversation being deleted,
    including the conversation ID and any relevant metadata.
    """
    
    kafka_topic: ClassVar[str] = "conversation.deleted"
    
    # Core fields
    conversation_id: str = Field(..., description="ID of the deleted conversation")
    
    # Timing
    deleted_at: datetime = Field(default_factory=datetime.now, description="When this conversation was deleted")
    
    # Context
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Optional fields
    agent_id: Optional[str] = Field(None, description="ID of the associated agent, if applicable")
    user_id: Optional[str] = Field(None, description="ID of the user who deleted the conversation, if applicable")
    reason: Optional[str] = Field(None, description="Reason for deletion, if provided")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "conversation_id": "conv-123",
                    "deleted_at": "2023-01-01T00:00:00Z",
                    "metadata": {
                        "archived": True,
                        "storage_path": "s3://backup/conv-123"
                    },
                    "reason": "User requested deletion"
                }
            ]
        }
    )
    
    # Set event_type to a consistent value
    @field_validator("event_type", mode="before")
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type to a consistent value."""
        return "conversation_deleted"
