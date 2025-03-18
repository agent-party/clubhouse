"""
Message schemas for Kafka CLI.

This module defines the structure of messages exchanged between the CLI and 
the Clubhouse components through Kafka.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be sent/received."""
    
    # Command types (CLI to Clubhouse)
    COMMAND_CREATE_AGENT = "command.create_agent"
    COMMAND_DELETE_AGENT = "command.delete_agent"
    COMMAND_PROCESS_MESSAGE = "command.process_message"
    
    # Response types (Clubhouse to CLI)
    RESPONSE_AGENT_CREATED = "response.agent_created"
    RESPONSE_AGENT_DELETED = "response.agent_deleted"
    RESPONSE_MESSAGE_PROCESSED = "response.message_processed"
    
    # Event types (Clubhouse to CLI and vice versa)
    EVENT_AGENT_THINKING = "event.agent_thinking"
    EVENT_AGENT_ERROR = "event.agent_error"
    EVENT_AGENT_STATE_CHANGED = "event.agent_state_changed"


class BaseMessage(BaseModel):
    """Base model for all messages."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CreateAgentCommand(BaseMessage):
    """Command to create a new agent."""
    
    message_type: MessageType = MessageType.COMMAND_CREATE_AGENT
    agent_id: str
    personality_type: str = "researcher"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeleteAgentCommand(BaseMessage):
    """Command to delete an agent."""
    
    message_type: MessageType = MessageType.COMMAND_DELETE_AGENT
    agent_id: str


class ProcessMessageCommand(BaseMessage):
    """Command to process a message."""
    
    message_type: MessageType = MessageType.COMMAND_PROCESS_MESSAGE
    agent_id: str
    content: str
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentCreatedResponse(BaseMessage):
    """Response indicating an agent was created."""
    
    message_type: MessageType = MessageType.RESPONSE_AGENT_CREATED
    agent_id: str
    agent_name: str
    agent_role: str
    personality_type: str
    status: str = "created"


class AgentDeletedResponse(BaseMessage):
    """Response indicating an agent was deleted."""
    
    message_type: MessageType = MessageType.RESPONSE_AGENT_DELETED
    agent_id: str
    status: str = "deleted"


class MessageProcessedResponse(BaseMessage):
    """Response to a processed message."""
    
    message_type: MessageType = MessageType.RESPONSE_MESSAGE_PROCESSED
    agent_id: str
    content: str
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentThinkingEvent(BaseMessage):
    """Event indicating an agent is thinking."""
    
    message_type: MessageType = MessageType.EVENT_AGENT_THINKING
    agent_id: str


class AgentErrorEvent(BaseMessage):
    """Event indicating an error occurred with an agent."""
    
    message_type: MessageType = MessageType.EVENT_AGENT_ERROR
    agent_id: str
    error_message: str
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentStateChangedEvent(BaseMessage):
    """Event indicating an agent's state has changed."""
    
    message_type: MessageType = MessageType.EVENT_AGENT_STATE_CHANGED
    agent_id: str
    previous_state: Optional[str] = None
    current_state: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
