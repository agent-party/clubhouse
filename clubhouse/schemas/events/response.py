"""
Response Event Schema.

This module defines the event schemas for response events that represent
results from agent capability execution and system responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List, Union
from uuid import UUID

from pydantic import Field, field_validator, ConfigDict

from clubhouse.schemas.events.base import EventBase


class ResponseStatus(str, Enum):
    """Status of response events."""
    
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"
    THROTTLED = "throttled"
    UNAUTHORIZED = "unauthorized"
    INVALID_REQUEST = "invalid_request"


class ResponseEvent(EventBase):
    """
    Events representing responses to commands.
    
    These events encapsulate results from capability execution or
    system processing of commands.
    """
    
    kafka_topic: ClassVar[str] = "agent.responses"
    
    # Core response fields
    response_id: UUID = Field(..., description="Unique identifier for this response")
    command_id: UUID = Field(..., description="ID of the command this responds to")
    session_id: UUID = Field(..., description="Session ID matching the command")
    responder_id: str = Field(..., description="ID of the responder (agent or service)")
    recipient_id: str = Field(..., description="ID of the recipient (user or agent)")
    
    # Response details
    status: ResponseStatus = Field(..., description="Status of the response")
    result: Dict[str, Any] = Field(
        default_factory=dict, description="Result data from the command execution"
    )
    
    # Error information
    error_code: Optional[str] = Field(
        default=None, description="Error code if status is not success"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if status is not success"
    )
    
    # Streaming and chunking
    is_streaming: bool = Field(
        default=False, description="Whether this is part of a streaming response"
    )
    is_final: bool = Field(
        default=True, description="Whether this is the final part of a chunked response"
    )
    sequence_number: Optional[int] = Field(
        default=None, description="Sequence number for chunked responses"
    )
    total_chunks: Optional[int] = Field(
        default=None, description="Total number of chunks in a chunked response"
    )
    
    # Performance metrics
    execution_time_ms: Optional[int] = Field(
        default=None, description="Execution time in milliseconds"
    )
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, description="Token usage statistics for LLM operations"
    )
    
    # Additional information
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type for response events."""
        return "response"
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "123e4567-e89b-12d3-a456-426614174000",
                    "event_type": "response",
                    "response_id": "890j1234-e89b-12d3-a456-426614174000",
                    "command_id": "567h8901-e89b-12d3-a456-426614174000",
                    "session_id": "789i0123-e89b-12d3-a456-426614174000",
                    "responder_id": "agent-123",
                    "recipient_id": "user-789",
                    "status": "success",
                    "result": {
                        "search_results": [
                            {"title": "Introduction to Machine Learning", "url": "https://example.com/ml-intro"},
                            {"title": "Advanced ML Techniques", "url": "https://example.com/ml-advanced"}
                        ],
                        "total_matches": 128
                    },
                    "execution_time_ms": 450,
                    "token_usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 350,
                        "total_tokens": 470
                    },
                    "producer_id": "search_capability",
                    "timestamp": "2025-03-16T17:16:00Z"
                }
            ]
        }
    )
