"""
Base event models for the Clubhouse platform.

This module defines the base event models that all other event types inherit from,
ensuring consistent structure and metadata across all events.
"""

from datetime import datetime
from typing import Any, Dict, Optional, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator


class EventBase(BaseModel):
    """
    Base class for all event models in the system.
    
    All events in the system should inherit from this class to ensure
    consistent structure and metadata.
    """
    
    model_config = ConfigDict(
        frozen=True,  # Events are immutable once created
        json_schema_extra={"examples": [{"event_id": "123e4567-e89b-12d3-a456-426614174000"}]},
    )
    
    # Common fields for all events
    event_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this event")
    event_type: str = Field(..., description="Type of event for routing and processing")
    event_version: str = Field(default="1.0", description="Schema version for forward compatibility")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the event occurred")
    producer_id: str = Field(..., description="ID of the component that produced this event")
    correlation_id: Optional[UUID] = Field(default=None, description="ID to correlate related events")
    causation_id: Optional[UUID] = Field(default=None, description="ID of the event that caused this event")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")
    
    # Class variable to define the Kafka topic for this event type
    kafka_topic: ClassVar[str] = "events"
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_default_event_type(cls, v: Optional[str]) -> str:
        """Set default event_type based on class name if not provided."""
        if v is not None:
            return v
        # Convert CamelCase to snake_case and remove "Event" suffix
        class_name = cls.__name__
        if class_name.endswith("Event"):
            class_name = class_name[:-5]  # Remove "Event" suffix
        
        # Convert CamelCase to snake_case
        result = ""
        for i, char in enumerate(class_name):
            if char.isupper() and i > 0:
                result += "_" + char.lower()
            else:
                result += char.lower()
                
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary for Kafka serialization."""
        return self.model_dump()
