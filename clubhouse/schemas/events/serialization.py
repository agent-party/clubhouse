"""
Event Serialization Utilities.

This module provides utilities for serializing and deserializing events
for Kafka integration, including JSON serialization and schema registration.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Type, Union, get_origin, get_args
from uuid import UUID

from pydantic import BaseModel

from clubhouse.schemas.events.base import EventBase

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects and UUIDs."""

    def default(self, obj: Any) -> Any:
        """Convert special Python objects to JSON serializable objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)


def serialize_event(event: EventBase) -> str:
    """
    Serialize an event to JSON format for Kafka.
    
    Args:
        event: The event object to serialize
        
    Returns:
        JSON string representation of the event
    """
    return json.dumps(event.model_dump(), cls=DateTimeEncoder)


def deserialize_event(event_data: str, event_type: Type[EventBase]) -> EventBase:
    """
    Deserialize JSON data into an event object.
    
    Args:
        event_data: JSON string representation of the event
        event_type: The event class to deserialize into
        
    Returns:
        Instantiated event object
    """
    data = json.loads(event_data)
    return event_type.model_validate(data)


def get_kafka_topic_for_event(event: EventBase) -> str:
    """
    Get the Kafka topic for an event.
    
    Args:
        event: The event object
        
    Returns:
        Kafka topic name
    """
    return event.kafka_topic


def get_event_key(event: EventBase) -> str:
    """
    Get the Kafka message key for an event.
    
    The key is used for partitioning. For example, all events for a specific
    agent should go to the same partition for ordering guarantees.
    
    Args:
        event: The event object
        
    Returns:
        String to use as Kafka message key
    """
    # Use agent_id, user_id, or another appropriate field as the key
    if hasattr(event, "agent_id"):
        return f"agent:{event.agent_id}"
    elif hasattr(event, "user_id"):
        return f"user:{event.user_id}"
    elif hasattr(event, "session_id"):
        return f"session:{event.session_id}"
    else:
        # Fall back to event_id to ensure all events have a key
        return f"event:{event.event_id}"
