"""Test utilities for event schema tests."""

import json
from datetime import datetime
from typing import Any, Dict, Type
from uuid import UUID, uuid4

from pydantic import BaseModel

from clubhouse.schemas.events.base import EventBase


def create_minimal_event_data(event_class: Type[EventBase]) -> Dict[str, Any]:
    """
    Create minimal valid data for an event class.
    
    This utility creates the minimum required data to instantiate a valid
    event of the given class, with sensible default values.
    
    Args:
        event_class: The event class to create data for
        
    Returns:
        Dictionary with minimal valid data for the event class
    """
    # Start with common fields for all events
    data = {
        "event_id": str(uuid4()),
        "event_type": event_class.__name__.lower().replace("event", ""),
        "event_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "producer_id": "test_producer",
    }
    
    # Get field definitions from the model
    for name, field in event_class.model_fields.items():
        # Skip fields that are already set or have defaults
        if name in data or field.default is not None or field.default_factory is not None:
            continue
        
        # Skip ClassVar fields
        if getattr(field, "annotation", None) and "ClassVar" in str(field.annotation):
            continue
            
        # Set appropriate test values based on field type
        if field.annotation == str:
            data[name] = f"test_{name}"
        elif field.annotation == int:
            data[name] = 42
        elif field.annotation == float:
            data[name] = 42.0
        elif field.annotation == bool:
            data[name] = True
        elif field.annotation == UUID or getattr(field, "annotation", None) and "UUID" in str(field.annotation):
            data[name] = str(uuid4())
        elif field.annotation == Dict or getattr(field, "annotation", None) and "Dict" in str(field.annotation):
            data[name] = {"test_key": "test_value"}
        elif field.annotation == list or getattr(field, "annotation", None) and "List" in str(field.annotation):
            data[name] = ["test_item"]
        # Handle enum fields
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == list:
            element_type = field.annotation.__args__[0]
            if hasattr(element_type, "__members__"):  # It's an Enum
                data[name] = [list(element_type.__members__.keys())[0]]
            else:
                data[name] = ["test_item"]
        elif hasattr(field.annotation, "__members__"):  # It's an Enum
            data[name] = list(field.annotation.__members__.keys())[0]

    return data


def assert_serialization_roundtrip(event_class: Type[EventBase], event_data: Dict[str, Any]) -> None:
    """
    Test that an event can be serialized to JSON and back.
    
    Args:
        event_class: The event class to test
        event_data: The event data to use for testing
    """
    # Create the event instance
    event = event_class.model_validate(event_data)
    
    # Serialize to JSON
    json_data = event.model_dump_json()
    
    # Deserialize from JSON
    event_dict = json.loads(json_data)
    deserialized = event_class.model_validate(event_dict)
    
    # Check if the deserialized event is equal to the original
    assert event == deserialized, f"Serialization roundtrip failed for {event_class.__name__}"
