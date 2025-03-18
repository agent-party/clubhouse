"""Tests for the event serialization utilities."""

import json
import uuid
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any
from pydantic import Field

import pytest

from clubhouse.schemas.events.base import EventBase
from clubhouse.schemas.events.agent_lifecycle import AgentLifecycleEvent, AgentLifecycleEventType
from clubhouse.schemas.events.command import CommandEvent
from clubhouse.schemas.events.serialization import (
    serialize_event,
    deserialize_event,
    get_kafka_topic_for_event,
    get_event_key,
    DateTimeEncoder
)


# Custom event classes for testing
class TestEnum(str, Enum):
    """Test enum for serialization testing."""
    ONE = "one"
    TWO = "two"


class TestEvent(EventBase):
    """Test event for serialization utilities."""
    
    agent_id: str
    enum_value: TestEnum
    optional_value: Dict[str, Any] = {}
    
    # Required fields for proper schema validation based on EventBase
    response_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique identifier for this response")
    command_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="ID of the command this responds to")
    session_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Session ID matching the command")
    responder_id: str = Field(default="test_responder", description="ID of the responder")
    recipient_id: str = Field(default="test_recipient", description="ID of the recipient")
    execution_time_ms: int = Field(default=100, description="Execution time in milliseconds")
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        },
        description="Token usage information"
    )


class TestEventWithSessionId(EventBase):
    """Test event with session_id for key generation testing."""
    
    session_id: uuid.UUID


class TestEventWithUserId(EventBase):
    """Test event with user_id for key generation testing."""
    
    user_id: str


class TestSerializationUtils:
    """Tests for the event serialization utilities."""
    
    def test_serialize_event(self):
        """Test serializing an event to JSON."""
        # Create a test event with various field types
        event_id = uuid.uuid4()
        timestamp = datetime.now()
        
        event = TestEvent(
            event_id=event_id,
            timestamp=timestamp,
            producer_id="test_producer",
            agent_id="test_agent",
            enum_value=TestEnum.ONE,
            event_type="test_event"
        )
        
        # Serialize to JSON
        json_str = serialize_event(event)
        
        # Parse the JSON to verify
        data = json.loads(json_str)
        
        assert data["event_id"] == str(event_id)
        assert data["timestamp"] == timestamp.isoformat()
        assert data["producer_id"] == "test_producer"
        assert data["agent_id"] == "test_agent"
        assert data["enum_value"] == "one"  # Enum value should be serialized to string
        
    def test_deserialize_event(self):
        """Test deserializing JSON data to an event object."""
        event_id = uuid.uuid4()
        timestamp = datetime.now()
        
        # Create JSON data for a test event
        json_data = json.dumps({
            "event_id": str(event_id),
            "event_type": "test",
            "event_version": "1.0",
            "timestamp": timestamp.isoformat(),
            "producer_id": "test_producer",
            "agent_id": "test_agent",
            "enum_value": "two",
            "optional_value": {"key": "value"}
        }, cls=DateTimeEncoder)
        
        # Deserialize to event object
        event = deserialize_event(json_data, TestEvent)
        
        assert event.event_id == event_id
        assert event.timestamp.isoformat() == timestamp.isoformat()  # Comparing isoformat to handle microsecond precision
        assert event.producer_id == "test_producer"
        assert event.agent_id == "test_agent"
        assert event.enum_value == TestEnum.TWO
        assert event.optional_value == {"key": "value"}
        
    def test_datetime_encoder(self):
        """Test the DateTimeEncoder."""
        # Create an object with datetime, UUID, enum, and BaseModel
        event = TestEvent(
            producer_id="test_producer",
            agent_id="test_agent",
            enum_value=TestEnum.ONE,
            event_type="test_event"
        )
        
        data = {
            "datetime": datetime.now(),
            "uuid": uuid.uuid4(),
            "enum": TestEnum.TWO,
            "model": event,
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }
        
        # Encode to JSON
        json_str = json.dumps(data, cls=DateTimeEncoder)
        
        # Should serialize without errors
        assert isinstance(json_str, str)
        
        # Parse the JSON to verify
        parsed = json.loads(json_str)
        
        # Check that special types were serialized correctly
        assert isinstance(parsed["datetime"], str)  # datetime -> string
        assert isinstance(parsed["uuid"], str)      # UUID -> string
        assert parsed["enum"] == "two"              # enum -> string value
        assert isinstance(parsed["model"], dict)    # BaseModel -> dict
        assert parsed["list"] == [1, 2, 3]          # list unchanged
        assert parsed["dict"] == {"key": "value"}   # dict unchanged
        
    def test_get_kafka_topic(self):
        """Test getting the Kafka topic for an event."""
        # Create test events with different kafka_topic class variables
        class TopicTestEvent(EventBase):
            kafka_topic: str = "test_topic"
            event_type: str = "topic_test_event"
            
        topic_event = TopicTestEvent(producer_id="test_producer")
        
        # Command event with its own kafka_topic
        command_event = CommandEvent(
            producer_id="test_producer",
            command_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            sender_id="user-123",
            recipient_id="agent-456",
            capability="search",
            event_type="command"
        )
        
        # AgentLifecycleEvent with its own kafka_topic
        lifecycle_event = AgentLifecycleEvent(
            producer_id="test_producer",
            agent_id="agent-123",
            lifecycle_type=AgentLifecycleEventType.CREATED,
            current_state={"name": "Test Agent"},
            initiated_by="user-456",
            event_type="agent_lifecycle"
        )
        
        # Get Kafka topics
        assert get_kafka_topic_for_event(topic_event) == "test_topic"
        assert get_kafka_topic_for_event(command_event) == "agent.commands"
        assert get_kafka_topic_for_event(lifecycle_event) == "agent.lifecycle"
        
    def test_get_event_key(self):
        """Test getting the Kafka message key for an event."""
        # Event with agent_id
        agent_event = TestEvent(
            producer_id="test_producer",
            agent_id="agent-123",
            enum_value=TestEnum.ONE,
            event_type="test_event"
        )
        
        # Event with user_id
        user_event = TestEventWithUserId(
            producer_id="test_producer",
            user_id="user-456",
            event_type="user_event"
        )
        
        # Event with session_id
        session_id = uuid.uuid4()
        session_event = TestEventWithSessionId(
            producer_id="test_producer",
            session_id=session_id,
            event_type="session_event"
        )
        
        # Event with only event_id
        base_event = EventBase(
            producer_id="test_producer",
            event_type="base_event"
        )
        
        # Get Kafka message keys
        assert get_event_key(agent_event) == "agent:agent-123"
        assert get_event_key(user_event) == "user:user-456"
        assert get_event_key(session_event) == f"session:{session_id}"
        assert get_event_key(base_event) == f"event:{base_event.event_id}"
