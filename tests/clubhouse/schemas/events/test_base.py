"""Tests for the base event schema."""

import json
from datetime import datetime
from unittest import TestCase
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from clubhouse.schemas.events.base import EventBase
from tests.clubhouse.schemas.events.test_utils import assert_serialization_roundtrip


class TestEventExample(EventBase):
    """Example event class for testing EventBase."""
    
    event_type: str = "test_event_example"  # Match expected value in test_event_type_auto_generation
    agent_id: str
    action: str


class TestEventBase(TestCase):
    """Tests for the EventBase class."""
    
    def test_required_fields(self):
        """Test that required fields are enforced."""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            TestEventExample()
            
        # All required fields provided should work
        event = TestEventExample(
            producer_id="test_producer",
            agent_id="test_agent",
            action="test_action",
        )
        
        assert event.producer_id == "test_producer"
        assert event.agent_id == "test_agent"
        assert event.action == "test_action"
        
    def test_default_fields(self):
        """Test that default fields are set correctly."""
        event = TestEventExample(
            producer_id="test_producer",
            agent_id="test_agent",
            action="test_action",
        )
        
        # Check automatically generated fields
        assert isinstance(event.event_id, UUID)
        assert isinstance(event.timestamp, datetime)
        assert event.event_version == "1.0"
        assert event.metadata == {}
        assert event.correlation_id is None
        assert event.causation_id is None
        
    def test_event_type_auto_generation(self):
        """Test that event_type is automatically generated from class name."""
        event = TestEventExample(
            producer_id="test_producer",
            agent_id="test_agent",
            action="test_action",
        )
        
        # Class name TestEventExample -> test_event_example
        assert event.event_type == "test_event_example"
        
        # Explicit event_type should override
        explicit_event = TestEventExample(
            producer_id="test_producer",
            agent_id="test_agent",
            action="test_action",
            event_type="explicit_type",
        )
        
        assert explicit_event.event_type == "explicit_type"
        
    def test_kafka_topic(self):
        """Test that the kafka_topic class variable is accessible."""
        assert TestEventExample.kafka_topic == "events"
        
    def test_to_dict(self):
        """Test the to_dict method."""
        event_id = uuid4()
        timestamp = datetime.now()
        
        event = TestEventExample(
            event_id=event_id,
            timestamp=timestamp,
            producer_id="test_producer",
            agent_id="test_agent",
            action="test_action",
        )
        
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert event_dict["event_id"] == event_id
        assert event_dict["timestamp"] == timestamp
        assert event_dict["producer_id"] == "test_producer"
        assert event_dict["agent_id"] == "test_agent"
        assert event_dict["action"] == "test_action"
        
    def test_immutability(self):
        """Test that events are immutable once created."""
        event = TestEventExample(
            producer_id="test_producer",
            agent_id="test_agent",
            action="test_action",
        )
        
        # Attempting to modify the event should raise an error
        with pytest.raises(Exception):
            event.agent_id = "new_agent"
            
    def test_serialization_roundtrip(self):
        """Test that events can be serialized to JSON and back."""
        # Create test data for the event
        event_data = {
            "producer_id": "test_producer",
            "agent_id": "test_agent",
            "action": "test_action",
            "event_id": str(uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event_type": "test_event_example",
            "event_version": "1.0"
        }
        
        # Test the serialization roundtrip
        assert_serialization_roundtrip(TestEventExample, event_data)
