"""Tests for the command event schema."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from pydantic import ValidationError

from clubhouse.schemas.events.command import CommandEvent, CommandPriority
from tests.schemas.events.test_utils import assert_serialization_roundtrip, create_minimal_event_data


class TestCommandEvent:
    """Tests for the CommandEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(CommandEvent)
        command_id = uuid4()
        session_id = uuid4()
        # Add required fields specific to CommandEvent
        event_data.update({
            "command_id": str(command_id),
            "session_id": str(session_id),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search"
        })
        
        event = CommandEvent.model_validate(event_data)
        
        assert event.command_id == command_id
        assert event.session_id == session_id
        assert event.sender_id == "user-123"
        assert event.recipient_id == "agent-456"
        assert event.capability == "search"
        assert event.event_type == "command"
        assert event.parameters == {}  # Default empty dict
        assert event.priority == CommandPriority.NORMAL  # Default priority
        
    def test_with_parameters(self):
        """Test creating a command with parameters."""
        event_data = create_minimal_event_data(CommandEvent)
        event_data.update({
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search",
            "parameters": {
                "query": "machine learning techniques",
                "max_results": 5
            }
        })
        
        event = CommandEvent.model_validate(event_data)
        assert event.parameters == {
            "query": "machine learning techniques",
            "max_results": 5
        }
        
    def test_priority_levels(self):
        """Test creating commands with different priority levels."""
        event_data = create_minimal_event_data(CommandEvent)
        event_data.update({
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search"
        })
        
        for priority in CommandPriority:
            event_data["priority"] = priority.value
            event = CommandEvent.model_validate(event_data)
            assert event.priority == priority
            
    def test_timeout_and_expiry(self):
        """Test timeout and expiry calculation."""
        event_data = create_minimal_event_data(CommandEvent)
        timestamp = datetime.now()
        event_data.update({
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search",
            "timestamp": timestamp.isoformat(),
            "timeout_seconds": 30
        })
        
        event = CommandEvent.model_validate(event_data)
        # expires_at should be set based on timeout_seconds
        assert event.expires_at is not None
        expected_expiry = timestamp + timedelta(seconds=30)
        # Allow small deviation due to processing time
        assert abs((event.expires_at - expected_expiry).total_seconds()) < 1
        
        # Explicit expires_at should override timeout calculation
        explicit_expiry = timestamp + timedelta(seconds=60)
        event_data["expires_at"] = explicit_expiry.isoformat()
        event = CommandEvent.model_validate(event_data)
        assert event.expires_at == explicit_expiry
        
    def test_continuation_and_retry(self):
        """Test continuation and retry fields."""
        original_command_id = uuid4()
        continuation_of = uuid4()
        
        event_data = create_minimal_event_data(CommandEvent)
        event_data.update({
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search",
            "is_retry": True,
            "original_command_id": str(original_command_id),
            "continuation_of": str(continuation_of)
        })
        
        event = CommandEvent.model_validate(event_data)
        assert event.is_retry is True
        assert event.original_command_id == original_command_id
        assert event.continuation_of == continuation_of
        
    def test_context_information(self):
        """Test context information."""
        event_data = create_minimal_event_data(CommandEvent)
        event_data.update({
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search",
            "context": {
                "conversation_id": "conv-789",
                "theme": "research",
                "previous_topics": ["databases", "machine learning"]
            }
        })
        
        event = CommandEvent.model_validate(event_data)
        assert event.context == {
            "conversation_id": "conv-789",
            "theme": "research",
            "previous_topics": ["databases", "machine learning"]
        }
        
    def test_serialization_roundtrip(self):
        """Test that events can be serialized to JSON and back."""
        event_data = create_minimal_event_data(CommandEvent)
        event_data.update({
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "sender_id": "user-123",
            "recipient_id": "agent-456",
            "capability": "search",
            "parameters": {"query": "test"}
        })
        
        event = CommandEvent.model_validate(event_data)
        assert_serialization_roundtrip(event)
