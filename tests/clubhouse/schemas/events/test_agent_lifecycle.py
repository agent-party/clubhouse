"""Tests for the agent lifecycle event schema."""

import pytest
from pydantic import ValidationError

from clubhouse.schemas.events.agent_lifecycle import AgentLifecycleEvent, AgentLifecycleEventType
from tests.clubhouse.schemas.events.test_utils import assert_serialization_roundtrip, create_minimal_event_data


class TestAgentLifecycleEvent:
    """Tests for the AgentLifecycleEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(AgentLifecycleEvent)
        # Add required fields specific to AgentLifecycleEvent
        event_data.update({
            "agent_id": "agent-123",
            "lifecycle_type": AgentLifecycleEventType.CREATED.value,
            "current_state": {"name": "Test Agent", "capabilities": ["test"]},
            "initiated_by": "user-456"
        })
        
        event = AgentLifecycleEvent.model_validate(event_data)
        
        assert event.agent_id == "agent-123"
        assert event.lifecycle_type == AgentLifecycleEventType.CREATED
        assert event.current_state == {"name": "Test Agent", "capabilities": ["test"]}
        assert event.initiated_by == "user-456"
        assert event.event_type == "agent_lifecycle"
        
    def test_event_type_override(self):
        """Test that event_type is always set to agent_lifecycle."""
        event_data = create_minimal_event_data(AgentLifecycleEvent)
        # Add required fields specific to AgentLifecycleEvent
        event_data.update({
            "agent_id": "agent-123",
            "lifecycle_type": AgentLifecycleEventType.CREATED.value,
            "current_state": {"name": "Test Agent", "capabilities": ["test"]},
            "initiated_by": "user-456",
            "event_type": "custom_type"  # This should be overridden
        })
        
        event = AgentLifecycleEvent.model_validate(event_data)
        
        # The event_type should always be agent_lifecycle
        assert event.event_type == "agent_lifecycle"
        
    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        event_data = create_minimal_event_data(AgentLifecycleEvent)
        # Missing agent_id should raise ValidationError
        with pytest.raises(ValidationError):
            AgentLifecycleEvent.model_validate(event_data)
            
    def test_all_lifecycle_types(self):
        """Test creating events with all lifecycle types."""
        event_data = create_minimal_event_data(AgentLifecycleEvent)
        event_data.update({
            "agent_id": "agent-123",
            "current_state": {"name": "Test Agent", "capabilities": ["test"]},
            "initiated_by": "user-456"
        })
        
        for lifecycle_type in AgentLifecycleEventType:
            event_data["lifecycle_type"] = lifecycle_type.value
            event = AgentLifecycleEvent.model_validate(event_data)
            assert event.lifecycle_type == lifecycle_type
            
    def test_update_with_previous_state(self):
        """Test creating an update event with previous state."""
        event_data = create_minimal_event_data(AgentLifecycleEvent)
        event_data.update({
            "agent_id": "agent-123",
            "lifecycle_type": AgentLifecycleEventType.UPDATED.value,
            "current_state": {"name": "Updated Agent", "capabilities": ["test", "new"]},
            "previous_state": {"name": "Test Agent", "capabilities": ["test"]},
            "initiated_by": "user-456",
            "reason": "Adding new capability"
        })
        
        event = AgentLifecycleEvent.model_validate(event_data)
        
        assert event.agent_id == "agent-123"
        assert event.lifecycle_type == AgentLifecycleEventType.UPDATED
        assert event.current_state == {"name": "Updated Agent", "capabilities": ["test", "new"]}
        assert event.previous_state == {"name": "Test Agent", "capabilities": ["test"]}
        assert event.initiated_by == "user-456"
        assert event.reason == "Adding new capability"
        
    def test_serialization_roundtrip(self):
        """Test that events can be serialized to JSON and back."""
        event_data = create_minimal_event_data(AgentLifecycleEvent)
        event_data.update({
            "agent_id": "agent-123",
            "lifecycle_type": AgentLifecycleEventType.CREATED.value,
            "current_state": {"name": "Test Agent", "capabilities": ["test"]},
            "initiated_by": "user-456"
        })
        
        # Test serialization roundtrip with event class and event data
        assert_serialization_roundtrip(AgentLifecycleEvent, event_data)
