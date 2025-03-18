"""Tests for the agent interaction event schema."""

import pytest
from uuid import uuid4, UUID
from pydantic import ValidationError

from clubhouse.schemas.events.agent_interaction import AgentInteractionEvent, InteractionType
from tests.schemas.events.test_utils import assert_serialization_roundtrip, create_minimal_event_data


class TestAgentInteractionEvent:
    """Tests for the AgentInteractionEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        interaction_id = uuid4()
        # Add required fields specific to AgentInteractionEvent
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(interaction_id),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "model_name": "gpt-4",
            "model_provider": "openai"
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        
        assert event.agent_id == "agent-123"
        assert event.interaction_id == interaction_id
        assert event.interaction_type == InteractionType.GENERATE_RESPONSE
        assert event.prompt_tokens == 150
        assert event.completion_tokens == 50
        assert event.total_tokens == 200  # Should be calculated automatically
        assert event.model_name == "gpt-4"
        assert event.model_provider == "openai"
        assert event.event_type == "agent_interaction"
        
    def test_auto_calculate_total_tokens(self):
        """Test that total_tokens is calculated from prompt and completion tokens."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        # Add required fields specific to AgentInteractionEvent
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "model_name": "gpt-4",
            "model_provider": "openai"
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        assert event.total_tokens == 300
        
        # Explicit total should be respected
        event_data["total_tokens"] = 350
        event = AgentInteractionEvent.model_validate(event_data)
        assert event.total_tokens == 350
        
    def test_all_interaction_types(self):
        """Test creating events with all interaction types."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid4()),
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "model_name": "gpt-4",
            "model_provider": "openai"
        })
        
        for interaction_type in InteractionType:
            event_data["interaction_type"] = interaction_type.value
            event = AgentInteractionEvent.model_validate(event_data)
            assert event.interaction_type == interaction_type
            
    def test_error_tracking(self):
        """Test creating an event with error information."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid4()),
            "interaction_type": InteractionType.EXECUTE_CAPABILITY.value,
            "prompt_tokens": 150,
            "completion_tokens": 0,
            "model_name": "gpt-4",
            "model_provider": "openai",
            "capability_name": "search",
            "success": False,
            "error_type": "rate_limit_exceeded",
            "error_message": "API rate limit exceeded"
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        
        assert event.success is False
        assert event.error_type == "rate_limit_exceeded"
        assert event.error_message == "API rate limit exceeded"
        assert event.capability_name == "search"
        
    def test_cost_tracking(self):
        """Test creating an event with cost tracking."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "model_name": "gpt-4",
            "model_provider": "openai",
            "estimated_cost_usd": 0.0123
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        assert event.estimated_cost_usd == 0.0123
        
    def test_input_output_summaries(self):
        """Test creating an event with input and output summaries."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "model_name": "gpt-4",
            "model_provider": "openai",
            "input_summary": "Query about project architecture",
            "output_summary": "Description of event-driven architecture"
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        assert event.input_summary == "Query about project architecture"
        assert event.output_summary == "Description of event-driven architecture"
        
    def test_serialization_roundtrip(self):
        """Test that events can be serialized to JSON and back."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "model_name": "gpt-4",
            "model_provider": "openai"
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        assert_serialization_roundtrip(event)
