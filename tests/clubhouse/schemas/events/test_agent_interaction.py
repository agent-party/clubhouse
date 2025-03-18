"""Tests for the agent interaction event schema."""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from clubhouse.schemas.events.agent_interaction import (
    AgentInteractionEvent,
    InteractionType,
    ConversationCreatedEvent,
    MessageAddedEvent,
    ConversationDeletedEvent
)
from tests.clubhouse.schemas.events.test_utils import create_minimal_event_data, assert_serialization_roundtrip


class TestAgentInteractionEvent:
    """Tests for the AgentInteractionEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        interaction_id = uuid.uuid4()
        
        # Add required fields specific to AgentInteractionEvent
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(interaction_id),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,  # Required field
            "duration_ms": 450,  # Required field
            "model_name": "gpt-4",
            "model_provider": "openai"
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.agent_id == "agent-123"
        assert event.interaction_id == interaction_id
        assert event.interaction_type == InteractionType.GENERATE_RESPONSE
        assert event.prompt_tokens == 150
        assert event.completion_tokens == 50
        assert event.total_tokens == 200
        assert event.duration_ms == 450
        assert event.model_name == "gpt-4"
        assert event.model_provider == "openai"
        assert event.event_type == "agent_interaction"
    
    def test_full_valid_event(self):
        """Test creating a fully populated valid event."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        interaction_id = uuid.uuid4()
        
        # Add all possible fields
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(interaction_id),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
            "duration_ms": 450,
            "model_name": "gpt-4",
            "model_provider": "openai",
            "conversation_id": "conv-123",
            "message_id": "msg-456",
            "input_text": "Tell me about AI",
            "output_text": "AI stands for artificial intelligence...",
            "metadata": {"tags": ["ai", "explanation"]},
            "estimated_cost_usd": 0.0123
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.agent_id == "agent-123"
        assert event.interaction_id == interaction_id
        assert event.interaction_type == InteractionType.GENERATE_RESPONSE
        assert event.prompt_tokens == 150
        assert event.completion_tokens == 50
        assert event.total_tokens == 200
        assert event.duration_ms == 450
        assert event.model_name == "gpt-4"
        assert event.model_provider == "openai"
        assert event.conversation_id == "conv-123"
        assert event.message_id == "msg-456"
        assert event.input_text == "Tell me about AI"
        assert event.output_text == "AI stands for artificial intelligence..."
        assert event.metadata == {"tags": ["ai", "explanation"]}
        assert event.estimated_cost_usd == 0.0123
    
    def test_error_fields(self):
        """Test error-related fields."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid.uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 0,
            "total_tokens": 150,
            "duration_ms": 150,
            "model_name": "gpt-4",
            "model_provider": "openai",
            "error": "Rate limit exceeded",
            "error_type": "RateLimitError",
            "stack_trace": "Traceback: ...",
        })
        
        event = AgentInteractionEvent.model_validate(event_data)
        
        assert event.error == "Rate limit exceeded"
        assert event.error_type == "RateLimitError"
        assert event.stack_trace == "Traceback: ..."
    
    def test_error_fields_validation(self):
        """Test validation that error must be set if error_type is set."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid.uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 0,
            "total_tokens": 150,
            "duration_ms": 150,
            "model_name": "gpt-4",
            "model_provider": "openai",
            "error_type": "RateLimitError",  # Setting error_type without error
        })
        
        with pytest.raises(ValidationError) as exc_info:
            AgentInteractionEvent.model_validate(event_data)
        
        assert "Error must be set if error_type is set" in str(exc_info.value)
    
    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        event_data = create_minimal_event_data(AgentInteractionEvent)
        event_data.update({
            "agent_id": "agent-123",
            "interaction_id": str(uuid.uuid4()),
            "interaction_type": InteractionType.GENERATE_RESPONSE.value,
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
            "duration_ms": 450,
            "model_name": "gpt-4",
            "model_provider": "openai"
        })
        
        assert_serialization_roundtrip(AgentInteractionEvent, event_data)


class TestConversationCreatedEvent:
    """Tests for the ConversationCreatedEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(ConversationCreatedEvent)
        
        # Add required fields specific to ConversationCreatedEvent
        event_data.update({
            "conversation_id": "conv-123",
            "title": "Test Conversation"
        })
        
        event = ConversationCreatedEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.conversation_id == "conv-123"
        assert event.title == "Test Conversation"
        assert event.created_at is not None
        assert isinstance(event.metadata, dict)
    
    def test_full_valid_event(self):
        """Test creating a fully populated valid event."""
        event_data = create_minimal_event_data(ConversationCreatedEvent)
        created_at = datetime.now()
        
        # Add all possible fields
        event_data.update({
            "conversation_id": "conv-123",
            "title": "Discussion about AI",
            "created_at": created_at.isoformat(),
            "metadata": {"tags": ["ai", "discussion"], "importance": "high"},
            "agent_id": "agent-123",
            "user_id": "user-456"
        })
        
        event = ConversationCreatedEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.conversation_id == "conv-123"
        assert event.title == "Discussion about AI"
        assert event.metadata == {"tags": ["ai", "discussion"], "importance": "high"}
        assert event.agent_id == "agent-123"
        assert event.user_id == "user-456"
    
    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        event_data = create_minimal_event_data(ConversationCreatedEvent)
        event_data.update({
            "conversation_id": "conv-123",
            "title": "Test Conversation",
            "agent_id": "agent-123",
            "user_id": "user-456"
        })
        
        assert_serialization_roundtrip(ConversationCreatedEvent, event_data)


class TestMessageAddedEvent:
    """Tests for the MessageAddedEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(MessageAddedEvent)
        
        # Add required fields specific to MessageAddedEvent
        event_data.update({
            "conversation_id": "conv-123",
            "message_id": "msg-456",
            "content": "Hello, how can I help you today?",
            "sender": "agent-123"
        })
        
        event = MessageAddedEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.conversation_id == "conv-123"
        assert event.message_id == "msg-456"
        assert event.content == "Hello, how can I help you today?"
        assert event.sender == "agent-123"
        assert event.created_at is not None
        assert isinstance(event.metadata, dict)
    
    def test_full_valid_event(self):
        """Test creating a fully populated valid event."""
        event_data = create_minimal_event_data(MessageAddedEvent)
        created_at = datetime.now()
        
        # Add all possible fields
        event_data.update({
            "conversation_id": "conv-123",
            "message_id": "msg-456",
            "content": "Hello, how can I help you today?",
            "sender": "agent-123",
            "created_at": created_at.isoformat(),
            "metadata": {"sentiment": "positive", "tokens": 8}
        })
        
        event = MessageAddedEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.conversation_id == "conv-123"
        assert event.message_id == "msg-456"
        assert event.content == "Hello, how can I help you today?"
        assert event.sender == "agent-123"
        assert event.metadata == {"sentiment": "positive", "tokens": 8}
    
    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        event_data = create_minimal_event_data(MessageAddedEvent)
        event_data.update({
            "conversation_id": "conv-123",
            "message_id": "msg-456",
            "content": "Hello, how can I help you today?",
            "sender": "agent-123",
            "metadata": {"tokens": 8}
        })
        
        assert_serialization_roundtrip(MessageAddedEvent, event_data)


class TestConversationDeletedEvent:
    """Tests for the ConversationDeletedEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(ConversationDeletedEvent)
        
        # Add required fields specific to ConversationDeletedEvent
        event_data.update({
            "conversation_id": "conv-123"
        })
        
        event = ConversationDeletedEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.conversation_id == "conv-123"
        assert event.deleted_at is not None
        assert isinstance(event.metadata, dict)
        assert event.agent_id is None
        assert event.user_id is None
        assert event.reason is None
    
    def test_full_valid_event(self):
        """Test creating a fully populated valid event."""
        event_data = create_minimal_event_data(ConversationDeletedEvent)
        deleted_at = datetime.now()
        
        # Add all possible fields
        event_data.update({
            "conversation_id": "conv-123",
            "deleted_at": deleted_at.isoformat(),
            "metadata": {"archived": True, "storage_path": "s3://backup/conv-123"},
            "agent_id": "agent-123",
            "user_id": "user-456",
            "reason": "User requested deletion"
        })
        
        event = ConversationDeletedEvent.model_validate(event_data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.conversation_id == "conv-123"
        assert event.metadata == {"archived": True, "storage_path": "s3://backup/conv-123"}
        assert event.agent_id == "agent-123"
        assert event.user_id == "user-456"
        assert event.reason == "User requested deletion"
    
    def test_serialization_roundtrip(self):
        """Test serialization and deserialization."""
        event_data = create_minimal_event_data(ConversationDeletedEvent)
        event_data.update({
            "conversation_id": "conv-123",
            "agent_id": "agent-123",
            "user_id": "user-456",
            "reason": "User requested deletion"
        })
        
        assert_serialization_roundtrip(ConversationDeletedEvent, event_data)
