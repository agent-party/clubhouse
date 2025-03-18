"""
Tests for the ConversationCapability.

This module contains tests for the ConversationCapability, which manages
conversation context, history, and state for multi-turn interactions.
"""

import asyncio
import pytest
from typing import Dict, Any, List
import uuid

from clubhouse.agents.capability import BaseCapability
from clubhouse.agents.capabilities.conversation_capability import (
    ConversationCapability,
    ConversationParameters,
    ConversationMessage,
    ConversationContext
)
from clubhouse.agents.errors import ValidationError, ExecutionError

class TestConversationCapability:
    """Test cases for the ConversationCapability."""
    
    @pytest.fixture
    def conversation_capability(self) -> ConversationCapability:
        """Create a ConversationCapability for testing."""
        return ConversationCapability()
    
    def test_capability_properties(self, conversation_capability: ConversationCapability):
        """Test the basic properties of the capability."""
        assert conversation_capability.name == "conversation"
        assert "manage conversations" in conversation_capability.description.lower()
        assert isinstance(conversation_capability.parameters, dict)
        assert "message" in conversation_capability.parameters
        assert isinstance(conversation_capability, BaseCapability)
    
    def test_validate_parameters_valid(self, conversation_capability: ConversationCapability):
        """Test parameter validation with valid parameters."""
        # Basic message parameters
        params = {
            "message": "Hello, how are you?",
            "conversation_id": str(uuid.uuid4())
        }
        
        validated = conversation_capability.validate_parameters(**params)
        assert validated["message"] == params["message"]
        assert validated["conversation_id"] == params["conversation_id"]
        
        # Full parameters with optional fields
        full_params = {
            "message": "Tell me more about that",
            "conversation_id": str(uuid.uuid4()),
            "metadata": {"source": "user", "timestamp": "2025-03-16T22:00:00Z"},
            "context": {"topic": "AI agents", "reference_message_id": "12345"}
        }
        
        validated = conversation_capability.validate_parameters(**full_params)
        assert validated["message"] == full_params["message"]
        assert validated["metadata"] == full_params["metadata"]
        assert validated["context"]["topic"] == "AI agents"
    
    def test_validate_parameters_invalid(self, conversation_capability: ConversationCapability):
        """Test parameter validation with invalid parameters."""
        # Missing required message
        with pytest.raises(ValidationError):
            conversation_capability.validate_parameters(conversation_id="12345")
        
        # Invalid metadata type
        with pytest.raises(ValidationError):
            conversation_capability.validate_parameters(
                message="Hello",
                conversation_id="12345",
                metadata="not a dict"
            )
    
    @pytest.mark.asyncio
    async def test_add_message_to_conversation(self, conversation_capability: ConversationCapability):
        """Test adding a message to a conversation."""
        conversation_id = str(uuid.uuid4())
        message = "Hello, how can I help you today?"
        
        # Add a new message
        result = await conversation_capability.execute(
            message=message,
            conversation_id=conversation_id
        )
        
        # Check the result
        assert result.result["status"] == "success"
        assert "message_id" in result.result["data"]
        assert result.result["data"]["message"] == message
        
        # Check that the message was added to history
        history = conversation_capability.get_conversation_history(conversation_id)
        assert len(history) == 1
        assert history[0]["message"] == message
    
    @pytest.mark.asyncio
    async def test_get_conversation_context(self, conversation_capability: ConversationCapability):
        """Test retrieving conversation context."""
        conversation_id = str(uuid.uuid4())
        
        # Add several messages to build context
        await conversation_capability.execute(
            message="Hello, I'd like to learn about AI",
            conversation_id=conversation_id
        )
        
        await conversation_capability.execute(
            message="Specifically, I'm interested in agent-based systems",
            conversation_id=conversation_id,
            context={"topic": "AI systems", "language": "en"}
        )
        
        # Get the conversation context
        context = conversation_capability.get_conversation_context(conversation_id)
        
        # Verify context
        assert context.topic == "AI systems"
        assert context.conversation_id == conversation_id
        assert context.message_count == 2
        assert len(context.message_history) == 2
        
        # Verify we can access attributes
        assert "language" in context.attributes
        assert context.attributes["language"] == "en"
        # Also verify dynamic attribute access works
        assert context.language == "en"
    
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle(self, conversation_capability: ConversationCapability):
        """Test the full execution lifecycle of the capability."""
        # Set up test event tracking
        events_triggered = []
        
        def track_event(**kwargs):
            events_triggered.append(kwargs.get("event_type", "unknown"))
        
        # Register event handlers
        conversation_capability.register_event_handler("before_execution", track_event)
        conversation_capability.register_event_handler("after_execution", track_event)
        conversation_capability.register_event_handler("conversation.started", track_event)
        conversation_capability.register_event_handler("conversation.message_added", track_event)
        
        # Execute with lifecycle
        result = await conversation_capability.execute_with_lifecycle(
            message="Let's start a conversation",
            conversation_id="test-convo-123"
        )
        
        # Check result
        assert result["status"] == "success"
        assert "message_id" in result["data"]
        
        # Verify events were triggered in the right order
        assert "before_execution" in events_triggered
        assert "conversation.started" in events_triggered
        assert "conversation.message_added" in events_triggered
        assert "after_execution" in events_triggered
        
        # Verify order of events
        assert events_triggered.index("before_execution") < events_triggered.index("after_execution")
    
    @pytest.mark.asyncio
    async def test_context_persistence(self, conversation_capability: ConversationCapability):
        """Test that context persists across multiple messages in a conversation."""
        conversation_id = "persistent-context-test"
        
        # Add first message with context
        await conversation_capability.execute(
            message="Let's talk about Python",
            conversation_id=conversation_id,
            context={"topic": "programming", "language": "Python"}
        )
        
        # Add second message without explicit context
        await conversation_capability.execute(
            message="What are the best practices?",
            conversation_id=conversation_id
        )
        
        # Verify context persisted
        context = conversation_capability.get_conversation_context(conversation_id)
        assert context.topic == "programming"
        assert context.language == "Python"
        
        # Update context with third message
        await conversation_capability.execute(
            message="Actually, let's switch to discussing Java",
            conversation_id=conversation_id,
            context={"language": "Java"}
        )
        
        # Verify context was updated but maintained other fields
        updated_context = conversation_capability.get_conversation_context(conversation_id)
        assert updated_context.topic == "programming"  # Maintained
        assert updated_context.language == "Java"  # Updated
    
    @pytest.mark.asyncio
    async def test_reference_previous_messages(self, conversation_capability: ConversationCapability):
        """Test the ability to reference previous messages in a conversation."""
        conversation_id = "reference-test"
        
        # Add first message
        result1 = await conversation_capability.execute(
            message="What's the capital of France?",
            conversation_id=conversation_id
        )
        first_message_id = result1.result["data"]["message_id"]
        
        # Add second message referencing first
        result2 = await conversation_capability.execute(
            message="Why is it famous?",
            conversation_id=conversation_id,
            context={"reference_message_id": first_message_id}
        )
        
        # Verify reference was recorded
        history = conversation_capability.get_conversation_history(conversation_id)
        assert len(history) == 2
        assert history[1]["context"]["reference_message_id"] == first_message_id
        
        # Test retrieving a referenced message
        referenced_message = conversation_capability.get_message_by_id(
            conversation_id, first_message_id
        )
        assert referenced_message["message"] == "What's the capital of France?"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, conversation_capability: ConversationCapability):
        """Test error handling in the capability."""
        # Force an error by providing invalid conversation ID type
        with pytest.raises(ValidationError):
            await conversation_capability.execute(
                message="Hello",
                conversation_id=123  # Should be string, not int
            )
        
        # Test handling of execution errors
        conversation_capability._force_error = True  # Special testing flag
        
        result = await conversation_capability.execute_with_lifecycle(
            message="This should trigger an error",
            conversation_id="error-test"
        )
        
        assert result["status"] == "error"
        assert "error" in result
        assert "conversation" in result["error"].lower()
        
        # Reset for other tests
        conversation_capability._force_error = False

    @pytest.mark.asyncio
    async def test_resolve_reference_in_message(self, conversation_capability: ConversationCapability):
        """Test resolving references in messages."""
        # Step 1: Add an initial message
        initial_result = await conversation_capability.execute(
            conversation_id="reference-test-convo",
            message="I need information about the Python programming language",
        )
        
        # Verify the initial message was added
        assert initial_result.result["status"] == "success"
        initial_message_id = initial_result.result["data"]["message_id"]
        
        # Step 2: Add a follow-up message with a reference
        follow_up_result = await conversation_capability.execute(
            conversation_id="reference-test-convo",
            message="Tell me more about its history",
            resolve_references=True  # Enable reference resolution
        )
        
        # Verify the follow-up message was added
        assert follow_up_result.result["status"] == "success"
        
        # Step 3: Check that the context was resolved
        context = conversation_capability.get_conversation_context("reference-test-convo")
        
        # The latest message should have reference to the original topic
        latest_message = context.message_history[-1]
        assert "metadata" in latest_message
        assert "reference_resolved" in latest_message["metadata"]
        assert latest_message["metadata"]["reference_resolved"] is True
        
        # The referenced context should contain elements from the first message
        assert "referenced_entity" in latest_message["metadata"]
        assert "Python" in latest_message["metadata"]["referenced_entity"]
        
    @pytest.mark.asyncio
    async def test_query_chaining(self, conversation_capability: ConversationCapability):
        """Test chaining multiple queries with context preservation."""
        # First query
        await conversation_capability.execute(
            conversation_id="chain-test",
            message="What are the main features of Python 3.9?",
        )
        
        # Second query with implicit reference
        second_result = await conversation_capability.execute(
            conversation_id="chain-test",
            message="How do those compare to version 3.8?",
            resolve_references=True
        )
        
        # The second query should maintain context from the first
        assert second_result.result["status"] == "success"
        
        # Get the full context
        context = conversation_capability.get_conversation_context("chain-test")
        
        # Check that the context chain is maintained
        assert len(context.message_history) == 2
        assert "query_chain" in context.attributes
        assert len(context.attributes["query_chain"]) == 1  # Only the second message creates a chain entry
        
        # The second message should reference the first in some way
        second_message = context.message_history[-1]
        assert "metadata" in second_message
        assert "previous_query_reference" in second_message["metadata"]
        assert second_message["metadata"]["previous_query_reference"] is True
