"""
Tests for the clubhouse conversation manager service.

This module contains tests for the conversation manager service, which is responsible
for tracking and managing conversations between users and agents.
"""

import pytest
import unittest
from unittest.mock import MagicMock, patch, ANY
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from clubhouse.services.conversation_manager import (
    Conversation, 
    ConversationManager, 
    Message
)
from clubhouse.messaging.event_publisher import EventPublisherProtocol
from clubhouse.schemas.events.agent_interaction import (
    ConversationCreatedEvent, 
    MessageAddedEvent,
    ConversationDeletedEvent
)


@pytest.fixture
def message():
    """Fixture for a message object."""
    return Message(
        content="Hello",
        sender="user1",
        message_id="msg1",
        metadata={"key": "value"}
    )


@pytest.fixture
def conversation():
    """Fixture for a conversation object."""
    return Conversation(
        conversation_id="conv1",
        title="Test Conversation",
        metadata={"key": "value"}
    )


@pytest.fixture
def conversation_manager():
    """Fixture for a conversation manager."""
    return ConversationManager()


def test_message_initialization():
    """Test that a message initializes correctly."""
    message = Message(
        content="Hello",
        sender="user1",
        message_id="msg1",
        metadata={"key": "value"}
    )
    
    assert message.content == "Hello"
    assert message.sender == "user1"
    assert message.message_id == "msg1"
    assert message.metadata == {"key": "value"}
    assert isinstance(message.timestamp, datetime)


def test_message_to_dict(message):
    """Test converting a message to a dictionary."""
    message_dict = message.to_dict()
    
    assert message_dict["message_id"] == "msg1"
    assert message_dict["content"] == "Hello"
    assert message_dict["sender"] == "user1"
    assert "timestamp" in message_dict
    assert message_dict["metadata"] == {"key": "value"}


def test_message_from_dict():
    """Test creating a message from a dictionary."""
    now = datetime.now(timezone.utc)
    message_dict = {
        "message_id": "msg1",
        "content": "Hello",
        "sender": "user1",
        "timestamp": now.isoformat(),
        "metadata": {"key": "value"}
    }
    
    message = Message.from_dict(message_dict)
    
    assert message.message_id == "msg1"
    assert message.content == "Hello"
    assert message.sender == "user1"
    assert message.timestamp.isoformat() == now.isoformat()
    assert message.metadata == {"key": "value"}


def test_conversation_initialization():
    """Test that a conversation initializes correctly."""
    conversation = Conversation(
        conversation_id="conv1",
        title="Test Conversation",
        metadata={"key": "value"}
    )
    
    assert conversation.conversation_id == "conv1"
    assert conversation.title == "Test Conversation"
    assert conversation.metadata == {"key": "value"}
    assert conversation.messages == []
    assert isinstance(conversation.created_at, datetime)
    assert isinstance(conversation.updated_at, datetime)


def test_conversation_add_message(conversation, message):
    """Test adding a message to a conversation."""
    # Record the updated_at time before adding the message
    before_update = conversation.updated_at
    
    # Add the message
    conversation.add_message(message)
    
    # Verify the message was added
    assert len(conversation.messages) == 1
    assert conversation.messages[0] == message
    
    # Verify the updated_at time was updated
    assert conversation.updated_at > before_update


def test_conversation_get_messages(conversation):
    """Test getting messages from a conversation in chronological order."""
    # Create messages with different timestamps
    msg1 = Message(content="First", sender="user1")
    msg1.timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
    
    msg2 = Message(content="Second", sender="user1")
    msg2.timestamp = datetime(2023, 1, 2, tzinfo=timezone.utc)
    
    msg3 = Message(content="Third", sender="user1")
    msg3.timestamp = datetime(2023, 1, 3, tzinfo=timezone.utc)
    
    # Add messages in random order
    conversation.add_message(msg2)
    conversation.add_message(msg3)
    conversation.add_message(msg1)
    
    # Get messages
    messages = conversation.get_messages()
    
    # Verify messages are in chronological order
    assert len(messages) == 3
    assert messages[0] == msg1
    assert messages[1] == msg2
    assert messages[2] == msg3


def test_conversation_to_dict(conversation, message):
    """Test converting a conversation to a dictionary."""
    # Add a message
    conversation.add_message(message)
    
    # Convert to dictionary
    conv_dict = conversation.to_dict()
    
    assert conv_dict["conversation_id"] == "conv1"
    assert conv_dict["title"] == "Test Conversation"
    assert conv_dict["metadata"] == {"key": "value"}
    assert len(conv_dict["messages"]) == 1
    assert conv_dict["messages"][0]["message_id"] == "msg1"
    assert "created_at" in conv_dict
    assert "updated_at" in conv_dict


def test_conversation_from_dict():
    """Test creating a conversation from a dictionary."""
    now = datetime.now(timezone.utc)
    message_dict = {
        "message_id": "msg1",
        "content": "Hello",
        "sender": "user1",
        "timestamp": now.isoformat(),
        "metadata": {}
    }
    
    conv_dict = {
        "conversation_id": "conv1",
        "title": "Test Conversation",
        "metadata": {"key": "value"},
        "messages": [message_dict],
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }
    
    conversation = Conversation.from_dict(conv_dict)
    
    assert conversation.conversation_id == "conv1"
    assert conversation.title == "Test Conversation"
    assert conversation.metadata == {"key": "value"}
    assert len(conversation.messages) == 1
    assert conversation.messages[0].message_id == "msg1"
    assert conversation.created_at.isoformat() == now.isoformat()
    assert conversation.updated_at.isoformat() == now.isoformat()


def test_conversation_manager_create_conversation(conversation_manager):
    """Test creating a conversation with the manager."""
    # Create a conversation
    conversation = conversation_manager.create_conversation(
        title="Test Conversation",
        metadata={"key": "value"}
    )
    
    # Verify the conversation was created correctly
    assert conversation.title == "Test Conversation"
    assert conversation.metadata == {"key": "value"}
    
    # Verify the conversation was stored
    assert conversation.conversation_id in conversation_manager._conversations
    assert conversation_manager._conversations[conversation.conversation_id] == conversation


def test_conversation_manager_get_conversation(conversation_manager):
    """Test getting a conversation by ID."""
    # Create a conversation
    conversation = conversation_manager.create_conversation()
    
    # Get the conversation
    retrieved = conversation_manager.get_conversation(conversation.conversation_id)
    
    # Verify the correct conversation was returned
    assert retrieved == conversation


def test_conversation_manager_get_conversation_not_found(conversation_manager):
    """Test getting a conversation that doesn't exist."""
    # Attempt to get a non-existent conversation
    with pytest.raises(ValueError):
        conversation_manager.get_conversation("non-existent")


def test_conversation_manager_add_message(conversation_manager):
    """Test adding a message to a conversation with the manager."""
    # Create a conversation
    conversation = conversation_manager.create_conversation()
    
    # Add a message
    message = conversation_manager.add_message(
        conversation_id=conversation.conversation_id,
        content="Hello",
        sender="user1",
        metadata={"key": "value"}
    )
    
    # Verify the message was added correctly
    assert message.content == "Hello"
    assert message.sender == "user1"
    assert message.metadata == {"key": "value"}
    
    # Verify the message was added to the conversation
    assert len(conversation.messages) == 1
    assert conversation.messages[0] == message


def test_conversation_manager_add_message_conversation_not_found(conversation_manager):
    """Test adding a message to a conversation that doesn't exist."""
    # Attempt to add a message to a non-existent conversation
    with pytest.raises(ValueError):
        conversation_manager.add_message(
            conversation_id="non-existent",
            content="Hello",
            sender="user1"
        )


def test_conversation_manager_list_conversations(conversation_manager):
    """Test listing all conversations."""
    # Create conversations
    conv1 = conversation_manager.create_conversation(title="First")
    conv2 = conversation_manager.create_conversation(title="Second")
    
    # List conversations
    conversations = conversation_manager.list_conversations()
    
    # Verify the correct conversations were returned
    assert len(conversations) == 2
    assert conv1 in conversations
    assert conv2 in conversations


def test_conversation_manager_delete_conversation(conversation_manager):
    """Test deleting a conversation."""
    # Create a conversation
    conversation = conversation_manager.create_conversation()
    
    # Delete the conversation
    conversation_manager.delete_conversation(conversation.conversation_id)
    
    # Verify the conversation was removed
    assert conversation.conversation_id not in conversation_manager._conversations


def test_conversation_manager_delete_conversation_not_found(conversation_manager):
    """Test deleting a conversation that doesn't exist."""
    # Attempt to delete a non-existent conversation
    with pytest.raises(ValueError):
        conversation_manager.delete_conversation("non-existent")


def test_conversation_manager_get_conversation_history(conversation_manager):
    """Test getting the history of a conversation."""
    # Create a conversation
    conversation = conversation_manager.create_conversation()
    
    # Add messages
    msg1 = conversation_manager.add_message(conversation.conversation_id, "First", "user1")
    msg2 = conversation_manager.add_message(conversation.conversation_id, "Second", "agent1")
    msg3 = conversation_manager.add_message(conversation.conversation_id, "Third", "user1")
    
    # Get history without limit
    history = conversation_manager.get_conversation_history(conversation.conversation_id)
    
    # Verify all messages are returned
    assert len(history) == 3
    
    # Get history with limit
    limited_history = conversation_manager.get_conversation_history(
        conversation.conversation_id, 
        limit=2
    )
    
    # Verify only the most recent messages are returned
    assert len(limited_history) == 2
    assert limited_history[0] == msg2
    assert limited_history[1] == msg3


def test_conversation_manager_get_conversation_history_not_found(conversation_manager):
    """Test getting the history of a conversation that doesn't exist."""
    with pytest.raises(ValueError, match="No conversation found with ID nonexistent"):
        conversation_manager.get_conversation_history("nonexistent")


def test_conversation_manager_publish_event_on_create():
    """Test that an event is published when a conversation is created."""
    # Create a mock publisher
    mock_publisher = MagicMock(spec=EventPublisherProtocol)
    
    # Patch the ConversationCreatedEvent class to add the required fields
    with patch('clubhouse.services.conversation_manager.ConversationCreatedEvent') as mock_event_class:
        # Configure the mock event to include required fields
        mock_event = MagicMock()
        # Set up model_dump to return a dict with all required fields
        mock_event.model_dump.return_value = {
            "event_id": uuid4(),
            "event_type": "conversation_created",
            "producer_id": "test-conversation-manager",
            "timestamp": datetime.now(),
            "event_version": "1.0",
            "conversation_id": "will-be-replaced-in-test",
            "title": "will-be-replaced-in-test",
            "metadata": {}
        }
        # Return our configured mock when the ConversationCreatedEvent is instantiated
        mock_event_class.return_value = mock_event
        
        # Create a test conversation manager
        manager = ConversationManager()
        manager._event_publisher = mock_publisher
        
        # Create a conversation which should trigger event publishing
        conversation = manager.create_conversation(
            title="Test Event Conversation",
            metadata={"test": "value"}
        )
        
        # Verify the event class was instantiated with the right parameters
        mock_event_class.assert_called_once()
        event_call_kwargs = mock_event_class.call_args.kwargs
        assert event_call_kwargs["conversation_id"] == conversation.conversation_id
        assert event_call_kwargs["title"] == "Test Event Conversation"
        assert event_call_kwargs["metadata"] == {"test": "value"}
        
        # Check that the publisher was called with the event data and correct topic
        mock_publisher.publish_event.assert_called_once()
        call_args, call_kwargs = mock_publisher.publish_event.call_args
        assert call_args[1] == ConversationManager.CONVERSATION_CREATED_TOPIC


def test_conversation_manager_publish_event_on_add_message():
    """Test that an event is published when a message is added to a conversation."""
    # Create a mock publisher
    mock_publisher = MagicMock(spec=EventPublisherProtocol)
    
    # Patch both event classes
    with patch('clubhouse.services.conversation_manager.ConversationCreatedEvent') as mock_create_event:
        # Set up for conversation creation (needed to avoid validation errors)
        create_event = MagicMock()
        create_event.model_dump.return_value = {
            "event_id": uuid4(),
            "event_type": "conversation_created",
            "producer_id": "test-conversation-manager",
            "timestamp": datetime.now(),
            "event_version": "1.0",
            "conversation_id": "will-be-replaced-in-test",
            "title": "will-be-replaced-in-test", 
            "metadata": {}
        }
        mock_create_event.return_value = create_event
        
        with patch('clubhouse.services.conversation_manager.MessageAddedEvent') as mock_message_event:
            # Set up for message added event
            message_event = MagicMock()
            message_event.model_dump.return_value = {
                "event_id": uuid4(),
                "event_type": "message_added",
                "producer_id": "test-conversation-manager",
                "timestamp": datetime.now(),
                "event_version": "1.0",
                "conversation_id": "will-be-replaced-in-test",
                "message_id": "will-be-replaced-in-test",
                "content": "will-be-replaced-in-test",
                "sender": "will-be-replaced-in-test"
            }
            mock_message_event.return_value = message_event
            
            # Create a test conversation manager
            manager = ConversationManager()
            manager._event_publisher = mock_publisher
            
            # Create a conversation first
            conversation = manager.create_conversation(
                title="Test Event Conversation",
                metadata={"test": "value"}
            )
            
            # Reset the mock to clear the event from conversation creation
            mock_publisher.reset_mock()
            
            # Add a message which should trigger event publishing
            message = manager.add_message(
                conversation_id=conversation.conversation_id,
                content="Test message",
                sender="test_user",
                metadata={"test": "value"}
            )
            
            # Verify the event class was instantiated with the right parameters
            mock_message_event.assert_called_once()
            event_call_kwargs = mock_message_event.call_args.kwargs
            assert event_call_kwargs["conversation_id"] == conversation.conversation_id
            assert event_call_kwargs["message_id"] == message.message_id
            assert event_call_kwargs["content"] == "Test message"
            assert event_call_kwargs["sender"] == "test_user"
            
            # Check that the publisher was called
            mock_publisher.publish_event.assert_called_once()
            
            # Verify topic used for publish_event
            call_args, call_kwargs = mock_publisher.publish_event.call_args
            assert call_args[1] == ConversationManager.MESSAGE_ADDED_TOPIC


def test_conversation_manager_publish_event_on_delete():
    """Test that an event is published when a conversation is deleted."""
    # Create a mock publisher
    mock_publisher = MagicMock(spec=EventPublisherProtocol)
    
    # Patch both event classes
    with patch('clubhouse.services.conversation_manager.ConversationCreatedEvent') as mock_create_event:
        # Set up for conversation creation (needed to avoid validation errors)
        create_event = MagicMock()
        create_event.model_dump.return_value = {
            "event_id": uuid4(),
            "event_type": "conversation_created",
            "producer_id": "test-conversation-manager",
            "timestamp": datetime.now(),
            "event_version": "1.0",
            "conversation_id": "will-be-replaced-in-test",
            "title": "will-be-replaced-in-test", 
            "metadata": {}
        }
        mock_create_event.return_value = create_event
        
        with patch('clubhouse.services.conversation_manager.ConversationDeletedEvent') as mock_delete_event:
            # Set up for conversation deleted event
            delete_event = MagicMock()
            delete_event.model_dump.return_value = {
                "event_id": uuid4(),
                "event_type": "conversation_deleted",
                "producer_id": "test-conversation-manager",
                "timestamp": datetime.now(),
                "event_version": "1.0",
                "conversation_id": "will-be-replaced-in-test"
            }
            mock_delete_event.return_value = delete_event
            
            # Create a test conversation manager
            manager = ConversationManager()
            manager._event_publisher = mock_publisher
            
            # Create a conversation first
            conversation = manager.create_conversation(
                title="Test Event Conversation",
                metadata={"test": "value"}
            )
            
            # Reset the mock to clear the event from conversation creation
            mock_publisher.reset_mock()
            
            # Store the conversation ID before deletion
            conversation_id = conversation.conversation_id
            
            # Delete the conversation which should trigger event publishing
            manager.delete_conversation(conversation_id)
            
            # Verify the event class was instantiated with the right parameters
            mock_delete_event.assert_called_once()
            event_call_kwargs = mock_delete_event.call_args.kwargs
            assert event_call_kwargs["conversation_id"] == conversation_id
            
            # Check that the publisher was called
            mock_publisher.publish_event.assert_called_once()
            
            # Verify topic used for publish_event
            call_args, call_kwargs = mock_publisher.publish_event.call_args
            assert call_args[1] == ConversationManager.CONVERSATION_DELETED_TOPIC


def test_conversation_manager_update_context():
    """Test updating the context for a conversation."""
    manager = ConversationManager()
    
    # Create a conversation
    conversation = manager.create_conversation(title="Test Context")
    
    # Update context
    context = manager.update_context(
        conversation_id=conversation.conversation_id,
        context_updates={"key1": "value1", "key2": 42}
    )
    
    # Verify context was updated
    assert context == {"key1": "value1", "key2": 42}
    
    # Verify conversation's context was updated
    conversation = manager.get_conversation(conversation.conversation_id)
    assert conversation.context == {"key1": "value1", "key2": 42}


def test_conversation_manager_update_context_validation():
    """Test context validation when updating context."""
    manager = ConversationManager()
    
    # Create a conversation
    conversation = manager.create_conversation(title="Test Context Validation")
    
    # Test with non-dictionary
    with pytest.raises(ValueError, match="Context updates must be a dictionary"):
        manager.update_context(
            conversation_id=conversation.conversation_id,
            context_updates="not a dict"
        )
    
    # Test with non-string keys
    with pytest.raises(ValueError, match="Context keys must be strings"):
        manager.update_context(
            conversation_id=conversation.conversation_id,
            context_updates={123: "value"}
        )


def test_conversation_manager_get_context():
    """Test getting the context for a conversation."""
    manager = ConversationManager()
    
    # Create a conversation
    conversation = manager.create_conversation(title="Test Get Context")
    
    # Initialize context
    manager.update_context(
        conversation_id=conversation.conversation_id,
        context_updates={"key1": "value1"}
    )
    
    # Get context
    context = manager.get_context(conversation.conversation_id)
    
    # Verify context
    assert context == {"key1": "value1"}
