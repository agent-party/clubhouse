"""
Tests for the clubhouse message handlers.

This module contains tests for the message handlers that process
different types of messages received from the Kafka CLI.
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.handlers import (
    BaseHandler,
    CreateAgentHandler,
    DeleteAgentHandler,
    ProcessMessageHandler
)
from clubhouse.services.agent_manager import AgentManagerProtocol
from clubhouse.services.conversation_manager import ConversationManagerProtocol
from clubhouse.messaging.event_publisher import EventPublisherProtocol
from scripts.kafka_cli.message_schemas import MessageType


@pytest.fixture
def service_registry():
    """Fixture for a service registry."""
    registry = MagicMock(spec=ServiceRegistry)
    return registry


@pytest.fixture
def agent_manager():
    """Fixture for an agent manager."""
    manager = MagicMock(spec=AgentManagerProtocol)
    return manager


@pytest.fixture
def conversation_manager():
    """Fixture for a conversation manager."""
    manager = MagicMock(spec=ConversationManagerProtocol)
    return manager


@pytest.fixture
def event_publisher():
    """Fixture for an event publisher."""
    publisher = MagicMock(spec=EventPublisherProtocol)
    return publisher


class TestBaseHandler:
    """Tests for the BaseHandler class."""
    
    class ConcreteHandler(BaseHandler):
        """Concrete implementation of BaseHandler for testing."""
        
        def __init__(self, service_registry):
            super().__init__(service_registry)
            self.handled_types = [MessageType.COMMAND_CREATE_AGENT]
        
        def handle(self, message):
            return {"handled": True}
    
    def test_initialization(self, service_registry):
        """Test that the handler initializes correctly."""
        handler = self.ConcreteHandler(service_registry)
        assert handler._service_registry == service_registry
        assert handler.handled_types == [MessageType.COMMAND_CREATE_AGENT]
    
    def test_can_handle(self, service_registry):
        """Test that the handler determines which message types it can handle."""
        handler = self.ConcreteHandler(service_registry)
        
        # Test that the handler can handle its registered type
        assert handler.can_handle(MessageType.COMMAND_CREATE_AGENT.value)
        
        # Test that the handler cannot handle other types
        assert not handler.can_handle(MessageType.COMMAND_DELETE_AGENT.value)
        assert not handler.can_handle(MessageType.COMMAND_PROCESS_MESSAGE.value)
        
        # Test with invalid type
        assert not handler.can_handle("invalid_type")


class TestCreateAgentHandler:
    """Tests for the CreateAgentHandler class."""
    
    @pytest.fixture
    def handler(self, service_registry, agent_manager, event_publisher):
        """Fixture for a CreateAgentHandler with mocked dependencies."""
        service_registry.get.side_effect = lambda service_name: {
            "agent_manager": agent_manager,
            "event_publisher": event_publisher
        }.get(service_name)
        
        return CreateAgentHandler(service_registry)
    
    def test_initialization(self, handler):
        """Test that the handler initializes correctly."""
        assert handler.handled_types == [MessageType.COMMAND_CREATE_AGENT]
    
    def test_handle_create_agent(self, handler, agent_manager, event_publisher):
        """Test handling a create agent command."""
        # Set up a mock agent
        mock_agent = MagicMock()
        mock_agent.agent_id.return_value = "test-agent"
        mock_agent.name.return_value = "Test Agent"
        mock_agent.description.return_value = "A test agent"
        
        agent_manager.create_agent.return_value = mock_agent
        
        # Create a message
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_CREATE_AGENT.value,
            "agent_id": "test-agent",
            "personality_type": "researcher",
            "metadata": {"key": "value"}
        }
        
        # Handle the message
        response = handler.handle(message)
        
        # Verify the agent was created
        agent_manager.create_agent.assert_called_once_with(
            agent_id="test-agent",
            personality_type="researcher",
            metadata={"key": "value"}
        )
        
        # Verify the response
        assert response["message_type"] == MessageType.RESPONSE_AGENT_CREATED.value
        assert response["agent_id"] == "test-agent"
        assert response["agent_name"] == "Test Agent"
        assert response["agent_description"] == "A test agent"
        assert response["message_id"] != "msg-123"  # Should be a new message ID
        
        # Verify an event was published
        event_publisher.publish_event.assert_called_once()
        event_data = event_publisher.publish_event.call_args[0][0]
        assert event_data["message_type"] == MessageType.EVENT_AGENT_STATE_CHANGED.value
        assert event_data["agent_id"] == "test-agent"
        assert event_data["state"] == "ready"
    
    def test_handle_create_agent_error(self, handler, agent_manager, event_publisher):
        """Test handling a create agent command that results in an error."""
        # Set up the agent manager to raise an exception
        agent_manager.create_agent.side_effect = ValueError("Agent already exists")
        
        # Create a message
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_CREATE_AGENT.value,
            "agent_id": "test-agent",
            "personality_type": "researcher"
        }
        
        # Handle the message and expect an exception
        with pytest.raises(ValueError):
            handler.handle(message)
        
        # Verify no event was published
        event_publisher.publish_event.assert_not_called()


class TestDeleteAgentHandler:
    """Tests for the DeleteAgentHandler class."""
    
    @pytest.fixture
    def handler(self, service_registry, agent_manager, event_publisher):
        """Fixture for a DeleteAgentHandler with mocked dependencies."""
        service_registry.get.side_effect = lambda service_name: {
            "agent_manager": agent_manager,
            "event_publisher": event_publisher
        }.get(service_name)
        
        return DeleteAgentHandler(service_registry)
    
    def test_initialization(self, handler):
        """Test that the handler initializes correctly."""
        assert handler.handled_types == [MessageType.COMMAND_DELETE_AGENT]
    
    def test_handle_delete_agent(self, handler, agent_manager, event_publisher):
        """Test handling a delete agent command."""
        # Create a message
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_DELETE_AGENT.value,
            "agent_id": "test-agent"
        }
        
        # Handle the message
        response = handler.handle(message)
        
        # Verify the agent was deleted
        agent_manager.delete_agent.assert_called_once_with("test-agent")
        
        # Verify the response
        assert response["message_type"] == MessageType.RESPONSE_AGENT_DELETED.value
        assert response["agent_id"] == "test-agent"
        assert response["message_id"] != "msg-123"  # Should be a new message ID
        
        # Verify an event was published
        event_publisher.publish_event.assert_called_once()
        event_data = event_publisher.publish_event.call_args[0][0]
        assert event_data["message_type"] == MessageType.EVENT_AGENT_STATE_CHANGED.value
        assert event_data["agent_id"] == "test-agent"
        assert event_data["state"] == "deleted"
    
    def test_handle_delete_agent_error(self, handler, agent_manager, event_publisher):
        """Test handling a delete agent command that results in an error."""
        # Set up the agent manager to raise an exception
        agent_manager.delete_agent.side_effect = ValueError("Agent not found")
        
        # Create a message
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_DELETE_AGENT.value,
            "agent_id": "test-agent"
        }
        
        # Handle the message and expect an exception
        with pytest.raises(ValueError):
            handler.handle(message)
        
        # Verify no event was published
        event_publisher.publish_event.assert_not_called()


class TestProcessMessageHandler:
    """Tests for the ProcessMessageHandler class."""
    
    @pytest.fixture
    def handler(self, service_registry, agent_manager, conversation_manager, event_publisher):
        """Fixture for a ProcessMessageHandler with mocked dependencies."""
        service_registry.get.side_effect = lambda service_name: {
            "agent_manager": agent_manager,
            "conversation_manager": conversation_manager,
            "event_publisher": event_publisher
        }.get(service_name)
        
        return ProcessMessageHandler(service_registry)
    
    def test_initialization(self, handler):
        """Test that the handler initializes correctly."""
        assert handler.handled_types == [MessageType.COMMAND_PROCESS_MESSAGE]
    
    def test_handle_process_message(self, handler, agent_manager, conversation_manager, event_publisher):
        """Test handling a process message command."""
        # Set up a mock agent
        mock_agent = MagicMock()
        mock_agent.process_message.return_value = {
            "content": "This is a response",
            "conversation_id": "conv-123",
            "metadata": {}
        }
        
        agent_manager.get_agent.return_value = mock_agent
        
        # Create a message
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_PROCESS_MESSAGE.value,
            "agent_id": "test-agent",
            "conversation_id": "conv-123",
            "content": "Hello agent",
            "metadata": {}
        }
        
        # Handle the message
        response = handler.handle(message)
        
        # Verify the agent was retrieved
        agent_manager.get_agent.assert_called_once_with("test-agent")
        
        # Verify the message was processed
        mock_agent.process_message.assert_called_once()
        agent_message = mock_agent.process_message.call_args[0][0]
        assert agent_message["content"] == "Hello agent"
        assert agent_message["conversation_id"] == "conv-123"
        
        # Verify the response
        assert response["message_type"] == MessageType.RESPONSE_MESSAGE_PROCESSED.value
        assert response["agent_id"] == "test-agent"
        assert response["content"] == "This is a response"
        assert response["conversation_id"] == "conv-123"
        
        # Verify a thinking event was published
        event_publisher.publish_event.assert_called_once()
        event_data = event_publisher.publish_event.call_args[0][0]
        assert event_data["message_type"] == MessageType.EVENT_AGENT_THINKING.value
        assert event_data["agent_id"] == "test-agent"
    
    def test_handle_process_message_new_conversation(self, handler, agent_manager, conversation_manager, event_publisher):
        """Test handling a process message command for a new conversation."""
        # Set up a mock agent
        mock_agent = MagicMock()
        mock_agent.process_message.return_value = {
            "content": "This is a response",
            "conversation_id": "conv-123",
            "metadata": {}
        }
        
        agent_manager.get_agent.return_value = mock_agent
        
        # Set up the conversation manager to create a new conversation
        mock_conversation = MagicMock()
        mock_conversation.conversation_id = "conv-123"
        conversation_manager.create_conversation.return_value = mock_conversation
        
        # Create a message without a conversation ID
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_PROCESS_MESSAGE.value,
            "agent_id": "test-agent",
            "content": "Hello agent",
            "metadata": {}
        }
        
        # Handle the message
        response = handler.handle(message)
        
        # Verify a new conversation was created
        conversation_manager.create_conversation.assert_called_once()
        
        # Verify the response has the new conversation ID
        assert response["conversation_id"] == "conv-123"
    
    def test_handle_process_message_error(self, handler, agent_manager, event_publisher):
        """Test handling a process message command that results in an error."""
        # Set up the agent manager to raise an exception
        agent_manager.get_agent.side_effect = ValueError("Agent not found")
        
        # Create a message
        message = {
            "message_id": "msg-123",
            "message_type": MessageType.COMMAND_PROCESS_MESSAGE.value,
            "agent_id": "test-agent",
            "content": "Hello agent"
        }
        
        # Handle the message and expect an exception
        with pytest.raises(ValueError):
            handler.handle(message)
        
        # Verify no event was published
        event_publisher.publish_event.assert_not_called()
