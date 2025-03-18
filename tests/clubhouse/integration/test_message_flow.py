"""
Integration tests for the clubhouse message flow.

This module contains tests that verify the end-to-end flow of messages
through the clubhouse architecture, from receiving Kafka messages to
sending responses and events.
"""

import pytest
import uuid
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.event_publisher import EventPublisher
from clubhouse.messaging.handlers import (
    CreateAgentHandler,
    DeleteAgentHandler,
    ProcessMessageHandler
)
from clubhouse.messaging.message_router import MessageRouter
from clubhouse.services.agent_manager import AgentManager
from clubhouse.services.conversation_manager import ConversationManager
from clubhouse.clubhouse_main import ClubhouseMessageHandler
from scripts.kafka_cli.message_schemas import MessageType


class MockKafkaService:
    """Mock Kafka service for testing."""
    
    def __init__(self):
        """Initialize the mock service."""
        self.produced_messages = []
        
    def produce_message(self, message):
        """Record a produced message."""
        self.produced_messages.append(message)
        
    def get_produced_messages(self):
        """Get all produced messages."""
        return self.produced_messages


@pytest.fixture
def service_registry():
    """Fixture for a service registry with real services."""
    registry = ServiceRegistry()
    
    # Register mock Kafka service
    kafka_service = MockKafkaService()
    registry.register("kafka", kafka_service)
    
    # Create and register the factory first
    with patch("clubhouse.agents.factory.AgentFactory") as mock_factory_class:
        mock_factory = mock_factory_class.return_value
        
        # Configure mock factory to create mock agents
        def create_mock_agent(agent_id, personality_type, **kwargs):
            mock_agent = MagicMock()
            mock_agent.agent_id.return_value = agent_id
            mock_agent.name.return_value = f"{personality_type.capitalize()} Agent"
            mock_agent.description.return_value = f"A {personality_type} agent"
            
            # Configure process_message to return a response
            def process_message(message):
                return {
                    "content": f"Response to: {message.get('content')}",
                    "conversation_id": message.get("conversation_id", str(uuid.uuid4())),
                    "metadata": {}
                }
            
            mock_agent.process_message.side_effect = process_message
            return mock_agent
        
        mock_factory.create_agent.side_effect = create_mock_agent
        
        # Register the factory using both type and name registration
        registry.register(mock_factory_class, mock_factory)
        registry.register("agent_factory", mock_factory)
        
        # Register real services with mocked dependencies
        agent_manager = AgentManager(registry, agent_factory=mock_factory)
        registry.register("agent_manager", agent_manager)
        
        conversation_manager = ConversationManager()
        registry.register("conversation_manager", conversation_manager)
        
        event_publisher = EventPublisher(registry)
        registry.register("event_publisher", event_publisher)
        
        yield registry


@pytest.fixture
def message_router(service_registry):
    """Fixture for a message router with registered handlers."""
    router = MessageRouter(service_registry)
    
    # Register handlers
    router.register_handler(CreateAgentHandler(service_registry))
    router.register_handler(DeleteAgentHandler(service_registry))
    router.register_handler(ProcessMessageHandler(service_registry))
    
    return router


@pytest.fixture
def clubhouse_handler(message_router, service_registry):
    """Fixture for a clubhouse message handler."""
    event_publisher = service_registry.get("event_publisher")
    return ClubhouseMessageHandler(message_router, event_publisher)


def test_create_agent_flow(clubhouse_handler, service_registry):
    """Test the flow of creating an agent."""
    # Create a create agent message
    message_id = str(uuid.uuid4())
    message = {
        "message_id": message_id,
        "message_type": MessageType.COMMAND_CREATE_AGENT.value,
        "agent_id": "test-agent",
        "personality_type": "assistant",
        "metadata": {"key": "value"}
    }
    
    # Handle the message
    clubhouse_handler.handle(message)
    
    # Get the agent manager and verify the agent was created
    agent_manager = service_registry.get("agent_manager")
    assert "test-agent" in agent_manager._agents
    
    # Get the Kafka service and verify messages were produced
    kafka_service = service_registry.get("kafka")
    produced_messages = kafka_service.get_produced_messages()
    
    # Verify at least 2 messages were produced (response and event)
    assert len(produced_messages) >= 2
    
    # Find the response message
    response_message = None
    for msg in produced_messages:
        if msg.value.get("message_type") == MessageType.RESPONSE_AGENT_CREATED.value:
            response_message = msg
            break
    
    assert response_message is not None
    assert response_message.value.get("agent_id") == "test-agent"
    assert response_message.value.get("agent_name") == "Assistant Agent"
    
    # Find the state change event
    state_change_event = None
    for msg in produced_messages:
        if msg.value.get("message_type") == MessageType.EVENT_AGENT_STATE_CHANGED.value:
            state_change_event = msg
            break
    
    assert state_change_event is not None
    assert state_change_event.value.get("agent_id") == "test-agent"
    assert state_change_event.value.get("state") == "ready"


def test_process_message_flow(clubhouse_handler, service_registry):
    """Test the flow of processing a message."""
    # First create an agent
    agent_message = {
        "message_id": str(uuid.uuid4()),
        "message_type": MessageType.COMMAND_CREATE_AGENT.value,
        "agent_id": "chat-agent",
        "personality_type": "chat",
        "metadata": {}
    }
    clubhouse_handler.handle(agent_message)
    
    # Clear produced messages
    kafka_service = service_registry.get("kafka")
    kafka_service.produced_messages.clear()
    
    # Get the conversation manager and create a conversation first
    conversation_manager = service_registry.get("conversation_manager")
    conversation_id = str(uuid.uuid4())
    conversation = conversation_manager.create_conversation(
        title="Test Conversation",
        metadata={"agent_id": "chat-agent"}
    )
    conversation_id = conversation.conversation_id  # Use the actual conversation ID
    
    # Create a process message
    message = {
        "message_id": str(uuid.uuid4()),
        "message_type": MessageType.COMMAND_PROCESS_MESSAGE.value,
        "agent_id": "chat-agent",
        "conversation_id": conversation_id,
        "content": "Hello, agent!",
        "metadata": {}
    }
    
    # Handle the message
    clubhouse_handler.handle(message)
    
    # Get the Kafka service and verify messages were produced
    produced_messages = kafka_service.get_produced_messages()
    
    # Verify at least 2 messages were produced (thinking event and response)
    assert len(produced_messages) >= 2
    
    # Find the thinking event
    thinking_event = None
    for msg in produced_messages:
        if msg.value.get("message_type") == MessageType.EVENT_AGENT_THINKING.value:
            thinking_event = msg
            break
    
    assert thinking_event is not None
    assert thinking_event.value.get("agent_id") == "chat-agent"
    
    # Find the response message
    response_message = None
    for msg in produced_messages:
        if msg.value.get("message_type") == MessageType.RESPONSE_MESSAGE_PROCESSED.value:
            response_message = msg
            break
    
    assert response_message is not None
    assert response_message.value.get("agent_id") == "chat-agent"
    assert response_message.value.get("conversation_id") == conversation_id
    assert "Response to: Hello, agent!" in response_message.value.get("content")


def test_delete_agent_flow(clubhouse_handler, service_registry):
    """Test the flow of deleting an agent."""
    # First create an agent
    agent_message = {
        "message_id": str(uuid.uuid4()),
        "message_type": MessageType.COMMAND_CREATE_AGENT.value,
        "agent_id": "temp-agent",
        "personality_type": "assistant",
        "metadata": {}
    }
    clubhouse_handler.handle(agent_message)
    
    # Clear produced messages
    kafka_service = service_registry.get("kafka")
    kafka_service.produced_messages.clear()
    
    # Create a delete agent message
    message = {
        "message_id": str(uuid.uuid4()),
        "message_type": MessageType.COMMAND_DELETE_AGENT.value,
        "agent_id": "temp-agent"
    }
    
    # Handle the message
    clubhouse_handler.handle(message)
    
    # Get the agent manager and verify the agent was deleted
    agent_manager = service_registry.get("agent_manager")
    assert "temp-agent" not in agent_manager._agents
    
    # Get the Kafka service and verify messages were produced
    produced_messages = kafka_service.get_produced_messages()
    
    # Verify at least 2 messages were produced (response and event)
    assert len(produced_messages) >= 2
    
    # Find the response message
    response_message = None
    for msg in produced_messages:
        if msg.value.get("message_type") == MessageType.RESPONSE_AGENT_DELETED.value:
            response_message = msg
            break
    
    assert response_message is not None
    assert response_message.value.get("agent_id") == "temp-agent"
    
    # Find the state change event
    state_change_event = None
    for msg in produced_messages:
        if msg.value.get("message_type") == MessageType.EVENT_AGENT_STATE_CHANGED.value:
            state_change_event = msg
            break
    
    assert state_change_event is not None
    assert state_change_event.value.get("agent_id") == "temp-agent"
    assert state_change_event.value.get("state") == "deleted"


def test_error_handling_flow(clubhouse_handler, service_registry):
    """Test the flow of handling errors."""
    # Create a message for a non-existent agent
    message = {
        "message_id": str(uuid.uuid4()),
        "message_type": MessageType.COMMAND_PROCESS_MESSAGE.value,
        "agent_id": "non-existent-agent",
        "content": "Hello, agent!",
        "metadata": {}
    }
    
    # Handle the message, expecting an error
    try:
        clubhouse_handler.handle(message)
    except Exception:
        pass  # Error expected
    
    # Get the Kafka service and verify an error event was produced
    kafka_service = service_registry.get("kafka")
    produced_messages = kafka_service.get_produced_messages()
    
    # Find an error event
    error_event = None
    for msg in produced_messages:
        if (
            msg.value.get("message_type") == MessageType.EVENT_AGENT_ERROR.value and
            msg.value.get("agent_id") == "non-existent-agent"
        ):
            error_event = msg
            break
    
    assert error_event is not None
    assert "error_message" in error_event.value
    assert "error_type" in error_event.value
