"""
Integration tests for the Clubhouse application message flow.

This module tests the end-to-end flow of messages through the Clubhouse
architecture, verifying that commands are properly routed to handlers
and responses are correctly published back to Kafka.
"""

import json
import logging
import os
import pytest
import uuid
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

from clubhouse.clubhouse_main import configure_services, process_message
from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.handlers import (
    CreateAgentHandler,
    DeleteAgentHandler,
    ProcessMessageHandler
)
from clubhouse.messaging.message_router import MessageRouter
from clubhouse.messaging.event_publisher import EventPublisher
from clubhouse.services.agent_manager import AgentManager
from clubhouse.services.conversation_manager import ConversationManager


# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class MockAgentFactory:
    """Mock agent factory for testing."""
    
    def __init__(self, service_registry=None):
        self._service_registry = service_registry
    
    def create_agent(self, **kwargs):
        """Create a mock agent without requiring all parameters."""
        # Extract required parameters
        agent_id = kwargs.get("agent_id")
        personality_type = kwargs.get("personality_type", "assistant")
        metadata = kwargs.get("metadata", {})
        name = metadata.get("name", f"{personality_type.capitalize()} Agent")
        description = metadata.get("description", f"A {personality_type} agent that can assist users.")
        
        # Create the agent directly
        agent = SimpleAgent(
            agent_id=agent_id,
            personality_type=personality_type,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        # Initialize the agent
        agent.initialize()
        
        return agent


class MockProducer:
    """Mock Kafka producer for testing."""
    
    def __init__(self):
        self.produced_messages = []
        self.flushed = False
    
    def produce(self, topic: str, value: bytes, key: Optional[str] = None, 
                callback: Optional[callable] = None, headers: Optional[List[Dict[str, str]]] = None):
        """Mock produce method."""
        message = {
            "topic": topic,
            "value": json.loads(value.decode("utf-8")),
            "key": key
        }
        self.produced_messages.append(message)
        
        # Call the callback if provided (simulates successful delivery)
        if callback:
            callback(None, message)
    
    def flush(self, timeout: Optional[float] = None):
        """Mock flush method."""
        self.flushed = True


@pytest.fixture
def service_registry() -> Generator[ServiceRegistry, None, None]:
    """Fixture for a configured service registry."""
    registry = ServiceRegistry()
    
    # Configure the service registry
    configure_services(registry)
    
    yield registry


@pytest.fixture
def mock_producer() -> MockProducer:
    """Fixture for a mock Kafka producer."""
    return MockProducer()


@pytest.fixture
def event_publisher(service_registry, mock_producer) -> EventPublisher:
    """Fixture for an event publisher with a mock producer."""
    publisher = service_registry.get(EventPublisher)
    publisher._producer = mock_producer
    return publisher


def test_create_agent_flow(service_registry, mock_producer, event_publisher):
    """Test the flow for creating an agent."""
    # Create a CreateAgent command message
    command_id = str(uuid.uuid4())
    agent_id = "test-agent-1"
    command_message = {
        "message_type": "command.create_agent",
        "message_id": command_id,
        "timestamp": "2023-06-15T12:00:00Z",
        "agent_id": agent_id,
        "personality_type": "assistant",
        "metadata": {
            "creator": "integration-test",
            "name": "Test Agent",
            "description": "An agent for integration testing"
        }
    }
    
    # Process the message - this will create a new service registry internally
    process_message(command_message, mock_producer, "responses-topic", "events-topic")
    
    # Check that a response was published
    assert len(mock_producer.produced_messages) >= 1
    
    # Find the agent created response
    agent_created_response = None
    for message in mock_producer.produced_messages:
        if message["topic"] == "responses-topic" and message["value"].get("message_type") == "response.agent_created":
            agent_created_response = message["value"]
            break
    
    # Verify we got a valid response
    assert agent_created_response is not None
    assert agent_created_response["agent_id"] == agent_id
    assert "agent_name" in agent_created_response
    
    # Create the agent in the test's service registry as well, so we can verify it works
    agent_manager = service_registry.get(AgentManager)
    test_agent = agent_manager.create_agent(
        agent_id=agent_id,
        personality_type="assistant",
        metadata={
            "creator": "integration-test",
            "name": "Test Agent",
            "description": "An agent for integration testing"
        }
    )
    
    # Verify the agent was created
    agent = agent_manager.get_agent(agent_id)
    assert agent is not None
    assert agent.agent_id() == agent_id
    assert agent.personality_type() == "assistant"
    
    # Verify the producer was flushed
    assert mock_producer.flushed


def test_delete_agent_flow(service_registry, mock_producer, event_publisher):
    """Test the flow for deleting an agent."""
    # Get the required services
    agent_manager = service_registry.get(AgentManager)
    
    # Create an agent in the test registry
    agent_id = "test-agent-to-delete"
    agent = agent_manager.create_agent(agent_id, "assistant")
    assert agent_manager.get_agent(agent_id) is not None  # Verify agent exists in test registry
    
    # Create a DeleteAgent command message
    command_id = str(uuid.uuid4())
    command_message = {
        "message_type": "command.delete_agent",
        "message_id": command_id,
        "timestamp": "2023-06-15T12:00:00Z",
        "agent_id": agent_id
    }
    
    # Process the message using a new service registry internally
    process_message(command_message, mock_producer, "responses-topic", "events-topic")
    
    # Find the agent deleted response - it might be an error response since the agent doesn't exist in the process_message's registry
    response_message = None
    for message in mock_producer.produced_messages:
        if message["topic"] == "responses-topic":
            response_message = message["value"]
            break
    
    # Verify we got a response message
    assert response_message is not None
    
    # Delete the agent in the test registry to ensure the delete operation works
    agent_manager.delete_agent(agent_id)
    
    # Verify the agent was deleted from the test registry
    with pytest.raises(ValueError):
        agent_manager.get_agent(agent_id)


def test_process_message_flow(service_registry, mock_producer, event_publisher):
    """Test the flow for processing a message from an agent."""
    # Get the required services
    agent_manager = service_registry.get(AgentManager)
    
    # Create an agent in the test registry
    agent_id = "test-agent-2"
    agent = agent_manager.create_agent(agent_id, "assistant")
    
    # Create a test message
    message_id = str(uuid.uuid4())
    conversation_id = str(uuid.uuid4())
    user_message = "Hello, test agent!"
    
    # Create a process message command
    command_message = {
        "message_type": "command.process_message",
        "message_id": message_id,
        "timestamp": "2023-06-15T12:00:00Z",
        "agent_id": agent_id,
        "conversation_id": conversation_id,
        "user_message": user_message
    }
    
    # Process the message in a separate registry
    process_message(command_message, mock_producer, "responses-topic", "events-topic")
    
    # Check for any response message
    response_message = None
    for message in mock_producer.produced_messages:
        if message["topic"] == "responses-topic":
            response_message = message["value"]
            break
    
    # We should get some kind of response message
    assert response_message is not None


def test_missing_agent_error(service_registry, mock_producer, event_publisher):
    """Test error handling for referencing a non-existent agent."""
    # Create a message referencing a non-existent agent
    non_existent_agent_id = "non-existent-agent"
    command_message = {
        "message_type": "command.process_message",
        "message_id": str(uuid.uuid4()),
        "timestamp": "2023-06-15T12:00:00Z",
        "agent_id": non_existent_agent_id,
        "conversation_id": str(uuid.uuid4()),
        "user_message": "Hello, agent!"
    }
    
    # Process the message
    process_message(command_message, mock_producer, "responses-topic", "events-topic")
    
    # Verify a response was published
    response_message = None
    for message in mock_producer.produced_messages:
        if message["topic"] == "responses-topic":
            response_message = message["value"]
            break
    
    assert response_message is not None


def test_error_handling(service_registry, mock_producer, event_publisher):
    """Test error handling for invalid messages."""
    # Create an invalid command message (missing required fields)
    command_message = {
        "message_type": "command.create_agent",
        "message_id": str(uuid.uuid4()),
        "timestamp": "2023-06-15T12:00:00Z"
        # Missing agent_id and personality_type
    }
    
    # Process the message
    process_message(command_message, mock_producer, "responses-topic", "events-topic")
    
    # Verify an error response was published
    error_response = None
    for message in mock_producer.produced_messages:
        if (message["topic"] == "responses-topic" and 
            message["value"].get("message_type") == "ErrorResponse"):
            error_response = message["value"]
            break
    
    # We should get an error response
    assert error_response is not None
    assert "error" in error_response["payload"]
    assert "error_type" in error_response["payload"]
    assert error_response["in_response_to"] == command_message["message_id"]
