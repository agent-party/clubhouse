"""
Tests for the clubhouse message router.

This module contains tests for the message router component, which is
responsible for routing incoming messages to the appropriate handlers.
"""

import pytest
from unittest.mock import MagicMock, patch

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.message_router import MessageRouter
from scripts.kafka_cli.message_schemas import MessageType


class MockHandler:
    """Mock message handler for testing."""
    
    def __init__(self, handled_types):
        self.handled_types = handled_types
        self.handle_called = False
        self.last_message = None
    
    def can_handle(self, message_type):
        return message_type in self.handled_types
    
    def handle(self, message):
        self.handle_called = True
        self.last_message = message
        return {"response": "success", "message_id": message.get("message_id")}


@pytest.fixture
def service_registry():
    """Fixture for a service registry."""
    return MagicMock(spec=ServiceRegistry)


@pytest.fixture
def message_router(service_registry):
    """Fixture for a message router."""
    return MessageRouter(service_registry)


def test_message_router_initialization(message_router, service_registry):
    """Test that the message router initializes correctly."""
    assert message_router._service_registry == service_registry
    assert message_router._handlers == []


def test_register_handler(message_router):
    """Test registering a handler with the router."""
    handler = MockHandler([MessageType.COMMAND_CREATE_AGENT])
    message_router.register_handler(handler)
    assert len(message_router._handlers) == 1
    assert message_router._handlers[0] == handler


def test_route_message_to_handler(message_router):
    """Test routing a message to the appropriate handler."""
    # Register handlers
    create_handler = MockHandler([MessageType.COMMAND_CREATE_AGENT])
    process_handler = MockHandler([MessageType.COMMAND_PROCESS_MESSAGE])
    message_router.register_handler(create_handler)
    message_router.register_handler(process_handler)
    
    # Create a message
    message = {
        "message_id": "123",
        "message_type": MessageType.COMMAND_CREATE_AGENT.value,
        "agent_id": "test-agent"
    }
    
    # Route the message
    response = message_router.route_message(message)
    
    # Verify the message was routed to the correct handler
    assert create_handler.handle_called
    assert create_handler.last_message == message
    assert not process_handler.handle_called
    assert response == {"response": "success", "message_id": "123"}


def test_route_message_no_handler(message_router):
    """Test routing a message with no handler available."""
    # Register a handler that can't handle the message type
    handler = MockHandler([MessageType.COMMAND_CREATE_AGENT])
    message_router.register_handler(handler)
    
    # Create a message
    message = {
        "message_id": "123",
        "message_type": MessageType.COMMAND_PROCESS_MESSAGE.value,
        "agent_id": "test-agent"
    }
    
    # Route the message
    response = message_router.route_message(message)
    
    # Verify no handler was called
    assert not handler.handle_called
    assert response is None


def test_route_message_invalid_message(message_router):
    """Test routing a message with no message_type field."""
    # Register a handler
    handler = MockHandler([MessageType.COMMAND_CREATE_AGENT])
    message_router.register_handler(handler)
    
    # Create an invalid message
    message = {
        "message_id": "123",
        "agent_id": "test-agent"
    }
    
    # Attempt to route the message
    with pytest.raises(ValueError):
        message_router.route_message(message)
    
    # Verify no handler was called
    assert not handler.handle_called
