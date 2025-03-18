"""
Tests for the agent communication system.
"""
import pytest
import json
from enum import Enum
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import sys
from typing import Dict, List, Any, Optional, Union

from clubhouse.agents.communication import (
    AgentCommunicationService,
    RoutingStrategy,
    MessageStatus,
    MessageType,
    MessagePriority,
    EnhancedAgentMessage,
    MessageHandlerProtocol
)
from clubhouse.services.kafka_service import KafkaMessage
from clubhouse.core.utils.datetime_utils import utc_now


class TestEnhancedAgentMessage:
    """Test cases for EnhancedAgentMessage."""
    
    def test_create_message(self):
        """Test creating a new enhanced agent message."""
        # Create a message
        message = EnhancedAgentMessage.create(
            sender="test-agent",
            content={"command": "ping"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="recipient-agent"
        )
        
        # Verify required fields
        assert message["message_id"] is not None
        assert message["sender"] == "test-agent"
        assert message["content"] == {"command": "ping"}
        assert message["routing"]["strategy"] == RoutingStrategy.DIRECT.value
        assert message["routing"]["recipient"] == "recipient-agent"
        assert message["routing"]["status"] == MessageStatus.CREATED.value
        assert message["timestamp"] is not None
        
    def test_create_message_with_optional_fields(self):
        """Test creating a message with optional fields."""
        # Test timestamp
        test_time = utc_now().isoformat()
        
        # Create message with optional fields
        message = EnhancedAgentMessage.create(
            sender="test-agent",
            content={"command": "complex-command"},
            routing_strategy=RoutingStrategy.CAPABILITY,
            message_type=MessageType.COMMAND,
            priority=MessagePriority.HIGH,
            correlation_id="corr-123",
            expires_at=test_time,
            metadata={"test-key": "test-value"}
        )
        
        # Verify all fields
        assert message["message_id"] is not None
        assert message["sender"] == "test-agent"
        assert message["content"] == {"command": "complex-command"}
        assert message["routing"]["strategy"] == RoutingStrategy.CAPABILITY.value
        assert message["type"] == MessageType.COMMAND.value
        assert message["priority"] == MessagePriority.HIGH.value
        assert message["correlation_id"] == "corr-123"
        assert message["expires_at"] == test_time
        assert message["metadata"] == {"test-key": "test-value"}
    
    def test_create_response(self):
        """Test creating a response message."""
        # Create original message
        original_message = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "ping"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="receiver-agent",
            message_type=MessageType.COMMAND
        )
        
        # Create response
        response = original_message.create_response(
            sender="receiver-agent",
            content={"response": "pong"},
            success=True
        )
        
        # Verify response fields
        assert response["message_id"] is not None
        assert response["sender"] == "receiver-agent"
        assert response["content"] == {"response": "pong"}
        assert response["routing"]["strategy"] == RoutingStrategy.DIRECT.value
        assert response["routing"]["recipient"] == "sender-agent"
        assert response["type"] == MessageType.RESPONSE.value
        assert response["correlation_id"] == original_message["message_id"]
        assert response["in_response_to"] == original_message["message_id"]
        assert response["success"] is True
    
    def test_create_error_response(self):
        """Test creating an error response message."""
        # Create original message
        original_message = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "invalid-command"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="receiver-agent"
        )
        
        # Create error response
        error_response = original_message.create_response(
            sender="receiver-agent",
            content={"error": "Invalid command"},
            success=False,
            metadata={"details": "Command not supported"}
        )
        
        # Verify error response fields
        assert error_response["message_id"] is not None
        assert error_response["sender"] == "receiver-agent"
        assert error_response["content"] == {"error": "Invalid command"}
        assert error_response["routing"]["strategy"] == RoutingStrategy.DIRECT.value
        assert error_response["routing"]["recipient"] == "sender-agent"
        assert error_response["type"] == MessageType.ERROR.value
        assert error_response["correlation_id"] == original_message["message_id"]
        assert error_response["in_response_to"] == original_message["message_id"]
        assert error_response["success"] is False
        assert error_response["metadata"] == {"details": "Command not supported"}
    
    def test_create_message_with_optional_fields(self):
        """Test creating a message with optional fields."""
        # Create message with optional fields
        message = EnhancedAgentMessage.create(
            sender="test-agent",
            content={"command": "ping"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="target-agent",
            message_type=MessageType.COMMAND,
            priority=MessagePriority.HIGH,
            correlation_id="corr-123",
            expires_at="2023-12-31T23:59:59Z",
            metadata={"source": "test"}
        )
        
        # Verify optional fields
        assert message["correlation_id"] == "corr-123"
        assert message["expires_at"] == "2023-12-31T23:59:59Z"
        assert message["metadata"] == {"source": "test"}
        assert message["priority"] == MessagePriority.HIGH.value
    
    def test_create_response(self):
        """Test creating a response to a message."""
        # Create original message
        original = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "query"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="responder-agent",
            message_type=MessageType.QUERY
        )
        
        # Create response
        response = original.create_response(
            sender="responder-agent",
            content={"result": "data"},
            success=True
        )
        
        # Verify response structure
        assert "message_id" in response
        assert response["sender"] == "responder-agent"
        assert response["routing"]["strategy"] == RoutingStrategy.DIRECT.value
        assert response["routing"]["recipient"] == "sender-agent"
        assert response["type"] == MessageType.RESPONSE.value
        assert response["in_response_to"] == original["message_id"]
        assert response["content"] == {"result": "data"}
    
    def test_create_error_response(self):
        """Test creating an error response."""
        # Create original message
        original = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "query"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="responder-agent"
        )
        
        # Create error response
        response = original.create_response(
            sender="responder-agent",
            content={"error": "Failed to process query"},
            success=False
        )
        
        # Verify error response
        assert response["type"] == MessageType.ERROR.value
        assert response["content"] == {"error": "Failed to process query"}


# Define an async test handler for testing
class TestMessageHandler:
    """Test message handler implementation."""
    
    def __init__(self):
        self.messages_handled = []
        self.responses = {}
    
    async def handle_message(self, message):
        """Handle a test message."""
        self.messages_handled.append(message)
        
        # Return a predefined response or None
        message_id = message.get("message_id")
        return self.responses.get(message_id)


class TestAgentCommunicationService:
    """Test cases for AgentCommunicationService."""
    
    @pytest.fixture
    def mock_kafka_producer(self):
        """Create a mock Kafka producer."""
        mock_producer = MagicMock()
        
        # Add topic_exists and create_topic methods
        mock_producer.topic_exists = MagicMock(return_value=False)
        mock_producer.create_topic = MagicMock(return_value=None)
        
        # Mock the async produce method
        def mock_produce(topic, key, value):
            return None
        
        mock_producer.produce = mock_produce
        
        return mock_producer
    
    @pytest.fixture
    def mock_kafka_consumer(self):
        """Create a mock Kafka consumer."""
        mock_consumer = MagicMock()
        
        # Add subscribe method
        mock_consumer.subscribe = MagicMock(return_value=None)
        
        # Mock the start_consumer method to be awaitable
        async def mock_start_consumer(*args, **kwargs):
            return None
        
        mock_consumer.start_consumer = mock_start_consumer
        
        return mock_consumer
    
    @pytest.fixture
    def communication_service(self, mock_kafka_producer, mock_kafka_consumer):
        """Create a communication service for testing."""
        # Create a service instance with mock producer and consumer
        service = AgentCommunicationService(
            kafka_producer=mock_kafka_producer,
            kafka_consumer=mock_kafka_consumer,
            topic_prefix="test-agent"
        )
        
        return service
    
    def test_init_creates_standard_topics(self, communication_service, mock_kafka_producer):
        """Test that initialization creates all required topics."""
        # Verify topic_exists and create_topic were called
        assert mock_kafka_producer.topic_exists.call_count == 4
        assert mock_kafka_producer.create_topic.call_count == 4
        
        # Since we always return False for topic_exists, all topics should be created
        expected_topics = [
            "test-agent.direct",
            "test-agent.broadcast",
            "test-agent.capability",
            "test-agent.group"
        ]
        
        # Check that each topic was verified and created
        for topic in expected_topics:
            mock_kafka_producer.topic_exists.assert_any_call(topic)
            mock_kafka_producer.create_topic.assert_any_call(topic)
    
    def test_get_topic_for_strategy(self, communication_service):
        """Test getting the correct topic for each routing strategy."""
        # Direct strategy
        topic = communication_service._get_topic_for_strategy(RoutingStrategy.DIRECT)
        assert topic == "test-agent.direct"
        
        # Broadcast strategy
        topic = communication_service._get_topic_for_strategy(RoutingStrategy.BROADCAST)
        assert topic == "test-agent.broadcast"
        
        # Capability strategy
        topic = communication_service._get_topic_for_strategy(RoutingStrategy.CAPABILITY)
        assert topic == "test-agent.capability"
        
        # Group strategy
        topic = communication_service._get_topic_for_strategy(RoutingStrategy.GROUP)
        assert topic == "test-agent.group"
        
        # Invalid strategy - the implementation returns a default topic
        # This is a string that's not a valid RoutingStrategy enum value
        # The implementation handles this gracefully by returning the default topic
        topic = communication_service._get_topic_for_strategy("invalid")
        assert topic == "test-agent.direct"
    
    @pytest.mark.asyncio
    async def test_send_message(self, communication_service, mock_kafka_producer):
        """Test sending a message."""
        # Set up the mock producer to update message status properly
        def mock_produce(topic, key, value):
            # Parse the message from the value and update its status
            message_dict = json.loads(value)
            message_dict["routing"]["status"] = MessageStatus.SENT.value
            return True
        
        # Override the mock produce method
        mock_kafka_producer.produce = mock_produce
        
        # Create a message
        message = EnhancedAgentMessage.create(
            sender="test-agent",
            content={"command": "test"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="target-agent"
        )
        
        # Send the message
        await communication_service.send_message(message)
        
        # Check that message status was updated
        assert message["routing"]["status"] == MessageStatus.SENT.value
    
    def test_validate_message_valid(self, communication_service):
        """Test validating a valid message."""
        # Create a valid message
        message = EnhancedAgentMessage.create(
            sender="test-agent",
            content={"command": "test"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="target-agent"
        )
        
        # Validate the message
        result = communication_service._validate_message(message)
        
        # Should return True for valid message
        assert result is True
    
    def test_validate_message_missing_field(self, communication_service):
        """Test validating a message with missing required field."""
        # Manually create an invalid message missing sender
        message = {
            "message_id": "test-msg",
            "routing": {
                "strategy": RoutingStrategy.DIRECT.value,
                "recipient": "target-agent",
                "status": MessageStatus.CREATED.value
            },
            "content": {"command": "test"},
            "timestamp": utc_now().isoformat(),
            "type": MessageType.COMMAND.value,
            "priority": MessagePriority.NORMAL.value
            # Missing sender field
        }
        
        # Validate the message - should either raise ValueError or return False
        try:
            result = communication_service._validate_message(message)
            assert result is False
        except ValueError:
            # If it raises ValueError, that's acceptable too
            pass
        
    def test_validate_message_missing_recipient(self, communication_service):
        """Test validating a direct message with missing recipient."""
        # Manually create a message with direct routing but no recipient
        message = {
            "message_id": "test-msg",
            "sender": "test-agent",
            "timestamp": utc_now().isoformat(),
            "routing": {
                "strategy": RoutingStrategy.DIRECT.value,
                "status": MessageStatus.CREATED.value
                # Missing recipient field
            },
            "content": {"command": "test"},
            "type": MessageType.COMMAND.value,
            "priority": MessagePriority.NORMAL.value
        }
        
        # Validate the message - should either raise ValueError or return False
        try:
            result = communication_service._validate_message(message)
            assert result is False
        except ValueError:
            # If it raises ValueError, that's acceptable too
            pass
    
    def test_register_handler(self, communication_service):
        """Test registering a message handler."""
        # Create a test handler
        handler = MagicMock(spec=MessageHandlerProtocol)
        
        # Register the handler
        communication_service.register_handler("test-agent", handler)
        
        # Verify the handler was registered
        assert "test-agent" in communication_service.message_handlers
        assert handler in communication_service.message_handlers["test-agent"]
    
    def test_register_multiple_handlers(self, communication_service):
        """Test registering multiple handlers for an agent."""
        # Create test handlers
        handler1 = MagicMock(spec=MessageHandlerProtocol)
        handler2 = MagicMock(spec=MessageHandlerProtocol)
        
        # Register the handlers
        communication_service.register_handler("test-agent", handler1)
        communication_service.register_handler("test-agent", handler2)
        
        # Verify both handlers were registered
        assert len(communication_service.message_handlers["test-agent"]) == 2
        assert handler1 in communication_service.message_handlers["test-agent"]
        assert handler2 in communication_service.message_handlers["test-agent"]
    
    def test_unregister_handler(self, communication_service):
        """Test unregistering a specific handler."""
        # Create test handlers
        handler1 = MagicMock(spec=MessageHandlerProtocol)
        handler2 = MagicMock(spec=MessageHandlerProtocol)
        
        # Register the handlers
        communication_service.register_handler("test-agent", handler1)
        communication_service.register_handler("test-agent", handler2)
        
        # Unregister one handler
        communication_service.unregister_handler("test-agent", handler1)
        
        # Verify handler1 was unregistered but handler2 remains
        assert handler1 not in communication_service.message_handlers["test-agent"]
        assert handler2 in communication_service.message_handlers["test-agent"]
    
    def test_unregister_all_handlers(self, communication_service):
        """Test unregistering all handlers for an agent."""
        # Create test handlers
        handler1 = MagicMock(spec=MessageHandlerProtocol)
        handler2 = MagicMock(spec=MessageHandlerProtocol)
        
        # Register the handlers
        communication_service.register_handler("test-agent", handler1)
        communication_service.register_handler("test-agent", handler2)
        
        # Unregister all handlers
        communication_service.unregister_handler("test-agent")
        
        # Verify all handlers were unregistered
        assert "test-agent" not in communication_service.message_handlers
    
    @pytest.mark.asyncio
    async def test_start_consumers(self, communication_service, mock_kafka_consumer):
        """Test starting Kafka consumers."""
        # Set up the mock consumer
        mock_kafka_consumer.subscribe.return_value = None
        
        # Start consumers
        await communication_service.start_consumers()
        
        # Verify consumer was started for each topic
        assert mock_kafka_consumer.subscribe.call_count == 4
        
        # Verify the topic names
        topic_calls = [call_args[0][0] for call_args in mock_kafka_consumer.subscribe.call_args_list]
        expected_topics = [
            ["test-agent.direct"],
            ["test-agent.broadcast"],
            ["test-agent.capability"],
            ["test-agent.group"]
        ]
        for expected_topic in expected_topics:
            assert expected_topic in topic_calls
            
    @pytest.mark.asyncio
    async def test_message_consumer_handler_direct(self, communication_service):
        """Test handling a direct message from Kafka consumer."""
        # Create a mock handler
        handler = TestMessageHandler()
        
        # Register the handler
        communication_service.register_handler("test-agent", handler)
        
        # Create a test message
        message = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "test"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="test-agent"
        )
        
        # Convert to JSON as would come from Kafka
        message_json = json.dumps(message)
        
        # Call the handler directly
        await communication_service._message_consumer_handler(message_json)
        
        # Verify handler was called
        assert len(handler.messages_handled) == 1
        assert handler.messages_handled[0]["message_id"] == message["message_id"]
        
    @pytest.mark.asyncio
    async def test_message_consumer_handler_bytes(self, communication_service):
        """Test handling a message that comes as bytes from Kafka."""
        # Create a mock handler
        handler = TestMessageHandler()
        
        # Register the handler
        communication_service.register_handler("test-agent", handler)
        
        # Create a test message
        message = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "test"},
            routing_strategy=RoutingStrategy.DIRECT,
            recipient="test-agent"
        )
        
        # Convert to JSON bytes as would come from Kafka
        message_bytes = json.dumps(message).encode('utf-8')
        
        # Call the handler directly
        await communication_service._message_consumer_handler(message_bytes)
        
        # Verify handler was called
        assert len(handler.messages_handled) == 1
        assert handler.messages_handled[0]["message_id"] == message["message_id"]

    @pytest.mark.asyncio
    async def test_message_consumer_handler_broadcast(self, communication_service):
        """Test handling a broadcast message from Kafka consumer."""
        # Create mock handlers
        handler1 = TestMessageHandler()
        handler2 = TestMessageHandler()
        
        # Register the handlers
        communication_service.register_handler("agent1", handler1)
        communication_service.register_handler("agent2", handler2)
        
        # Create a test message
        message = EnhancedAgentMessage.create(
            sender="sender-agent",
            content={"command": "broadcast-test"},
            routing_strategy=RoutingStrategy.BROADCAST
        )
        
        # Convert to JSON as would come from Kafka
        message_json = json.dumps(message)
        
        # Call the handler directly
        await communication_service._message_consumer_handler(message_json)
        
        # Verify both handlers were called
        assert len(handler1.messages_handled) == 1
        assert len(handler2.messages_handled) == 1
        assert handler1.messages_handled[0]["message_id"] == message["message_id"]
        assert handler2.messages_handled[0]["message_id"] == message["message_id"]

    @pytest.mark.asyncio
    async def test_dispatch_message(self, communication_service):
        """Test dispatching a message to a handler."""
        # Create a test handler with synchronous implementation
        handler = TestMessageHandler()
        
        # Register the handler
        communication_service.register_handler("test-agent", handler)
        
        # Create a test message
        message = {
            "message_id": "test-msg-id",
            "sender": "test-sender",
            "routing": {
                "status": MessageStatus.DELIVERED.value
            },
            "content": {"command": "test"}
        }
        
        # Dispatch the message
        await communication_service._dispatch_message("test-agent", message)
        
        # Verify the handler was called
        assert len(handler.messages_handled) == 1
        assert handler.messages_handled[0]["message_id"] == "test-msg-id"
        
        # Verify the message status was updated
        assert message["routing"]["status"] == MessageStatus.PROCESSING.value
        
    @pytest.mark.asyncio
    async def test_dispatch_message_async_handler(self, communication_service):
        """Test dispatching a message to an async handler."""
        class AsyncTestHandler(MessageHandlerProtocol):
            def __init__(self):
                self.messages_handled = []
                
            async def handle_message(self, message):
                self.messages_handled.append(message)
                return None
        
        # Create a test handler with async implementation
        async_handler = AsyncTestHandler()
        
        # Register the handler
        communication_service.register_handler("test-agent", async_handler)
        
        # Create a test message
        message = {
            "message_id": "test-async-msg",
            "sender": "test-sender",
            "routing": {
                "status": MessageStatus.DELIVERED.value
            },
            "content": {"command": "test"}
        }
        
        # Dispatch the message
        await communication_service._dispatch_message("test-agent", message)
        
        # Verify the handler was called
        assert len(async_handler.messages_handled) == 1
        assert async_handler.messages_handled[0]["message_id"] == "test-async-msg"
        
        # Verify the message status was updated
        assert message["routing"]["status"] == MessageStatus.PROCESSING.value
        
    @pytest.mark.asyncio
    async def test_dispatch_message_error_handling(self, communication_service):
        """Test error handling when dispatching a message."""
        class ErrorTestHandler(MessageHandlerProtocol):
            def handle_message(self, message):
                raise Exception("Test error")
        
        # Create a test handler that raises an exception
        error_handler = ErrorTestHandler()
        
        # Register the handler
        communication_service.register_handler("test-agent", error_handler)
        
        # Create a test message
        message = {
            "message_id": "test-error-msg",
            "sender": "test-sender",
            "routing": {
                "status": MessageStatus.DELIVERED.value
            },
            "content": {"command": "test"}
        }
        
        # Dispatch the message - should not raise exception
        await communication_service._dispatch_message("test-agent", message)
        
        # Verify the message status was updated to failed
        assert message["routing"]["status"] == MessageStatus.FAILED.value
