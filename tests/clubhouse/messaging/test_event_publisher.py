"""
Tests for the clubhouse event publisher.

This module contains tests for the event publisher service, which is
responsible for publishing events from the clubhouse to Kafka.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.event_publisher import EventPublisher
from clubhouse.services.confluent_kafka_service import ConfluentKafkaService, KafkaMessage
from scripts.kafka_cli.message_schemas import MessageType


@pytest.fixture
def mock_kafka_service():
    """Fixture for a mock Kafka service."""
    service = MagicMock(spec=ConfluentKafkaService)
    return service


@pytest.fixture
def service_registry(mock_kafka_service):
    """Fixture for a service registry with a mock Kafka service."""
    registry = MagicMock(spec=ServiceRegistry)
    registry.get.return_value = mock_kafka_service
    return registry


@pytest.fixture
def event_publisher(service_registry):
    """Fixture for an event publisher with mocked dependencies."""
    return EventPublisher(service_registry, events_topic="test-events")


def test_event_publisher_initialization(event_publisher, service_registry):
    """Test that the event publisher initializes correctly."""
    assert event_publisher._service_registry == service_registry
    assert event_publisher._events_topic == "test-events"
    service_registry.get.assert_called_with("kafka")


def test_publish_event(event_publisher, mock_kafka_service):
    """Test publishing an event to Kafka."""
    # Create an event message
    event = {
        "message_type": MessageType.EVENT_AGENT_THINKING.value,
        "agent_id": "test-agent"
    }
    
    # Publish the event
    event_publisher.publish_event(event)
    
    # Verify a Kafka message was produced
    mock_kafka_service.produce_message.assert_called_once()
    
    # Get the Kafka message that was produced
    kafka_message = mock_kafka_service.produce_message.call_args[0][0]
    
    # Verify the message was constructed correctly
    assert isinstance(kafka_message, KafkaMessage)
    assert kafka_message.topic == "test-events"
    assert kafka_message.value["message_type"] == MessageType.EVENT_AGENT_THINKING.value
    assert kafka_message.value["agent_id"] == "test-agent"
    assert "message_id" in kafka_message.value
    assert "timestamp" in kafka_message.value
    assert kafka_message.key == "test-agent"
    assert kafka_message.headers == {"message_type": MessageType.EVENT_AGENT_THINKING.value}


def test_publish_event_with_custom_topic(event_publisher, mock_kafka_service):
    """Test publishing an event to a custom topic."""
    # Create an event message
    event = {
        "message_type": MessageType.EVENT_AGENT_THINKING.value,
        "agent_id": "test-agent"
    }
    
    # Publish the event to a custom topic
    event_publisher.publish_event(event, topic="custom-topic")
    
    # Get the Kafka message that was produced
    kafka_message = mock_kafka_service.produce_message.call_args[0][0]
    
    # Verify the custom topic was used
    assert kafka_message.topic == "custom-topic"


def test_publish_event_with_existing_message_id(event_publisher, mock_kafka_service):
    """Test publishing an event with an existing message ID."""
    # Create an event message with an ID
    event = {
        "message_id": "existing-id",
        "message_type": MessageType.EVENT_AGENT_THINKING.value,
        "agent_id": "test-agent"
    }
    
    # Publish the event
    event_publisher.publish_event(event)
    
    # Get the Kafka message that was produced
    kafka_message = mock_kafka_service.produce_message.call_args[0][0]
    
    # Verify the existing ID was preserved
    assert kafka_message.value["message_id"] == "existing-id"


def test_publish_event_with_existing_timestamp(event_publisher, mock_kafka_service):
    """Test publishing an event with an existing timestamp."""
    # Create a timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Create an event message with a timestamp
    event = {
        "message_type": MessageType.EVENT_AGENT_THINKING.value,
        "agent_id": "test-agent",
        "timestamp": timestamp
    }
    
    # Publish the event
    event_publisher.publish_event(event)
    
    # Get the Kafka message that was produced
    kafka_message = mock_kafka_service.produce_message.call_args[0][0]
    
    # Verify the existing timestamp was preserved
    assert kafka_message.value["timestamp"] == timestamp


def test_publish_event_without_message_type(event_publisher):
    """Test publishing an event without a message type."""
    # Create an invalid event message
    event = {
        "agent_id": "test-agent"
    }
    
    # Attempt to publish the event
    with pytest.raises(ValueError):
        event_publisher.publish_event(event)
