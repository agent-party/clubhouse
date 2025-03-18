"""
Event publisher for the clubhouse.

This module provides a service for publishing events from the clubhouse
to Kafka topics, allowing the CLI to receive notifications about agent
activities and system state changes.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, cast

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.services.confluent_kafka_service import ConfluentKafkaService, KafkaMessage

logger = logging.getLogger(__name__)


class EventPublisherProtocol(Protocol):
    """Protocol for event publisher service."""
    
    def publish_event(
        self, 
        event_message: Dict[str, Any], 
        topic: Optional[str] = None
    ) -> None:
        """
        Publish an event to Kafka.
        
        Args:
            event_message: The event message to publish
            topic: Optional override of the default topic
            
        Raises:
            ValueError: If the message is invalid
        """
        ...


class EventPublisher:
    """
    Service for publishing events to Kafka.
    
    This service is responsible for formatting event messages and
    publishing them to the appropriate Kafka topics.
    """
    
    def __init__(
        self, 
        service_registry: ServiceRegistry,
        events_topic: str = "clubhouse-events",
        responses_topic: str = "clubhouse-responses"
    ) -> None:
        """
        Initialize the event publisher.
        
        Args:
            service_registry: Registry for accessing required services
            events_topic: The default topic for events
            responses_topic: The default topic for responses
        """
        self._service_registry = service_registry
        self._events_topic = events_topic
        self._responses_topic = responses_topic
        self._producer = None
        
        # Try to get the Kafka service if available
        try:
            self._kafka_service = cast(
                ConfluentKafkaService, 
                service_registry.get("kafka")
            )
        except Exception as e:
            logger.debug(f"Kafka service not yet available: {e}")
            self._kafka_service = None
    
    def publish_event(
        self, 
        event_message: Dict[str, Any], 
        topic: Optional[str] = None
    ) -> None:
        """
        Publish an event to Kafka.
        
        Args:
            event_message: The event message to publish
            topic: Optional override of the default topic
            
        Raises:
            ValueError: If the message is invalid
        """
        event_topic = topic or self._events_topic
        
        if "message_id" not in event_message:
            event_message["message_id"] = str(uuid.uuid4())
            
        if "timestamp" not in event_message:
            event_message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if "message_type" not in event_message:
            raise ValueError("Event message must have a message_type field")
        
        logger.debug(f"Publishing event of type {event_message['message_type']} to {event_topic}")
        
        # Use direct producer if available (for testing and direct integration)
        if self._producer is not None:
            try:
                # Convert to JSON string
                event_json = json.dumps(event_message).encode("utf-8")
                
                # Get key
                key = event_message.get("agent_id")
                if key is None:
                    key = event_message.get("message_id")
                
                # Produce the message
                self._producer.produce(
                    topic=event_topic,
                    key=key,
                    value=event_json,
                    headers=[
                        ("message_type", event_message["message_type"].encode("utf-8"))
                    ]
                )
                
                # Flush to ensure delivery
                self._producer.flush()
                
                logger.debug(f"Event published directly: {event_message['message_id']}")
                return
                
            except Exception as e:
                logger.error(f"Error publishing event directly: {e}")
                # Fall back to Kafka service if direct publishing fails
        
        # Use Kafka service if available
        if self._kafka_service is not None:
            try:
                # Create Kafka message
                kafka_message = KafkaMessage(
                    topic=event_topic,
                    value=event_message,
                    key=event_message.get("agent_id", event_message.get("message_id")),
                    headers={"message_type": event_message["message_type"]}
                )
                
                # Publish the message
                self._kafka_service.produce_message(kafka_message)
                
                logger.debug(f"Event published via service: {event_message['message_id']}")
                return
                
            except Exception as e:
                logger.error(f"Error publishing event via Kafka service: {e}")
        
        # Log if we couldn't publish
        logger.warning(f"Could not publish event: {event_message['message_id']} - No working publisher available")
