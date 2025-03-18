"""
Kafka service implementation.

This module provides a high-level interface for interacting with Kafka.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, TypedDict, Self, NotRequired, cast

from confluent_kafka import KafkaError, KafkaException
from pydantic import BaseModel, ValidationError

from clubhouse.services.kafka_protocol import (
    KafkaConsumerProtocol,
    KafkaProducerProtocol,
    MessageHandlerProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class KafkaMessageHeaders(TypedDict):
    """Type definition for Kafka message headers."""
    name: str
    value: str


class KafkaMessageValue(TypedDict):
    """Base type definition for Kafka message values."""
    pass


class KafkaMessageWithMetadata(TypedDict):
    """Type definition for Kafka message with metadata."""
    topic: str
    value: Dict[str, Any]  
    key: NotRequired[str]
    headers: NotRequired[Dict[str, str]]


class KafkaServiceError(Exception):
    """Base exception for Kafka service errors."""
    pass


class MessageProducerError(KafkaServiceError):
    """Exception raised when producing a message fails."""
    pass


class MessageConsumerError(KafkaServiceError):
    """Exception raised when consuming a message fails."""
    pass


class KafkaMessage(BaseModel):
    """Kafka message model."""

    topic: str
    value: Dict[str, Any]
    key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class KafkaService:
    """
    Kafka service for producing and consuming messages.

    This class provides a high-level interface for interacting with Kafka.
    """

    def __init__(
        self,
        producer: KafkaProducerProtocol,
        consumer: KafkaConsumerProtocol,
    ) -> None:
        """
        Initialize the Kafka service.

        Args:
            producer: The Kafka producer
            consumer: The Kafka consumer
        """
        self._producer = producer
        self._consumer = consumer
        self._running = False

    def produce_message(self, message: KafkaMessage) -> None:
        """
        Produce a message to Kafka.

        Args:
            message: The message to produce

        Raises:
            ValueError: If topic is missing
            ValidationError: If message fails validation
            MessageProducerError: If there's an error producing the message
        """
        try:
            # Convert message to a dictionary
            if isinstance(message.value, BaseModel):  
                value_dict = message.value.dict()
            else:
                value_dict = message.value

            # Validate the message
            if not message.topic:
                raise ValueError("Topic is required")

            # Produce the message
            self._producer.produce(
                topic=message.topic,
                value=json.dumps(value_dict).encode("utf-8"),
                key=message.key.encode("utf-8") if message.key else None,
                headers=message.headers,
            )

            # Flush to ensure delivery
            self._producer.flush()

        except ValidationError as e:
            logger.error("Validation error producing message: %s", e)
            raise
        except KafkaException as e:
            logger.error("Kafka error producing message: %s", e)
            raise MessageProducerError(f"Failed to produce message: {e}") from e
        except Exception as e:
            logger.error("Unexpected error producing message: %s", e)
            raise MessageProducerError(f"Unexpected error producing message: {e}") from e

    def consume_messages(
        self,
        topics: List[str],
        handler: Callable[
            [Dict[str, Any], Optional[str], Optional[Dict[str, str]]], None
        ],
        timeout: float = 1.0,
    ) -> None:
        """
        Consume messages from Kafka.

        Args:
            topics: Topics to consume from
            handler: Function to handle messages
            timeout: Poll timeout in seconds

        Raises:
            ValueError: If no topics are provided
            MessageConsumerError: If there's an error consuming messages
        """
        if not topics:
            raise ValueError("At least one topic is required")

        try:
            # Subscribe to topics
            self._consumer.subscribe(topics)
            logger.info("Subscribed to topics: %s", topics)

            # Set running flag
            self._running = True

            # Consume messages
            while self._running:
                # Poll for messages
                msg = self._consumer.poll(timeout)

                if msg is None:
                    continue

                if msg.error():
                    match msg.error().code():
                        case KafkaError._PARTITION_EOF:
                            # End of partition event - not an error
                            logger.debug("Reached end of partition")
                        case _:
                            # Error
                            logger.error("Consumer error: %s", msg.error())
                    continue

                # Process the message
                try:
                    # Parse the value as JSON
                    value: Dict[str, Any] = json.loads(msg.value().decode("utf-8"))
                    key: Optional[str] = msg.key().decode("utf-8") if msg.key() else None
                    headers: Optional[Dict[str, str]] = (
                        msg.headers()
                        if hasattr(msg, "headers") and callable(msg.headers)
                        else None
                    )

                    # Call the handler with the message value, key, and headers
                    handler(value, key, headers)

                except json.JSONDecodeError as e:
                    logger.error("Error decoding message: %s", e)
                except Exception as e:
                    logger.error("Error processing message: %s", e)

        except KafkaException as e:
            logger.error("Kafka error during consumption: %s", e)
            raise MessageConsumerError(f"Failed to consume messages: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during consumption: %s", e)
            raise MessageConsumerError(f"Unexpected error during consumption: {e}") from e
        finally:
            # Close the consumer
            try:
                if self._consumer and hasattr(self._consumer, "close"):
                    self._consumer.close()
                    logger.info("Consumer closed")
            except Exception as e:
                logger.warning("Error closing consumer: %s", e)

    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self._running = False
        try:
            self._consumer.close()
            logger.info("Stopped consuming messages")
        except Exception as e:
            logger.warning("Error stopping consumer: %s", e)
            
    @classmethod
    def create(cls, config: Dict[str, Any]) -> Self:
        """
        Create a new KafkaService instance from configuration.
        
        Args:
            config: Configuration dictionary for the Kafka service
            
        Returns:
            A new KafkaService instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        from confluent_kafka import Consumer, Producer
        
        try:
            producer_config = config.get("producer", {})
            consumer_config = config.get("consumer", {})
            
            producer = Producer(producer_config)
            consumer = Consumer(consumer_config)
            
            return cls(producer=producer, consumer=consumer)
        except Exception as e:
            raise ValueError(f"Failed to create KafkaService: {e}") from e
