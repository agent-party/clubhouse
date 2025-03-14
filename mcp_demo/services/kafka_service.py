"""
Kafka service implementation.

This module provides a high-level interface for interacting with Kafka.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, cast

from pydantic import BaseModel, ValidationError
from confluent_kafka import KafkaError, KafkaException

from mcp_demo.services.kafka_protocol import (
    KafkaProducerProtocol,
    KafkaConsumerProtocol,
    MessageHandlerProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')

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
    ):
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
        """
        try:
            # Convert message to a dictionary
            if isinstance(message.value, BaseModel):  # pragma: no cover
                value_dict = message.value.dict()
            else:
                value_dict = message.value
            
            # Validate the message
            if not message.topic:  # pragma: no cover
                raise ValueError("Topic is required")
                
            # Produce the message
            self._producer.produce(
                topic=message.topic,
                value=json.dumps(value_dict).encode('utf-8'),
                key=message.key.encode('utf-8') if message.key else None,
                headers=message.headers
            )
            
            # Flush to ensure delivery
            self._producer.flush()
            
        except (ValidationError, KafkaException) as e:  # pragma: no cover
            logger.error("Error producing message: %s", e)
            raise
    
    def consume_messages(
        self,
        topics: List[str],
        handler: Callable[[Dict[str, Any], Optional[str], Optional[Dict[str, str]]], None],
        timeout: float = 1.0
    ) -> None:
        """
        Consume messages from Kafka.
        
        Args:
            topics: Topics to consume from
            handler: Function to handle messages
            timeout: Poll timeout in seconds
        """
        if not topics:  # pragma: no cover
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
                
                if msg is None:  # pragma: no cover
                    continue
                
                if msg.error():  # pragma: no cover
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event - not an error
                        logger.debug("Reached end of partition")
                    else:
                        # Error
                        logger.error("Consumer error: %s", msg.error())
                    continue
                
                # Process the message
                try:
                    # Parse the value as JSON
                    value = json.loads(msg.value().decode('utf-8'))
                    key = msg.key().decode('utf-8') if msg.key() else None
                    headers = msg.headers() if hasattr(msg, 'headers') and callable(msg.headers) else None
                    
                    # Call the handler with the message value, key, and headers
                    handler(value, key, headers)
                    
                except json.JSONDecodeError as e:  # pragma: no cover
                    logger.error("Error decoding message: %s", e)
                except Exception as e:  # pragma: no cover
                    logger.error("Error processing message: %s", e)
        
        except KafkaException as e:  # pragma: no cover
            logger.error("Kafka error: %s", e)
            raise
        finally:
            # Close the consumer
            if self._consumer and hasattr(self._consumer, 'close'):
                self._consumer.close()
                logger.info("Consumer closed")
    
    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self._running = False
        self._consumer.close()
        logger.info("Stopped consuming messages")
