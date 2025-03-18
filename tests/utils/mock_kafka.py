"""
Mock Kafka implementation for testing.

This module provides mock implementations of Kafka components to allow
testing without requiring a running Kafka broker.
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import MagicMock

from confluent_kafka import Producer as ConfluentKafkaProducer
from confluent_kafka import Consumer as ConfluentKafkaConsumer
from confluent_kafka import Message as ConfluentKafkaMessage

logger = logging.getLogger(__name__)


class MockKafkaMessage:
    """Mock implementation of a Kafka message."""
    
    def __init__(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[tuple]] = None,
        partition: int = 0,
        offset: int = 0,
        error: int = 0,
    ):
        """
        Initialize a mock Kafka message.
        
        Args:
            topic: Topic name
            value: Message value as bytes
            key: Optional message key as bytes
            headers: Optional message headers as list of tuples
            partition: Partition number
            offset: Message offset
            error: Error code
        """
        self._topic = topic
        self._value = value
        self._key = key
        self._headers = headers or []
        self._partition = partition
        self._offset = offset
        self._error = error
    
    def topic(self) -> str:
        """Get the topic name."""
        return self._topic
    
    def value(self) -> bytes:
        """Get the message value."""
        return self._value
    
    def key(self) -> Optional[bytes]:
        """Get the message key."""
        return self._key
    
    def headers(self) -> List[tuple]:
        """Get the message headers."""
        return self._headers
    
    def partition(self) -> int:
        """Get the partition number."""
        return self._partition
    
    def offset(self) -> int:
        """Get the message offset."""
        return self._offset
    
    def error(self) -> int:
        """Get the error code."""
        return self._error


class MockKafkaProducer:
    """Mock implementation of a Confluent Kafka producer."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a mock Kafka producer.
        
        Args:
            config: Producer configuration
        """
        self.config = config or {}
        self.produced_messages = []
        self.callbacks = []
        self.flush_timeout = 0.0
    
    def produce(
        self,
        topic: str,
        value: bytes = None,
        key: bytes = None,
        headers: List[tuple] = None,
        partition: int = -1,
        on_delivery: Callable = None,
        timestamp: int = 0,
    ) -> None:
        """
        Mock produce a message to a topic.
        
        Args:
            topic: Topic name
            value: Message value
            key: Message key
            headers: Message headers
            partition: Target partition
            on_delivery: Delivery callback
            timestamp: Message timestamp
        """
        logger.debug(f"Mock producing to topic {topic}: {value}")
        
        # Store the message for later inspection
        message = {
            "topic": topic,
            "value": value,
            "key": key,
            "headers": headers,
            "partition": partition,
            "timestamp": timestamp,
        }
        self.produced_messages.append(message)
        
        # Store the callback for later delivery
        if on_delivery:
            self.callbacks.append((on_delivery, message))
    
    def flush(self, timeout: float = -1) -> int:
        """
        Mock flush any outstanding messages.
        
        Args:
            timeout: Flush timeout in seconds
            
        Returns:
            Number of outstanding messages
        """
        self.flush_timeout = timeout if timeout > 0 else 0.0
        
        # Call all delivery callbacks
        for callback, message in self.callbacks:
            mock_message = MockKafkaMessage(
                topic=message["topic"],
                value=message["value"],
                key=message["key"],
                headers=message["headers"],
                partition=message["partition"] if message["partition"] >= 0 else 0,
                offset=len(self.produced_messages) - 1,
                error=0,  # No error
            )
            callback(None, mock_message)  # No error, successful delivery
        
        # Clear the callbacks
        self.callbacks = []
        
        return 0  # No messages left
    
    def poll(self, timeout: float = 0) -> None:
        """
        Mock poll for events.
        
        Args:
            timeout: Poll timeout in seconds
        """
        pass


class MockKafkaConsumer:
    """Mock implementation of a Confluent Kafka consumer."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a mock Kafka consumer.
        
        Args:
            config: Consumer configuration
        """
        self.config = config or {}
        self.subscribed_topics = []
        self.assignment = []
        self.committed_offsets = {}
        self.message_queue = []
        self.closed = False
    
    def subscribe(self, topics: List[str], on_assign: Callable = None) -> None:
        """
        Mock subscribe to topics.
        
        Args:
            topics: List of topics to subscribe to
            on_assign: Assignment callback
        """
        logger.debug(f"Mock subscribing to topics: {topics}")
        self.subscribed_topics = topics
        
        # Create a mock assignment
        self.assignment = [{"topic": topic, "partition": 0} for topic in topics]
        
        # Call the assignment callback if provided
        if on_assign:
            on_assign(self, self.assignment)
    
    def poll(self, timeout: float = None) -> Optional[MockKafkaMessage]:
        """
        Mock poll for messages.
        
        Args:
            timeout: Poll timeout in seconds
            
        Returns:
            A message if available, None otherwise
        """
        if not self.message_queue:
            return None
        
        return self.message_queue.pop(0)
    
    def add_mock_message(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[tuple]] = None,
        partition: int = 0,
        offset: int = 0,
    ) -> None:
        """
        Add a mock message to the consumer queue.
        
        Args:
            topic: Topic name
            value: Message value
            key: Message key
            headers: Message headers
            partition: Partition number
            offset: Message offset
        """
        if topic not in self.subscribed_topics:
            logger.warning(f"Adding message for topic {topic} not in subscribed topics")
        
        message = MockKafkaMessage(
            topic=topic,
            value=value,
            key=key,
            headers=headers,
            partition=partition,
            offset=offset,
        )
        self.message_queue.append(message)
    
    def close(self) -> None:
        """Close the consumer."""
        logger.debug("Mock closing consumer")
        self.closed = True
    
    def commit(self, message: Optional[MockKafkaMessage] = None, asynchronous: bool = True) -> None:
        """
        Mock commit offsets.
        
        Args:
            message: Optional message to commit
            asynchronous: Whether to commit asynchronously
        """
        if message:
            topic = message.topic()
            partition = message.partition()
            offset = message.offset()
            self.committed_offsets.setdefault(topic, {})[partition] = offset + 1
        else:
            # Commit all current position (not implemented in mock)
            pass


def create_mock_producer() -> MagicMock:
    """
    Create a MagicMock for a Confluent Kafka Producer.
    
    Returns:
        MagicMock with Producer interface
    """
    producer = MagicMock(spec=ConfluentKafkaProducer)
    producer.produce.return_value = None
    producer.flush.return_value = 0
    
    return producer


def create_mock_consumer() -> MagicMock:
    """
    Create a MagicMock for a Confluent Kafka Consumer.
    
    Returns:
        MagicMock with Consumer interface
    """
    consumer = MagicMock(spec=ConfluentKafkaConsumer)
    consumer.subscribe.return_value = None
    consumer.poll.return_value = None
    consumer.close.return_value = None
    consumer.commit.return_value = None
    
    return consumer
