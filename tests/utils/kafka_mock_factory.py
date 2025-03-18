"""
Kafka Mock Factory.

This module provides realistic mock implementations for Kafka components
to facilitate unit testing without requiring a running Kafka broker.
"""

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Set, Tuple

from unittest.mock import MagicMock

from clubhouse.services.kafka_protocol import (
    KafkaProducerProtocol,
    KafkaConsumerProtocol,
    MessageHandlerProtocol,
    SerializerProtocol,
    DeserializerProtocol
)

logger = logging.getLogger(__name__)


class MockMessage:
    """Mock implementation of a Kafka message with realistic behavior."""
    
    def __init__(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[Tuple[str, bytes]]] = None,
        partition: int = 0,
        offset: int = 0,
        error: int = 0,
        timestamp: Optional[int] = None
    ):
        """Initialize a mock Kafka message."""
        self._topic = topic
        self._value = value
        self._key = key
        self._headers = headers or []
        self._partition = partition
        self._offset = offset
        self._error = error
        self._timestamp = timestamp or int(time.time() * 1000)
    
    def topic(self) -> str:
        return self._topic
    
    def value(self) -> bytes:
        return self._value
    
    def key(self) -> Optional[bytes]:
        return self._key
    
    def headers(self) -> List[Tuple[str, bytes]]:
        return self._headers
    
    def partition(self) -> int:
        return self._partition
    
    def offset(self) -> int:
        return self._offset
    
    def error(self) -> int:
        return self._error
    
    def timestamp(self) -> Tuple[int, int]:
        """Return timestamp with creation time type (0)."""
        return (0, self._timestamp)


class KafkaTopicPartition:
    """Simulates a Kafka topic partition with offset tracking."""
    
    def __init__(self, topic: str, partition: int = 0):
        """Initialize a topic partition."""
        self.topic = topic
        self.partition = partition
        self.messages: List[MockMessage] = []
        self.next_offset = 0
    
    def add_message(self, message: MockMessage) -> int:
        """Add a message to the partition and return its offset."""
        message._offset = self.next_offset
        message._partition = self.partition
        self.messages.append(message)
        self.next_offset += 1
        return message._offset


class KafkaClusterState:
    """Simulates a Kafka cluster state for testing."""
    
    def __init__(self):
        """Initialize the cluster state."""
        self.topic_partitions: Dict[str, List[KafkaTopicPartition]] = {}
        self.consumer_groups: Dict[str, Dict[str, Dict[int, int]]] = {}
        self._lock = threading.RLock()
    
    def get_or_create_topic(self, topic: str, num_partitions: int = 1) -> List[KafkaTopicPartition]:
        """Get or create a topic with the specified number of partitions."""
        with self._lock:
            if topic not in self.topic_partitions:
                self.topic_partitions[topic] = [
                    KafkaTopicPartition(topic, i) for i in range(num_partitions)
                ]
            return self.topic_partitions[topic]
    
    def add_message(self, topic: str, value: bytes, key: Optional[bytes] = None, 
                  headers: Optional[List[Tuple[str, bytes]]] = None) -> MockMessage:
        """Add a message to a topic and return the created message."""
        with self._lock:
            partitions = self.get_or_create_topic(topic)
            # Simple partition selection - could be enhanced to match real Kafka
            # Use key for consistent partition selection if provided
            partition_idx = 0
            if key:
                partition_idx = hash(key) % len(partitions)
            
            message = MockMessage(topic, value, key, headers, partition_idx)
            partitions[partition_idx].add_message(message)
            return message
    
    def get_consumer_position(self, group_id: str, topic: str, partition: int) -> int:
        """Get the current offset for a consumer group on a topic-partition."""
        with self._lock:
            if group_id not in self.consumer_groups:
                self.consumer_groups[group_id] = {}
            
            group = self.consumer_groups[group_id]
            if topic not in group:
                group[topic] = {}
            
            topic_offsets = group[topic]
            if partition not in topic_offsets:
                topic_offsets[partition] = 0
                
            return topic_offsets[partition]
    
    def commit_offset(self, group_id: str, topic: str, partition: int, offset: int) -> None:
        """Commit an offset for a consumer group on a topic-partition."""
        with self._lock:
            if group_id not in self.consumer_groups:
                self.consumer_groups[group_id] = {}
            
            group = self.consumer_groups[group_id]
            if topic not in group:
                group[topic] = {}
            
            group[topic][partition] = offset


# Global shared state for consistent behavior across mock instances
_kafka_cluster = KafkaClusterState()


class MockProducer(KafkaProducerProtocol):
    """Mock Kafka producer with realistic behavior."""
    
    def __init__(self, serializer: SerializerProtocol, cluster_state: KafkaClusterState = None):
        """Initialize the mock producer."""
        self._serializer = serializer
        self._cluster = cluster_state or _kafka_cluster
        self._delivery_callbacks: Dict[str, Callable] = {}
        
    def produce(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        on_delivery: Optional[Callable[[Optional[Exception], Any], None]] = None,
    ) -> None:
        """Produce a message to a topic."""
        try:
            # Serialize message
            serialized_value = self._serializer.serialize(value)
            serialized_key = key.encode('utf-8') if key else None
            
            # Convert headers to expected format
            message_headers = None
            if headers:
                message_headers = [(k, v.encode('utf-8')) for k, v in headers.items()]
            
            # Add message to cluster state
            message = self._cluster.add_message(topic, serialized_value, serialized_key, message_headers)
            
            # Execute delivery callback if provided
            if on_delivery:
                on_delivery(None, message)
                
        except Exception as e:
            logger.error(f"Error producing message to {topic}: {e}")
            if on_delivery:
                on_delivery(e, None)
            else:
                raise
    
    def flush(self, timeout: Optional[float] = None) -> int:
        """Mock flush operation."""
        return 0
    
    def poll(self, timeout: Optional[float] = None) -> int:
        """Mock poll operation."""
        return 0


class MockConsumer(KafkaConsumerProtocol):
    """Mock Kafka consumer with realistic behavior."""
    
    def __init__(
        self, 
        deserializer: DeserializerProtocol,
        group_id: str = "mock-group",
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = True,
        cluster_state: KafkaClusterState = None
    ):
        """Initialize the mock consumer."""
        self._deserializer = deserializer
        self._group_id = group_id
        self._auto_offset_reset = auto_offset_reset
        self._enable_auto_commit = enable_auto_commit
        self._cluster = cluster_state or _kafka_cluster
        
        self._subscribed_topics: List[str] = []
        self._running = False
        self._closed = False
        self._poll_interval = 0.1
    
    def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics."""
        if self._closed:
            raise RuntimeError("Cannot subscribe on a closed consumer")
        self._subscribed_topics = topics
        
        # Ensure topics exist in cluster state
        for topic in topics:
            self._cluster.get_or_create_topic(topic)
    
    def poll(self, timeout: Optional[float] = None) -> Optional[MockMessage]:
        """Poll for messages."""
        if self._closed:
            raise RuntimeError("Cannot poll on a closed consumer")
            
        if not self._subscribed_topics:
            return None
        
        # Check each subscribed topic for new messages
        for topic in self._subscribed_topics:
            partitions = self._cluster.get_or_create_topic(topic)
            
            for partition in partitions:
                current_offset = self._cluster.get_consumer_position(
                    self._group_id, topic, partition.partition
                )
                
                # Check if there are messages available at or after the current offset
                if current_offset < len(partition.messages):
                    message = partition.messages[current_offset]
                    
                    # Auto-commit if enabled
                    if self._enable_auto_commit:
                        self._cluster.commit_offset(
                            self._group_id, topic, partition.partition, current_offset + 1
                        )
                    
                    return message
                    
        # No messages found
        return None
    
    def close(self) -> None:
        """Close the consumer."""
        self._closed = True
        self._running = False
    
    def commit(self, message: Optional[Any] = None, asynchronous: bool = True) -> None:
        """Commit offsets."""
        if self._closed:
            raise RuntimeError("Cannot commit on a closed consumer")
            
        if message is None:
            # Nothing to do in this mock for a consumer-wide commit
            return
        
        # Commit the specific message's offset
        self._cluster.commit_offset(
            self._group_id,
            message.topic(),
            message.partition(),
            message.offset() + 1
        )


def create_mock_producer(serializer: SerializerProtocol = None) -> MockProducer:
    """Create a mock Kafka producer."""
    # Create a default JSON serializer if none provided
    if serializer is None:
        class DefaultSerializer(SerializerProtocol):
            def serialize(self, data: Dict[str, Any]) -> bytes:
                return json.dumps(data).encode('utf-8')
        
        serializer = DefaultSerializer()
    
    return MockProducer(serializer)


def create_mock_consumer(
    deserializer: DeserializerProtocol = None,
    group_id: str = f"mock-group-{uuid.uuid4()}"
) -> MockConsumer:
    """Create a mock Kafka consumer."""
    # Create a default JSON deserializer if none provided
    if deserializer is None:
        class DefaultDeserializer(DeserializerProtocol):
            def deserialize(self, data: bytes) -> Dict[str, Any]:
                return json.loads(data.decode('utf-8'))
        
        deserializer = DefaultDeserializer()
    
    return MockConsumer(deserializer, group_id=group_id)
