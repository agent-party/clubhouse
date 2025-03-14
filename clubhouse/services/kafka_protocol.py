"""
Kafka protocol definitions for the application.

This module defines the protocol interfaces for Kafka producers and consumers
using the Protocol class from typing to ensure proper interface contracts.
"""

from typing import Protocol, Dict, Any, Optional, List, Callable, TypeVar, Generic, Union
import json


# Type variables for generic schemas
T = TypeVar('T')
K = TypeVar('K')


class SchemaRegistryProtocol(Protocol):
    """Protocol for schema registry client operations."""
    
    def register(self, subject: str, schema: Dict[str, Any]) -> int:  # pragma: no cover
        """
        Register a new schema with the schema registry.
        
        Args:
            subject: Subject name
            schema: Schema definition
            
        Returns:
            Schema ID
        """
        ...
    
    def get_schema(self, schema_id: int) -> Dict[str, Any]:  # pragma: no cover
        """
        Get a schema by ID.
        
        Args:
            schema_id: Schema ID
            
        Returns:
            Schema definition
        """
        ...
    
    def get_latest_schema(self, subject: str) -> tuple[int, Dict[str, Any]]:  # pragma: no cover
        """
        Get the latest schema for a subject.
        
        Args:
            subject: Subject name
            
        Returns:
            Tuple of (schema_id, schema_definition)
        """
        ...


class SerializerProtocol(Protocol, Generic[T]):
    """Protocol for serializing messages."""
    
    def serialize(self, topic: str, data: T) -> bytes:  # pragma: no cover
        """
        Serialize data for a specific topic.
        
        Args:
            topic: Destination topic
            data: Data to serialize
            
        Returns:
            Serialized bytes
        """
        ...


class DeserializerProtocol(Protocol, Generic[T]):
    """Protocol for deserializing messages."""
    
    def deserialize(self, topic: str, data: bytes) -> T:  # pragma: no cover
        """
        Deserialize data from a specific topic.
        
        Args:
            topic: Source topic
            data: Data to deserialize
            
        Returns:
            Deserialized data
        """
        ...


class KafkaProducerProtocol(Protocol):
    """Protocol defining the Kafka producer interface."""
    
    def produce(
        self, 
        topic: str, 
        value: bytes, 
        key: Optional[bytes] = None, 
        headers: Optional[Dict[str, str]] = None,
        on_delivery: Optional[Callable[[Optional[Exception], Any], None]] = None
    ) -> None:  # pragma: no cover
        """
        Produce a message to a topic.
        
        Args:
            topic: Topic to produce to
            value: Message value
            key: Optional message key
            headers: Optional message headers
            on_delivery: Optional callback function to execute on delivery
        """
        ...
    
    def flush(self, timeout: Optional[float] = None) -> int:  # pragma: no cover
        """
        Flush the producer.
        
        Args:
            timeout: Maximum time to block in seconds
            
        Returns:
            Number of messages still in queue
        """
        ...
    
    def poll(self, timeout: Optional[float] = None) -> int:  # pragma: no cover
        """
        Poll the producer for events.
        
        Args:
            timeout: Maximum time to block in seconds
            
        Returns:
            Number of events processed
        """
        ...


class KafkaConsumerProtocol(Protocol):
    """Protocol defining the Kafka consumer interface."""
    
    def subscribe(self, topics: List[str]) -> None:  # pragma: no cover
        """
        Subscribe to a list of topics.
        
        Args:
            topics: List of topics to subscribe to
        """
        ...
    
    def poll(self, timeout: Optional[float] = None) -> Any:  # pragma: no cover
        """
        Poll for new messages.
        
        Args:
            timeout: Maximum time to block in seconds
            
        Returns:
            Message object or None
        """
        ...
    
    def close(self) -> None:  # pragma: no cover
        """Close the consumer."""
        ...
    
    def commit(self, message: Optional[Any] = None, asynchronous: bool = True) -> None:  # pragma: no cover
        """
        Commit offsets to Kafka.
        
        Args:
            message: Optional message to commit offsets for
            asynchronous: Whether to commit asynchronously
        """
        ...


class MessageHandlerProtocol(Protocol, Generic[T, K]):
    """Protocol for message handlers."""
    
    def handle(self, value: T, key: Optional[K] = None, headers: Optional[Dict[str, str]] = None) -> None:  # pragma: no cover
        """
        Handle a message.
        
        Args:
            value: Message value
            key: Optional message key
            headers: Optional message headers
        """
        ...
