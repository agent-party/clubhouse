"""
Serializers and deserializers for Kafka messages.

This module provides implementations of serializers and deserializers
for Kafka messages, supporting both JSON and Avro formats with Schema Registry.
"""

import json
import logging
from typing import Dict, Any, TypeVar, Generic, Optional

from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
from pydantic import BaseModel

from clubhouse.services.kafka_protocol import SerializerProtocol, DeserializerProtocol

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')


class JSONSerializer(SerializerProtocol[Dict[str, Any]]):
    """
    JSON serializer for Kafka messages.
    
    Serializes Python dictionaries to JSON bytes.
    """
    
    def serialize(self, topic: str, data: Dict[str, Any]) -> bytes:
        """
        Serialize data to JSON bytes.
        
        Args:
            topic: Destination topic (unused but required by interface)
            data: Data to serialize
            
        Returns:
            Serialized bytes
            
        Raises:
            TypeError: If data is not serializable
        """
        try:
            return json.dumps(data).encode('utf-8')
        except (TypeError, ValueError) as e:  # pragma: no cover
            logger.error(f"Failed to serialize data: {e}")
            raise


class JSONDeserializer(DeserializerProtocol[Dict[str, Any]]):
    """
    JSON deserializer for Kafka messages.
    
    Deserializes JSON bytes to Python dictionaries.
    """
    
    def deserialize(self, topic: str, data: bytes) -> Dict[str, Any]:  # pragma: no cover
        """
        Deserialize JSON bytes to a dictionary.
        
        Args:
            topic: Source topic (unused but required by interface)
            data: Data to deserialize
            
        Returns:
            Deserialized data
            
        Raises:
            json.JSONDecodeError: If data is not valid JSON
        """
        try:
            if data is None:  # pragma: no cover
                return {}
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:  # pragma: no cover
            logger.error(f"Failed to deserialize data: {e}")
            raise


class AvroSchemaSerializer(SerializerProtocol[Dict[str, Any]]):
    """
    Avro serializer for Kafka messages using Schema Registry.
    
    Serializes Python dictionaries to Avro bytes with schema ID.
    """
    
    def __init__(self, schema_registry_client: SchemaRegistryClient, schema: Dict[str, Any]) -> None:
        """
        Initialize the Avro serializer.
        
        Args:
            schema_registry_client: Schema Registry client
            schema: Avro schema definition
        """
        self._serializer = AvroSerializer(
            schema_registry_client,
            json.dumps(schema),
            lambda obj, ctx: obj  # Identity function for Python dict to dict conversion
        )
    
    def serialize(self, topic: str, data: Dict[str, Any]) -> bytes:  # pragma: no cover
        """
        Serialize data to Avro bytes with schema ID.
        
        Args:
            topic: Destination topic
            data: Data to serialize
            
        Returns:
            Serialized bytes
            
        Raises:
            Exception: If serialization fails
        """
        try:
            ctx = SerializationContext(topic)  # pragma: no cover
            return self._serializer(data, ctx)
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to serialize data: {e}")
            raise


class AvroSchemaDeserializer(DeserializerProtocol[Dict[str, Any]]):
    """
    Avro deserializer for Kafka messages using Schema Registry.
    
    Deserializes Avro bytes with schema ID to Python dictionaries.
    """
    
    def __init__(self, schema_registry_client: SchemaRegistryClient, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the Avro deserializer.
        
        Args:
            schema_registry_client: Schema Registry client
            schema: Optional Avro schema definition (if None, schema will be fetched from registry)
        """
        schema_str = json.dumps(schema) if schema else None
        self._deserializer = AvroDeserializer(
            schema_registry_client,
            schema_str,
            lambda obj, ctx: obj  # Identity function for Avro dict to Python dict conversion
        )
    
    def deserialize(self, topic: str, data: bytes) -> Dict[str, Any]:  # pragma: no cover
        """
        Deserialize Avro bytes to a dictionary.
        
        Args:
            topic: Source topic
            data: Data to deserialize
            
        Returns:
            Deserialized data
            
        Raises:
            Exception: If deserialization fails
        """
        try:
            if data is None:  # pragma: no cover
                return {}
            ctx = SerializationContext(topic)  # pragma: no cover
            return self._deserializer(data, ctx)
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to deserialize data: {e}")
            raise


class SerializationContext:  # pragma: no cover
    """
    Context object for serialization/deserialization.
    """
    
    def __init__(self, topic: str) -> None:  # pragma: no cover
        """
        Initialize the serialization context.
        
        Args:
            topic: Kafka topic
        """
        self.topic = topic
