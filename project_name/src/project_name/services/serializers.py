"""
Serializers and deserializers for Kafka messages.

This module provides implementations of serializers and deserializers
for Kafka messages, supporting both JSON and Avro formats with Schema Registry.
"""

import json
import logging
import struct
from typing import Dict, Any, Optional, TypeVar, Generic, cast, Type, Union

import fastavro
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer

from project_name.services.kafka_protocol import SerializerProtocol, DeserializerProtocol

logger = logging.getLogger(__name__)

# Type variables for generic data types
T = TypeVar('T')
K = TypeVar('K')

# Magic byte for Confluent Wire Format
MAGIC_BYTE = 0


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
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data: {e}")
            raise


class JSONDeserializer(DeserializerProtocol[Dict[str, Any]]):
    """
    JSON deserializer for Kafka messages.
    
    Deserializes JSON bytes to Python dictionaries.
    """
    
    def deserialize(self, topic: str, data: bytes) -> Dict[str, Any]:
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
            if data is None:
                return {}
            return json.loads(data.decode('utf-8'))
        except json.JSONDecodeError as e:
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
    
    def serialize(self, topic: str, data: Dict[str, Any]) -> bytes:
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
            return self._serializer(data, SerializationContext(topic))
        except Exception as e:
            logger.error(f"Failed to serialize data for topic {topic}: {e}")
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
    
    def deserialize(self, topic: str, data: bytes) -> Dict[str, Any]:
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
            if data is None:
                return {}
            return self._deserializer(data, SerializationContext(topic))
        except Exception as e:
            logger.error(f"Failed to deserialize data from topic {topic}: {e}")
            raise


class SerializationContext:
    """Context object for serialization/deserialization."""
    
    def __init__(self, topic: str) -> None:
        """
        Initialize the serialization context.
        
        Args:
            topic: Kafka topic
        """
        self.topic = topic
