"""
Confluent Kafka service implementation.

This module provides a comprehensive implementation of Kafka services using
the Confluent Kafka client library with Schema Registry support.
"""

import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, cast, Type, Union
from datetime import datetime

from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.schema_registry import SchemaRegistryClient
from pydantic import BaseModel, Field, field_validator

from clubhouse.services.kafka_protocol import (
    KafkaProducerProtocol, 
    KafkaConsumerProtocol,
    MessageHandlerProtocol,
    SerializerProtocol,
    DeserializerProtocol
)
from clubhouse.services.serializers import (
    JSONSerializer, 
    JSONDeserializer,
    AvroSchemaSerializer,
    AvroSchemaDeserializer
)
from clubhouse.services.schema_registry import ConfluentSchemaRegistry, SchemaRegistryProtocol

logger = logging.getLogger(__name__)

# Type variables for generic data types
T = TypeVar('T')
K = TypeVar('K')


class KafkaConfig(BaseModel):
    """Configuration for Kafka producers and consumers."""
    
    bootstrap_servers: str = Field(..., description="Kafka bootstrap servers")
    client_id: Optional[str] = Field(None, description="Client ID for Kafka")
    group_id: Optional[str] = Field(None, description="Consumer group ID")
    auto_offset_reset: str = Field("earliest", description="Auto offset reset policy")
    enable_auto_commit: bool = Field(True, description="Enable auto commit")
    schema_registry_url: Optional[str] = Field(None, description="Schema Registry URL")
    
    @field_validator('bootstrap_servers')
    @classmethod
    def validate_bootstrap_servers(cls, v: str) -> str:
        """Validate bootstrap servers configuration."""
        if not v:
            raise ValueError("bootstrap_servers cannot be empty")
        return v


class KafkaMessage(BaseModel):
    """Base model for Kafka messages with validation."""
    
    topic: str = Field(..., description="Kafka topic")
    value: Dict[str, Any] = Field(..., description="Message value")
    key: Optional[str] = Field(None, description="Optional message key")
    headers: Optional[Dict[str, str]] = Field(None, description="Optional message headers")
    
    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Validate topic name."""
        if not v:
            raise ValueError("topic cannot be empty")
        return v


class ConfluentKafkaProducer(KafkaProducerProtocol):
    """
    Confluent Kafka producer implementation.
    
    This class provides an implementation of the KafkaProducerProtocol using
    the Confluent Kafka library.
    """
    
    def __init__(
        self, 
        config: KafkaConfig,
        serializer: SerializerProtocol[Dict[str, Any]],
        key_serializer: Optional[SerializerProtocol[str]] = None
    ) -> None:
        """
        Initialize the Confluent Kafka producer.
        
        Args:
            config: Kafka configuration
            serializer: Serializer for message values
            key_serializer: Optional serializer for message keys
        """
        producer_config = {
            'bootstrap.servers': config.bootstrap_servers,
        }
        
        if config.client_id:
            producer_config['client.id'] = config.client_id
        
        self._producer = Producer(producer_config)
        self._serializer = serializer
        self._key_serializer = key_serializer or JSONSerializer()
        self._default_delivery_callback = self._delivery_report
    
    def _delivery_report(self, err: Optional[KafkaException], msg: Any) -> None:
        """
        Delivery report callback for produced messages.
        
        Args:
            err: Error if message delivery failed
            msg: Message that was delivered
        """
        if err is not None:
            logger.error(f"Message delivery failed: {err}")  # pragma: no cover
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
    
    def produce(
        self, 
        topic: str, 
        value: Dict[str, Any], 
        key: Optional[str] = None, 
        headers: Optional[Dict[str, str]] = None,
        on_delivery: Optional[Callable[[Optional[Exception], Any], None]] = None
    ) -> None:
        """
        Produce a message to a Kafka topic.
        
        Args:
            topic: Topic to produce to
            value: Message value
            key: Optional message key
            headers: Optional message headers
            on_delivery: Optional callback function to execute on delivery
            
        Raises:
            ValueError: If serialization fails
            KafkaException: If there is an error producing the message
        """
        try:
            # Serialize the message value
            serialized_value = self._serializer.serialize(topic, value)
            
            # Serialize the message key if provided
            serialized_key = self._key_serializer.serialize(topic, key) if key else None
            
            # Convert headers to a list of tuples if provided
            kafka_headers = [(k, v.encode('utf-8')) for k, v in headers.items()] if headers else None
            
            # Use the provided callback or the default
            callback = on_delivery or self._default_delivery_callback
            
            # Produce the message
            self._producer.produce(
                topic=topic,
                value=serialized_value,
                key=serialized_key,
                headers=kafka_headers,
                on_delivery=callback
            )
            
        except (ValueError, KafkaException) as e:  # pragma: no cover
            logger.error(f"Error producing message to topic {topic}: {e}")
            raise
    
    def flush(self, timeout: Optional[float] = None) -> int:
        """
        Flush the producer.
        
        Args:
            timeout: Maximum time to block in seconds
            
        Returns:
            Number of messages still in queue
            
        Raises:
            KafkaException: If there is an error flushing the producer
        """
        try:
            return self._producer.flush(timeout=timeout if timeout is not None else -1)
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error flushing producer: {e}")
            raise
    
    def poll(self, timeout: Optional[float] = None) -> int:
        """
        Poll the producer for events.
        
        Args:
            timeout: Maximum time to block in seconds
            
        Returns:
            Number of events processed
            
        Raises:
            KafkaException: If there is an error polling the producer
        """
        try:
            return self._producer.poll(timeout=timeout if timeout is not None else 0)
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error polling producer: {e}")
            raise


class ConfluentKafkaConsumer(KafkaConsumerProtocol):
    """
    Confluent Kafka consumer implementation.
    
    This class provides an implementation of the KafkaConsumerProtocol using
    the Confluent Kafka library.
    """
    
    def __init__(
        self, 
        config: KafkaConfig,
        deserializer: DeserializerProtocol[Dict[str, Any]],
        key_deserializer: Optional[DeserializerProtocol[str]] = None
    ) -> None:
        """
        Initialize the Confluent Kafka consumer.
        
        Args:
            config: Kafka configuration
            deserializer: Deserializer for message values
            key_deserializer: Optional deserializer for message keys
        """
        if not config.group_id:
            raise ValueError("group_id is required for consumers")
        
        consumer_config = {
            'bootstrap.servers': config.bootstrap_servers,
            'group.id': config.group_id,
            'auto.offset.reset': config.auto_offset_reset,
            'enable.auto.commit': str(config.enable_auto_commit).lower()
        }
        
        if config.client_id:
            consumer_config['client.id'] = config.client_id
        
        self._consumer = Consumer(consumer_config)
        self._deserializer = deserializer
        self._key_deserializer = key_deserializer or JSONDeserializer()
        self._running = False
    
    def subscribe(self, topics: List[str]) -> None:
        """
        Subscribe to a list of topics.
        
        Args:
            topics: List of topics to subscribe to
            
        Raises:
            KafkaException: If there is an error subscribing to topics
        """
        try:
            self._consumer.subscribe(topics)
            logger.info(f"Subscribed to topics: {topics}")
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error subscribing to topics {topics}: {e}")
            raise
    
    def poll(self, timeout: Optional[float] = None) -> Any:
        """
        Poll for new messages.
        
        Args:
            timeout: Maximum time to block in seconds
            
        Returns:
            Message object or None
            
        Raises:
            KafkaException: If there is an error polling for messages
        """
        try:
            return self._consumer.poll(timeout=timeout if timeout is not None else 1.0)
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error polling for messages: {e}")
            raise
    
    def close(self) -> None:
        """
        Close the consumer.
        
        Raises:
            KafkaException: If there is an error closing the consumer
        """
        try:
            self._running = False
            self._consumer.close()
            logger.info("Consumer closed")
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error closing consumer: {e}")
            raise
    
    def commit(self, message: Optional[Any] = None, asynchronous: bool = True) -> None:
        """
        Commit offsets to Kafka.
        
        Args:
            message: Optional message to commit offsets for
            asynchronous: Whether to commit asynchronously
            
        Raises:
            KafkaException: If there is an error committing offsets
        """
        try:
            self._consumer.commit(message=message, asynchronous=asynchronous)
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error committing offsets: {e}")
            raise


class ConfluentKafkaService:
    """
    Comprehensive Kafka service using Confluent Kafka with Schema Registry.
    
    This service provides a high-level interface for producing and consuming
    Kafka messages with schema validation and serialization.
    """
    
    def __init__(
        self, 
        config: KafkaConfig,
        schema_registry: Optional[SchemaRegistryProtocol] = None
    ) -> None:
        """
        Initialize the Confluent Kafka service.
        
        Args:
            config: Kafka configuration
            schema_registry: Optional schema registry client
        """
        self._config = config
        self._schema_registry = schema_registry
        
        # Default serializers (JSON)
        self._value_serializer = JSONSerializer()
        self._value_deserializer = JSONDeserializer()
        
        # Initialize schema registry client if URL is provided
        if config.schema_registry_url and not schema_registry:
            self._schema_registry = ConfluentSchemaRegistry(config.schema_registry_url)
        
        # Producers and consumers will be created on-demand
        self._producer = None
        self._consumer = None
        self._running = False
    
    def get_producer(self) -> ConfluentKafkaProducer:
        """
        Get or create a Kafka producer.
        
        Returns:
            Kafka producer instance
        """
        if not self._producer:
            self._producer = ConfluentKafkaProducer(
                config=self._config,
                serializer=self._value_serializer
            )
        return self._producer
    
    def get_consumer(self) -> ConfluentKafkaConsumer:
        """
        Get or create a Kafka consumer.
        
        Returns:
            Kafka consumer instance
            
        Raises:
            ValueError: If group_id is not provided in config
        """
        if not self._consumer:
            self._consumer = ConfluentKafkaConsumer(
                config=self._config,
                deserializer=self._value_deserializer
            )
        return self._consumer
    
    def set_avro_serializer(self, schema: Dict[str, Any]) -> None:
        """
        Set an Avro serializer for message values.
        
        Args:
            schema: Avro schema definition
            
        Raises:
            ValueError: If schema registry URL is not provided
        """
        if not self._config.schema_registry_url:
            raise ValueError("Schema registry URL is required for Avro serialization")
        
        client = SchemaRegistryClient({'url': self._config.schema_registry_url})
        self._value_serializer = AvroSchemaSerializer(client, schema)
    
    def set_avro_deserializer(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Set an Avro deserializer for message values.
        
        Args:
            schema: Optional Avro schema definition (if None, schema will be fetched from registry)
            
        Raises:
            ValueError: If schema registry URL is not provided
        """
        if not self._config.schema_registry_url:
            raise ValueError("Schema registry URL is required for Avro deserialization")
        
        client = SchemaRegistryClient({'url': self._config.schema_registry_url})
        self._value_deserializer = AvroSchemaDeserializer(client, schema)
    
    def produce_message(self, message: KafkaMessage) -> None:
        """
        Produce a message to a Kafka topic.
        
        Args:
            message: The message to produce
            
        Raises:
            ValueError: If message validation fails
            KafkaException: If there is an error producing the message
        """
        producer = self.get_producer()
        producer.produce(
            topic=message.topic,
            value=message.value,
            key=message.key,
            headers=message.headers
        )
        producer.flush()
    
    def consume_messages(
        self, 
        topics: List[str], 
        handler: MessageHandlerProtocol[Dict[str, Any], str],
        timeout: float = 1.0
    ) -> None:
        """
        Consume messages from Kafka topics.
        
        Args:
            topics: List of topics to consume from
            handler: Handler for consumed messages
            timeout: Poll timeout in seconds
            
        Raises:
            ValueError: If handler is not provided
            KafkaException: If there is an error consuming messages
        """
        if not handler:
            raise ValueError("Message handler is required")
        
        consumer = self.get_consumer()
        consumer.subscribe(topics)
        self._running = True
        
        logger.info(f"Started consuming from topics: {topics}")
        
        try:
            while self._running:
                msg = consumer.poll(timeout)
                
                if msg is None:
                    continue
                
                if msg.error():  # pragma: no cover
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event - not an error
                        logger.debug("Reached end of partition")
                    else:
                        # Error
                        logger.error(f"Error while consuming: {msg.error()}")
                else:
                    # Process the message
                    try:
                        # Deserialize value and key
                        value = self._value_deserializer.deserialize(msg.topic(), msg.value())
                        key = msg.key().decode('utf-8') if msg.key() else None
                        
                        # Convert headers to dictionary if present
                        headers = None
                        if msg.headers():  # pragma: no cover
                            headers = {k: v.decode('utf-8') for k, v in msg.headers()}
                        
                        # Handle the message
                        handler.handle(value, key, headers)
                        
                        # Commit the offset if auto commit is disabled
                        if not self._config.enable_auto_commit:
                            consumer.commit(msg)
                            
                    except Exception as e:  # pragma: no cover
                        logger.error(f"Error processing message: {e}")
        
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Kafka error: {e}")
            raise
        finally:
            # Close the consumer
            self.stop_consuming()
    
    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self._running = False
        if self._consumer:
            self._consumer.close()
            logger.info("Stopped consuming messages")
