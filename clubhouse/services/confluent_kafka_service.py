"""
Confluent Kafka service implementation.

This module provides a comprehensive implementation of Kafka services using
the Confluent Kafka client library with Schema Registry support.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from confluent_kafka import Consumer as ConfluentKafkaConsumer, KafkaError, KafkaException, Producer as ConfluentKafkaProducer
from confluent_kafka import KafkaError as KafkaError  # Explicitly import KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from pydantic import BaseModel, Field, field_validator

from clubhouse.services.kafka_protocol import (
    DeserializerProtocol,
    KafkaConsumerProtocol,
    KafkaProducerProtocol,
    MessageHandlerProtocol,
    SerializerProtocol,
)
from clubhouse.services.schema_registry import (
    ConfluentSchemaRegistry,
    SchemaRegistryProtocol,
)
from clubhouse.services.serializers import (
    AvroSchemaDeserializer,
    AvroSchemaSerializer,
    JSONDeserializer,
    JSONSerializer,
)

logger = logging.getLogger(__name__)

# Type variables for generic data types
T = TypeVar("T")
K = TypeVar("K")


class KafkaConfig(BaseModel):
    """Configuration for Kafka producers and consumers."""

    bootstrap_servers: str = Field(..., description="Kafka bootstrap servers")
    client_id: Optional[str] = Field(None, description="Client ID for Kafka")
    group_id: Optional[str] = Field(None, description="Consumer group ID")
    auto_offset_reset: str = Field("earliest", description="Auto offset reset policy")
    enable_auto_commit: bool = Field(True, description="Enable auto commit")
    schema_registry_url: Optional[str] = Field(None, description="Schema Registry URL")

    @field_validator("bootstrap_servers")
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
    headers: Optional[Dict[str, str]] = Field(
        None, description="Optional message headers"
    )

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Validate topic name."""
        if not v:
            raise ValueError("topic cannot be empty")
        
        # Kafka has a limit of 249 characters for topic names
        if len(v) > 249:
            raise ValueError("Topic name is too long (max 249 characters)")
            
        # Check for other Kafka topic name constraints
        if "." in v and ".." in v:
            raise ValueError("Topic name cannot contain '..'")
            
        # All good
        return v


class ConfluentBaseKafkaProducer(KafkaProducerProtocol):
    """
    Confluent Kafka producer implementation.

    This class provides an implementation of the KafkaProducerProtocol using
    the Confluent Kafka library.
    """

    def __init__(
        self,
        config: KafkaConfig,
        serializer: SerializerProtocol[Dict[str, Any]],
        key_serializer: Optional[SerializerProtocol[str]] = None,
        producer: Optional[ConfluentKafkaProducer] = None,
    ) -> None:
        """
        Initialize the Confluent Kafka producer.

        Args:
            config: Kafka configuration
            serializer: Serializer for message values
            key_serializer: Optional serializer for message keys
            producer: Optional pre-configured producer for testing
        """
        producer_config = {
            "bootstrap.servers": config.bootstrap_servers,
        }

        if config.client_id:
            producer_config["client.id"] = config.client_id

        self._producer = producer or ConfluentKafkaProducer(producer_config)
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
            logger.debug(
                f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

    def produce(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        on_delivery: Optional[Callable[[Optional[Exception], Any], None]] = None,
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
            kafka_headers = (
                [(k, v.encode("utf-8")) for k, v in headers.items()]
                if headers
                else None
            )

            # Use the provided callback or the default
            callback = on_delivery or self._default_delivery_callback

            # Produce the message
            self._producer.produce(
                topic=topic,
                value=serialized_value,
                key=serialized_key,
                headers=kafka_headers,
                on_delivery=callback,
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
            return self._producer.flush(timeout=timeout if timeout is not None else -1)  # type: ignore[any_return]
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
            return self._producer.poll(timeout=timeout if timeout is not None else 0)  # type: ignore[any_return]
        except KafkaException as e:  # pragma: no cover
            logger.error(f"Error polling producer: {e}")
            raise


class ConfluentBaseKafkaConsumer(KafkaConsumerProtocol):
    """
    Confluent Kafka consumer implementation.

    This class provides an implementation of the KafkaConsumerProtocol using
    the Confluent Kafka library.
    """

    def __init__(
        self,
        config: KafkaConfig,
        deserializer: DeserializerProtocol[Dict[str, Any]],
        key_deserializer: Optional[DeserializerProtocol[str]] = None,
        consumer: Optional[ConfluentKafkaConsumer] = None,
    ) -> None:
        """
        Initialize the Confluent Kafka consumer.

        Args:
            config: Kafka configuration
            deserializer: Deserializer for message values
            key_deserializer: Optional deserializer for message keys
            consumer: Optional pre-configured consumer for testing
        """
        if not config.group_id:
            raise ValueError("group_id is required for consumers")

        consumer_config = {
            "bootstrap.servers": config.bootstrap_servers,
            "group.id": config.group_id,
            "auto.offset.reset": config.auto_offset_reset,
            "enable.auto.commit": str(config.enable_auto_commit).lower(),
        }

        if config.client_id:
            consumer_config["client.id"] = config.client_id

        self._consumer = consumer or ConfluentKafkaConsumer(consumer_config)
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
        schema_registry: Optional[SchemaRegistryProtocol] = None,
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

    def get_producer(self, producer: Optional[KafkaProducerProtocol] = None) -> KafkaProducerProtocol:
        """
        Get or create a Kafka producer.

        Args:
            producer: Optional pre-configured producer for testing

        Returns:
            Kafka producer instance
        """
        if producer:
            return producer

        if not self._producer:
            # Set proper connection timeout to prevent indefinite hanging
            producer_config = {
                "bootstrap.servers": self._config.bootstrap_servers,
                "socket.timeout.ms": 10000,  # 10 second socket timeout
                "request.timeout.ms": 15000,  # 15 second request timeout
                "message.timeout.ms": 20000,  # 20 second message delivery timeout
            }

            if self._config.client_id:
                producer_config["client.id"] = self._config.client_id

            try:
                # Initialize the producer with proper timeouts
                self._producer = ConfluentBaseKafkaProducer(
                    self._config,
                    self._value_serializer,
                    producer=ConfluentKafkaProducer(producer_config)
                )
            except Exception as e:
                logger.error(f"Failed to create Kafka producer: {e}")
                raise

        return self._producer

    def get_consumer(self, consumer: Optional[KafkaConsumerProtocol] = None) -> KafkaConsumerProtocol:
        """
        Get or create a Kafka consumer.

        Args:
            consumer: Optional pre-configured consumer for testing

        Returns:
            Kafka consumer instance

        Raises:
            ValueError: If group_id is not provided in config
        """
        if consumer:
            return consumer

        if not self._consumer:
            if not self._config.group_id:
                raise ValueError("group_id is required for consumers")

            # Set proper connection timeout to prevent indefinite hanging
            consumer_config = {
                "bootstrap.servers": self._config.bootstrap_servers,
                "group.id": self._config.group_id,
                "auto.offset.reset": self._config.auto_offset_reset,
                "enable.auto.commit": str(self._config.enable_auto_commit).lower(),
                "session.timeout.ms": 10000,       # 10 second session timeout
                "max.poll.interval.ms": 30000,     # 30 second poll interval timeout
                "socket.timeout.ms": 10000,        # 10 second socket timeout
                "request.timeout.ms": 15000,       # 15 second request timeout
                "metadata.request.timeout.ms": 10000,  # 10 second metadata request timeout
            }

            if self._config.client_id:
                consumer_config["client.id"] = self._config.client_id

            try:
                # Initialize consumer with proper timeouts
                self._consumer = ConfluentBaseKafkaConsumer(
                    self._config,
                    self._value_deserializer,
                    consumer=ConfluentKafkaConsumer(consumer_config)
                )
            except Exception as e:
                logger.error(f"Failed to create Kafka consumer: {e}")
                raise

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

        try:
            client = SchemaRegistryClient({"url": self._config.schema_registry_url})
            self._value_serializer = AvroSchemaSerializer(client, schema)  # type: ignore[type_assignment]
            logger.info(f"Avro serializer configured with schema: {schema.get('name', 'unknown')}")
            
            # Reset the producer so it will be recreated with the new serializer
            self._producer = None
        except Exception as e:
            logger.error(f"Failed to configure Avro serializer: {e}")
            raise

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

        try:
            client = SchemaRegistryClient({"url": self._config.schema_registry_url})
            self._value_deserializer = AvroSchemaDeserializer(client, schema)  # type: ignore[type_assignment]
            logger.info(f"Avro deserializer configured{' with schema' if schema else ' to fetch schema from registry'}")
            
            # Reset the consumer so it will be recreated with the new deserializer
            self._consumer = None
        except Exception as e:
            logger.error(f"Failed to configure Avro deserializer: {e}")
            raise

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
            headers=message.headers,
        )
        producer.flush()

    def consume_messages(
        self,
        topics: List[str],
        handler: MessageHandlerProtocol[Dict[str, Any], str],
        poll_timeout: float = 1.0,
        max_runtime: float = 10.0,
    ) -> None:
        """
        Consume messages from Kafka topics.

        Args:
            topics: List of topics to consume from
            handler: Handler for consumed messages
            poll_timeout: Timeout for polling messages in seconds
            max_runtime: Maximum time to run consumer loop in seconds before exiting

        Raises:
            ValueError: If handler is not provided
            KafkaException: If there is an error with Kafka operations
        """
        if not handler:
            raise ValueError("Message handler must be provided")

        # Store handler reference if it supports service access
        if hasattr(handler, "service"):
            handler.service = self

        consumer = self.get_consumer()
        
        try:
            # Subscribe to topics
            consumer.subscribe(topics)
            self._running = True
            logger.info(f"Started consuming from topics: {topics}")

            # Set up absolute timeout to prevent indefinite hanging
            start_time = time.time()
            empty_poll_count = 0
            max_empty_polls = 5  # Break after multiple consecutive empty polls
            
            # Main consumer loop with proper timeouts
            while self._running:
                # Stop consuming if max_runtime exceeded
                elapsed_time = time.time() - start_time
                if elapsed_time > max_runtime:
                    logger.warning(f"Kafka consumer exceeded max runtime of {max_runtime}s, stopping")
                    break

                # Calculate remaining time for poll
                remaining_time = max(0.1, min(poll_timeout, max_runtime - elapsed_time))
                
                # Poll for messages with dynamic timeout
                msg = consumer.poll(remaining_time)

                if msg is None:
                    empty_poll_count += 1
                    logger.debug(f"No message received, poll count: {empty_poll_count}")
                    
                    # Break if we've had multiple empty polls and approaching timeout
                    if empty_poll_count >= max_empty_polls and elapsed_time > (max_runtime * 0.8):
                        logger.info(f"Breaking after {empty_poll_count} empty polls, elapsed time: {elapsed_time:.2f}s")
                        break
                    continue
                
                # Reset empty poll counter when we get a message
                empty_poll_count = 0

                # Handle any errors
                if msg.error():
                    error_code = msg.error().code()
                    
                    # End of partition is not an error, just means we've read all available messages
                    if error_code == KafkaError._PARTITION_EOF:
                        logger.debug(f"Reached end of partition for {msg.topic()}/{msg.partition()}")
                        continue
                    # Ignore timeout errors - they are expected when polling with a timeout
                    elif error_code == KafkaError._TIMED_OUT:
                        continue
                    else:
                        # For all other errors, log and decide whether to break
                        error_str = str(msg.error()).lower()
                        
                        # Check for critical errors that should stop the consumer
                        is_critical = (
                            # Check for specific error conditions using the error string
                            "unknown topic" in error_str or
                            "unknown partition" in error_str or
                            "invalid arg" in error_str or
                            "auth" in error_str or
                            "fatal" in error_str
                        )
                        
                        if is_critical:
                            logger.error(f"Critical Kafka error: {msg.error()}")
                            # Store the error to raise outside the loop
                            consumer_error = msg.error()
                            break
                        else:
                            logger.warning(f"Kafka error: {msg.error()}")
                            continue
                try:
                    # Get value
                    value = self._value_deserializer.deserialize(msg.topic(), msg.value())

                    # Get key if present
                    key = None
                    if msg.key() is not None:
                        if hasattr(consumer, '_key_deserializer'):
                            # Use consumer's key deserializer if available
                            key = consumer._key_deserializer.deserialize(msg.topic(), msg.key())
                        else:
                            # Fallback to basic UTF-8 decoding
                            key = msg.key().decode('utf-8')

                    # Get headers if present
                    headers = None
                    if msg.headers():
                        headers = {}
                        for k, v in msg.headers():
                            if isinstance(v, bytes):
                                v = v.decode('utf-8')
                            headers[k] = v

                    # Handle message
                    handler.handle(value, key, headers)
                    
                    # Commit offset for processed message if auto-commit is disabled
                    if not self._config.enable_auto_commit:
                        consumer.commit(msg)
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Continue processing other messages instead of breaking
                    # This is more resilient to individual message processing failures
            
            # Gracefully stop the consumer
            logger.info(f"Consumer loop completed after {time.time() - start_time:.2f}s")
            
        except KafkaException as e:
            logger.error(f"Kafka error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
        finally:
            # Ensure we clean up consumer resources
            self._running = False
            try:
                consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.warning(f"Error closing consumer: {e}")

    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self._running = False
        if self._consumer:
            self._consumer.close()
            self._consumer = None  # Reset consumer instance
            logger.info("Stopped consuming messages")