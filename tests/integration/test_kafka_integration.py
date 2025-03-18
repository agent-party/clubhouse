"""
Kafka Integration Tests.

This module provides comprehensive integration tests for Kafka functionality using real Kafka brokers.
Tests in this module require a running Kafka broker and Schema Registry as configured in docker-compose.yml.
"""

import json
import logging
import os
import pytest
import time
import uuid
from typing import Dict, Any, List, Optional

from clubhouse.services.confluent_kafka_service import (
    ConfluentKafkaService,
    KafkaConfig,
    KafkaMessage
)
from clubhouse.services.schema_registry import ConfluentSchemaRegistry
from clubhouse.schemas.events.base import EventBase
from clubhouse.schemas.events.command import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPayload(BaseModel):
    """Test payload model for Kafka messages."""
    test_id: str = Field(..., description="Unique test identifier")
    message: str = Field(..., description="Test message content")
    timestamp: int = Field(..., description="Message timestamp")


class TestEvent(EventBase):
    """Test event model for Kafka messages."""
    payload: TestPayload


class TestCommand(Command):
    """Test command model for Kafka messages."""
    payload: TestPayload


@pytest.mark.integration
class TestKafkaIntegration:
    """Integration tests with a real Kafka broker."""

    @pytest.fixture(scope="class")
    def kafka_bootstrap_servers(self) -> str:
        """Get the Kafka bootstrap servers from environment or use default."""
        return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    @pytest.fixture(scope="class")
    def schema_registry_url(self) -> str:
        """Get the Schema Registry URL from environment or use default."""
        return os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")

    @pytest.fixture(scope="class")
    def kafka_config(self, kafka_bootstrap_servers: str, schema_registry_url: str) -> KafkaConfig:
        """Create Kafka configuration."""
        config = KafkaConfig(
            bootstrap_servers=kafka_bootstrap_servers,
            client_id=f"test-client-{uuid.uuid4()}",
            group_id=f"test-group-{uuid.uuid4()}",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url=schema_registry_url
        )
        logger.info(f"Created Kafka config with bootstrap servers: {config.bootstrap_servers}")
        return config

    @pytest.fixture(scope="class")
    def schema_registry(self, schema_registry_url: str) -> ConfluentSchemaRegistry:
        """Create Schema Registry client."""
        client = ConfluentSchemaRegistry(schema_registry_url)
        
        # Verify connection to Schema Registry
        try:
            client.check_connection()
            logger.info(f"Connected to Schema Registry at {schema_registry_url}")
            return client
        except Exception as e:
            pytest.skip(f"Schema Registry not available at {schema_registry_url}: {e}")

    @pytest.fixture(scope="class")
    def kafka_service(self, kafka_config: KafkaConfig, schema_registry: ConfluentSchemaRegistry) -> ConfluentKafkaService:
        """Create Kafka service."""
        service = ConfluentKafkaService(kafka_config, schema_registry)
        logger.info("Created Kafka service")
        return service

    @pytest.fixture
    def test_topic(self) -> str:
        """Create a unique test topic name."""
        return f"test-topic-{uuid.uuid4()}"
    
    @pytest.fixture
    def avro_test_topic(self) -> str:
        """Create a unique test topic name for Avro tests."""
        return f"test-avro-topic-{uuid.uuid4()}"

    def test_produce_consume_json_message(self, kafka_service: ConfluentKafkaService, test_topic: str):
        """Test basic produce and consume functionality with JSON serialization."""
        # Arrange
        test_id = str(uuid.uuid4())
        test_message = f"Test message {test_id}"
        
        message = KafkaMessage(
            topic=test_topic,
            value={
                "test_id": test_id,
                "message": test_message,
                "timestamp": int(time.time())
            },
            key=test_id
        )
        
        received_messages = []
        
        class TestHandler:
            def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
                logger.info(f"Received message: {value}")
                received_messages.append((value, key, headers))
                if len(received_messages) >= 1:
                    kafka_service.stop_consuming()
        
        # Act
        kafka_service.produce_message(message)
        logger.info(f"Produced message to topic {test_topic}")
        
        # Create consumer and subscribe to topic
        kafka_service.consume_messages([test_topic], TestHandler(), poll_timeout=1.0)
        
        # Assert
        assert len(received_messages) == 1
        received_value, received_key, _ = received_messages[0]
        assert received_value["test_id"] == test_id
        assert received_value["message"] == test_message
        assert received_key == test_id

    def test_produce_consume_multiple_messages(self, kafka_service: ConfluentKafkaService, test_topic: str):
        """Test producing and consuming multiple messages."""
        # Arrange
        message_count = 5
        test_messages = []
        
        for i in range(message_count):
            test_id = str(uuid.uuid4())
            test_messages.append(KafkaMessage(
                topic=test_topic,
                value={
                    "test_id": test_id,
                    "message": f"Test message {i}",
                    "timestamp": int(time.time())
                },
                key=test_id
            ))
        
        received_messages = []
        
        class TestHandler:
            def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
                logger.info(f"Received message: {value}")
                received_messages.append((value, key, headers))
                if len(received_messages) >= message_count:
                    kafka_service.stop_consuming()
        
        # Act
        for message in test_messages:
            kafka_service.produce_message(message)
            logger.info(f"Produced message to topic {test_topic}")
        
        # Create consumer and subscribe to topic
        kafka_service.consume_messages([test_topic], TestHandler(), poll_timeout=1.0)
        
        # Assert
        assert len(received_messages) == message_count
        received_ids = [msg[0]["test_id"] for msg in received_messages]
        sent_ids = [msg.value["test_id"] for msg in test_messages]
        
        # Check that all sent messages were received (order may vary)
        for sent_id in sent_ids:
            assert sent_id in received_ids

    def test_avro_serialization_deserialization(self, kafka_service: ConfluentKafkaService, 
                                              schema_registry: ConfluentSchemaRegistry, 
                                              avro_test_topic: str):
        """Test Avro serialization and deserialization with Schema Registry."""
        # Skip test if Schema Registry is not available
        if not schema_registry:
            pytest.skip("Schema Registry not available")
            
        # Arrange
        test_id = str(uuid.uuid4())
        test_message = f"Test Avro message {test_id}"
        
        # Create a simplified schema to avoid unnecessary complexity
        # This ensures we're testing just the Avro serialization/deserialization
        # without getting caught up in complex schema issues
        from fastavro import schema, parse_schema
        
        # Define a simple schema directly
        simple_schema = {
            "type": "record",
            "name": "SimpleEvent",
            "namespace": "com.clubhouse.test",
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "message", "type": "string"},
                {"name": "timestamp", "type": "long"}
            ]
        }
        
        # Register the schema
        subject_name = f"{avro_test_topic}-value"
        schema_id = schema_registry.register(subject_name, simple_schema)
        logger.info(f"Registered test schema with ID: {schema_id}")
        
        # Set Avro serializer and deserializer
        kafka_service.set_avro_serializer(simple_schema)
        kafka_service.set_avro_deserializer()
        
        # Create test message 
        simple_event = {
            "id": test_id,
            "message": test_message,
            "timestamp": int(time.time())
        }
        
        message = KafkaMessage(
            topic=avro_test_topic,
            value=simple_event,
            key=test_id
        )
        
        received_messages = []
        
        class TestHandler:
            def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
                logger.info(f"Received Avro message: {value}")
                received_messages.append((value, key, headers))
                if len(received_messages) >= 1:
                    kafka_service.stop_consuming()
        
        # Act
        kafka_service.produce_message(message)
        logger.info(f"Produced Avro message to topic {avro_test_topic}")
        
        # Create consumer and subscribe to topic
        kafka_service.consume_messages([avro_test_topic], TestHandler(), poll_timeout=1.0, max_runtime=20.0)
        
        # Assert
        assert len(received_messages) == 1, "Should have received exactly one message"
        received_value, received_key, _ = received_messages[0]
        assert received_value["id"] == test_id, f"Expected id {test_id}, got {received_value.get('id')}"
        assert received_value["message"] == test_message, f"Expected message {test_message}, got {received_value.get('message')}"
        assert "timestamp" in received_value, "Timestamp field missing in received message"
        logger.info("Successfully verified Avro serialization and deserialization")

    def test_multiple_consumer_groups(self, kafka_config: KafkaConfig, schema_registry: ConfluentSchemaRegistry, test_topic: str):
        """Test that multiple consumer groups receive all messages."""
        # Arrange
        test_id = str(uuid.uuid4())
        test_message = f"Test message {test_id}"
        
        # Create two services with different consumer groups
        group1_config = KafkaConfig(
            bootstrap_servers=kafka_config.bootstrap_servers,
            client_id=f"test-client-group1-{uuid.uuid4()}",
            group_id=f"test-group1-{uuid.uuid4()}",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url=kafka_config.schema_registry_url
        )
        
        group2_config = KafkaConfig(
            bootstrap_servers=kafka_config.bootstrap_servers,
            client_id=f"test-client-group2-{uuid.uuid4()}",
            group_id=f"test-group2-{uuid.uuid4()}",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url=kafka_config.schema_registry_url
        )
        
        service1 = ConfluentKafkaService(group1_config, schema_registry)
        service2 = ConfluentKafkaService(group2_config, schema_registry)
        
        message = KafkaMessage(
            topic=test_topic,
            value={
                "test_id": test_id,
                "message": test_message,
                "timestamp": int(time.time())
            },
            key=test_id
        )
        
        group1_messages = []
        group2_messages = []
        
        class Group1Handler:
            def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
                logger.info(f"Group 1 received message: {value}")
                group1_messages.append((value, key, headers))
                if len(group1_messages) >= 1:
                    service1.stop_consuming()
        
        class Group2Handler:
            def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
                logger.info(f"Group 2 received message: {value}")
                group2_messages.append((value, key, headers))
                if len(group2_messages) >= 1:
                    service2.stop_consuming()
        
        # Act
        # Produce message first to ensure it's available to both consumers
        service1.produce_message(message)
        logger.info(f"Produced message to topic {test_topic}")
        
        # Start both consumers in separate threads
        import threading
        
        def consume_for_group1():
            service1.consume_messages([test_topic], Group1Handler(), poll_timeout=1.0, max_runtime=30.0)
            
        def consume_for_group2():
            service2.consume_messages([test_topic], Group2Handler(), poll_timeout=1.0, max_runtime=30.0)
        
        group1_thread = threading.Thread(target=consume_for_group1)
        group2_thread = threading.Thread(target=consume_for_group2)
        
        group1_thread.start()
        group2_thread.start()
        
        # Wait for consumers to receive messages
        group1_thread.join(timeout=45)
        group2_thread.join(timeout=45)
        
        # Ensure consumers are stopped
        service1.stop_consuming()
        service2.stop_consuming()
        
        # Assert both consumer groups received the message
        assert len(group1_messages) == 1
        assert len(group2_messages) == 1
        
        assert group1_messages[0][0]["test_id"] == test_id
        assert group2_messages[0][0]["test_id"] == test_id

    def test_error_handling_invalid_topic(self, kafka_service: ConfluentKafkaService):
        """Test error handling when producing to an invalid topic."""
        # Arrange
        # Use a topic name that would be rejected by the topic validation
        # Kafka doesn't allow very long topic names
        invalid_topic_name = "a" * 300  # Create an excessively long topic name (Kafka has a limit of 249 characters)
        test_id = str(uuid.uuid4())
        
        try:
            # Create a message with an invalid topic - this should fail validation
            message = KafkaMessage(
                topic=invalid_topic_name,
                value={
                    "test_id": test_id,
                    "message": "This should fail",
                    "timestamp": int(time.time())
                },
                key=test_id
            )
            
            # If we get here, the validation didn't work as expected
            # Try to produce it and expect an error
            kafka_service.produce_message(message)
            pytest.fail("Should have raised a validation error")
            
        except Exception as e:
            # We expect either a ValueError from our custom validator 
            # or a Pydantic ValidationError
            assert "too long" in str(e).lower(), f"Expected error about topic length, got: {e}"
            logger.info(f"Successfully caught error for invalid topic: {invalid_topic_name}")

    def test_idempotent_schema_registration(self, kafka_service: ConfluentKafkaService, 
                                          schema_registry: ConfluentSchemaRegistry, 
                                          avro_test_topic: str):
        """Test that registering the same schema multiple times returns the same schema ID."""
        # Skip test if Schema Registry is not available
        if not schema_registry:
            pytest.skip("Schema Registry not available")
            
        # Arrange
        subject_name = f"{avro_test_topic}-value"
        
        # Define a simple schema directly rather than using a complex Pydantic model
        # This ensures we're testing just the schema registration logic
        simple_schema = {
            "type": "record",
            "name": "SimpleRegistrationEvent",
            "namespace": "com.clubhouse.test",
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "message", "type": "string"},
                {"name": "timestamp", "type": "long"}
            ]
        }
        
        # Act - Register the same schema twice
        # The schema registry service should return the same ID for both registrations
        schema_id1 = schema_registry.register(subject_name, simple_schema)
        logger.info(f"First registration returned schema ID: {schema_id1}")
        
        schema_id2 = schema_registry.register(subject_name, simple_schema)
        logger.info(f"Second registration returned schema ID: {schema_id2}")
        
        # Assert
        assert schema_id1 == schema_id2, f"Expected same schema ID for identical schemas, got {schema_id1} and {schema_id2}"
        logger.info("Successfully verified idempotent schema registration")
