"""
Unit tests for the Confluent Kafka service.

This module provides comprehensive tests for the Confluent Kafka service
implementation using real infrastructure following test-driven development practices.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from confluent_kafka import Producer as ConfluentKafkaProducer
from confluent_kafka.admin import AdminClient, NewTopic, TopicMetadata

from clubhouse.services.confluent_kafka_service import (
    KafkaConfig,
    KafkaMessage,
    ConfluentKafkaService,
)
from clubhouse.services.kafka_protocol import MessageHandlerProtocol


def wait_for_kafka_ready(admin_client: AdminClient, timeout: float = 30) -> bool:
    """Wait for Kafka to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            cluster_metadata = admin_client.list_topics(timeout=2)
            if cluster_metadata is not None:
                return True
        except Exception:
            time.sleep(1)
    return False


def wait_for_topic_ready(admin_client: AdminClient, topic: str, timeout: float = 30) -> bool:
    """Wait for a topic to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            topics_metadata = admin_client.list_topics(topic=topic, timeout=2)
            if topic in topics_metadata.topics:
                return True
        except Exception:
            time.sleep(1)
    return False


class TestKafkaConfig:
    """Unit tests for the KafkaConfig class."""
    
    def test_valid_config(self) -> None:
        """Test valid Kafka configuration."""
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            client_id="test-client",
            group_id="test-group",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url="http://localhost:8081"
        )
        
        assert config.bootstrap_servers == "localhost:9092"
        assert config.client_id == "test-client"
        assert config.group_id == "test-group"
        assert config.auto_offset_reset == "earliest"
        assert config.enable_auto_commit is True
        assert config.schema_registry_url == "http://localhost:8081"
    
    def test_minimal_config(self) -> None:
        """Test minimal valid Kafka configuration."""
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        
        assert config.bootstrap_servers == "localhost:9092"
        assert config.client_id is None
        assert config.group_id is None
        assert config.auto_offset_reset == "earliest"
        assert config.enable_auto_commit is True
        assert config.schema_registry_url is None
    
    def test_invalid_bootstrap_servers(self) -> None:
        """Test that empty bootstrap servers raises ValueError."""
        with pytest.raises(ValueError):
            KafkaConfig(bootstrap_servers="")


class TestKafkaMessage:
    """Unit tests for the KafkaMessage class."""
    
    def test_valid_message(self) -> None:
        """Test valid Kafka message."""
        message = KafkaMessage(
            topic="test-topic",
            value={"key": "value"},
            key="test-key",
            headers={"header1": "value1"}
        )
        
        assert message.topic == "test-topic"
        assert message.value == {"key": "value"}
        assert message.key == "test-key"
        assert message.headers == {"header1": "value1"}
    
    def test_minimal_message(self) -> None:
        """Test minimal valid Kafka message."""
        message = KafkaMessage(
            topic="test-topic",
            value={"key": "value"}
        )
        
        assert message.topic == "test-topic"
        assert message.value == {"key": "value"}
        assert message.key is None
        assert message.headers is None
    
    def test_invalid_topic(self) -> None:
        """Test that empty topic raises ValueError."""
        with pytest.raises(ValueError):
            KafkaMessage(topic="", value={"key": "value"})


class MessageHandler:
    """Real implementation of MessageHandlerProtocol for testing."""
    
    def __init__(self) -> None:
        """Initialize the handler."""
        self.handled_messages = []
    
    def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        """Handle a message."""
        self.handled_messages.append((value, key, headers))


@pytest.fixture(scope="session")
def kafka_admin() -> AdminClient:
    """Fixture that provides a Kafka admin client."""
    admin_client = AdminClient({"bootstrap.servers": "localhost:9092"})
    assert wait_for_kafka_ready(admin_client), "Kafka not ready"
    return admin_client


@pytest.fixture(scope="session")
def kafka_config() -> KafkaConfig:
    """Fixture that provides a Kafka configuration."""
    return KafkaConfig(
        bootstrap_servers="localhost:9092",
        client_id="test-service",
        group_id="test-group",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        schema_registry_url="http://localhost:8081"
    )


@pytest.fixture(scope="session")
def kafka_service(kafka_config: KafkaConfig) -> ConfluentKafkaService:
    """Fixture that provides a Confluent Kafka service."""
    return ConfluentKafkaService(config=kafka_config)


@pytest.fixture(autouse=True)
def setup_test_topic(kafka_admin: AdminClient) -> None:
    """Setup and cleanup test topic."""
    topic = "test-topic"
    topics = [NewTopic(topic, num_partitions=1, replication_factor=1)]
    
    # Create topic
    kafka_admin.create_topics(topics)
    
    # Wait for topic to be ready
    assert wait_for_topic_ready(kafka_admin, topic), "Topic not ready"
    
    yield
    
    # Cleanup topic
    kafka_admin.delete_topics([topic])


class TestConfluentKafkaService:
    """Integration tests for the ConfluentKafkaService class."""
    
    def test_produce_and_consume_message(self, kafka_service: ConfluentKafkaService) -> None:
        """Test producing and consuming a message using real Kafka."""
        # Arrange
        message = KafkaMessage(
            topic="test-topic",
            value={"key": "value"},
            key="test-key",
            headers={"header1": "value1"}
        )
        handler = MessageHandler()
        
        # Act - Produce message
        kafka_service.produce_message(message)
        
        # Add a small delay to ensure message is available
        time.sleep(1)
        
        # Act - Consume message
        kafka_service.consume_messages(
            topics=["test-topic"],
            handler=handler,
            poll_timeout=1.0,  # Increased timeout
            max_runtime=5.0    # Increased runtime
        )
        
        # Assert
        assert len(handler.handled_messages) == 1
        value, key, headers = handler.handled_messages[0]
        assert value == {"key": "value"}
        assert key == "test-key"
        assert headers == {"header1": "value1"}
    
    def test_avro_serialization(self, kafka_service: ConfluentKafkaService) -> None:
        """Test Avro serialization with Schema Registry."""
        # Define Avro schema
        schema = {
            "type": "record",
            "name": "test",
            "fields": [
                {"name": "field1", "type": "string"}
            ]
        }
        
        # Set up Avro serialization
        kafka_service.set_avro_serializer(schema)
        kafka_service.set_avro_deserializer(schema)
        
        # Create and send message
        message = KafkaMessage(
            topic="test-topic",
            value={"field1": "test value"},
            key="test-key"
        )
        
        handler = MessageHandler()
        
        # Act - Produce message
        kafka_service.produce_message(message)
        
        # Add a small delay to ensure message is available
        time.sleep(1)
        
        # Act - Consume message
        kafka_service.consume_messages(
            topics=["test-topic"],
            handler=handler,
            poll_timeout=1.0,  # Increased timeout
            max_runtime=5.0    # Increased runtime
        )
        
        # Assert
        assert len(handler.handled_messages) == 1
        value, key, _ = handler.handled_messages[0]
        assert value == {"field1": "test value"}
        assert key == "test-key"
