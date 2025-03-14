"""
Unit tests for the Confluent Kafka service.

This module provides comprehensive tests for the Confluent Kafka service
implementation following test-driven development practices.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call, ANY
from typing import Dict, Any, Optional, List, Callable, cast

from clubhouse.services.confluent_kafka_service import (
    KafkaConfig,
    KafkaMessage,
    ConfluentKafkaProducer,
    ConfluentKafkaConsumer,
    ConfluentKafkaService
)
from clubhouse.services.kafka_protocol import MessageHandlerProtocol


class TestKafkaConfig:
    """Unit tests for the KafkaConfig class."""
    
    def test_valid_config(self) -> None:
        """Test valid Kafka configuration."""
        # Arrange & Act
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            client_id="test-client",
            group_id="test-group",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url="http://localhost:8081"
        )
        
        # Assert
        assert config.bootstrap_servers == "localhost:9092"
        assert config.client_id == "test-client"
        assert config.group_id == "test-group"
        assert config.auto_offset_reset == "earliest"
        assert config.enable_auto_commit is True
        assert config.schema_registry_url == "http://localhost:8081"
    
    def test_minimal_config(self) -> None:
        """Test minimal valid Kafka configuration."""
        # Arrange & Act
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        
        # Assert
        assert config.bootstrap_servers == "localhost:9092"
        assert config.client_id is None
        assert config.group_id is None
        assert config.auto_offset_reset == "earliest"
        assert config.enable_auto_commit is True
        assert config.schema_registry_url is None
    
    def test_invalid_bootstrap_servers(self) -> None:
        """Test that empty bootstrap servers raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            KafkaConfig(bootstrap_servers="")


class TestKafkaMessage:
    """Unit tests for the KafkaMessage class."""
    
    def test_valid_message(self) -> None:
        """Test valid Kafka message."""
        # Arrange & Act
        message = KafkaMessage(
            topic="test-topic",
            value={"key": "value"},
            key="test-key",
            headers={"header1": "value1"}
        )
        
        # Assert
        assert message.topic == "test-topic"
        assert message.value == {"key": "value"}
        assert message.key == "test-key"
        assert message.headers == {"header1": "value1"}
    
    def test_minimal_message(self) -> None:
        """Test minimal valid Kafka message."""
        # Arrange & Act
        message = KafkaMessage(
            topic="test-topic",
            value={"key": "value"}
        )
        
        # Assert
        assert message.topic == "test-topic"
        assert message.value == {"key": "value"}
        assert message.key is None
        assert message.headers is None
    
    def test_invalid_topic(self) -> None:
        """Test that empty topic raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            KafkaMessage(topic="", value={"key": "value"})


class MockMessageHandler(MessageHandlerProtocol[Dict[str, Any], str]):
    """Mock implementation of MessageHandlerProtocol for testing."""
    
    def __init__(self) -> None:
        """Initialize the mock handler."""
        self.handled_messages = []
    
    def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Handle a message.
        
        Args:
            value: Message value
            key: Optional message key
            headers: Optional message headers
        """
        self.handled_messages.append((value, key, headers))


class TestConfluentKafkaProducer:
    """Unit tests for the ConfluentKafkaProducer class."""
    
    @pytest.fixture
    def config(self) -> KafkaConfig:
        """Fixture that provides a Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers="localhost:9092",
            client_id="test-producer"
        )
    
    @pytest.fixture
    def mock_producer(self) -> MagicMock:
        """Fixture that provides a mock Kafka producer."""
        with patch("clubhouse.services.confluent_kafka_service.Producer") as mock:
            producer_instance = MagicMock()
            mock.return_value = producer_instance
            yield producer_instance
    
    @pytest.fixture
    def mock_serializer(self) -> MagicMock:
        """Fixture that provides a mock serializer."""
        serializer = MagicMock()
        serializer.serialize.return_value = b'{"key": "value"}'
        return serializer
    
    @pytest.fixture
    def kafka_producer(self, config: KafkaConfig, mock_producer: MagicMock, mock_serializer: MagicMock) -> ConfluentKafkaProducer:
        """Fixture that provides a Confluent Kafka producer with mocked dependencies."""
        return ConfluentKafkaProducer(
            config=config,
            serializer=mock_serializer,
            key_serializer=mock_serializer
        )
    
    def test_init(self, config: KafkaConfig, mock_producer: MagicMock, mock_serializer: MagicMock) -> None:
        """Test the initialization of ConfluentKafkaProducer."""
        # Arrange & Act
        producer = ConfluentKafkaProducer(
            config=config,
            serializer=mock_serializer
        )
        
        # Assert
        assert producer._producer == mock_producer
        assert producer._serializer == mock_serializer
    
    def test_produce(self, kafka_producer: ConfluentKafkaProducer, mock_producer: MagicMock, mock_serializer: MagicMock) -> None:
        """Test producing a message."""
        # Arrange
        topic = "test-topic"
        value = {"key": "value"}
        key = "test-key"
        headers = {"header1": "value1"}
        
        # Act
        kafka_producer.produce(topic, value, key, headers)
        
        # Assert
        mock_serializer.serialize.assert_any_call(topic, value)
        mock_serializer.serialize.assert_any_call(topic, key)
        mock_producer.produce.assert_called_once_with(
            topic=topic,
            value=b'{"key": "value"}',
            key=b'{"key": "value"}',
            headers=[("header1", b"value1")],
            on_delivery=ANY
        )
    
    def test_flush(self, kafka_producer: ConfluentKafkaProducer, mock_producer: MagicMock) -> None:
        """Test flushing the producer."""
        # Arrange
        mock_producer.flush.return_value = 0
        
        # Act
        result = kafka_producer.flush(1.0)
        
        # Assert
        assert result == 0
        mock_producer.flush.assert_called_once_with(timeout=1.0)
    
    def test_poll(self, kafka_producer: ConfluentKafkaProducer, mock_producer: MagicMock) -> None:
        """Test polling the producer."""
        # Arrange
        mock_producer.poll.return_value = 1
        
        # Act
        result = kafka_producer.poll(1.0)
        
        # Assert
        assert result == 1
        mock_producer.poll.assert_called_once_with(timeout=1.0)


class TestConfluentKafkaConsumer:
    """Unit tests for the ConfluentKafkaConsumer class."""
    
    @pytest.fixture
    def config(self) -> KafkaConfig:
        """Fixture that provides a Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers="localhost:9092",
            client_id="test-consumer",
            group_id="test-group"
        )
    
    @pytest.fixture
    def mock_consumer(self) -> MagicMock:
        """Fixture that provides a mock Kafka consumer."""
        with patch("clubhouse.services.confluent_kafka_service.Consumer") as mock:
            consumer_instance = MagicMock()
            mock.return_value = consumer_instance
            yield consumer_instance
    
    @pytest.fixture
    def mock_deserializer(self) -> MagicMock:
        """Fixture that provides a mock deserializer."""
        deserializer = MagicMock()
        deserializer.deserialize.return_value = {"key": "value"}
        return deserializer
    
    @pytest.fixture
    def kafka_consumer(self, config: KafkaConfig, mock_consumer: MagicMock, mock_deserializer: MagicMock) -> ConfluentKafkaConsumer:
        """Fixture that provides a Confluent Kafka consumer with mocked dependencies."""
        return ConfluentKafkaConsumer(
            config=config,
            deserializer=mock_deserializer,
            key_deserializer=mock_deserializer
        )
    
    def test_init(self, config: KafkaConfig, mock_consumer: MagicMock, mock_deserializer: MagicMock) -> None:
        """Test the initialization of ConfluentKafkaConsumer."""
        # Arrange & Act
        consumer = ConfluentKafkaConsumer(
            config=config,
            deserializer=mock_deserializer
        )
        
        # Assert
        assert consumer._consumer == mock_consumer
        assert consumer._deserializer == mock_deserializer
    
    def test_init_without_group_id(self, mock_deserializer: MagicMock) -> None:
        """Test that initialization without group_id raises ValueError."""
        # Arrange
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        
        # Act & Assert
        with pytest.raises(ValueError):
            ConfluentKafkaConsumer(config=config, deserializer=mock_deserializer)
    
    def test_subscribe(self, kafka_consumer: ConfluentKafkaConsumer, mock_consumer: MagicMock) -> None:
        """Test subscribing to topics."""
        # Arrange
        topics = ["test-topic"]
        
        # Act
        kafka_consumer.subscribe(topics)
        
        # Assert
        mock_consumer.subscribe.assert_called_once_with(topics)
    
    def test_poll(self, kafka_consumer: ConfluentKafkaConsumer, mock_consumer: MagicMock) -> None:
        """Test polling for messages."""
        # Arrange
        mock_message = MagicMock()
        mock_consumer.poll.return_value = mock_message
        
        # Act
        result = kafka_consumer.poll(1.0)
        
        # Assert
        assert result == mock_message
        mock_consumer.poll.assert_called_once_with(timeout=1.0)
    
    def test_close(self, kafka_consumer: ConfluentKafkaConsumer, mock_consumer: MagicMock) -> None:
        """Test closing the consumer."""
        # Arrange & Act
        kafka_consumer.close()
        
        # Assert
        assert not kafka_consumer._running
        mock_consumer.close.assert_called_once()
    
    def test_commit(self, kafka_consumer: ConfluentKafkaConsumer, mock_consumer: MagicMock) -> None:
        """Test committing offsets."""
        # Arrange
        mock_message = MagicMock()
        
        # Act
        kafka_consumer.commit(mock_message, False)
        
        # Assert
        mock_consumer.commit.assert_called_once_with(message=mock_message, asynchronous=False)


class TestConfluentKafkaService:
    """Unit tests for the ConfluentKafkaService class."""
    
    @pytest.fixture
    def config(self) -> KafkaConfig:
        """Fixture that provides a Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers="localhost:9092",
            client_id="test-client",
            group_id="test-group",
            schema_registry_url="http://localhost:8081"
        )
    
    @pytest.fixture
    def mock_schema_registry(self) -> MagicMock:
        """Fixture that provides a mock schema registry."""
        return MagicMock()
    
    @pytest.fixture
    def mock_producer(self) -> MagicMock:
        """Fixture that provides a mock Kafka producer."""
        return MagicMock()
    
    @pytest.fixture
    def mock_consumer(self) -> MagicMock:
        """Fixture that provides a mock Kafka consumer."""
        return MagicMock()
    
    @pytest.fixture
    def kafka_service(
        self, 
        config: KafkaConfig, 
        mock_schema_registry: MagicMock,
        mock_producer: MagicMock,
        mock_consumer: MagicMock
    ) -> ConfluentKafkaService:
        """Fixture that provides a Confluent Kafka service with mocked dependencies."""
        service = ConfluentKafkaService(config=config, schema_registry=mock_schema_registry)
        service._producer = mock_producer
        service._consumer = mock_consumer
        return service
    
    def test_init(self, config: KafkaConfig, mock_schema_registry: MagicMock) -> None:
        """Test the initialization of ConfluentKafkaService."""
        # Arrange & Act
        service = ConfluentKafkaService(config=config, schema_registry=mock_schema_registry)
        
        # Assert
        assert service._config == config
        assert service._schema_registry == mock_schema_registry
        assert service._producer is None
        assert service._consumer is None
        assert not service._running
    
    def test_init_with_schema_registry_url(self) -> None:
        """Test initialization with schema registry URL."""
        # Arrange
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            schema_registry_url="http://localhost:8081"
        )
        
        # Act
        with patch("clubhouse.services.confluent_kafka_service.ConfluentSchemaRegistry") as mock:
            service = ConfluentKafkaService(config=config)
            
            # Assert
            mock.assert_called_once_with("http://localhost:8081")
    
    def test_get_producer(self) -> None:
        """Test getting a producer."""
        # Arrange
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        
        # Act
        with patch("clubhouse.services.confluent_kafka_service.ConfluentKafkaProducer") as mock:
            producer_instance = MagicMock()
            mock.return_value = producer_instance
            
            service = ConfluentKafkaService(config=config)
            producer = service.get_producer()
            
            # Assert
            assert producer == producer_instance
            mock.assert_called_once()
    
    def test_get_consumer(self) -> None:
        """Test getting a consumer."""
        # Arrange
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group"
        )
        
        # Act
        with patch("clubhouse.services.confluent_kafka_service.ConfluentKafkaConsumer") as mock:
            consumer_instance = MagicMock()
            mock.return_value = consumer_instance
            
            service = ConfluentKafkaService(config=config)
            consumer = service.get_consumer()
            
            # Assert
            assert consumer == consumer_instance
            mock.assert_called_once()
    
    def test_set_avro_serializer(self) -> None:
        """Test setting an Avro serializer."""
        # Arrange
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            schema_registry_url="http://localhost:8081"
        )
        schema = {"type": "record", "name": "test", "fields": [{"name": "field1", "type": "string"}]}
        
        # Act
        with patch("clubhouse.services.confluent_kafka_service.SchemaRegistryClient") as client_mock:
            with patch("clubhouse.services.confluent_kafka_service.AvroSchemaSerializer") as serializer_mock:
                service = ConfluentKafkaService(config=config)
                service.set_avro_serializer(schema)
                
                # Assert
                client_mock.assert_called_once_with({"url": "http://localhost:8081"})
                serializer_mock.assert_called_once()
    
    def test_set_avro_serializer_without_url(self) -> None:
        """Test that setting an Avro serializer without schema registry URL raises ValueError."""
        # Arrange
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        schema = {"type": "record", "name": "test", "fields": [{"name": "field1", "type": "string"}]}
        service = ConfluentKafkaService(config=config)
        
        # Act & Assert
        with pytest.raises(ValueError):
            service.set_avro_serializer(schema)
    
    def test_set_avro_deserializer(self) -> None:
        """Test setting an Avro deserializer."""
        # Arrange
        config = KafkaConfig(
            bootstrap_servers="localhost:9092",
            schema_registry_url="http://localhost:8081"
        )
        schema = {"type": "record", "name": "test", "fields": [{"name": "field1", "type": "string"}]}
        
        # Act
        with patch("clubhouse.services.confluent_kafka_service.SchemaRegistryClient") as client_mock:
            with patch("clubhouse.services.confluent_kafka_service.AvroSchemaDeserializer") as deserializer_mock:
                service = ConfluentKafkaService(config=config)
                service.set_avro_deserializer(schema)
                
                # Assert
                client_mock.assert_called_once_with({"url": "http://localhost:8081"})
                deserializer_mock.assert_called_once()
    
    def test_set_avro_deserializer_without_url(self) -> None:
        """Test that setting an Avro deserializer without schema registry URL raises ValueError."""
        # Arrange
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        schema = {"type": "record", "name": "test", "fields": [{"name": "field1", "type": "string"}]}
        service = ConfluentKafkaService(config=config)
        
        # Act & Assert
        with pytest.raises(ValueError):
            service.set_avro_deserializer(schema)
    
    def test_produce_message(self, kafka_service: ConfluentKafkaService, mock_producer: MagicMock) -> None:
        """Test producing a message."""
        # Arrange
        message = KafkaMessage(
            topic="test-topic",
            value={"key": "value"},
            key="test-key",
            headers={"header1": "value1"}
        )
        
        # Act
        kafka_service.produce_message(message)
        
        # Assert
        mock_producer.produce.assert_called_once_with(
            topic="test-topic",
            value={"key": "value"},
            key="test-key",
            headers={"header1": "value1"}
        )
        mock_producer.flush.assert_called_once()
    
    def test_consume_messages(
        self, 
        kafka_service: ConfluentKafkaService, 
        mock_consumer: MagicMock
    ) -> None:
        """Test consuming messages."""
        # Arrange
        topics = ["test-topic"]
        handler = MockMessageHandler()
        
        # Configure the mock to return a message on the first poll and None on the second to exit the loop
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.topic.return_value = "test-topic"
        mock_message.value.return_value = b'{"key": "value"}'
        mock_message.key.return_value = b"test-key"
        mock_message.headers.return_value = [("header1", b"value1")]
        
        mock_consumer.poll.side_effect = [mock_message, None]
        
        # Setup the service to exit after processing one message
        def set_running_to_false(*args, **kwargs):
            kafka_service._running = False
        
        mock_consumer.poll.side_effect = [mock_message, None]
        
        # Act
        with patch.object(kafka_service, "_value_deserializer") as mock_deserializer:
            mock_deserializer.deserialize.return_value = {"key": "value"}
            
            # Force the service to exit after one message
            kafka_service._running = True
            mock_consumer.poll.side_effect = lambda timeout: (
                mock_message if kafka_service._running else None
            )
            
            # After handling the message, stop the service
            original_handle = handler.handle
            def handle_and_stop(*args, **kwargs):
                result = original_handle(*args, **kwargs)
                kafka_service._running = False
                return result
            
            handler.handle = handle_and_stop
            
            kafka_service.consume_messages(topics, handler)
        
        # Assert
        mock_consumer.subscribe.assert_called_once_with(topics)
        assert len(handler.handled_messages) == 1
        assert handler.handled_messages[0][0] == {"key": "value"}
        assert handler.handled_messages[0][1] == "test-key"
        assert handler.handled_messages[0][2] == {"header1": "value1"}
    
    def test_stop_consuming(self, kafka_service: ConfluentKafkaService, mock_consumer: MagicMock) -> None:
        """Test stopping message consumption."""
        # Arrange
        kafka_service._running = True
        
        # Act
        kafka_service.stop_consuming()
        
        # Assert
        assert not kafka_service._running
        mock_consumer.close.assert_called_once()
