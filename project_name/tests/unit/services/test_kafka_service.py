import json
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, Optional, List

from project_name.services.kafka_service import KafkaService, KafkaMessage


class TestKafkaService:
    """Unit tests for the KafkaService class."""
    
    @pytest.fixture
    def mock_producer(self):
        """Fixture that provides a mock Kafka producer."""
        with patch("project_name.services.kafka_service.Producer") as mock:
            producer_instance = MagicMock()
            mock.return_value = producer_instance
            yield producer_instance
    
    @pytest.fixture
    def mock_consumer(self):
        """Fixture that provides a mock Kafka consumer."""
        with patch("project_name.services.kafka_service.Consumer") as mock:
            consumer_instance = MagicMock()
            mock.return_value = consumer_instance
            yield consumer_instance
    
    @pytest.fixture
    def kafka_service(self, mock_producer, mock_consumer):
        """Fixture that provides a KafkaService with mocked dependencies."""
        producer_config = {"bootstrap.servers": "localhost:9092"}
        consumer_config = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "test-group",
            "auto.offset.reset": "earliest"
        }
        return KafkaService(producer_config, consumer_config)
    
    def test_init(self, mock_producer, mock_consumer):
        """Test the initialization of KafkaService."""
        # Arrange
        producer_config = {"bootstrap.servers": "localhost:9092"}
        consumer_config = {
            "bootstrap.servers": "localhost:9092",
            "group.id": "test-group"
        }
        
        # Act
        service = KafkaService(producer_config, consumer_config)
        
        # Assert
        assert service._producer == mock_producer
        assert service._consumer == mock_consumer
        assert service._running is False
    
    def test_produce_message(self, kafka_service, mock_producer):
        """Test producing a message to Kafka."""
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
        mock_producer.produce.assert_called_once()
        call_args = mock_producer.produce.call_args[1]
        assert call_args["topic"] == "test-topic"
        assert json.loads(call_args["value"].decode("utf-8")) == {"key": "value"}
        assert call_args["key"] == b"test-key"
        assert call_args["headers"] == {"header1": "value1"}
        mock_producer.flush.assert_called_once()
    
    def test_consume_messages(self, kafka_service, mock_consumer):
        """Test consuming messages from Kafka."""
        # Arrange
        topics = ["test-topic"]
        handler = MagicMock()
        
        # Configure mock to return None after first poll to exit the loop
        mock_consumer.poll.side_effect = [
            MagicMock(
                error=lambda: None,
                value=lambda: json.dumps({"key": "value"}).encode("utf-8"),
                key=lambda: b"test-key"
            ),
            None
        ]
        kafka_service._running = True
        
        # Stop consuming after first message
        def set_running_to_false(*args, **kwargs):
            kafka_service._running = False
            return None
            
        handler.side_effect = set_running_to_false
        
        # Act
        kafka_service.consume_messages(topics, handler)
        
        # Assert
        mock_consumer.subscribe.assert_called_once_with(topics)
        assert mock_consumer.poll.call_count == 2
        handler.assert_called_once_with({"key": "value"}, "test-key")
        mock_consumer.close.assert_called_once()
    
    def test_stop_consuming(self, kafka_service, mock_consumer):
        """Test stopping message consumption."""
        # Arrange
        kafka_service._running = True
        
        # Act
        kafka_service.stop_consuming()
        
        # Assert
        assert kafka_service._running is False
        mock_consumer.close.assert_called_once()
