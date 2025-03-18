import json
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, Optional, List

from clubhouse.services.kafka_service import KafkaService, KafkaMessage
from clubhouse.services.kafka_protocol import KafkaProducerProtocol, KafkaConsumerProtocol


class TestKafkaService:
    """Unit tests for the KafkaService class."""
    
    @pytest.fixture
    def mock_producer(self):
        """Fixture that provides a mock Kafka producer."""
        producer = MagicMock(spec=KafkaProducerProtocol)
        return producer
    
    @pytest.fixture
    def mock_consumer(self):
        """Fixture that provides a mock Kafka consumer."""
        consumer = MagicMock(spec=KafkaConsumerProtocol)
        return consumer
    
    @pytest.fixture
    def kafka_service(self, mock_producer, mock_consumer):
        """Fixture that provides a KafkaService with mocked dependencies."""
        return KafkaService(mock_producer, mock_consumer)
    
    def test_init(self, mock_producer, mock_consumer):
        """Test the initialization of KafkaService."""
        # Act
        service = KafkaService(mock_producer, mock_consumer)
        
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
        args, kwargs = mock_producer.produce.call_args
        assert kwargs["topic"] == "test-topic"
        assert kwargs["value"] == json.dumps({"key": "value"}).encode("utf-8")
        assert kwargs["key"] == b"test-key"
        assert kwargs["headers"] == {"header1": "value1"}
        mock_producer.flush.assert_called_once()
    
    def test_consume_messages(self, kafka_service, mock_consumer):
        """Test consuming messages from Kafka."""
        # Arrange
        topics = ["test-topic"]
        handler = MagicMock()
        
        # Configure mock message
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.value.return_value = json.dumps({"key": "value"}).encode("utf-8")
        mock_message.key.return_value = b"test-key"
        mock_message.headers.return_value = None
        
        # Configure consumer to return the message once
        mock_consumer.poll.return_value = mock_message
        
        # Stop consuming after first message
        def stop_consuming(*args, **kwargs):
            kafka_service._running = False
        
        handler.side_effect = stop_consuming
        
        # Act
        kafka_service.consume_messages(topics, handler)
        
        # Assert
        mock_consumer.subscribe.assert_called_once_with(topics)
        assert mock_consumer.poll.call_count == 1
        handler.assert_called_once_with({"key": "value"}, "test-key", None)
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
