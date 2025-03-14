import os
import pytest
from typing import Dict, Any, Generator, Protocol
from unittest.mock import Mock, MagicMock

# Type definitions for better protocol definitions
class KafkaProducerProtocol(Protocol):
    """Protocol defining the Kafka producer interface."""
    
    def produce(self, topic: str, value: bytes, key: bytes = None, headers: Dict[str, str] = None) -> None:
        """Produce a message to a topic."""
        ...
    
    def flush(self, timeout: float = None) -> int:
        """Flush the producer."""
        ...


class KafkaConsumerProtocol(Protocol):
    """Protocol defining the Kafka consumer interface."""
    
    def subscribe(self, topics: list[str]) -> None:
        """Subscribe to a list of topics."""
        ...
    
    def poll(self, timeout: float = None) -> Any:
        """Poll for new messages."""
        ...
    
    def close(self) -> None:
        """Close the consumer."""
        ...


@pytest.fixture
def mock_kafka_producer() -> Generator[KafkaProducerProtocol, None, None]:
    """
    Fixture that provides a mock Kafka producer.
    
    Returns:
        A mock object implementing the KafkaProducerProtocol.
    """
    mock_producer = MagicMock(spec=KafkaProducerProtocol)
    
    # Configure the mock to return appropriate values
    mock_producer.produce.return_value = None
    mock_producer.flush.return_value = 0
    
    yield mock_producer


@pytest.fixture
def mock_kafka_consumer() -> Generator[KafkaConsumerProtocol, None, None]:
    """
    Fixture that provides a mock Kafka consumer.
    
    Returns:
        A mock object implementing the KafkaConsumerProtocol.
    """
    mock_consumer = MagicMock(spec=KafkaConsumerProtocol)
    
    # Configure the mock to return appropriate values
    mock_consumer.poll.return_value = None
    
    yield mock_consumer
    
    # Ensure the consumer is closed after the test
    mock_consumer.close.assert_called_once()


@pytest.fixture
def temp_env_vars() -> Generator[None, None, None]:
    """
    Fixture to temporarily set environment variables for tests.
    
    Restores the original environment after the test.
    """
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"
    os.environ["KAFKA_TOPIC_PREFIX"] = "test-"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Add fixture for service registry pattern
class ServiceRegistry:
    """A simple service registry for dependency injection in tests."""
    
    def __init__(self):
        self._services = {}
    
    def register(self, service_name: str, service_instance: Any) -> None:
        """Register a service in the registry."""
        self._services[service_name] = service_instance
    
    def get(self, service_name: str) -> Any:
        """Get a service from the registry."""
        return self._services.get(service_name)


@pytest.fixture
def service_registry() -> ServiceRegistry:
    """
    Fixture that provides a service registry for dependency injection.
    
    Returns:
        A ServiceRegistry instance.
    """
    return ServiceRegistry()


# Integration test fixtures
@pytest.fixture(scope="session")
def kafka_integration_config() -> Dict[str, str]:
    """
    Fixture that provides Kafka integration configuration.
    Only used in integration tests when real Kafka is available.
    
    Returns:
        A dictionary with Kafka configuration.
    """
    return {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": "test-consumer-group",
        "auto.offset.reset": "earliest",
    }