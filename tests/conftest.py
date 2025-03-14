import os
import pytest
from typing import Dict, Any, Generator, Protocol, List
from unittest.mock import Mock, MagicMock

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.services.kafka_protocol import KafkaProducerProtocol, KafkaConsumerProtocol
from clubhouse.services.mcp_protocol import MCPServerProtocol, MCPIntegrationProtocol


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


# Mock MCP classes for testing
class MockFastMCP(MCPServerProtocol):
    """Mock implementation of FastMCP for testing."""
    
    def __init__(self):
        self.tools = []
        self.resources = []
        
    def tool(self, *args, **kwargs):
        """Mock tool decorator."""
        def decorator(func):
            self.tools.append((func, args, kwargs))
            return func
        return decorator
        
    def resource(self, uri_template, *args, **kwargs):
        """Mock resource decorator."""
        def decorator(func):
            self.resources.append((uri_template, func, args, kwargs))
            return func
        return decorator
        
    def middleware(self, func):
        """Mock middleware decorator."""
        return func
        
    async def start(self, host="127.0.0.1", port=8000):
        """Mock start method."""
        pass
        
    async def stop(self):
        """Mock stop method."""
        pass


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


@pytest.fixture
def service_registry() -> ServiceRegistry:
    """
    Fixture that provides a service registry for dependency injection.
    
    Returns:
        A ServiceRegistry instance.
    """
    return ServiceRegistry()


# MCP test fixtures

@pytest.fixture
def mock_mcp_server() -> MockFastMCP:
    """
    Fixture that provides a mock MCP server for testing integration.
    
    Returns:
        A mock FastMCP server instance.
    """
    return MockFastMCP()


@pytest.fixture
def mock_mcp_integrated_service() -> MCPIntegrationProtocol:
    """
    Fixture that provides a mock service implementing MCPIntegrationProtocol.
    
    Returns:
        A mock service that implements MCPIntegrationProtocol.
    """
    service = Mock(spec=MCPIntegrationProtocol)
    service.register_with_mcp = Mock()
    return service


@pytest.fixture
def mcp_service_registry():
    """
    Fixture that provides an MCPServiceRegistry for dependency injection.
    
    Returns:
        An MCPServiceRegistry instance.
    """
    # Import here to avoid circular imports
    from clubhouse.core.mcp_service_registry import MCPServiceRegistry
    return MCPServiceRegistry()


# Integration test fixtures
@pytest.fixture(scope="session")
def kafka_integration_config() -> Dict[str, str]:
    """
    Fixture that provides Kafka integration test configuration.
    Only used in integration tests when real Kafka is available.
    
    Returns:
        A dictionary with Kafka configuration.
    """
    return {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": "test-consumer-group",
        "auto.offset.reset": "earliest",
    }