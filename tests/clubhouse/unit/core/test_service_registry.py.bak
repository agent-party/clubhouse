import pytest
from typing import Protocol

from clubhouse.core.service_registry import ServiceRegistry


class DummyServiceProtocol(Protocol):
    """A protocol for testing the service registry."""
    
    def do_something(self) -> str:
        """A dummy method for testing."""
        ...


class DummyService:
    """A dummy service implementation for testing."""
    
    def do_something(self) -> str:
        """Return a dummy value."""
        return "something"


class AnotherService:
    """Another service implementation for testing."""
    
    def do_something_else(self) -> str:
        """Return a dummy value."""
        return "something else"


class TestServiceRegistry:
    """Tests for the ServiceRegistry class."""
    
    def test_register_and_get_service(self) -> None:
        """Test registering and retrieving a service."""
        # Arrange
        registry = ServiceRegistry()
        service = DummyService()
        
        # Act
        registry.register("dummy", service)
        retrieved = registry.get("dummy")
        
        # Assert
        assert retrieved is service
    
    def test_get_nonexistent_service_raises_key_error(self) -> None:
        """Test that getting a nonexistent service raises a KeyError."""
        # Arrange
        registry = ServiceRegistry()
        
        # Act & Assert
        with pytest.raises(KeyError):
            registry.get("nonexistent")
    
    def test_get_typed_with_correct_type(self) -> None:
        """Test get_typed with a service of the correct type."""
        # Arrange
        registry = ServiceRegistry()
        service = DummyService()
        registry.register("dummy", service)
        
        # Act
        retrieved = registry.get_typed("dummy", DummyService)
        
        # Assert
        assert retrieved is service
    
    def test_get_typed_with_wrong_type_raises_type_error(self) -> None:
        """Test that get_typed with a service of the wrong type raises a TypeError."""
        # Arrange
        registry = ServiceRegistry()
        service = DummyService()
        registry.register("dummy", service)
        
        # Act & Assert
        with pytest.raises(TypeError):
            registry.get_typed("dummy", AnotherService)
    
    def test_has_service(self) -> None:
        """Test has_service method."""
        # Arrange
        registry = ServiceRegistry()
        service = DummyService()
        
        # Act
        registry.register("dummy", service)
        
        # Assert
        assert registry.has_service("dummy") is True
        assert registry.has_service("nonexistent") is False
