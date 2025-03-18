"""
Tests for the enhanced service registry functionality.

These tests ensure that the enhanced ServiceRegistry properly supports
Protocol-based registration, lifecycle hooks, and plugin architecture.
"""
import pytest
from typing import Dict, List, Protocol, runtime_checkable
from unittest.mock import MagicMock, call

from clubhouse.core.service_registry import ServiceProtocol, ServiceRegistry


class TestService(ServiceProtocol):
    """A simple test service that implements the ServiceProtocol."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.shutdown_called = False
        self.reset_called = False
        
    def initialize(self) -> None:
        self.initialized = True
        
    def shutdown(self) -> None:
        self.shutdown_called = True
        
    def health_check(self) -> bool:
        return self.initialized and not self.shutdown_called
        
    def reset(self) -> None:
        self.reset_called = True
        self.initialized = False
        self.shutdown_called = False


@runtime_checkable
class TestPluginProtocol(Protocol):
    """Protocol for test plugins."""
    
    def process(self, data: str) -> str:
        """Process the given data."""
        ...


class TestPlugin:
    """A test plugin implementation."""
    
    def __init__(self, name: str):
        self.name = name
        
    def process(self, data: str) -> str:
        return f"Processed by {self.name}: {data}"


class TestPluginExtension(ServiceProtocol):
    """A test plugin extension that also implements ServiceProtocol."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.shutdown_called = False
        
    def initialize(self) -> None:
        self.initialized = True
        
    def shutdown(self) -> None:
        self.shutdown_called = True
        
    def health_check(self) -> bool:
        return self.initialized and not self.shutdown_called
        
    def reset(self) -> None:
        self.initialized = False
        self.shutdown_called = False
        
    def apply_extension(self, target: object) -> None:
        """Apply this extension to the target object."""
        setattr(target, f"extended_by_{self.name}", True)


@pytest.fixture
def service_registry() -> ServiceRegistry:
    """Fixture for a fresh ServiceRegistry."""
    return ServiceRegistry()


@pytest.fixture
def test_service() -> TestService:
    """Fixture for a test service."""
    return TestService("test_service")


class TestEnhancedServiceRegistry:
    """Tests for the enhanced ServiceRegistry functionality."""
    
    def test_protocol_registration_and_retrieval(self, service_registry: ServiceRegistry, test_service: TestService):
        """Test registering and retrieving a service using protocol."""
        # Register a service with a protocol
        service_registry.register_protocol(ServiceProtocol, test_service)
        
        # Retrieve using protocol
        retrieved_service = service_registry.get_protocol(ServiceProtocol)
        
        assert retrieved_service is test_service
        assert service_registry.has_protocol(ServiceProtocol)
        
    def test_lifecycle_methods(self, service_registry: ServiceRegistry, test_service: TestService):
        """Test lifecycle methods (initialize, shutdown)."""
        # Register a service with a protocol
        service_registry.register_protocol(ServiceProtocol, test_service)
        
        # Initialize all services
        service_registry.initialize_all()
        
        assert test_service.initialized
        assert service_registry.get_lifecycle_stage() == "running"
        
        # Shutdown all services
        service_registry.shutdown_all()
        
        assert test_service.shutdown_called
        assert service_registry.get_lifecycle_stage() == "shutdown"
        
    def test_plugin_extension_registration(self, service_registry: ServiceRegistry):
        """Test registering and using plugin extensions."""
        # Create a plugin extension
        plugin_extension = TestPluginExtension("test_extension")
        
        # Register the plugin extension
        service_registry.register_plugin_extension(plugin_extension)
        
        # Initialize all services (including plugin extensions)
        service_registry.initialize_all()
        
        assert plugin_extension.initialized
        
        # Test applying the extension to a target
        target = MagicMock()
        plugin_extension.apply_extension(target)
        
        # Check that the extension was applied
        assert getattr(target, f"extended_by_{plugin_extension.name}", False)
        
    def test_extension_point_system(self, service_registry: ServiceRegistry):
        """Test the extension point system."""
        # Register an extension point
        service_registry.register_extension_point("data_processors", TestPluginProtocol)
        
        # Create some extensions
        plugin1 = TestPlugin("plugin1")
        plugin2 = TestPlugin("plugin2")
        
        # Register the extensions
        service_registry.register_extension("data_processors", "plugin1", plugin1)
        service_registry.register_extension("data_processors", "plugin2", plugin2)
        
        # Get all extensions for the extension point
        extensions = service_registry.get_extensions("data_processors")
        
        assert len(extensions) == 2
        assert extensions["plugin1"] is plugin1
        assert extensions["plugin2"] is plugin2
        
        # Get a specific extension
        plugin1_retrieved = service_registry.get_extension("data_processors", "plugin1")
        assert plugin1_retrieved is plugin1
        
        # Check has_extension
        assert service_registry.has_extension("data_processors", "plugin1")
        assert not service_registry.has_extension("data_processors", "non_existent")
        
        # Test using the extensions
        data = "test data"
        for ext_name, extension in extensions.items():
            processed = extension.process(data)
            assert processed == f"Processed by {extension.name}: {data}"
            
        # Unregister an extension
        service_registry.unregister_extension("data_processors", "plugin1")
        assert not service_registry.has_extension("data_processors", "plugin1")
        assert len(service_registry.get_extensions("data_processors")) == 1
        
        # Unregister the extension point
        service_registry.unregister_extension_point("data_processors")
        assert not service_registry.has_extension_point("data_processors")
        
    def test_initialization_order(self, service_registry: ServiceRegistry):
        """Test that initialization happens in the correct order."""
        # Create mocked services
        plugin_extension = MagicMock(spec=ServiceProtocol)
        protocol_service = MagicMock(spec=ServiceProtocol)
        named_service = MagicMock(spec=ServiceProtocol)
        
        # Register services
        service_registry.register_plugin_extension(plugin_extension)
        service_registry.register_protocol(ServiceProtocol, protocol_service)
        service_registry.register("named_service", named_service)
        
        # Initialize all
        service_registry.initialize_all()
        
        # Check initialization order using call_args_list on the mocks
        initialization_calls = [
            call.initialize(),  # plugin_extension should be initialized first
            call.initialize(),  # protocol_service should be initialized second
            call.initialize()   # named_service should be initialized last
        ]
        
        assert plugin_extension.initialize.called
        assert protocol_service.initialize.called
        assert named_service.initialize.called
        
    def test_error_handling_during_initialization(self, service_registry: ServiceRegistry):
        """Test error handling during initialization."""
        # Create a service that raises an exception during initialization
        failing_service = MagicMock(spec=ServiceProtocol)
        failing_service.initialize.side_effect = RuntimeError("Initialization failed")
        
        # Register the failing service
        service_registry.register_protocol(ServiceProtocol, failing_service)
        
        # Initialize all should raise an exception
        with pytest.raises(RuntimeError):
            service_registry.initialize_all()
            
        # Check that the lifecycle stage is set to "failed"
        assert service_registry.get_lifecycle_stage() == "failed"
        
        # Even though initialization failed, shutdown should be called with force=True
        assert not service_registry._initialized
        
        # Now try to shutdown without the force parameter
        service_registry.shutdown_all()  # force=False by default
        
        # No services should be shut down as initialization failed
        assert failing_service.shutdown.call_count == 0
        
        # With force=True, shutdown should be called
        service_registry.shutdown_all(force=True)
        assert failing_service.shutdown.call_count == 1
