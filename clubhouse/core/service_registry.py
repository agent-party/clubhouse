"""
Service registry for dependency management.

This module provides a centralized registry for managing service dependencies using
the service locator pattern, enabling better testability and loose coupling.
"""
from typing import Any, Dict, Generic, Optional, Protocol, Type, TypeVar, List, cast, get_type_hints, runtime_checkable
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S", bound="ServiceProtocol")


@runtime_checkable
class ServiceProtocol(Protocol):
    """
    Base protocol for all services in the application.
    
    All services should implement this protocol to ensure they can be properly
    registered and retrieved from the ServiceRegistry.
    """
    
    def initialize(self) -> None:
        """Initialize the service with any required setup."""
        ...
        
    def shutdown(self) -> None:
        """Shutdown the service and perform cleanup."""
        ...
        
    def health_check(self) -> bool:
        """
        Check if the service is healthy and operational.
        
        Returns:
            bool: True if the service is healthy, False otherwise
        """
        ...
        
    def reset(self) -> None:
        """
        Reset the service to its initial state.
        
        This is primarily used for testing or recovery scenarios.
        """
        ...


class ServiceRegistry:
    """
    A service registry for dependency management following the service locator pattern.

    This registry allows for dependency injection and service management, making
    components more testable and loosely coupled.
    
    Example:
        ```python
        # Create a registry
        registry = ServiceRegistry()
        
        # Register protocol-based services
        registry.register_protocol(KafkaServiceProtocol, KafkaService(...))
        registry.register_protocol(SchemaRegistryProtocol, SchemaRegistryService(...))
        
        # Get service by protocol
        kafka_service: Any = registry.get_protocol(KafkaServiceProtocol)
        
        # Register plugin extensions
        registry.register_plugin_extension(MyPluginExtension())
        
        # Legacy: Register services by name
        registry.register("legacy_service", LegacyService(...))
        
        # Legacy: Get service by name
        legacy_service = registry.get("legacy_service")
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty service registry."""
        self._services: Dict[str, Any] = {}
        self._protocol_services: Dict[Type, Any] = {}
        self._plugin_extensions: Dict[str, Any] = {}
        self._extension_points: Dict[str, Dict[str, Any]] = {}
        self._initialized: bool = False
        self._lifecycle_stage: str = "created"
        
    def initialize_all(self) -> None:
        """
        Initialize all registered services that implement the ServiceProtocol.
        
        This method calls the initialize method on all registered services
        in the following order:
        1. Plugin extensions
        2. Protocol-based services
        3. Named services
        
        If any service fails to initialize, it will attempt to shutdown already
        initialized services and set the lifecycle stage to "failed".
        
        Raises:
            Exception: If any service fails to initialize
        """
        logger.info("Initializing all services...")
        
        if self._initialized:
            logger.warning("Services already initialized")
            return
            
        self._lifecycle_stage = "initializing"
        
        # Keep track of initialized extensions and services so we can shut them down if initialization fails
        initialized_extensions = []
        initialized_protocol_services = []
        initialized_named_services = []
        
        try:
            # Initialize plugin extensions first
            for ext_name, extension in self._plugin_extensions.items():
                if isinstance(extension, ServiceProtocol):
                    logger.debug(f"Initializing plugin extension: {ext_name}")
                    extension.initialize()
                    initialized_extensions.append((ext_name, extension))
            
            # Initialize protocol-based services
            for protocol_type, service in self._protocol_services.items():
                if isinstance(service, ServiceProtocol):
                    logger.debug(f"Initializing service for protocol: {protocol_type.__name__}")
                    service.initialize()
                    initialized_protocol_services.append((protocol_type, service))
            
            # Initialize name-based services
            for service_name, service in self._services.items():
                if isinstance(service, ServiceProtocol):
                    logger.debug(f"Initializing service: {service_name}")
                    service.initialize()
                    initialized_named_services.append((service_name, service))
                    
            self._initialized = True
            self._lifecycle_stage = "running"
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            self._lifecycle_stage = "failed"
            self._initialized = False  # Ensure initialized flag is set to False
            
            # Shutdown already initialized services in reverse order
            for service_name, service in reversed(initialized_named_services):
                logger.debug(f"Shutting down service after initialization failure: {service_name}")
                try:
                    service.shutdown()
                except Exception as shutdown_error:
                    logger.error(f"Error shutting down service {service_name} after initialization failure: {shutdown_error}")
                    
            for protocol_type, service in reversed(initialized_protocol_services):
                logger.debug(f"Shutting down service after initialization failure: {protocol_type.__name__}")
                try:
                    service.shutdown()
                except Exception as shutdown_error:
                    logger.error(f"Error shutting down service for protocol {protocol_type.__name__} after initialization failure: {shutdown_error}")
                    
            for ext_name, extension in reversed(initialized_extensions):
                logger.debug(f"Shutting down plugin extension after initialization failure: {ext_name}")
                try:
                    extension.shutdown()
                except Exception as shutdown_error:
                    logger.error(f"Error shutting down plugin extension {ext_name} after initialization failure: {shutdown_error}")
            
            raise

    def shutdown_all(self, force: bool = False) -> None:
        """
        Shutdown all registered services.
        
        This method calls the shutdown method on all registered services
        that implement the ServiceProtocol.
        
        Args:
            force: If True, shutdown services even if initialization failed or never completed
        """
        if not self._initialized and not force:
            logger.warning("Services not initialized, nothing to shut down")
            return
            
        logger.info("Shutting down all services...")
        
        # Keep track of the previous lifecycle stage for error cases
        previous_stage = self._lifecycle_stage
        
        # Only update the lifecycle stage if not already in failed state
        if self._lifecycle_stage != "failed":
            self._lifecycle_stage = "shutting_down"
        
        # Skip actual shutdown unless initialized or force=True
        if self._initialized or force:
            # Shutdown plugin extensions first
            for ext_name, extension in self._plugin_extensions.items():
                if isinstance(extension, ServiceProtocol):
                    logger.debug(f"Shutting down plugin extension: {ext_name}")
                    try:
                        extension.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down plugin extension {ext_name}: {e}")
            
            # Shutdown protocol-based services
            for protocol_type, service in self._protocol_services.items():
                if isinstance(service, ServiceProtocol):
                    logger.debug(f"Shutting down service for protocol: {protocol_type.__name__}")
                    try:
                        service.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down service for protocol {protocol_type.__name__}: {e}")
            
            # Shutdown name-based services
            for service_name, service in self._services.items():
                if isinstance(service, ServiceProtocol):
                    logger.debug(f"Shutting down service: {service_name}")
                    try:
                        service.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down service {service_name}: {e}")
        
        self._initialized = False
        
        # Only update the lifecycle stage if not in failed state
        if previous_stage != "failed":
            self._lifecycle_stage = "shutdown"
            logger.info("All services shut down successfully")
        else:
            # Keep the failed state
            logger.info("All services shut down after initialization failure")

    def register(self, service_name: str, service_instance: Any) -> None:
        """
        Register a service in the registry using a string identifier.
        
        Note: This method is maintained for backward compatibility.
        Consider using register_protocol for new code.

        Args:
            service_name: A unique identifier for the service
            service_instance: The service instance to register
            
        Raises:
            ValueError: If the service_name is empty or already registered
        """
        if not service_name:
            raise ValueError("Service name cannot be empty")
            
        if service_name in self._services:
            raise ValueError(f"Service '{service_name}' is already registered")
            
        self._services[service_name] = service_instance

    def register_protocol(self, protocol: Type[T], implementation: T) -> None:
        """
        Register a service implementation for a specific protocol.

        Args:
            protocol: The protocol interface type
            implementation: The service implementation instance
            
        Raises:
            ValueError: If the protocol is already registered
            TypeError: If the implementation does not conform to the protocol
        """
        if protocol in self._protocol_services:
            raise ValueError(f"Protocol '{protocol.__name__}' is already registered")
            
        # Validate that the implementation conforms to the protocol
        # This is a basic check that can be enhanced in the future
        if not self._validate_protocol_implementation(protocol, implementation):
            raise TypeError(
                f"Implementation {implementation.__class__.__name__} does not conform to protocol {protocol.__name__}"
            )
            
        self._protocol_services[protocol] = implementation

    def register_plugin_extension(self, extension: Any) -> None:
        """
        Register a plugin extension.

        Args:
            extension: The plugin extension instance
            
        Raises:
            ValueError: If the extension is already registered
        """
        ext_name = extension.__class__.__name__
        if ext_name in self._plugin_extensions:
            raise ValueError(f"Plugin extension '{ext_name}' is already registered")
            
        self._plugin_extensions[ext_name] = extension

    def _validate_protocol_implementation(self, protocol: Type, implementation: Any) -> bool:
        """
        Validate that an implementation conforms to a protocol.
        
        This is a basic validation that checks if the implementation has all
        the methods defined in the protocol.
        
        Args:
            protocol: The protocol interface type
            implementation: The service implementation instance
            
        Returns:
            True if the implementation conforms to the protocol, False otherwise
        """
        # Special case for unittest.mock.Mock objects
        if hasattr(implementation, '_mock_methods') or hasattr(implementation, '_spec_class'):
            # For mocks with a spec, we trust that the spec is correct
            return True
            
        # Check if the implementation directly claims to implement the protocol via isinstance
        # This works for @runtime_checkable protocols
        if isinstance(protocol, type) and isinstance(implementation, protocol):
            return True
        
        # Basic check - just ensure the implementation has the methods defined in the protocol
        # This doesn't check parameter types or return types
        protocol_methods = {name for name, _ in inspect.getmembers(protocol, inspect.isfunction)}
        implementation_methods = {name for name, _ in inspect.getmembers(implementation, inspect.ismethod)}
        
        # For protocols, exclude dunder methods
        protocol_methods = {name for name in protocol_methods if not (name.startswith('__') and name.endswith('__'))}
        
        # Check if all protocol methods are implemented
        return all(method in implementation_methods for method in protocol_methods)

    def get(self, service_name: str) -> Any:
        """
        Get a service from the registry by name.
        
        Note: This method is maintained for backward compatibility.
        Consider using get_protocol for new code.

        Args:
            service_name: The identifier for the service to retrieve

        Returns:
            The registered service instance

        Raises:
            KeyError: If the service is not registered
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not registered")
        return self._services[service_name]

    def get_protocol(self, protocol: Type[T]) -> T:
        """
        Get a service implementation for a specific protocol.
        
        Args:
            protocol: The protocol interface type
            
        Returns:
            The registered service implementation
            
        Raises:
            KeyError: If no implementation is registered for the protocol
        """
        if protocol not in self._protocol_services:
            raise KeyError(f"No implementation registered for protocol '{protocol.__name__}'")
        return cast(T, self._protocol_services[protocol])

    def get_typed(self, service_name: str, expected_type: Type[T]) -> T:
        """
        Get a service with type checking.

        Args:
            service_name: The identifier for the service to retrieve
            expected_type: The expected type of the service

        Returns:
            The registered service instance casted to the expected type

        Raises:
            KeyError: If the service is not registered
            TypeError: If the service is not of the expected type
        """
        service = self.get(service_name)
        if not isinstance(service, expected_type):
            raise TypeError(
                f"Service '{service_name}' is not of type {expected_type.__name__}"
            )
        return service

    def has_service(self, service_name: str) -> bool:
        """
        Check if a service is registered by name.

        Args:
            service_name: The identifier for the service to check

        Returns:
            True if the service is registered, False otherwise
        """
        return service_name in self._services
        
    def has_protocol(self, protocol: Type) -> bool:
        """
        Check if an implementation is registered for a protocol.
        
        Args:
            protocol: The protocol interface type
            
        Returns:
            True if an implementation is registered, False otherwise
        """
        return protocol in self._protocol_services
        
    def unregister(self, service_name: str) -> None:
        """
        Remove a service from the registry.
        
        Args:
            service_name: The identifier for the service to remove
            
        Raises:
            KeyError: If the service is not registered
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not registered")
        del self._services[service_name]
        
    def unregister_protocol(self, protocol: Type) -> None:
        """
        Unregister an implementation for a protocol.
        
        Args:
            protocol: The protocol interface type
            
        Raises:
            KeyError: If no implementation is registered for the protocol
        """
        if protocol not in self._protocol_services:
            raise KeyError(f"No implementation registered for protocol '{protocol.__name__}'")
        del self._protocol_services[protocol]

    def unregister_plugin_extension(self, extension_name: str) -> None:
        """
        Unregister a plugin extension.
        
        Args:
            extension_name: The name of the plugin extension to remove
            
        Raises:
            KeyError: If the plugin extension is not registered
        """
        if extension_name not in self._plugin_extensions:
            raise KeyError(f"Plugin extension '{extension_name}' not registered")
        del self._plugin_extensions[extension_name]

    def register_extension_point(self, extension_point_name: str, extension_point_type: Type) -> None:
        """
        Register an extension point that plugins can implement.
        
        Args:
            extension_point_name: Unique name for the extension point
            extension_point_type: The expected type or protocol for extensions
            
        Raises:
            ValueError: If the extension point already exists
        """
        if extension_point_name in self._extension_points:
            raise ValueError(f"Extension point '{extension_point_name}' is already registered")
            
        self._extension_points[extension_point_name] = {
            "type": extension_point_type,
            "extensions": {}
        }
        logger.debug(f"Registered extension point: {extension_point_name}")
        
    def register_extension(self, extension_point_name: str, extension_name: str, extension: Any) -> None:
        """
        Register an extension for a specific extension point.
        
        Args:
            extension_point_name: Name of the extension point
            extension_name: Unique name for this extension
            extension: The extension implementation
            
        Raises:
            KeyError: If the extension point doesn't exist
            ValueError: If the extension already exists or doesn't match the expected type
        """
        if extension_point_name not in self._extension_points:
            raise KeyError(f"Extension point '{extension_point_name}' does not exist")
            
        extension_point = self._extension_points[extension_point_name]
        extension_type = extension_point["type"]
        
        # Check if extension already exists
        if extension_name in extension_point["extensions"]:
            raise ValueError(f"Extension '{extension_name}' is already registered for extension point '{extension_point_name}'")
            
        # Validate extension type
        if not isinstance(extension, extension_type):
            raise ValueError(
                f"Extension '{extension_name}' does not match the required type for extension point '{extension_point_name}'"
            )
            
        extension_point["extensions"][extension_name] = extension
        logger.debug(f"Registered extension '{extension_name}' for extension point '{extension_point_name}'")
        
    def get_extensions(self, extension_point_name: str) -> Dict[str, Any]:
        """
        Get all extensions for a specific extension point.
        
        Args:
            extension_point_name: Name of the extension point
            
        Returns:
            Dictionary mapping extension names to extension instances
            
        Raises:
            KeyError: If the extension point doesn't exist
        """
        if extension_point_name not in self._extension_points:
            raise KeyError(f"Extension point '{extension_point_name}' does not exist")
            
        return self._extension_points[extension_point_name]["extensions"].copy()  # type: ignore[any_return]
        
    def get_extension(self, extension_point_name: str, extension_name: str) -> Any:
        """
        Get a specific extension for an extension point.
        
        Args:
            extension_point_name: Name of the extension point
            extension_name: Name of the extension
            
        Returns:
            The extension instance
            
        Raises:
            KeyError: If the extension point or extension doesn't exist
        """
        if extension_point_name not in self._extension_points:
            raise KeyError(f"Extension point '{extension_point_name}' does not exist")
            
        extensions = self._extension_points[extension_point_name]["extensions"]
        if extension_name not in extensions:
            raise KeyError(f"Extension '{extension_name}' not found for extension point '{extension_point_name}'")
            
        return extensions[extension_name]
        
    def has_extension_point(self, extension_point_name: str) -> bool:
        """
        Check if an extension point exists.
        
        Args:
            extension_point_name: Name of the extension point
            
        Returns:
            True if the extension point exists, False otherwise
        """
        return extension_point_name in self._extension_points
        
    def has_extension(self, extension_point_name: str, extension_name: str) -> bool:
        """
        Check if a specific extension exists for an extension point.
        
        Args:
            extension_point_name: Name of the extension point
            extension_name: Name of the extension
            
        Returns:
            True if the extension exists, False otherwise
        """
        if not self.has_extension_point(extension_point_name):
            return False
            
        return extension_name in self._extension_points[extension_point_name]["extensions"]
        
    def unregister_extension(self, extension_point_name: str, extension_name: str) -> None:
        """
        Unregister an extension from an extension point.
        
        Args:
            extension_point_name: Name of the extension point
            extension_name: Name of the extension
            
        Raises:
            KeyError: If the extension point or extension doesn't exist
        """
        if extension_point_name not in self._extension_points:
            raise KeyError(f"Extension point '{extension_point_name}' does not exist")
            
        extensions = self._extension_points[extension_point_name]["extensions"]
        if extension_name not in extensions:
            raise KeyError(f"Extension '{extension_name}' not found for extension point '{extension_point_name}'")
            
        del extensions[extension_name]
        logger.debug(f"Unregistered extension '{extension_name}' from extension point '{extension_point_name}'")
        
    def unregister_extension_point(self, extension_point_name: str) -> None:
        """
        Unregister an extension point and all its extensions.
        
        Args:
            extension_point_name: Name of the extension point
            
        Raises:
            KeyError: If the extension point doesn't exist
        """
        if extension_point_name not in self._extension_points:
            raise KeyError(f"Extension point '{extension_point_name}' does not exist")
            
        del self._extension_points[extension_point_name]
        logger.debug(f"Unregistered extension point '{extension_point_name}'")
        
    def get_lifecycle_stage(self) -> str:
        """
        Get the current lifecycle stage of the service registry.
        
        Returns:
            Current lifecycle stage ('created', 'initializing', 'running', 'shutdown', 'failed')
        """
        return self._lifecycle_stage