from typing import Any, Dict, Optional, Protocol, Type, TypeVar

T = TypeVar("T")


class ServiceProtocol(Protocol):
    """Base protocol for all services in the application."""

    pass


class ServiceRegistry:
    """
    A service registry for dependency management following the service locator pattern.

    This registry allows for dependency injection and service management, making
    components more testable and loosely coupled.
    """

    def __init__(self) -> None:
        """Initialize an empty service registry."""
        self._services: Dict[str, Any] = {}

    def register(self, service_name: str, service_instance: Any) -> None:
        """
        Register a service in the registry.

        Args:
            service_name: A unique identifier for the service
            service_instance: The service instance to register
        """
        self._services[service_name] = service_instance

    def get(self, service_name: str) -> Any:
        """
        Get a service from the registry.

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
        Check if a service is registered.

        Args:
            service_name: The identifier for the service to check

        Returns:
            True if the service is registered, False otherwise
        """
        return service_name in self._services
