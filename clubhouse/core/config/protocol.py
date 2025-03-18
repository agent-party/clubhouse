"""
Protocol definitions for the hierarchical configuration system.

This module defines the core interfaces that all configuration providers
must adhere to, ensuring consistency across the framework.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel
from typing import cast, List, Dict, Any, Type


class ConfigUpdateEvent(BaseModel):
    """
    Event fired when configuration is updated.

    This event contains information about what was updated, including the path
    to the updated section, the old and new values, and the source of the update.
    """

    path: List[
        str
    ]  # Path to the updated config section (e.g., ["kafka", "bootstrap_servers"])
    old_value: Optional[Any] = None
    new_value: Any
    source: str  # Source of the update (e.g., "env", "file", "api")


# Generic type variable for the configuration model - making it covariant for protocol use
T_co = TypeVar("T_co", bound=BaseModel, covariant=True)

# Type for configuration update callbacks
ConfigUpdateCallback = Callable[[ConfigUpdateEvent], None]

# Type for unsubscribe function
Unsubscribe = Callable[[], None]


class ConfigProtocol(Protocol, Generic[T_co]):
    """
    Protocol defining the interface for configuration providers.

    This protocol specifies the contract that all configuration providers
    must adhere to, including methods for getting the current configuration,
    updating configuration values, and subscribing to configuration changes.
    """

    def get_model(self) -> Type[T_co]:
        """
        Get the configuration model class.

        Returns:
            The Pydantic model class used for this configuration.
        """
        ...

    def get(self) -> T_co:
        """
        Get the current configuration.

        Returns:
            The current configuration as a Pydantic model instance.
        """
        ...

    def get_value(self, path: List[str]) -> Any:
        """
        Get a specific configuration value by path.

        Args:
            path: List of keys forming the path to the configuration value.
                 E.g., ["kafka", "bootstrap_servers"]

        Returns:
            The configuration value at the specified path, or None if not found.
        """
        ...

    def set_value(self, path: List[str], value: Any, layer: str = "api") -> None:
        """
        Set a specific configuration value by path.

        Args:
            path: List of keys forming the path to the configuration value.
                 E.g., ["kafka", "bootstrap_servers"]
            value: The value to set.
            layer: The layer to update (defaults to "api").
        """
        ...

    def subscribe(self, callback: ConfigUpdateCallback) -> Unsubscribe:
        """
        Subscribe to configuration updates.

        Args:
            callback: Function to call when configuration is updated.

        Returns:
            Function to call to unsubscribe.
        """
        ...


class ConfigLayerProtocol(Protocol):
    """
    Protocol defining the interface for a configuration layer.

    A configuration layer represents a single source of configuration values,
    such as default values, environment variables, or a configuration file.
    """

    def name(self) -> str:
        """
        Get the name of this configuration layer.

        Returns:
            String identifier for this layer (e.g., "defaults", "env", "file").
        """
        ...

    def priority(self) -> int:
        """
        Get the priority of this configuration layer.

        Higher priority layers override values from lower priority layers.

        Returns:
            Integer priority value (higher = more precedence).
        """
        ...

    def get(self, key: str) -> Optional[Any]:
        """
        Get a configuration value from this layer.

        Args:
            key: Configuration key to retrieve (can be a dotted path).

        Returns:
            The configuration value if found, or None if not present in this layer.
        """
        ...

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values from this layer.

        Returns:
            Dictionary of all configuration values in this layer.
        """
        ...

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value in this layer.

        Args:
            key: Configuration key to set (can be a dotted path).
            value: Value to set.
        """
        ...

    def refresh(self) -> Dict[str, Any]:
        """
        Refresh configuration values from the source.

        This is used for layers that can change externally, such as
        environment variables or files that might be modified.

        Returns:
            Dictionary of all configuration values after refresh.
        """
        ...
