"""
Configuration provider implementation for the hierarchical configuration system.

This module provides the main configuration provider class that implements
the ConfigProtocol and manages multiple configuration layers.
"""

import os
import logging
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, create_model

from clubhouse.core.config.layers import ConfigLayer
from clubhouse.core.config.protocol import (
    ConfigLayerProtocol,
    ConfigProtocol,
    ConfigUpdateCallback,
    ConfigUpdateEvent,
)

T_co = TypeVar("T_co", bound=BaseModel, covariant=True)


class ConfigurationProvider(Generic[T_co], ConfigProtocol[T_co]):
    """
    Implementation of the ConfigProtocol that manages multiple configuration layers.

    This class combines values from multiple layers based on their priorities,
    handles updates, and notifies subscribers of changes.
    """

    def __init__(
        self, model_type: Type[T_co], layers: Optional[List[ConfigLayerProtocol]] = None
    ) -> None:
        """
        Initialize the configuration provider.

        Args:
            model_type: Pydantic model class that defines the schema for this configuration.
            layers: Optional list of configuration layers to use.
        """
        self._model_type = model_type
        self._layers = layers or []
        self._subscribers: Dict[str, List[ConfigUpdateCallback]] = {}
        self._cached_config: Optional[T_co] = None

        # Sort layers by priority (highest first)
        self._sort_layers()

        # Initialize cached config
        self._refresh_config()

    @property
    def current(self) -> T_co:
        """
        Property accessor for the current configuration.
        This is an alias for get() to maintain compatibility.

        Returns:
            The current configuration as a Pydantic model instance.
        """
        return self.get()

    @property
    def model(self) -> Type[T_co]:
        """
        Property accessor for the model type.
        This is an alias for get_model() to maintain compatibility.

        Returns:
            The Pydantic model class used for this configuration.
        """
        return self.get_model()

    def _sort_layers(self) -> None:
        """Sort layers by priority (highest first)."""
        self._layers.sort(key=lambda layer: layer.priority(), reverse=True)

    def _refresh_config(self) -> None:
        """Refresh the cached configuration from all layers."""
        # Start with empty config data
        config_data: Dict[str, Any] = {}

        # Apply layers from lowest to highest priority to ensure proper overrides
        # First sort layers by priority (lowest first)
        sorted_layers = sorted(self._layers, key=lambda layer: layer.priority())

        # Apply each layer in order of increasing priority
        for layer in sorted_layers:
            self._apply_layer(config_data, layer)

        # Create a new config instance
        try:
            self._cached_config = self._model_type(**config_data)
        except Exception:
            # If validation fails, create a dynamic model that accepts any fields
            field_defs: Dict[str, Any] = {
                k: (type(v), v) for k, v in config_data.items()
            }

            # Create a properly typed dynamic model using keyword arguments
            model_name = f"Dynamic{self._model_type.__name__}"

            # Pass fields as named arguments to create_model
            # First, prepare the keyword arguments with __base__ parameter
            kwargs: Dict[str, Any] = {"__base__": self._model_type}
            kwargs.update(field_defs)

            # Create the dynamic model and handle the typing with cast
            DynamicModel = cast(Type[T_co], create_model(model_name, **kwargs))

            # Create an instance of the dynamic model
            self._cached_config = DynamicModel(**config_data)

    def _apply_layer(
        self, config_data: Dict[str, Any], layer: ConfigLayerProtocol
    ) -> None:
        """
        Apply a configuration layer to the config data.

        Args:
            config_data: Dictionary to update with layer values.
            layer: Configuration layer to apply.
        """
        layer_data = layer.get_all()
        self._merge_dicts(config_data, layer_data)

    def _merge_dicts(
        self,
        target: Dict[str, Any],
        source: Dict[str, Any],
        path: Optional[List[str]] = None,
    ) -> None:
        """
        Recursively merge dictionaries.

        Args:
            target: Dictionary to update.
            source: Dictionary with values to merge.
            path: Current path for nested dictionaries.
        """
        if path is None:
            path = []

        for key, value in source.items():
            # Build the current path
            current_path = path + [key]

            # If both are dicts, merge recursively
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_dicts(target[key], value, current_path)
            else:
                # Otherwise, just update the value
                target[key] = value

    def _get_layer_by_name(self, name: str) -> Optional[ConfigLayerProtocol]:
        """
        Get a layer by name.

        Args:
            name: Name of the layer to find.

        Returns:
            The layer if found, or None if not found.
        """
        for layer in self._layers:
            if layer.name() == name:
                return layer
        return None

    def add_layer(self, layer: ConfigLayerProtocol) -> None:
        """
        Add a new configuration layer.

        Args:
            layer: Configuration layer to add.
        """
        # Remove existing layer with the same name if present
        existing = self._get_layer_by_name(layer.name())
        if existing:
            self._layers.remove(existing)

        # Add the new layer
        self._layers.append(layer)

        # Re-sort layers
        self._sort_layers()

        # Refresh the cached config
        self._refresh_config()

    def remove_layer(self, name: str) -> None:
        """
        Remove a configuration layer by name.

        Args:
            name: Name of the layer to remove.
        """
        layer = self._get_layer_by_name(name)
        if layer:
            self._layers.remove(layer)
            self._refresh_config()

    def get_model(self) -> Type[T_co]:
        """
        Get the configuration model class.

        Returns:
            The Pydantic model class used for this configuration.
        """
        return self._model_type

    def get(self) -> T_co:
        """
        Get the current configuration.

        Returns:
            The current configuration as a Pydantic model instance.
        """
        if self._cached_config is None:
            # Initialize an empty model if none exists
            self._cached_config = self._model_type()
        return self._cached_config

    def get_value(self, path: List[str]) -> Any:
        """
        Get a specific configuration value by path.

        Args:
            path: List of keys forming the path to the configuration value.
                 E.g., ["kafka", "bootstrap_servers"]

        Returns:
            The configuration value at the specified path, or None if not found.
        """
        config = self.get()
        value = config
        for key in path:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def get_section(self, path: List[str]) -> Any:
        """
        Get a configuration section by path.

        Args:
            path: List of keys forming the path to the configuration section.
                 E.g., ["kafka"] would return the kafka configuration section.
                 An empty list returns the entire configuration.

        Returns:
            The configuration section at the specified path, or None if not found.
        """
        if not path:
            # If path is empty, return the entire configuration
            return self.get()

        # Otherwise, get the specific section
        return self.get_value(path)

    def update(self, updates: Dict[str, Any], source: str) -> List[ConfigUpdateEvent]:
        """
        Update multiple configuration values at once.

        Args:
            updates: Dictionary of configuration values to update.
                    Keys should be top-level configuration keys.
            source: Source identifier for the update.

        Returns:
            List of ConfigUpdateEvent objects describing the changes.
        """
        events: List[ConfigUpdateEvent] = []

        # Find the "api" layer or create one if it doesn't exist
        api_layer = self._get_layer_by_name("api")
        if not api_layer:
            api_layer = ConfigLayer(name="api", priority=1000)
            self.add_layer(api_layer)

        # Get current configuration
        current_config = self.get()

        # Apply each update and generate events
        for key, value in updates.items():
            # Get the old value
            old_value = getattr(current_config, key, None)

            # Update the layer
            api_layer.set(key, value)

            # Create an event
            event = ConfigUpdateEvent(
                path=[key], old_value=old_value, new_value=value, source=source
            )
            events.append(event)

            # Notify subscribers about this event
            self._notify_subscribers(event)

        # Refresh the configuration
        self._refresh_config()

        # Return the events
        return events

    def set_value(self, path: List[str], value: Any, layer_name: str = "api") -> None:
        """
        Set a specific configuration value by path.

        Args:
            path: List of keys forming the path to the configuration value.
                 E.g., ["kafka", "bootstrap_servers"]
            value: The value to set.
            layer_name: The layer to update (defaults to "api").
        """
        # Get or create the API layer
        layer = self._get_layer_by_name(layer_name)

        if layer is None:
            # If layer doesn't exist, create it with high priority
            api_layer = ConfigLayer(name=layer_name, priority=1000)
            self._layers.append(api_layer)
            self._sort_layers()
            layer = api_layer

        # Build the dotted key
        key = ".".join(path)

        # Set the value
        assert layer is not None  # This helps mypy understand we handled the None case
        layer.set(key, value)

        # Update the cached config
        old_config = self.get()
        old_dict = old_config.model_dump()

        # Refresh the cached config
        self._refresh_config()

        # Get the new config
        new_config = self.get()
        new_dict = new_config.model_dump()

        # Generate update events
        events: List[ConfigUpdateEvent] = []
        self._find_changes(old_dict, new_dict, [], layer_name, events)

        # Notify subscribers
        for event in events:
            self._notify_subscribers(event)

    def _notify_subscribers(self, event: ConfigUpdateEvent) -> None:
        """
        Notify subscribers of a configuration update event.

        Args:
            event: Configuration update event to notify subscribers about.
        """
        # Notify key-specific subscribers
        key_path = ".".join(event.path)
        if key_path in self._subscribers:
            for callback in self._subscribers[key_path]:
                callback(event)

        # Notify global subscribers (key="*")
        if "*" in self._subscribers:
            for callback in self._subscribers["*"]:
                callback(event)

    def subscribe(
        self, callback: ConfigUpdateCallback, keys: Optional[List[str]] = None
    ) -> Callable[[], None]:
        """
        Subscribe to configuration updates.

        Args:
            callback: Function to call when configuration is updated.
            keys: Optional list of specific keys to subscribe to.
                If not provided, subscribes to all updates.
                Keys should be top-level configuration keys.

        Returns:
            Function to call to unsubscribe.
        """
        # Track all subscriptions for this callback
        subscription_keys = []

        if keys:
            # Subscribe to each key
            for key in keys:
                # Create subscription list if it doesn't exist
                if key not in self._subscribers:
                    self._subscribers[key] = []

                # Add callback to subscribers
                self._subscribers[key].append(callback)
                subscription_keys.append(key)
        else:
            # Use "*" for global subscription if keys is not provided
            if "*" not in self._subscribers:
                self._subscribers["*"] = []

            # Add callback to global subscribers
            self._subscribers["*"].append(callback)
            subscription_keys.append("*")

        # Return unsubscribe function
        def unsubscribe() -> None:
            for key in subscription_keys:
                if key in self._subscribers and callback in self._subscribers[key]:
                    self._subscribers[key].remove(callback)
                    # Clean up empty subscriber lists
                    if not self._subscribers[key]:
                        del self._subscribers[key]

        return unsubscribe

    def refresh_layer(self, name: str) -> List[ConfigUpdateEvent]:
        """
        Refresh a specific configuration layer and update the configuration.

        Args:
            name: Name of the layer to refresh.

        Returns:
            List of ConfigUpdateEvent objects for each updated field.
        """
        layer = self._get_layer_by_name(name)
        if not layer:
            return []

        # Get current config as dictionary
        old_config = self.get()
        old_dict = old_config.model_dump()

        # Refresh the layer
        layer.refresh()

        # Refresh the cached config
        self._refresh_config()

        # Get new config as dictionary
        new_config = self.get()
        new_dict = new_config.model_dump()

        # Generate update events for changed fields
        events: List[ConfigUpdateEvent] = []

        # Compare old and new configs to find changes
        self._find_changes(old_dict, new_dict, [], name, events)

        # Notify subscribers for each event
        for event in events:
            self._notify_subscribers(event)

        return events

    def _find_changes(
        self,
        old_dict: Dict[str, Any],
        new_dict: Dict[str, Any],
        path: List[str],
        source: str,
        events: List[ConfigUpdateEvent],
    ) -> None:
        """
        Recursively find changes between two dictionaries.

        Args:
            old_dict: Old configuration dictionary.
            new_dict: New configuration dictionary.
            path: Current path in the configuration.
            source: Source of the update.
            events: List to append update events to.
        """
        # Check keys in new dict that were added or changed
        for key, new_value in new_dict.items():
            current_path = path + [key]

            if key not in old_dict:
                # Key was added
                events.append(
                    ConfigUpdateEvent(
                        path=current_path,
                        old_value=None,
                        new_value=new_value,
                        source=source,
                    )
                )
            elif isinstance(new_value, dict) and isinstance(old_dict[key], dict):
                # Both are dicts, recursively check
                self._find_changes(
                    old_dict[key], new_value, current_path, source, events
                )
            elif new_value != old_dict[key]:
                # Value changed
                events.append(
                    ConfigUpdateEvent(
                        path=current_path,
                        old_value=old_dict[key],
                        new_value=new_value,
                        source=source,
                    )
                )

        # Check keys in old dict that were removed
        for key, old_value in old_dict.items():
            if key not in new_dict:
                # Key was removed
                events.append(
                    ConfigUpdateEvent(
                        path=path + [key],
                        old_value=old_value,
                        new_value=None,
                        source=source,
                    )
                )

    def refresh_all(self) -> List[ConfigUpdateEvent]:
        """
        Refresh all configuration layers and update the configuration.

        Returns:
            List of ConfigUpdateEvent objects for each updated field.
        """
        # Get current config as dictionary
        old_config = self.get()
        old_dict = old_config.model_dump()

        # Refresh all layers
        for layer in self._layers:
            layer.refresh()

        # Refresh the cached config
        self._refresh_config()

        # Get new config as dictionary
        new_config = self.get()
        new_dict = new_config.model_dump()

        # Generate update events for changed fields
        events: List[ConfigUpdateEvent] = []

        # Compare old and new configs to find changes
        self._find_changes(old_dict, new_dict, [], "refresh_all", events)

        # Notify subscribers for each event
        for event in events:
            self._notify_subscribers(event)

        return events
