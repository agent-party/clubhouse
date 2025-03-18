"""Tests for the ConfigurationProvider."""
import pytest
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from clubhouse.core.config import (
    ConfigurationProvider,
    DefaultsLayer,
    ConfigLayer,
    ConfigUpdateEvent,
)


class TestConfig(BaseModel):
    """Test configuration model."""
    string_value: str = "default"
    int_value: int = 10
    bool_value: bool = False
    nested: Dict[str, Any] = Field(default_factory=lambda: {"key": "value"})


def test_provider_initialization():
    """Test basic provider initialization."""
    # Create a provider with no layers
    provider = ConfigurationProvider(model_type=TestConfig)
    
    # Should have default values from the model
    config = provider.current
    assert config.string_value == "default"
    assert config.int_value == 10
    assert config.bool_value is False
    assert config.nested["key"] == "value"
    
    # Model type should be correct
    assert provider.model == TestConfig


def test_provider_with_layers():
    """Test provider with multiple configuration layers."""
    # Create layers with different priorities
    defaults = DefaultsLayer(
        name="defaults",
        priority=0,
        defaults={
            "string_value": "from_defaults",
            "int_value": 100,
        }
    )
    
    overrides = ConfigLayer(
        name="overrides",
        priority=100
    )
    overrides.set("string_value", "from_overrides")
    overrides.set("bool_value", True)
    
    # Create provider with these layers
    provider = ConfigurationProvider(
        model_type=TestConfig,
        layers=[defaults, overrides]
    )
    
    # Values should be merged with priority
    config = provider.current
    assert config.string_value == "from_overrides"  # From high priority layer
    assert config.int_value == 100  # From defaults layer
    assert config.bool_value is True  # From overrides layer
    assert config.nested["key"] == "value"  # From model default


def test_add_and_remove_layers():
    """Test adding and removing layers dynamically."""
    # Create a provider with no layers
    provider = ConfigurationProvider(model_type=TestConfig)
    
    # Add a layer
    layer1 = DefaultsLayer(
        name="layer1",
        priority=10,
        defaults={"string_value": "from_layer1"}
    )
    provider.add_layer(layer1)
    
    # Check that the value is applied
    assert provider.current.string_value == "from_layer1"
    
    # Add a higher priority layer
    layer2 = ConfigLayer(
        name="layer2",
        priority=20
    )
    layer2.set("string_value", "from_layer2")
    provider.add_layer(layer2)
    
    # Higher priority should win
    assert provider.current.string_value == "from_layer2"
    
    # Remove the higher priority layer
    provider.remove_layer("layer2")
    
    # Should fall back to layer1
    assert provider.current.string_value == "from_layer1"
    
    # Remove remaining layer
    provider.remove_layer("layer1")
    
    # Should fall back to model default
    assert provider.current.string_value == "default"


def test_get_section():
    """Test getting configuration sections."""
    # Create a provider with nested configuration
    layer = DefaultsLayer(
        name="test",
        priority=10,
        defaults={
            "string_value": "test",
            "nested": {
                "level1": {
                    "level2": "nested_value"
                },
                "sibling": 42
            }
        }
    )
    
    provider = ConfigurationProvider(
        model_type=TestConfig,
        layers=[layer]
    )
    
    # Get entire config
    assert provider.get_section([]) == provider.current
    
    # Get top-level value
    assert provider.get_section(["string_value"]) == "test"
    
    # Get nested dictionary
    nested = provider.get_section(["nested"])
    assert nested["level1"]["level2"] == "nested_value"
    assert nested["sibling"] == 42
    
    # Get deeply nested value
    assert provider.get_section(["nested", "level1", "level2"]) == "nested_value"
    
    # Get non-existent section
    assert provider.get_section(["nonexistent"]) is None


def test_update():
    """Test updating configuration values."""
    # Create a provider
    provider = ConfigurationProvider(model_type=TestConfig)
    
    # Update some values
    events = provider.update(
        updates={
            "string_value": "updated",
            "int_value": 999,
        },
        source="test_update"
    )
    
    # Check new values
    config = provider.current
    assert config.string_value == "updated"
    assert config.int_value == 999
    
    # Check events
    assert len(events) == 2
    
    string_event = next(e for e in events if e.path == ["string_value"])
    assert string_event.old_value == "default"
    assert string_event.new_value == "updated"
    assert string_event.source == "test_update"
    
    int_event = next(e for e in events if e.path == ["int_value"])
    assert int_event.old_value == 10
    assert int_event.new_value == 999
    assert int_event.source == "test_update"
    
    # Update a nested value
    events = provider.update(
        updates={"nested": {"key": "new_value", "new_key": "added"}},
        source="nested_update"
    )
    
    # Check the nested update
    assert provider.current.nested["key"] == "new_value"
    assert provider.current.nested["new_key"] == "added"
    
    # There should be one event for the entire nested update
    assert len(events) == 1
    assert events[0].path == ["nested"]
    assert events[0].source == "nested_update"


def test_subscribe():
    """Test subscribing to configuration updates."""
    provider = ConfigurationProvider(model_type=TestConfig)
    
    # Track received events
    received_events = []
    
    # Subscribe to all updates
    unsubscribe_all = provider.subscribe(
        lambda event: received_events.append(("all", event))
    )
    
    # Subscribe to specific key
    unsubscribe_specific = provider.subscribe(
        lambda event: received_events.append(("specific", event)),
        keys=["string_value"]
    )
    
    # Update multiple values
    provider.update(
        updates={
            "string_value": "will_trigger_both",
            "int_value": 42
        },
        source="test"
    )
    
    # Should have 3 events: all/string, all/int, specific/string
    assert len(received_events) == 3
    
    # Check specific subscription
    specific_events = [e for role, e in received_events if role == "specific"]
    assert len(specific_events) == 1
    assert specific_events[0].path == ["string_value"]
    assert specific_events[0].new_value == "will_trigger_both"
    
    # Check all subscription
    all_events = [e for role, e in received_events if role == "all"]
    assert len(all_events) == 2
    assert {e.path[0] for e in all_events} == {"string_value", "int_value"}
    
    # Test unsubscribe
    received_events.clear()
    unsubscribe_specific()
    
    # Update again
    provider.update(
        updates={"string_value": "after_specific_unsub"},
        source="test"
    )
    
    # Should only have all/string event
    assert len(received_events) == 1
    assert received_events[0][0] == "all"
    
    # Unsubscribe all
    received_events.clear()
    unsubscribe_all()
    
    # Update again
    provider.update(
        updates={"string_value": "after_all_unsub"},
        source="test"
    )
    
    # Should have no events
    assert len(received_events) == 0


def test_refresh_layer():
    """Test refreshing a configuration layer."""
    # Create a layer with values
    layer = ConfigLayer(name="test_layer", priority=10)
    layer.set("string_value", "initial")
    
    # Create provider with this layer
    provider = ConfigurationProvider(
        model_type=TestConfig,
        layers=[layer]
    )
    
    assert provider.current.string_value == "initial"
    
    # Mock layer refresh by directly modifying values
    layer._values["string_value"] = "refreshed"
    
    # Refresh the layer
    events = provider.refresh_layer("test_layer")
    
    # Check the value was updated
    assert provider.current.string_value == "refreshed"
    
    # Check events
    assert len(events) == 1
    assert events[0].path == ["string_value"]
    assert events[0].old_value == "initial"
    assert events[0].new_value == "refreshed"
    assert events[0].source == "test_layer"


def test_refresh_all():
    """Test refreshing all configuration layers."""
    # Create multiple layers
    layer1 = ConfigLayer(name="layer1", priority=10)
    layer1.set("string_value", "layer1_initial")
    
    layer2 = ConfigLayer(name="layer2", priority=20)
    layer2.set("int_value", 100)
    
    # Create provider with these layers
    provider = ConfigurationProvider(
        model_type=TestConfig,
        layers=[layer1, layer2]
    )
    
    assert provider.current.string_value == "layer1_initial"
    assert provider.current.int_value == 100
    
    # Mock layer refresh by directly modifying values
    layer1._values["string_value"] = "layer1_refreshed"
    layer2._values["int_value"] = 200
    
    # Refresh all layers
    events = provider.refresh_all()
    
    # Check values were updated
    assert provider.current.string_value == "layer1_refreshed"
    assert provider.current.int_value == 200
    
    # Check events
    assert len(events) == 2
    
    string_event = next(e for e in events if e.path == ["string_value"])
    assert string_event.old_value == "layer1_initial"
    assert string_event.new_value == "layer1_refreshed"
    
    int_event = next(e for e in events if e.path == ["int_value"])
    assert int_event.old_value == 100
    assert int_event.new_value == 200
