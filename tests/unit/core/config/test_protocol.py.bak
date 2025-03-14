"""Tests for configuration protocol interfaces."""
import pytest
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field

from clubhouse.core.config.protocol import ConfigProtocol, ConfigUpdateEvent


class TestConfig(BaseModel):
    """Test configuration model."""
    name: str = "default"
    value: int = 10
    nested: Dict[str, Any] = Field(default_factory=lambda: {"key": "value"})


class MockConfigProvider:
    """Mock implementation of the ConfigProtocol for testing."""
    
    def __init__(self, initial_config: Optional[Dict[str, Any]] = None):
        self._model_type = TestConfig
        self._config = TestConfig(**(initial_config or {}))
        self._subscribers = []
    
    @property
    def model(self) -> Type[TestConfig]:
        """Get the Pydantic model type for this configuration."""
        return self._model_type
    
    @property
    def current(self) -> TestConfig:
        """Get the current configuration value."""
        return self._config
    
    def get_section(self, section_path: List[str]) -> Any:
        """Get a specific section of the configuration."""
        if not section_path:
            return self._config
            
        current = self._config.model_dump()  
        for part in section_path:
            if part not in current:
                return None
            current = current[part]
        return current
    
    def update(self, updates: Dict[str, Any], source: str) -> List[ConfigUpdateEvent]:
        """Update configuration with new values."""
        events = []
        old_config = self._config.model_dump()  
        
        # Create a new config with updates
        new_config = self._config.model_copy(update=updates)  
        self._config = new_config
        
        # Generate events for each updated field
        for key, value in updates.items():
            events.append(
                ConfigUpdateEvent(
                    path=[key],
                    old_value=old_config.get(key),
                    new_value=value,
                    source=source
                )
            )
            
        # Notify subscribers
        for subscriber in self._subscribers:
            subscriber(events)
            
        return events
    
    def subscribe(self, handler):
        """Subscribe to configuration updates."""
        self._subscribers.append(handler)
        return lambda: self._subscribers.remove(handler)


def test_config_protocol_implementation():
    """Test that MockConfigProvider correctly implements the ConfigProtocol."""
    # This test verifies that MockConfigProvider properly implements the ConfigProtocol
    provider = MockConfigProvider()
    
    # Check that provider implements required ConfigProtocol attributes/methods
    assert hasattr(provider, "model")
    assert hasattr(provider, "current")
    assert hasattr(provider, "get_section")
    assert hasattr(provider, "update")
    assert hasattr(provider, "subscribe")


def test_config_current():
    """Test getting the current configuration."""
    initial_config = {"name": "test", "value": 42}
    provider = MockConfigProvider(initial_config)
    
    # Access current config and verify it matches the initial values
    config = provider.current
    assert config.name == "test"
    assert config.value == 42
    assert config.nested["key"] == "value"


def test_config_update():
    """Test updating configuration values."""
    provider = MockConfigProvider()
    
    # Verify initial values
    assert provider.current.name == "default"
    assert provider.current.value == 10
    
    # Update values
    updates = {"name": "updated", "value": 99}
    events = provider.update(updates, "test")
    
    # Verify updates were applied
    assert provider.current.name == "updated"
    assert provider.current.value == 99
    
    # Verify events were generated correctly
    assert len(events) == 2
    
    name_event = next(e for e in events if e.path == ["name"])
    assert name_event.old_value == "default"
    assert name_event.new_value == "updated"
    assert name_event.source == "test"
    
    value_event = next(e for e in events if e.path == ["value"])
    assert value_event.old_value == 10
    assert value_event.new_value == 99
    assert value_event.source == "test"


def test_config_get_section():
    """Test getting a specific section of the configuration."""
    initial_config = {
        "name": "test", 
        "value": 42,
        "nested": {
            "key": "custom",
            "extra": {
                "deep": "value"
            }
        }
    }
    provider = MockConfigProvider(initial_config)
    
    # Get root section
    root = provider.get_section([])
    assert root.name == "test"
    assert root.value == 42
    
    # Get nested section
    nested = provider.get_section(["nested"])
    assert nested == {"key": "custom", "extra": {"deep": "value"}}
    
    # Get deep nested value
    deep = provider.get_section(["nested", "extra", "deep"])
    assert deep == "value"
    
    # Get non-existent section
    missing = provider.get_section(["missing"])
    assert missing is None


def test_config_subscribe():
    """Test subscribing to configuration updates."""
    provider = MockConfigProvider()
    received_events = []
    
    # Define a subscription handler
    def handler(events):
        received_events.extend(events)
    
    # Subscribe to updates
    unsubscribe = provider.subscribe(handler)
    
    # Update configuration and verify handler was called
    provider.update({"name": "updated"}, "test")
    assert len(received_events) == 1
    assert received_events[0].path == ["name"]
    assert received_events[0].old_value == "default"
    assert received_events[0].new_value == "updated"
    
    # Unsubscribe and verify handler is no longer called
    unsubscribe()
    provider.update({"value": 999}, "test")
    assert len(received_events) == 1  # Still only the first event
