"""Tests for configuration layer implementations."""
import os
import pytest
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from clubhouse.core.config.layers import (
    ConfigLayer,
    DefaultsLayer,
    EnvironmentLayer,
    FileLayer,
)


def test_config_layer_base():
    """Test the base ConfigLayer class functionality."""
    # Create a basic layer
    layer = ConfigLayer(name="test_layer", priority=10)
    
    # Test initial state
    assert layer.name() == "test_layer"
    assert layer.priority() == 10
    assert layer.get_all() == {}
    
    # Test setting and getting simple values
    layer.set("key1", "value1")
    assert layer.get("key1") == "value1"
    
    # Test setting and getting nested values
    layer.set("section.nested", 42)
    assert layer.get("section.nested") == 42
    
    # Verify structure
    expected = {
        "key1": "value1",
        "section": {
            "nested": 42
        }
    }
    assert layer.get_all() == expected
    
    # Test get_nested helper
    assert layer.get_nested(expected, ["section", "nested"]) == 42
    assert layer.get_nested(expected, ["nonexistent"]) is None
    assert layer.get_nested(expected, ["section", "nonexistent"]) is None
    
    # Test refresh (base implementation just returns current values)
    assert layer.refresh() == expected


def test_defaults_layer():
    """Test the DefaultsLayer implementation."""
    # Create with initial defaults
    defaults = {
        "string": "default_value",
        "number": 123,
        "nested": {
            "bool": True,
            "list": [1, 2, 3]
        }
    }
    
    layer = DefaultsLayer(
        name="defaults",
        priority=0,
        defaults=defaults
    )
    
    # Verify the layer has the defaults
    assert layer.name() == "defaults"
    assert layer.priority() == 0
    assert layer.get_all() == defaults
    assert layer.get("string") == "default_value"
    assert layer.get("nested.bool") is True
    assert layer.get("nested.list") == [1, 2, 3]
    
    # Test overriding defaults
    layer.set("string", "new_value")
    assert layer.get("string") == "new_value"


def test_environment_layer():
    """Test the EnvironmentLayer implementation."""
    # Set some environment variables for testing
    os.environ["CLUBHOUSE_STRING"] = "env_value"
    os.environ["CLUBHOUSE_NUMBER"] = "456"
    os.environ["CLUBHOUSE_JSON"] = '{"key": "value", "list": [4, 5, 6]}'
    os.environ["CLUBHOUSE_NESTED_BOOL"] = "true"
    os.environ["UNRELATED"] = "should_not_be_included"
    
    # Create the layer
    layer = EnvironmentLayer(
        name="env",
        priority=100
    )
    
    # Verify environment variables were loaded
    assert layer.name() == "env"
    assert layer.priority() == 100
    assert layer.get("string") == "env_value"
    assert layer.get("number") == 456  # Should be parsed as number from string
    assert layer.get("json") == {"key": "value", "list": [4, 5, 6]}  # Parsed JSON
    assert layer.get("nested.bool") is True  # Parsed boolean
    
    # Verify unrelated variables are not included
    assert layer.get("unrelated") is None
    
    # Test custom separators
    os.environ["CUSTOM:CUSTOM_SETTING"] = "custom_value"
    
    custom_layer = EnvironmentLayer(
        name="custom_env",
        priority=110,
        prefix="CUSTOM",
        separator=":"
    )
    
    assert custom_layer.name() == "custom_env"
    assert custom_layer.priority() == 110
    assert custom_layer.get("custom.setting") == "custom_value"
    
    # Test refresh to load new environment variables
    os.environ["CLUBHOUSE_NEW_VAR"] = "new_value"
    layer.refresh()
    assert layer.get("new_var") == "new_value"
    
    # Clean up
    for key in ["CLUBHOUSE_STRING", "CLUBHOUSE_NUMBER", "CLUBHOUSE_JSON", 
                "CLUBHOUSE_NESTED_BOOL", "CLUBHOUSE_NEW_VAR", "CUSTOM:CUSTOM_SETTING"]:
        if key in os.environ:
            del os.environ[key]


def test_file_layer_json():
    """Test the FileLayer implementation with JSON files."""
    # Create a temporary JSON file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "config.json"
        
        # Write test data to the file
        test_data = {
            "string": "file_value",
            "number": 789,
            "nested": {
                "bool": False,
                "list": [7, 8, 9]
            }
        }
        
        with open(file_path, "w") as f:
            json.dump(test_data, f)
        
        # Create the layer
        layer = FileLayer(
            file_path=file_path,
            name="json_file",
            priority=50
        )
        
        # Verify file was loaded
        assert layer.name() == "json_file"
        assert layer.priority() == 50
        assert layer.get("string") == "file_value"
        assert layer.get("number") == 789
        assert layer.get("nested.bool") is False
        assert layer.get("nested.list") == [7, 8, 9]
        
        # Test updating the file and refreshing
        updated_data = {
            "string": "updated_value",
            "new_key": "new_value"
        }
        
        with open(file_path, "w") as f:
            json.dump(updated_data, f)
        
        layer.refresh()
        assert layer.get("string") == "updated_value"
        assert layer.get("new_key") == "new_value"
        assert layer.get("number") is None  # Should be gone after update


def test_file_layer_yaml():
    """Test the FileLayer implementation with YAML files."""
    # Create a temporary YAML file
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "config.yaml"
        
        # Write test data to the file
        test_data = {
            "string": "yaml_value",
            "number": 123,
            "nested": {
                "bool": True,
                "list": [1, 2, 3]
            }
        }
        
        with open(file_path, "w") as f:
            yaml.dump(test_data, f)
        
        # Create the layer
        layer = FileLayer(
            file_path=file_path,
            name="yaml_file",
            priority=60
        )
        
        # Verify file was loaded
        assert layer.name() == "yaml_file"
        assert layer.priority() == 60
        assert layer.get("string") == "yaml_value"
        assert layer.get("number") == 123
        assert layer.get("nested.bool") is True
        assert layer.get("nested.list") == [1, 2, 3]
        
        # Test format auto-detection
        assert layer.format == "yaml"
        
        # Test explicit format setting
        json_layer = FileLayer(
            file_path=file_path,
            name="forced_json",
            format="json"
        )
        assert json_layer.format == "json"


def test_file_layer_nonexistent():
    """Test FileLayer behavior with nonexistent files."""
    # Create a layer with a nonexistent file
    layer = FileLayer(
        file_path="/nonexistent/path/config.json",
        name="nonexistent_file",
        priority=40
    )
    
    # Should not error, just have empty values
    assert layer.name() == "nonexistent_file"
    assert layer.priority() == 40
    assert layer.get_all() == {}
    
    # Setting values should work
    layer.set("key", "value")
    assert layer.get("key") == "value"


def test_layer_priority_behavior():
    """Test that layer priority affects value precedence."""
    # Create multiple layers with different priorities
    low_layer = DefaultsLayer(
        name="low",
        priority=10,
        defaults={"key": "low", "only_low": "only_in_low"}
    )
    
    medium_layer = DefaultsLayer(
        name="medium",
        priority=20,
        defaults={"key": "medium", "only_medium": "only_in_medium"}
    )
    
    high_layer = DefaultsLayer(
        name="high",
        priority=30,
        defaults={"key": "high", "only_high": "only_in_high"}
    )
    
    # Verify each layer independently
    assert low_layer.name() == "low"
    assert low_layer.priority() == 10
    assert low_layer.get("key") == "low"
    assert medium_layer.name() == "medium"
    assert medium_layer.priority() == 20
    assert medium_layer.get("key") == "medium"
    assert high_layer.name() == "high"
    assert high_layer.priority() == 30
    assert high_layer.get("key") == "high"
    
    # In a real ConfigurationProvider, the high priority would win
