"""Tests for configuration management."""
import pytest
import os
from typing import Dict, Any
from unittest.mock import patch

from pydantic import ValidationError


class TestMCPConfiguration:
    """Test cases for MCP configuration."""
    
    def test_mcp_config_model(self):
        """Test that the MCP config model has the expected fields and validation."""
        from clubhouse.core.config import MCPConfig
        
        # Check required fields
        fields = MCPConfig.model_fields
        assert "host" in fields
        assert "port" in fields
        assert "log_level" in fields
        
        # Test default values
        config = MCPConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.log_level.value == "info"  # Access enum value since it's now a proper enum
    
    def test_mcp_config_validation(self):
        """Test that the MCP config model validates input correctly."""
        from clubhouse.core.config import MCPConfig
        
        # Valid config
        valid_config = MCPConfig(host="0.0.0.0", port=9000, log_level="debug")
        assert valid_config.host == "0.0.0.0"
        assert valid_config.port == 9000
        assert valid_config.log_level.value == "debug"  # Access enum value
        
        # Invalid port (too high)
        with pytest.raises(ValidationError):
            MCPConfig(port=70000)
            
        # Invalid log level
        with pytest.raises(ValidationError):
            MCPConfig(log_level="invalid_level")
    
    def test_from_environment(self):
        """Test that config can be loaded from environment variables."""
        from clubhouse.core.config import MCPConfig, load_config_from_env
        
        # Setup test environment variables
        with patch.dict(os.environ, {
            "MCP_HOST": "localhost",
            "MCP_PORT": "5000",
            "MCP_LOG_LEVEL": "debug"
        }):
            config = load_config_from_env()
            
            assert isinstance(config, MCPConfig)
            assert config.host == "localhost"
            assert config.port == 5000
            assert config.log_level.value == "debug"  # Access enum value
    
    def test_from_environment_defaults(self):
        """Test that defaults are used when environment variables are not set."""
        from clubhouse.core.config import load_config_from_env
        
        # Clear relevant environment variables
        with patch.dict(os.environ, {
            "MCP_HOST": "",
            "MCP_PORT": "",
            "MCP_LOG_LEVEL": ""
        }, clear=True):
            config = load_config_from_env()
            
            assert config.host == "127.0.0.1"
            assert config.port == 8000
            assert config.log_level.value == "info"  # Access enum value
    
    def test_kafka_config(self):
        """Test the Kafka configuration model."""
        from clubhouse.core.config import KafkaConfig
        
        # Test default values
        config = KafkaConfig()
        assert config.bootstrap_servers == "localhost:9092"
        assert config.topic_prefix == ""
        
        # Test custom values
        custom_config = KafkaConfig(
            bootstrap_servers="broker1:9092,broker2:9092",
            topic_prefix="test-",
            group_id="test-group",
            client_id="test-client"
        )
        assert custom_config.bootstrap_servers == "broker1:9092,broker2:9092"
        assert custom_config.topic_prefix == "test-"
        assert custom_config.group_id == "test-group"
        assert custom_config.client_id == "test-client"
    
    def test_schema_registry_config(self):
        """Test the Schema Registry configuration model."""
        from clubhouse.core.config import SchemaRegistryConfig
        
        # Test default values
        config = SchemaRegistryConfig()
        assert config.url == "http://localhost:8081"
        
        # Test custom values
        custom_config = SchemaRegistryConfig(
            url="http://schema-registry:8081",
            basic_auth_user_info="user:pass"
        )
        assert custom_config.url == "http://schema-registry:8081"
        assert custom_config.basic_auth_user_info == "user:pass"
