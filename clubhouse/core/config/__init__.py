"""
Hierarchical configuration system for the MCP framework.

This module provides a protocol-based approach to configuration management
with multiple layers, inheritance, and runtime updates.
"""
from typing import Optional

# Import Protocol interfaces
from clubhouse.core.config.protocol import (
    ConfigProtocol,
    ConfigLayerProtocol,
    ConfigUpdateEvent,
    ConfigUpdateCallback,
)

# Import Layer implementations
from clubhouse.core.config.layers import (
    ConfigLayer,
    DefaultsLayer,
    EnvironmentLayer,
    FileLayer,
)

# Import Provider
from clubhouse.core.config.provider import ConfigurationProvider

# Import configuration models
from clubhouse.core.config.models.log_level import LogLevel
from clubhouse.core.config.models.mcp_config import MCPConfig
from clubhouse.core.config.models.kafka_config import KafkaConfig
from clubhouse.core.config.models.schema_registry_config import SchemaRegistryConfig
from clubhouse.core.config.models.app_config import AppConfig
from clubhouse.core.config.models.loaders import (
    load_config_from_env,
    load_kafka_config_from_env,
    load_schema_registry_config_from_env,
    load_app_config_from_env,
)

# Singleton instance of the configuration provider
_config_provider: Optional[ConfigurationProvider] = None


def get_config_provider() -> Optional[ConfigurationProvider]:
    """
    Get the global configuration provider instance.
    
    Returns:
        Optional[ConfigurationProvider]: The global ConfigurationProvider instance, 
        or None if not initialized.
    """
    return _config_provider


def set_config_provider(provider: ConfigurationProvider) -> None:
    """
    Set the global configuration provider instance.
    
    Args:
        provider: The ConfigurationProvider instance to use globally.
    """
    global _config_provider
    _config_provider = provider


__all__ = [
    # Protocol interfaces
    "ConfigProtocol",
    "ConfigLayerProtocol",
    "ConfigUpdateEvent",
    "ConfigUpdateCallback",
    
    # Layer implementations
    "ConfigLayer",
    "DefaultsLayer",
    "EnvironmentLayer",
    "FileLayer",
    
    # Provider
    "ConfigurationProvider",
    
    # Models
    "LogLevel",
    "MCPConfig",
    "KafkaConfig",
    "SchemaRegistryConfig",
    "AppConfig",
    
    # Loader functions
    "load_config_from_env",
    "load_kafka_config_from_env",
    "load_schema_registry_config_from_env",
    "load_app_config_from_env",
    
    # Global provider access
    "get_config_provider",
    "set_config_provider",
]
