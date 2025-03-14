"""
Configuration loader functions.

This module provides functions for loading configuration from environment variables.
"""
import os
from typing import Optional, Dict, Any, cast

from mcp_demo.core.config.models.mcp_config import MCPConfig
from mcp_demo.core.config.models.kafka_config import KafkaConfig
from mcp_demo.core.config.models.schema_registry_config import SchemaRegistryConfig
from mcp_demo.core.config.models.app_config import AppConfig
from mcp_demo.core.config.models.log_level import LogLevel


def load_config_from_env() -> MCPConfig:
    """
    Load MCP configuration from environment variables.
    
    Environment variables:
        MCP_HOST: Host to bind MCP server to
        MCP_PORT: Port to bind MCP server to
        MCP_LOG_LEVEL: Log level for MCP server
        MCP_TIMEOUT_SECONDS: Timeout for MCP operations in seconds
    
    Returns:
        MCPConfig: Configuration object with values from environment variables or defaults
    """
    host = os.environ.get("MCP_HOST")
    port_str = os.environ.get("MCP_PORT")
    log_level_str = os.environ.get("MCP_LOG_LEVEL")
    timeout_str = os.environ.get("MCP_TIMEOUT_SECONDS")
    
    # Build kwargs with only the values that are set
    kwargs: Dict[str, Any] = {}
    if host:
        kwargs["host"] = host
    if port_str:
        try:
            kwargs["port"] = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port number: {port_str}")
    if log_level_str:
        kwargs["log_level"] = log_level_str
    if timeout_str:
        try:
            kwargs["timeout_seconds"] = float(timeout_str)
        except ValueError:
            raise ValueError(f"Invalid timeout seconds: {timeout_str}")
            
    return MCPConfig(**kwargs)


def load_kafka_config_from_env() -> KafkaConfig:
    """
    Load Kafka configuration from environment variables.
    
    Environment variables:
        KAFKA_BOOTSTRAP_SERVERS: Comma-separated list of Kafka broker addresses
        KAFKA_TOPIC_PREFIX: Prefix for all Kafka topics
        KAFKA_GROUP_ID: Consumer group ID
        KAFKA_CLIENT_ID: Client ID for Kafka producer/consumer
        KAFKA_AUTO_OFFSET_RESET: Auto offset reset policy (earliest, latest, none)
    
    Returns:
        KafkaConfig: Configuration object with values from environment variables or defaults
    """
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    topic_prefix = os.environ.get("KAFKA_TOPIC_PREFIX")
    group_id = os.environ.get("KAFKA_GROUP_ID")
    client_id = os.environ.get("KAFKA_CLIENT_ID")
    auto_offset_reset = os.environ.get("KAFKA_AUTO_OFFSET_RESET")
    
    # Build kwargs with only the values that are set
    kwargs: Dict[str, Any] = {}
    if bootstrap_servers:
        kwargs["bootstrap_servers"] = bootstrap_servers
    if topic_prefix:
        kwargs["topic_prefix"] = topic_prefix
    if group_id:
        kwargs["group_id"] = group_id
    if client_id:
        kwargs["client_id"] = client_id
    if auto_offset_reset:
        kwargs["auto_offset_reset"] = auto_offset_reset
            
    return KafkaConfig(**kwargs)


def load_schema_registry_config_from_env() -> SchemaRegistryConfig:
    """
    Load Schema Registry configuration from environment variables.
    
    Environment variables:
        SCHEMA_REGISTRY_URL: URL of the Schema Registry server
        SCHEMA_REGISTRY_BASIC_AUTH: Basic auth credentials in format username:password
    
    Returns:
        SchemaRegistryConfig: Configuration object with values from environment variables or defaults
    """
    url = os.environ.get("SCHEMA_REGISTRY_URL")
    basic_auth = os.environ.get("SCHEMA_REGISTRY_BASIC_AUTH")
    
    # Build kwargs with only the values that are set
    kwargs: Dict[str, Any] = {}
    if url:
        kwargs["url"] = url
    if basic_auth:
        kwargs["basic_auth_user_info"] = basic_auth
            
    return SchemaRegistryConfig(**kwargs)


def load_app_config_from_env() -> AppConfig:
    """
    Load all application configuration from environment variables.
    
    Returns:
        AppConfig: Complete application configuration object with all sections
    """
    return AppConfig(
        mcp=load_config_from_env(),
        kafka=load_kafka_config_from_env(),
        schema_registry=load_schema_registry_config_from_env()
    )
