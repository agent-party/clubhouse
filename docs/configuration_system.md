# Clubhouse Configuration System

## Overview

The Clubhouse project implements a robust, hierarchical configuration system designed to handle various configuration sources in a type-safe manner. The configuration system follows Protocol-based design principles and leverages Pydantic for validation.

## Configuration Structure

The configuration system is organized into the following structure:

```
clubhouse/core/config/
├── __init__.py                # Exports all configuration components
├── models/                    # Directory containing configuration models
│   ├── __init__.py            # Exports all models
│   ├── app_config.py          # Root application configuration
│   ├── kafka_config.py        # Kafka-specific configuration
│   ├── log_level.py           # LogLevel enum definition
│   ├── loaders.py             # Configuration loaders for environment variables
│   ├── mcp_config.py          # MCP server configuration
│   └── schema_registry_config.py # Schema Registry configuration
├── layers.py                  # Configuration layers implementation
├── protocol.py                # Protocol interfaces for configuration
└── provider.py                # Configuration provider implementation
```

## Key Components

### Configuration Protocol

Defines the service contract for all configuration providers.

### Configuration Layers

Implements a layered approach to configuration, allowing values to be sourced from different locations with priority.

### Configuration Provider

Orchestrates the loading and refreshing of configuration values across layers.

### Configuration Models

Pydantic models that define the schema and validation rules for each configuration type.

### LogLevel Enum

The LogLevel enum defines valid logging levels and ensures type safety:

```python
class LogLevel(str, Enum):
    """Valid log levels for the application."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
```

### Configuration Loaders

The `loaders.py` module provides functions to load configuration from environment variables:

- `load_config_from_env()`: Loads MCPConfig from environment variables
- `load_kafka_config_from_env()`: Loads KafkaConfig from environment variables
- `load_schema_registry_config_from_env()`: Loads SchemaRegistryConfig from environment variables

## Environment Variables

The configuration system supports loading values from environment variables:

| Environment Variable | Configuration Property | Default Value |
|----------------------|------------------------|---------------|
| `MCP_HOST` | `mcp.host` | 127.0.0.1 |
| `MCP_PORT` | `mcp.port` | 8000 |
| `MCP_LOG_LEVEL` | `mcp.log_level` | info |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka.bootstrap_servers` | localhost:9092 |
| `KAFKA_TOPIC_PREFIX` | `kafka.topic_prefix` | "" |
| `KAFKA_GROUP_ID` | `kafka.group_id` | "" |
| `KAFKA_CLIENT_ID` | `kafka.client_id` | "" |
| `SCHEMA_REGISTRY_URL` | `schema_registry.url` | http://localhost:8081 |
| `SCHEMA_REGISTRY_BASIC_AUTH` | `schema_registry.basic_auth_user_info` | "" |

## Usage Examples

### Basic Usage

```python
from clubhouse.core.config import configuration
from clubhouse.core.config.models.kafka_config import KafkaConfig

# Get the Kafka configuration
kafka_config = configuration.get(KafkaConfig)

# Use the configuration
kafka_bootstrap_servers = kafka_config.bootstrap_servers
```

### Working with Kafka Configuration

```python
from clubhouse.core.config import KafkaConfig, load_kafka_config_from_env

# Create Kafka configuration with default values
kafka_config = KafkaConfig()

# Create Kafka configuration with custom values
custom_kafka_config = KafkaConfig(
    bootstrap_servers="broker1:9092,broker2:9092",
    topic_prefix="test-",
    group_id="consumer-group-1",
    client_id="client-1"
)

# Load Kafka configuration from environment variables
env_kafka_config = load_kafka_config_from_env()
```

### Complete Application Configuration

```python
from clubhouse.core.config import AppConfig, MCPConfig, KafkaConfig, SchemaRegistryConfig

# Create a complete application configuration
app_config = AppConfig(
    mcp=MCPConfig(host="0.0.0.0", port=9000),
    kafka=KafkaConfig(bootstrap_servers="broker:9092"),
    schema_registry=SchemaRegistryConfig(url="http://schema-registry:8081")
)

# Access configuration sections
mcp_config = app_config.mcp
kafka_config = app_config.kafka
schema_registry_config = app_config.schema_registry
```

## Advanced Configuration

For more advanced configuration needs, the system implements a layered approach with protocols and providers as described in the architecture document [ADR-002-hierarchical-config.md](architecture/ADR-002-hierarchical-config.md).

The layered approach allows:
- Multiple configuration sources (defaults, files, environment, dynamic updates)
- Clear precedence rules
- Runtime configuration changes with validation
- Event-based notification for configuration changes

## Design Decisions

1. **Modular Organization**: Configuration models are organized in separate files for better maintainability
2. **Type Safety**: All configuration uses Pydantic models with proper validation
3. **Defaults First**: Sensible defaults are provided for all configuration options
4. **Environment Overrides**: Environment variables can override default values
5. **Extensibility**: The system can be extended with additional configuration sections as needed

## Architecture

The configuration system follows a layered architecture:

```
┌─────────────────┐
│ ConfigProtocol  │
└────────┬────────┘
         │
┌────────▼────────┐
│ ConfigProvider  │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Layers  │
    └────┬────┘
         │
  ┌──────▼──────┐
  │ Data Sources │
  └─────────────┘
```

Each layer represents a different source of configuration values, and they are applied in order of priority:

1. Default values (lowest priority)
2. Configuration files (YAML, JSON)
3. Environment variables
4. Command-line arguments (highest priority)

## Subscribing to Configuration Changes

The configuration system supports a publish-subscribe pattern for configuration changes:

```python
from clubhouse.core.config import configuration
from clubhouse.core.config.models.kafka_config import KafkaConfig

def on_config_change(old_config: KafkaConfig, new_config: KafkaConfig) -> None:
    print(f"Configuration changed: {old_config} -> {new_config}")

# Subscribe to configuration changes
unsubscribe = configuration.subscribe(KafkaConfig, on_config_change)

# Later, when no longer needed:
unsubscribe()
```

## Adding Custom Configuration Models

To add a custom configuration model:

1. Create a Pydantic model in the `clubhouse.core.config.models` package
2. Register it with the configuration system
3. Add appropriate environment variable mappings or default values

```python
from pydantic import BaseModel, Field

class MyCustomConfig(BaseModel):
    setting1: str = Field(default="default_value", description="My setting")
    setting2: int = Field(default=42, description="Another setting")
```

## Type Safety

All configuration components are fully type-annotated and checked with mypy to ensure type safety throughout the system. This helps catch configuration-related errors at development time rather than at runtime.
