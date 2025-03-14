# ADR 002: Hierarchical Configuration Architecture

## Status

Proposed

## Context

Enterprise frameworks require sophisticated configuration management that supports multiple sources, hierarchical inheritance, and runtime updates. Our current configuration system uses Pydantic models but lacks hierarchical capabilities and dynamic updates. We need a configuration system that:

1. Supports multiple layers (defaults, files, environment, dynamic)
2. Enables component-specific configuration that inherits from global settings
3. Provides type safety and validation at all levels
4. Allows runtime updates with proper validation
5. Securely handles sensitive configuration values

## Decision

We will implement a hierarchical configuration architecture with the following characteristics:

1. **`ConfigProtocol` Interface**: Define a protocol that specifies the contract for all config providers
2. **Layered Configuration**: Support multiple layers with clear precedence rules
3. **Component Configuration**: Enable components to define their own config schema while inheriting from parents
4. **Dynamic Updates**: Support runtime configuration changes with validation
5. **Event-Based Notification**: Allow components to subscribe to configuration changes
6. **Secure Storage**: Provide secure handling of sensitive configuration values

## Implementation Details

### `ConfigProtocol` Interface

```python
from typing import Protocol, Dict, Any, Optional, TypeVar, Generic, Type, List, Callable
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class ConfigUpdateEvent(BaseModel):
    """Event fired when configuration is updated."""
    path: List[str]  # Path to the updated config section (e.g., ["kafka", "bootstrap_servers"])
    old_value: Optional[Any] = None
    new_value: Any
    source: str  # Source of the update (e.g., "env", "file", "api")

class ConfigProtocol(Protocol, Generic[T]):
    """Protocol defining the interface for configuration providers."""
    
    @property
    def model(self) -> Type[T]:
        """Get the Pydantic model type for this configuration."""
        ...
    
    @property
    def current(self) -> T:
        """Get the current configuration value."""
        ...
    
    def get_section(self, section_path: List[str]) -> Any:
        """Get a specific section of the configuration."""
        ...
    
    def update(self, updates: Dict[str, Any], source: str) -> List[ConfigUpdateEvent]:
        """Update configuration with new values."""
        ...
    
    def subscribe(self, callback: Callable[[ConfigUpdateEvent], None]) -> Callable[[], None]:
        """Subscribe to configuration updates. Returns an unsubscribe function."""
        ...
```

### Layered Configuration

We'll implement a layered configuration approach with clear precedence:

1. **Default values**: Defined in Pydantic models
2. **Configuration files**: YAML/JSON config files
3. **Environment variables**: Override from environment
4. **Dynamic updates**: Runtime updates via API

Each layer will override values from the previous layers, with validation at each step.

### Component Configuration

Components will define their own configuration models that can inherit values from parent configs:

```python
from pydantic import BaseModel, Field
from typing import Optional

class BaseComponentConfig(BaseModel):
    """Base configuration for all components."""
    enabled: bool = True
    log_level: str = "INFO"

class KafkaComponentConfig(BaseComponentConfig):
    """Kafka-specific configuration."""
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: Optional[str] = None
    
    @classmethod
    def from_parent(cls, parent_config: BaseComponentConfig, **overrides):
        """Create from parent config with overrides."""
        # Start with parent values for shared fields
        values = {
            "enabled": parent_config.enabled,
            "log_level": parent_config.log_level
        }
        # Apply any overrides
        values.update(overrides)
        return cls(**values)
```

### Configuration Manager

We'll implement a central configuration manager to coordinate all config operations:

```python
class ConfigManager:
    """Central manager for hierarchical configuration."""
    
    def __init__(self, base_config: BaseModel):
        self._base_config = base_config
        self._component_configs = {}
        self._subscribers = []
        self._sources = {
            "default": self._load_defaults(),
            "file": {},
            "env": {},
            "dynamic": {}
        }
    
    def get_component_config(self, component_name: str, config_type: Type[T]) -> T:
        """Get or create a component-specific configuration."""
        if component_name not in self._component_configs:
            # Create component config inheriting from base
            self._component_configs[component_name] = config_type.from_parent(
                self._base_config
            )
        return self._component_configs[component_name]
    
    def update_from_source(self, updates: Dict[str, Any], source: str) -> List[ConfigUpdateEvent]:
        """Update configuration from a specific source."""
        # Implementation details omitted for brevity
        # Updates the appropriate source dictionary and recalculates merged config
        pass
```

### Secure Configuration

For sensitive values:

```python
from pydantic import SecretStr, Field

class SecurityConfig(BaseModel):
    """Security configuration with sensitive values."""
    api_key: SecretStr = Field(..., description="API key for external service")
    
    # When converting to dict, ensure secrets are handled properly
    def dict(self, *args, **kwargs):
        exclude_secrets = kwargs.pop("exclude_secrets", False)
        result = super().dict(*args, **kwargs)
        if exclude_secrets:
            for key, value in result.items():
                if isinstance(getattr(self, key), SecretStr):
                    result[key] = "***REDACTED***"
        return result
```

## Consequences

### Advantages

1. **Flexibility**: Multiple configuration sources with clear precedence
2. **Type Safety**: Full validation through Pydantic models
3. **Modularity**: Components can define their own configuration needs
4. **Runtime Updates**: Configuration can change while the system is running
5. **Security**: Sensitive values are properly protected

### Disadvantages

1. **Complexity**: More complex than simple flat configuration
2. **Performance**: Some overhead for validation and layer merging
3. **Learning Curve**: Developers need to understand the hierarchical approach

## Alternatives Considered

1. **Environment Variables Only**: Simpler but lacks hierarchical support and type safety
2. **Config Files Only**: Lacks dynamic update capabilities
3. **Third-Party Libraries**: Adding external dependencies increases complexity

## References

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [12-Factor App: Config](https://12factor.net/config)
- [Python contextvars](https://docs.python.org/3/library/contextvars.html)
