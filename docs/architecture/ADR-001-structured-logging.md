# ADR 001: Structured Logging Architecture

## Status

Proposed

## Context

As we build an enterprise-grade framework, we need a standardized approach to logging that supports modern observability requirements. Our logging system needs to:

1. Support structured data in a machine-readable format
2. Provide context propagation across service boundaries
3. Enable log correlation with trace IDs
4. Allow for flexible configuration of log levels, formats, and outputs
5. Maintain good performance characteristics

## Decision

We will implement a structured logging architecture with the following characteristics:

1. **`StructuredLogger` Protocol**: Define a protocol interface that specifies the contract for all logger implementations
2. **JSON-formatted logs**: All logs will be structured as JSON for machine readability
3. **Context propagation**: Use contextvars to propagate context (trace ID, etc.) across async boundaries
4. **Separation of concerns**: 
   - Log creation (generating structured log data)
   - Log transport (writing to files, sending to external systems)
   - Log configuration (controlling levels, formats)
5. **Performance optimization**: Lazy evaluation of expensive operations

## Implementation Details

### `LoggingProtocol` Interface

```python
from typing import Protocol, Dict, Any, Optional, Union
import enum

class LogLevel(enum.IntEnum):
    """Log levels with numeric values matching standard Python logging."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class LoggingProtocol(Protocol):
    """Protocol defining the interface for structured loggers."""
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get the current logger context."""
        ...
    
    def with_context(self, **context_updates: Any) -> "LoggingProtocol":
        """Create a new logger with updated context."""
        ...
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        ...
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        ...
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        ...
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        ...
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        ...
```

### Context Propagation

We will use Python's `contextvars` module to propagate context across async boundaries:

```python
import contextvars
from typing import Dict, Any

# Global context variable
log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)
```

### Configuration Model

We will use Pydantic for configuration:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict

class LogHandlerConfig(BaseModel):
    """Configuration for a log handler."""
    type: str = Field(..., description="Handler type (file, console, etc.)")
    level: str = Field("INFO", description="Minimum log level for this handler")
    format: str = Field("json", description="Log format (json, text)")
    
    # Handler-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict)

class LoggingConfig(BaseModel):
    """Configuration for the logging system."""
    default_level: str = Field("INFO", description="Default log level")
    propagate_context: bool = Field(True, description="Propagate context across boundaries")
    include_timestamps: bool = Field(True, description="Include timestamps in logs")
    handlers: List[LogHandlerConfig] = Field(
        default_factory=lambda: [
            LogHandlerConfig(type="console", level="INFO", format="json")
        ]
    )
```

## Consequences

### Advantages

1. **Consistency**: All components will use the same logging approach
2. **Observability**: Structured logs enable better search and analysis
3. **Correlation**: Context propagation allows tracing requests across components
4. **Flexibility**: Configuration options allow adapting to different environments
5. **Performance**: Lazy evaluation minimizes impact on hot paths

### Disadvantages

1. **Complexity**: More complex than simple unstructured logging
2. **Learning curve**: Requires developers to understand the system
3. **Overhead**: Some performance impact compared to no logging

## Alternatives Considered

1. **Use an existing logging library**: While libraries like structlog exist, implementing our own gives us more control and alignment with our architectural patterns.
2. **Unstructured logging**: Simpler but lacks the benefits of structured data.
3. **External logging service**: Would add external dependencies we want to avoid at this stage.

## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Structlog](https://www.structlog.org/en/stable/)
- [OpenTelemetry Logging](https://opentelemetry.io/docs/reference/specification/logs/overview/)
