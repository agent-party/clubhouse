# ADR 003: Error Handling and Resilience Patterns

## Status

Proposed

## Context

Enterprise-grade frameworks require robust error handling and resilience mechanisms to maintain system stability under various failure conditions. Our current implementation lacks a standardized approach to error handling, classification, and recovery. We need an error handling system that:

1. Provides a consistent approach across all components
2. Classifies errors to enable appropriate responses
3. Includes contextual information for debugging
4. Implements resilience patterns for fault tolerance
5. Supports observability to detect and diagnose issues

## Decision

We will implement a structured error handling and resilience architecture with the following characteristics:

1. **Error Hierarchy**: Define a structured hierarchy of exception classes
2. **Error Context**: Include contextual information in exceptions
3. **Resilience Patterns**: Implement circuit breaker, retry, and bulkhead patterns
4. **Error Boundary**: Create error boundary mechanisms to contain failures
5. **Observability Hooks**: Provide hooks for logging and monitoring errors

## Implementation Details

### Error Hierarchy

```python
from typing import Dict, Any, Optional, List, Type
import enum

class ErrorSeverity(enum.Enum):
    """Classification of error severity."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(enum.Enum):
    """Classification of error types."""
    VALIDATION = "validation"  # Input validation errors
    AUTHENTICATION = "authentication"  # Auth failures
    AUTHORIZATION = "authorization"  # Permission issues
    RESOURCE = "resource"  # Resource not found/unavailable
    DEPENDENCY = "dependency"  # External dependency failures
    TIMEOUT = "timeout"  # Operation timeouts
    CONFIGURATION = "configuration"  # Config issues
    INTERNAL = "internal"  # Unexpected internal errors
    THROTTLING = "throttling"  # Rate limiting issues

class BaseFrameworkError(Exception):
    """Base class for all framework exceptions."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.cause = cause
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a dictionary representation."""
        result = {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
        }
        if self.cause:
            result["cause"] = str(self.cause)
        return result

# Specific error types
class ValidationError(BaseFrameworkError):
    """Error raised when input validation fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(
            message, 
            category=ErrorCategory.VALIDATION, 
            severity=ErrorSeverity.WARNING,
            context=context,
            cause=cause
        )

class DependencyError(BaseFrameworkError):
    """Error raised when an external dependency fails."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(
            message, 
            category=ErrorCategory.DEPENDENCY, 
            severity=ErrorSeverity.ERROR,
            context=context,
            cause=cause
        )
```

### Error Context Capture

To ensure errors have sufficient context for debugging:

```python
from functools import wraps
import traceback
import inspect
from typing import Callable, TypeVar, cast

T = TypeVar('T')

def with_error_context(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add context to errors raised from a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseFrameworkError as e:
            # Already a framework error, just add call context
            e.context.update({
                "function": func.__name__,
                "module": func.__module__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            raise
        except Exception as e:
            # Convert to framework error with context
            raise BaseFrameworkError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                cause=e,
                context={
                    "function": func.__name__,
                    "module": func.__module__,
                    "traceback": traceback.format_exc(),
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
            )
    return cast(Callable[..., T], wrapper)
```

### Resilience Patterns

#### Circuit Breaker

```python
import time
import asyncio
from enum import Enum, auto
from typing import Callable, TypeVar, Generic, Optional, Dict, Any, Awaitable, Union, cast

T = TypeVar('T')
R = TypeVar('R')

class CircuitState(Enum):
    """States for the circuit breaker pattern."""
    CLOSED = auto()  # Normal operation, requests pass through
    OPEN = auto()    # Circuit is open, requests fail fast
    HALF_OPEN = auto()  # Testing if service is recovered

class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5      # Number of failures before opening
    recovery_timeout: float = 30.0  # Seconds to wait before half-open
    test_calls: int = 1             # Calls to allow during half-open
    
    # Optional list of exception types to count as failures
    failure_exceptions: List[Type[Exception]] = []

class CircuitBreaker(Generic[T, R]):
    """Implementation of the circuit breaker pattern."""
    
    def __init__(self, 
                 name: str,
                 config: CircuitBreakerConfig = CircuitBreakerConfig()):
        self._name = name
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._successful_test_calls = 0
    
    async def execute(self, 
                      func: Callable[..., Awaitable[R]], 
                      *args, **kwargs) -> R:
        """Execute a function with circuit breaker protection."""
        self._check_state_transition()
        
        if self._state == CircuitState.OPEN:
            raise DependencyError(
                f"Circuit '{self._name}' is open, failing fast",
                context={
                    "circuit": self._name,
                    "state": self._state.name,
                    "failures": self._failure_count,
                    "last_failure": self._last_failure_time
                }
            )
        
        try:
            result = await func(*args, **kwargs)
            
            # If we're half-open and succeeded, record the success
            if self._state == CircuitState.HALF_OPEN:
                self._successful_test_calls += 1
                if self._successful_test_calls >= self._config.test_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._successful_test_calls = 0
            
            return result
            
        except Exception as e:
            should_count = (
                not self._config.failure_exceptions or  # Count all exceptions if list is empty
                any(isinstance(e, ex_type) for ex_type in self._config.failure_exceptions)
            )
            
            if should_count:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                if (self._state == CircuitState.CLOSED and 
                    self._failure_count >= self._config.failure_threshold):
                    self._state = CircuitState.OPEN
            
            raise
    
    def _check_state_transition(self) -> None:
        """Check if the circuit state should transition based on time."""
        if (self._state == CircuitState.OPEN and 
            time.time() - self._last_failure_time >= self._config.recovery_timeout):
            self._state = CircuitState.HALF_OPEN
            self._successful_test_calls = 0
```

#### Retry Policy

```python
import random
import asyncio
from typing import TypeVar, Callable, Awaitable, Optional, Type, List, Union

T = TypeVar('T')

class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_exceptions: List[Type[Exception]] = []  # Empty means retry all exceptions

class RetryPolicy:
    """Implementation of retry policy with backoff."""
    
    def __init__(self, config: RetryConfig = RetryConfig()):
        self._config = config
    
    async def execute(self, 
                      func: Callable[..., Awaitable[T]], 
                      *args, **kwargs) -> T:
        """Execute a function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception type
                should_retry = (
                    not self._config.retry_exceptions or
                    any(isinstance(e, ex_type) for ex_type in self._config.retry_exceptions)
                )
                
                if not should_retry or attempt == self._config.max_attempts:
                    raise
                
                # Calculate backoff delay
                delay = min(
                    self._config.base_delay * (self._config.backoff_factor ** (attempt - 1)),
                    self._config.max_delay
                )
                
                # Add jitter if configured
                if self._config.jitter:
                    delay = delay * (0.5 + random.random())
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # This should never be reached due to the raise in the loop
        assert last_exception is not None
        raise last_exception
```

### Error Boundary

To contain errors and provide graceful degradation:

```python
from typing import TypeVar, Callable, Awaitable, Optional, Any, Dict, Union, Generic

T = TypeVar('T')
F = TypeVar('F')  # Fallback type

class ErrorBoundary(Generic[T, F]):
    """
    Error boundary that contains failures and provides fallback mechanisms.
    Similar to React's error boundaries or a fallback pattern.
    """
    
    def __init__(self, 
                 fallback_value: Optional[F] = None,
                 fallback_factory: Optional[Callable[..., F]] = None,
                 on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None):
        """
        Initialize an error boundary.
        
        Args:
            fallback_value: Static fallback value to return on error
            fallback_factory: Function to call to create a fallback value on error
            on_error: Callback to execute when an error occurs
        """
        self._fallback_value = fallback_value
        self._fallback_factory = fallback_factory
        self._on_error = on_error
    
    async def execute(self, 
                      func: Callable[..., Awaitable[T]], 
                      *args, 
                      context: Optional[Dict[str, Any]] = None,
                      **kwargs) -> Union[T, F]:
        """
        Execute a function with error boundary protection.
        
        Args:
            func: Async function to execute
            *args: Arguments to pass to the function
            context: Additional context to provide on error
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Either the function result or the fallback value
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_context = context or {}
            error_context.update({
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
            })
            
            # Call the error handler if provided
            if self._on_error:
                try:
                    self._on_error(e, error_context)
                except Exception:
                    # Ignore errors in the error handler
                    pass
            
            # Return fallback value or create one
            if self._fallback_factory:
                return self._fallback_factory(e, error_context)
            return self._fallback_value
```

## Consequences

### Advantages

1. **Consistency**: Standardized approach to error handling across the system
2. **Contextual Information**: Errors include information needed for debugging
3. **Resilience**: Circuit breaker and retry patterns improve system stability
4. **Observability**: Structured errors can be properly logged and monitored
5. **Containment**: Error boundaries prevent cascading failures

### Disadvantages

1. **Complexity**: More complex than simple try/except blocks
2. **Performance**: Some overhead for context capture and decision logic
3. **Learning Curve**: Developers need to understand when to use which pattern

## Alternatives Considered

1. **Simple Exceptions**: Easier to implement but lacks classification and context
2. **Third-Party Libraries**: Could leverage resilience4j or similar, but adds dependencies
3. **No Structured Error Handling**: Simplest approach but inadequate for enterprise requirements

## References

- [Python Exception Handling](https://docs.python.org/3/tutorial/errors.html)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Retry Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/retry)
- [Bulkhead Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/bulkhead)
