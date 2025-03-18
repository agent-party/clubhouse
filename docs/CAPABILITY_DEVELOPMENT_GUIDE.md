# Capability Development Guide

This guide outlines the standardized patterns and best practices for developing agent capabilities in the Agent Orchestration Platform. Following these patterns ensures consistency, maintainability, and reliability across all capabilities.

## Table of Contents

1. [Capability Structure](#capability-structure)
2. [Parameter Validation](#parameter-validation)
3. [Lifecycle Management](#lifecycle-management)
4. [Error Handling](#error-handling)
5. [Event Handling](#event-handling)
6. [Cost Tracking](#cost-tracking)
7. [Response Formatting](#response-formatting)
8. [Testing Guidelines](#testing-guidelines)

## Capability Structure

All capabilities must inherit from `BaseCapability` and implement the required methods and properties.

### Required Properties

- `name` - String identifier for the capability (must be unique)
- `description` - Human-readable description of the capability's purpose
- `parameters_schema` - Pydantic model defining the expected parameters

### Required Methods

- `validate_parameters(**kwargs)` - Validate input parameters using Pydantic models
- `execute(**kwargs)` - Core implementation of the capability functionality
- `execute_and_handle_lifecycle(**kwargs)` or use `execute_with_lifecycle(**kwargs)` directly

### Example Skeleton

```python
from pydantic import BaseModel, Field
from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError

class MyCapabilityParameters(BaseModel):
    """Model for parameter validation."""
    param1: str = Field(..., description="Description of parameter 1")
    param2: int = Field(10, description="Description of parameter 2")

class MyCapability(BaseCapability):
    """My capability description."""
    
    name = "my_capability"
    description = "Detailed description of what this capability does"
    parameters_schema = MyCapabilityParameters
    
    def validate_parameters(self, **kwargs):
        """Validate input parameters."""
        try:
            params = self.parameters_schema(**kwargs)
            return params.model_dump()
        except Exception as e:
            raise ValidationError(f"Invalid parameters: {str(e)}", self.name)
            
    async def execute(self, **kwargs):
        """Execute the capability."""
        # Implementation of the capability
        # Should return a CapabilityResult
        
    async def execute_and_handle_lifecycle(self, **kwargs):
        """Execute with lifecycle handling."""
        return await self.execute_with_lifecycle(**kwargs)
```

## Parameter Validation

All capabilities must use Pydantic models for parameter validation to ensure type safety and proper validation.

### Best Practices

1. Define a dedicated Pydantic model for the capability's parameters
2. Use Field annotations to provide descriptions and constraints
3. Handle validation errors consistently by converting them to ValidationError
4. Never use manual parameter validation - always leverage Pydantic

### Example

```python
class SearchParameters(BaseModel):
    """Model for search parameters validation."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(5, description="Maximum number of results to return")
    sources: list[str] = Field(
        default_factory=lambda: ["knowledge_base"],
        description="Sources to search in"
    )

def validate_parameters(self, **kwargs):
    """Validate the parameters for the capability."""
    try:
        # Special handling for parameters that need conversion
        if "max_results" in kwargs and isinstance(kwargs["max_results"], str):
            try:
                kwargs["max_results"] = int(kwargs["max_results"])
            except ValueError:
                raise ValidationError(
                    f"Invalid max_results parameter: cannot convert '{kwargs['max_results']}' to integer",
                    self.name
                )
                
        # Use Pydantic model for validation
        params = self.parameters_schema(**kwargs)
        return params.model_dump()
    except PydanticValidationError as e:
        # Convert Pydantic validation errors to our ValidationError
        raise ValidationError(f"Invalid parameters: {str(e)}", self.name)
```

## Lifecycle Management

Capabilities should leverage the `execute_with_lifecycle` method from `BaseCapability` to ensure consistent lifecycle management and event triggering.

### Implementation Options

1. **Recommended Approach**: Create an `execute_and_handle_lifecycle` method that delegates to `execute_with_lifecycle`
2. **Alternative**: Call `execute_with_lifecycle` directly when appropriate

### Example

```python
async def execute_and_handle_lifecycle(self, **kwargs):
    """Execute the capability with full lifecycle handling."""
    return await self.execute_with_lifecycle(**kwargs)
```

## Error Handling

Capabilities must use the centralized error framework for consistent error handling.

### Error Types

- `ValidationError` - For parameter validation failures
- `ExecutionError` - For failures during capability execution

### Best Practices

1. Catch and wrap specific exceptions with appropriate error types
2. Include meaningful error messages with context
3. Log errors with appropriate severity levels
4. Return consistent error response structures

### Example

```python
async def execute(self, **kwargs):
    """Execute the capability."""
    try:
        # Implementation
        return CapabilityResult(
            result={"status": "success", "data": result},
            metadata={"cost": self.get_operation_cost()}
        )
    except Exception as e:
        error_message = f"Operation failed: {str(e)}"
        logger.error(error_message)
        
        return CapabilityResult(
            result={"status": "error", "error": error_message},
            metadata={
                "cost": self.get_operation_cost(),
                "error_type": type(e).__name__
            }
        )
```

## Event Handling

Capabilities must trigger standard events during execution to allow for proper monitoring and extension.

### Standard Events

- `before_execution` - Triggered before capability execution
- `after_execution` - Triggered after successful execution
- `execution_error` - Triggered when an error occurs

### Legacy/Custom Events

Some capabilities may need to maintain backward compatibility with custom events. In these cases:

1. Trigger both standard events and legacy events
2. Document the custom events clearly

### Example

```python
# When using execute_with_lifecycle, events are triggered automatically

# When manually triggering events:
self.trigger_event("before_execution", params=params)
# Legacy/custom events for backward compatibility
self.trigger_event("my_capability_started", param1=param1, param2=param2)
```

## Cost Tracking

Capabilities must implement consistent cost tracking to enable proper accounting.

### Best Practices

1. Reset costs at the beginning of execution
2. Track detailed costs for different operations
3. Include total cost in the result metadata

### Example

```python
def reset_operation_cost(self):
    """Reset the operation cost tracking."""
    self._operation_costs = {}
    
def add_operation_cost(self, operation, cost):
    """Add a cost for a specific operation."""
    if operation in self._operation_costs:
        self._operation_costs[operation] += cost
    else:
        self._operation_costs[operation] = cost
        
def get_operation_cost(self):
    """Get the operation costs."""
    costs = self._operation_costs.copy()
    total = round(sum(costs.values()), 2) if costs else 0
    costs["total"] = total
    return costs
```

## Response Formatting

Capabilities must return responses in a consistent format to ensure compatibility with the platform.

### Response Structure

- For successful operations:
  ```python
  {
      "status": "success",
      "result_data": {...},  # Capability-specific result data
      "metadata": {
          "execution_time": 0.123,  # Optional execution metrics
          "other_metadata": "..."
      }
  }
  ```

- For errors:
  ```python
  {
      "status": "error",
      "error": "Error message",
      "metadata": {
          "error_type": "ValidationError",
          "cost": {"total": 0.0}
      }
  }
  ```

### Best Practices

1. Always include a "status" field with either "success" or "error"
2. For errors, include a descriptive error message and type
3. Include relevant metadata for both success and error cases
4. Structure results consistently for all capabilities

## Testing Guidelines

Capabilities should be thoroughly tested following Test-Driven Development principles.

### Test Coverage Requirements

1. Parameter validation (both success and failure cases)
2. Core functionality execution
3. Error handling
4. Event triggering
5. Cost tracking
6. Lifecycle management

### Example Test Structure

```python
def test_validation_success(self):
    """Test parameter validation with valid parameters."""
    
def test_validation_failure(self):
    """Test parameter validation with invalid parameters."""
    
@pytest.mark.asyncio
async def test_execution_success(self):
    """Test successful execution."""
    
@pytest.mark.asyncio
async def test_execution_error(self):
    """Test error handling during execution."""
    
@pytest.mark.asyncio
async def test_event_triggering(self):
    """Test event triggering during execution."""
    
def test_cost_tracking(self):
    """Test operation cost tracking."""
```

## Migrating Existing Capabilities

When updating existing capabilities to follow these standards:

1. Maintain backward compatibility with existing tests and clients
2. Implement the new patterns alongside legacy behavior where needed
3. Add tests for the new patterns
4. Document changes and migration path

By following these guidelines, you ensure all capabilities in the Agent Orchestration Platform are consistent, maintainable, and reliable.
