"""
Base capability implementation and standard capability patterns.

This module provides the core classes for implementing agent capabilities in a
standardized way, following the CapabilityProtocol interface. It includes a
robust BaseCapability class with common functionality and utilities for
capability development.
"""

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic, Union, cast, get_type_hints, Protocol, Type, Awaitable, AsyncIterator, Callable, Coroutine
from uuid import uuid4

from pydantic import ValidationError as PydanticValidationError, create_model, Field
import inspect
import time
from uuid import uuid4
from datetime import datetime
import traceback
import logging

# Import errors from the centralized error module
from clubhouse.agents.errors import (
    CapabilityError,
    ValidationError,
    ExecutionError
)
from clubhouse.agents.schemas import validate_capability_params

# Configure logger
logger = logging.getLogger(__name__)

# We're no longer defining these error classes here since they're imported
# from the errors module


class CapabilityResult:
    """Result of a capability execution.
    
    This class represents the result of a capability execution,
    including the result value and any metadata.
    """
    
    def __init__(self, result: Any, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a capability result.
        
        Args:
            result: The result value
            metadata: Optional metadata about the execution
        """
        self.result = result
        self.metadata = metadata or {}
        
    def __repr__(self) -> str:
        """Return string representation."""
        return f"CapabilityResult(result={self.result}, metadata={self.metadata})"


# Type variables for generic typing
P = TypeVar("P", bound=Dict[str, Any])  # Parameter type
R = TypeVar("R", bound=Dict[str, Any])  # Result type
CoroT = TypeVar('CoroT')  # TypeVar for coroutine return types


class EventCallback(Protocol):
    """Protocol for event callbacks."""
    async def __call__(self, **kwargs: Any) -> None:
        """Call the event callback."""
        ...


# Handler for fire-and-forget tasks to properly log exceptions
def _task_callback(task: asyncio.Task) -> None:
    """Callback to handle exceptions from fire-and-forget tasks.
    
    This function is used to prevent unhandled exceptions in background tasks
    from being silently ignored, which could lead to hard-to-debug issues.
    
    Args:
        task: The completed task to check for exceptions
    """
    try:
        # Get the result to handle exceptions
        if task.done() and not task.cancelled():
            task.result()
    except Exception as e:
        # Log the exception but don't let it propagate
        # We use print here as the logger might not be initialized yet
        logger.error(f"Unhandled exception in background task: {e}")


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    """
    Schedule a coroutine to run in the background without awaiting it.
    
    This function is used when we need to call an async function from a sync context
    and we don't need to wait for the result, such as for event handlers.
    
    Args:
        coro: The coroutine to run in the background
    """
    # Create the task and register an error handler
    # This pattern is recommended for fire-and-forget tasks
    task = asyncio.create_task(coro)
    task.add_done_callback(_task_callback)

    # The key fix: explicitly cast to None to tell mypy we're deliberately discarding the task
    _ = cast(None, task)


class BaseCapability(ABC):
    """
    Base implementation of the CapabilityProtocol with common functionality.
    
    This class provides implementations for common capability functionality like
    event handling, standardized responses, and error management. Concrete
    capability implementations should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, requires_human_approval: bool = False) -> None:
        """
        Initialize the capability with common properties.
        
        Args:
            requires_human_approval: Whether this capability requires human approval
                                     before execution
        """
        self._requires_approval = requires_human_approval
        self._event_handlers: Dict[str, List[EventCallback]] = {}
        self._operation_cost: Dict[str, float] = {}
        self._execution_history: List[Dict[str, Any]] = []
        
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the unique identifier for this capability.
        
        This name should be unique within an agent and is used for capability lookup
        and routing.
        
        Returns:
            The capability name as a string
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get a human-readable description of what this capability does.
        
        This description should clearly explain the purpose and function of the
        capability to both developers and end-users.
        
        Returns:
            Description string
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters required by this capability.
        
        This metadata is used for validation and documentation.
        
        Returns:
            Dictionary mapping parameter names to descriptions or schemas
        """
        pass
    
    @property
    def version(self) -> str:
        """
        Get the version of this capability.
        
        Version information is important for tracking API changes over time.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        return "1.0.0"
    
    def requires_human_approval(self) -> bool:
        """
        Determine if this capability requires human approval before execution.
        
        This is a critical security and safety feature that allows certain
        capabilities to be gated behind human review.
        
        Returns:
            True if human approval is required, False otherwise
        """
        return self._requires_approval
    
    def register_event_handler(self, event_type: str, handler: EventCallback) -> None:
        """
        Register an event handler for capability lifecycle events.
        
        Event handlers allow for extending capability behavior without modifying
        the core implementation.
        
        Args:
            event_type: The event type to register for (e.g., "started", "completed")
            handler: The handler function to call when the event occurs
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
    
    def trigger_event(self, event_name: str, **kwargs: Any) -> None:
        """
        Trigger an event by name.
        
        This method calls all registered event handlers for the given event name.
        
        Args:
            event_name: The name of the event to trigger
            **kwargs: Additional parameters to pass to the event handlers
        """
        if "capability_name" not in kwargs:
            kwargs["capability_name"] = self.name
        
        # Always include the event name as the event_type in kwargs
        if "event_type" not in kwargs:
            kwargs["event_type"] = event_name
        
        if event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    # For coroutine functions, create a task intentionally without awaiting
                    # For regular functions, call directly
                    if asyncio.iscoroutinefunction(handler):
                        # Use our helper to properly handle the coroutine
                        fire_and_forget(handler(**kwargs))
                    else:
                        handler(**kwargs)
                except Exception as e:
                    logger.warning(f"Error in event handler for '{event_name}': {str(e)}")
    
    def get_operation_cost(self) -> Dict[str, float]:
        """
        Get the cost details for the last executed operation.
        
        This is used for accounting and monitoring resource usage.
        
        Returns:
            Dictionary with cost details (tokens, API calls, etc.)
        """
        # Ensure we always return a dict with total cost
        result = self._operation_cost.copy()
        if "total" not in result:
            total = sum(cost for cost in result.values())
            result["total"] = total
        return result
    
    def record_operation_cost(self, cost_type: str, amount: float) -> None:
        """
        Record a cost associated with the capability execution.
        
        Args:
            cost_type: The type of cost (e.g., "tokens", "api_calls")
            amount: The amount to record
        """
        if cost_type in self._operation_cost:
            self._operation_cost[cost_type] += amount
        else:
            self._operation_cost[cost_type] = amount
    
    def reset_operation_cost(self) -> None:
        """Reset the operation cost tracking."""
        self._operation_cost = {}
    
    def record_execution(self, parameters: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Record an execution for history and auditing purposes.
        
        Args:
            parameters: The parameters used for execution
            result: The execution result
        """
        execution_record = {
            "execution_id": str(uuid4()),
            "timestamp": asyncio.get_event_loop().time(),
            "parameters": parameters,
            "result": result,
            "cost": self.get_operation_cost()  # Use get_operation_cost to ensure total is included
        }
        self._execution_history.append(execution_record)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history for this capability.
        
        Returns:
            List of execution records
        """
        return self._execution_history
    
    def validate_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Validate the parameters for the capability.
        
        Args:
            **kwargs: The parameters to validate
            
        Returns:
            The validated parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # If using pydantic model and schema is defined, validate with pydantic
        if hasattr(self, "parameters_schema") and self.parameters_schema:
            try:
                # Create a model instance with the parameters
                params_model = self.parameters_schema(**kwargs)
                # Convert to dict for compatibility
                return dict(params_model.model_dump())
            except PydanticValidationError as e:
                # Convert pydantic validation error to our ValidationError
                raise ValidationError(str(e), self.name)
        else:
            # Fall back to a simple pass-through validation for backwards compatibility
            # Just return the parameters as-is
            return kwargs
    
    def create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized success response.
        
        Args:
            data: The data to include in the response
            
        Returns:
            Standardized success response dictionary
        """
        return {
            "status": "success",
            "data": data
        }
    
    def create_error_response(self, error: Union[str, Exception]) -> Dict[str, Any]:
        """
        Create a standardized error response.
        
        Args:
            error: The error message or exception to include
            
        Returns:
            Standardized error response dictionary
        """
        if isinstance(error, Exception):
            error_message = str(error)
        else:
            error_message = error
            
        return {
            "status": "error",
            "error": error_message
        }
    
    async def execute_step_by_step(self) -> AsyncIterator[CapabilityResult]:
        """Execute the capability step by step, yielding partial results.
        
        This is a template method that should be implemented by subclasses.
        By default, it calls the execute method and yields the result.
        """
        result = await self.execute()
        yield result
    
    async def execute_with_lifecycle(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the capability with full lifecycle handling.
        
        This method handles parameter validation, event triggering, and
        execution within a consistent pattern.
        
        Args:
            **kwargs: The parameters for capability execution
            
        Returns:
            Dictionary with execution results or error information
        """
        # First, validate all parameters
        try:
            validated_params = self.validate_parameters(**kwargs)
        except ValidationError as e:
            # Parameter validation error handling
            error_message = f"Validation error in capability '{self.name}': {str(e)}"
            logger.error(error_message)
            
            error_result = {
                "status": "error", 
                "error": str(e)
            }
            
            # Trigger error events for validation errors - but NOT the "started" event
            self.trigger_event(f"{self.name}.error", error=e, params=kwargs, capability_name=self.name, error_type="ValidationError")
            
            return error_result
            
        # Trigger pre-execution events with both generic and specific event names
        self.trigger_event("before_execution", params=validated_params, capability_name=self.name)
        self.trigger_event(f"{self.name}.started", params=validated_params, capability_name=self.name)
        
        try:
            # Execute the capability with validated parameters
            result = await self.execute(**validated_params)
            
            # Create standardized success response, preserving the original result structure
            # Some capabilities might return a CapabilityResult, others return a dict directly
            if isinstance(result, CapabilityResult):
                success_response = result.result  # Use the result directly
                if "status" not in success_response:
                    success_response["status"] = "success"
            elif isinstance(result, dict):
                # If it's already a dict with status and data, use it directly
                if "status" in result and ("data" in result or "result" in result):
                    success_response = result
                else:
                    # Otherwise wrap it in a standard structure
                    success_response = {
                        "status": "success",
                        "data": result
                    }
            else:
                # For any other result type, wrap it
                success_response = {
                    "status": "success",
                    "data": {"result": result}
                }
            
            # Record this successful execution
            self.record_execution(validated_params, success_response)
            
            # Trigger post-execution events
            self.trigger_event("after_execution", result=success_response, params=validated_params, capability_name=self.name)
            self.trigger_event(f"{self.name}.completed", result=success_response, params=validated_params, capability_name=self.name)
            
            return success_response
            
        except Exception as e:
            # Execution error handling
            error_message = f"Error executing capability '{self.name}': {str(e)}"
            logger.error(error_message)
            
            error_result = {
                "status": "error", 
                "error": error_message
            }
            
            # Trigger error events with both generic and specific event names
            self.trigger_event("execution_error", error=e, result=error_result, capability_name=self.name)
            self.trigger_event(f"{self.name}.error", error=e, params=validated_params, capability_name=self.name)
            
            return error_result
    
    async def _handle_execution_error(self, e: Exception, **kwargs: Any) -> Dict[str, Any]:
        """Handle execution errors.
        
        Args:
            e: The exception that was raised
            **kwargs: Any additional parameters to pass to the result
            
        Returns:
            A dictionary with the error information
        """
        error_message = f"Error executing capability '{self.name}': {str(e)}"
        logger.error(error_message, exc_info=True)
        
        # Create an error response
        error_data = {
            "status": "error",
            "error": error_message
        }
        
        # Trigger execution error event
        self.trigger_event(f"{self.name}.error", params=kwargs, error=str(e))
        
        return error_data
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the capability with the provided parameters.
        
        This is the core method that implements the capability's functionality.
        Subclasses must implement this method.
        
        Args:
            **kwargs: The parameters for the capability execution
            
        Returns:
            CapabilityResult with the execution results
        """
        pass