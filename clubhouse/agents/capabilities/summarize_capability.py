"""
Summarize capability for creating summaries of text content.

This module provides a capability for summarizing text content across
various sources. It handles parameter validation, execution,
and proper event triggering during the summarization lifecycle.
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.schemas import validate_capability_params

# Set up logging
logger = logging.getLogger(__name__)

class SummarizeParameters(BaseModel):
    """Model for summarization parameters validation."""
    text: str = Field(..., description="The text content to summarize")
    max_length: int = Field(200, description="Maximum length of the summary in characters")
    format: str = Field(
        "concise", 
        description="Format of the summary (concise, detailed, bullet_points)"
    )

class SummarizeCapability(BaseCapability):
    """Capability for summarizing text content."""
    
    name = "summarize"
    description = "Generate summaries of text content with configurable length and format"
    
    # Cost constants for summarization operations
    _base_cost = 0.01
    _cost_per_character = 0.0001
    _cost_per_format = {
        "concise": 0.01,
        "detailed": 0.02,
        "bullet_points": 0.015
    }
    
    parameters_schema = SummarizeParameters
    
    def __init__(self) -> None:
        """Initialize the SummarizeCapability."""
        super().__init__()
        self._operation_costs: Dict[str, float] = {}
        
    def reset_operation_cost(self) -> None:
        """Reset the operation cost tracking."""
        self._operation_costs = {}
        
    def add_operation_cost(self, operation: str, cost: float) -> None:
        """
        Add a cost for a specific operation.
        
        Args:
            operation: The type of operation
            cost: The cost value to add
        """
        if operation in self._operation_costs:
            self._operation_costs[operation] += cost
        else:
            self._operation_costs[operation] = cost
    
    def get_operation_cost(self) -> Dict[str, float]:
        """
        Get the operation costs.
        
        Returns:
            Dictionary with operation types as keys and costs as values,
            plus a 'total' key with the sum of all costs.
        """
        costs = self._operation_costs.copy()
        # Round to avoid floating point precision issues
        total = round(sum(costs.values()), 2) if costs else 0
        costs["total"] = total
        return costs
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters specification for this capability.
        
        Returns:
            Dictionary mapping parameter names to specifications
        """
        return self.get_parameters_schema()
    
    def get_version(self) -> str:
        """
        Get the version of the capability.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        return "1.0.0"
    
    def get_parameters_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters schema for this capability.
        
        For backwards compatibility with older capabilities.
        
        Returns:
            The parameters schema dictionary
        """
        return {
            "text": {
                "type": "string",
                "description": "The text content to summarize",
                "required": True
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of the summary in characters",
                "default": 200
            },
            "format": {
                "type": "string",
                "description": "Format of the summary (concise, detailed, bullet_points)",
                "default": "concise"
            }
        }
    
    def validate_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Validate the parameters for the summarize capability.
        
        This method uses Pydantic for validation to ensure type safety
        and proper constraint checking.
        
        Args:
            **kwargs: The parameters to validate
            
        Returns:
            Dictionary with validated parameters
            
        Raises:
            ValidationError: If parameters fail validation
        """
        try:
            # Special handling for max_length if it's passed as a string
            if "max_length" in kwargs and isinstance(kwargs["max_length"], str):
                try:
                    kwargs["max_length"] = int(kwargs["max_length"])
                except ValueError:
                    raise ValidationError(f"Invalid max_length parameter: cannot convert '{kwargs['max_length']}' to integer", self.name)
            
            # Use Pydantic model for validation
            params = SummarizeParameters(**kwargs)
            return params.model_dump()
        except PydanticValidationError as e:
            # Convert Pydantic validation errors to our ValidationError
            raise ValidationError(f"Invalid parameters: {str(e)}", self.name)
    
    async def _perform_summarization(self, text: str, max_length: int, format: str) -> Dict[str, Any]:
        """
        Perform the actual summarization operation.
        
        This method would connect to actual NLP services in a real implementation.
        For demonstration purposes, it's a placeholder that mocks summarization results
        and demonstrates proper cost tracking.
        
        Args:
            text: Text content to summarize
            max_length: Maximum length of the summary in characters
            format: Format of the summary
            
        Returns:
            Dictionary with summary result and metadata
        """
        # Track the costs
        self.add_operation_cost("base", self._base_cost)
        self.add_operation_cost("text_length", self._cost_per_character * len(text))
        
        format_cost = self._cost_per_format.get(format, self._cost_per_format["concise"])
        self.add_operation_cost("format", format_cost)
        
        # Simulate processing time for demonstration purposes
        # In a real implementation, this would be connecting to actual NLP services
        await asyncio.sleep(0.1)
        
        # Create a mock summary based on the text length
        words = text.split()
        summary_word_count = min(len(words) // 3, max_length // 5)
        
        if format == "concise":
            summary = " ".join(words[:summary_word_count])
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
        elif format == "detailed":
            summary = " ".join(words[:summary_word_count * 2])
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
        elif format == "bullet_points":
            bullet_points = []
            chunk_size = summary_word_count // 3 if summary_word_count > 3 else 1
            for i in range(0, summary_word_count, chunk_size):
                point = " ".join(words[i:i+chunk_size])
                bullet_points.append(f"• {point}")
            summary = "\n".join(bullet_points)
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
        else:
            # Default to concise format
            summary = " ".join(words[:summary_word_count])
            if len(summary) > max_length:
                summary = summary[:max_length - 3] + "..."
            
        return {
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "format": format
        }
    
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the summarize capability.
        
        This method performs the actual summarization operation using the validated parameters.
        It ensures proper event triggering and cost tracking.
        
        Args:
            **kwargs: The parameters for the summarization operation
            
        Returns:
            CapabilityResult containing the summarization results and metadata
        """
        # Always track costs, even when the summarization is mocked in tests
        self.reset_operation_cost()
        
        try:
            # Extract validated parameters
            validated_params = self.validate_parameters(**kwargs)
            text = validated_params["text"]
            max_length = validated_params.get("max_length", 200)
            format = validated_params.get("format", "concise")
            
            # Trigger standard before_execution event
            self.trigger_event("before_execution", capability_name=self.name, params=validated_params)
            # Trigger legacy event for backward compatibility
            self.trigger_event("summarize_started", text=text, max_length=max_length, format=format)
            
            # Perform the summarization operation
            start_time = time.time()
            summary_result = await self._perform_summarization(text, max_length, format)
            execution_time = time.time() - start_time
            
            # Prepare the result object in the expected format
            result_data = {
                "status": "success",
                "summary": summary_result["summary"],
                "text_length": len(text),
                "summary_length": len(summary_result["summary"]),
                "format": format,
                "metadata": {
                    "original_length": len(text),
                    "max_length": max_length,
                    "execution_time": execution_time,
                    "format_details": summary_result
                }
            }
            
            # Trigger standard after_execution event
            self.trigger_event("after_execution", 
                capability_name=self.name, 
                result=result_data,
                execution_time=execution_time
            )
            # Trigger legacy event for backward compatibility
            self.trigger_event("summarize_completed", 
                text=text, 
                result=result_data,
                execution_time=execution_time
            )
            
            # Return the result with cost metadata
            return CapabilityResult(
                result=result_data,
                metadata={"cost": self.get_operation_cost()}
            )
        except ValidationError as ve:
            # Handle validation errors
            error_message = str(ve)
            logger.error(f"Summarization validation error: {error_message}")
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type="ValidationError")
            
            # Return error result with structure expected by tests
            return CapabilityResult(
                result={"status": "error", "error": error_message},
                metadata={
                    "cost": self.get_operation_cost(),
                    "error_type": type(ve).__name__
                }
            )
        except Exception as e:
            # Use centralized error handling with ExecutionError but preserve original error type for tests
            execution_error = ExecutionError(f"Summarization failed: {str(e)}", self.name)
            error_message = str(execution_error)
            logger.error(error_message)
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type=type(e).__name__)
            
            # Return error result with structure expected by tests
            return CapabilityResult(
                result={"status": "error", "error": error_message},
                metadata={
                    "cost": self.get_operation_cost(),
                    "error_type": type(e).__name__  # Use original exception type for tests
                }
            )
    
    async def execute_with_lifecycle(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the capability with parameters and standardized lifecycle events.
        
        This method handles parameter validation and triggers standard before/after events.
        
        Args:
            **kwargs: The parameters for the capability
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Validate parameters
            validated_params = self.validate_parameters(**kwargs)
            
            # Only trigger standard before_execution event once
            # (execute method already triggers legacy event)
            self.trigger_event("before_execution", capability_name=self.name, params=validated_params)
            
            # Call the actual execution method
            operation_start_time = time.time()
            result = await self._perform_summarization(
                validated_params["text"], 
                validated_params.get("max_length", 200),
                validated_params.get("format", "concise")
            )
            execution_time = time.time() - operation_start_time
            
            # Format the result
            result_data = {
                "status": "success",
                "summary": result["summary"],
                "metadata": {
                    "execution_time": execution_time,
                    "cost": self.get_operation_cost()
                }
            }
            
            # Only trigger standard after_execution event once
            # (execute method already triggers legacy event)
            self.trigger_event("after_execution", 
                capability_name=self.name, 
                result=result_data,
                execution_time=execution_time
            )
            
            return result_data
        except ValidationError as ve:
            error_message = str(ve)
            logger.error(f"Validation error in {self.name}: {error_message}")
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type="ValidationError")
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": "ValidationError"
            }
        except Exception as e:
            error_message = f"Error in {self.name}: {str(e)}"
            logger.error(error_message)
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type=type(e).__name__)
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": type(e).__name__
            }
    
    async def execute_and_handle_lifecycle(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the capability with full lifecycle handling.
        
        This method provides a standardized execution flow using the base class's
        execute_with_lifecycle method. It preserves backward compatibility
        while adopting the newer pattern.
        
        Args:
            **kwargs: The parameters for the capability
            
        Returns:
            Dictionary with execution results or error information
        """
        return await self.execute_with_lifecycle(**kwargs)
