"""
Centralized error handling framework for the Clubhouse agents module.

This module provides a standardized set of exception classes and utility functions
for handling errors in a consistent manner across the agent system. It defines
a hierarchy of errors that can be used for both raising exceptions and
generating standardized error responses.

The design follows the principle of having descriptive, well-typed errors
that provide clear information about what went wrong and where.
"""

from typing import Any, Dict, Optional, Union
import uuid
import logging
from typing import cast, List, Dict, Any, Type

# Configure logger
logger = logging.getLogger(__name__)


class ClubhouseError(Exception):
    """Base exception for all errors in the Clubhouse system."""
    
    def __init__(self, message: str) -> None:
        """
        Initialize a new ClubhouseError.
        
        Args:
            message: Human-readable error message
        """
        self.message = message
        super().__init__(message)


class CapabilityError(ClubhouseError):
    """Base exception for capability-related errors."""
    
    def __init__(self, message: str) -> None:
        """
        Initialize a new CapabilityError.
        
        Args:
            message: Human-readable error message
        """
        super().__init__(message)


class ExecutionError(CapabilityError):
    """Exception raised when a capability execution fails."""
    
    def __init__(self, message: str, capability_name: str) -> None:
        """
        Initialize a new ExecutionError.
        
        Args:
            message: Human-readable error message
            capability_name: Name of the capability that failed
        """
        self.capability_name = capability_name
        super().__init__(f"Error executing capability '{capability_name}': {message}")


class ValidationError(CapabilityError):
    """Exception raised when capability parameters fail validation."""
    
    def __init__(self, message: str, capability_name: str, parameter_name: Optional[str] = None) -> None:
        """
        Initialize a new ValidationError.
        
        Args:
            message: Human-readable error message
            capability_name: Name of the capability with validation error
            parameter_name: Optional name of the parameter that failed validation
        """
        self.capability_name = capability_name
        self.parameter_name = parameter_name
        
        param_info = f" (parameter: {parameter_name})" if parameter_name else ""
        error_message = f"Validation error for capability '{capability_name}'{param_info}: {message}"
        super().__init__(error_message)


class CapabilityNotFoundError(CapabilityError):
    """Exception raised when a requested capability is not found."""
    
    def __init__(self, capability_name: str) -> None:
        """
        Initialize a new CapabilityNotFoundError.
        
        Args:
            capability_name: Name of the capability that was not found
        """
        self.capability_name = capability_name
        error_message = f"Capability '{capability_name}' not found"
        super().__init__(error_message)


def format_error_response(
    error: Union[str, Exception],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format an error message or exception into a standardized response dictionary.
    
    Args:
        error: The error message or exception object
        context: Optional additional context to include in the response
        
    Returns:
        A dictionary with standardized error response format
    """
    # Extract error message from exception if needed
    if isinstance(error, Exception):
        message = str(error)
    else:
        message = error
    
    # Create the base response
    response: Dict[str, Any] = {
        "status": "error",
        "error": message
    }
    
    # Add context if provided
    if context:
        response["context"] = context
    
    return response


def create_error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response with optional metadata.
    
    This function provides a consistent structure for error responses
    that can be used throughout the system, including in API responses.
    
    Args:
        message: The human-readable error message
        error_code: Optional error code for categorizing errors
        details: Optional dictionary with additional error details
        request_id: Optional request identifier for tracking
        
    Returns:
        A dictionary with the standardized error response format
    """
    # Log the error for observability
    logger.error(f"Error: {message} (code: {error_code})")
    
    # Create the base response
    response: Dict[str, Any] = {
        "status": "error",
        "error": message
    }
    
    # Add optional fields if provided
    if error_code:
        response["error_code"] = error_code
    
    if details:
        response["details"] = details
    
    if request_id:
        response["request_id"] = request_id
    else:
        # Generate a request ID for tracking if not provided
        response["request_id"] = str(uuid.uuid4())
    
    return response
