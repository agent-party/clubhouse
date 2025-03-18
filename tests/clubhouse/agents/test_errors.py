"""
Tests for the centralized error handling framework in the Clubhouse agents module.

This module contains tests for the error classes and utility functions defined
in the clubhouse.agents.errors module.
"""

import pytest
from typing import Dict, Any
from uuid import uuid4

from clubhouse.agents.errors import (
    ClubhouseError,
    CapabilityError, 
    ExecutionError, 
    ValidationError,
    CapabilityNotFoundError,
    format_error_response,
    create_error_response
)


class TestErrorClasses:
    """Tests for the error class hierarchy."""
    
    def test_clubhouse_error_base(self):
        """Test that ClubhouseError is a proper base exception."""
        error = ClubhouseError("General error")
        assert str(error) == "General error"
        assert isinstance(error, Exception)
    
    def test_capability_error_inheritance(self):
        """Test that CapabilityError inherits from ClubhouseError."""
        error = CapabilityError("Capability error")
        assert str(error) == "Capability error"
        assert isinstance(error, ClubhouseError)
    
    def test_execution_error(self):
        """Test ExecutionError initialization and properties."""
        error = ExecutionError("Execution failed", "search")
        assert error.capability_name == "search"
        assert "search" in str(error)
        assert "Execution failed" in str(error)
        assert isinstance(error, CapabilityError)
    
    def test_validation_error(self):
        """Test ValidationError initialization and properties."""
        # Without parameter name
        error = ValidationError("Missing required field", "summarize")
        assert error.capability_name == "summarize"
        assert error.parameter_name is None
        assert "summarize" in str(error)
        assert "Missing required field" in str(error)
        assert isinstance(error, CapabilityError)
        
        # With parameter name
        error = ValidationError("Invalid value", "search", "query")
        assert error.capability_name == "search"
        assert error.parameter_name == "query"
        assert "search" in str(error)
        assert "query" in str(error)
        assert "Invalid value" in str(error)
    
    def test_capability_not_found_error(self):
        """Test CapabilityNotFoundError initialization and properties."""
        error = CapabilityNotFoundError("nonexistent_capability")
        assert error.capability_name == "nonexistent_capability"
        assert "nonexistent_capability" in str(error)
        assert isinstance(error, CapabilityError)


class TestErrorUtilities:
    """Tests for error utility functions."""
    
    def test_format_error_response(self):
        """Test formatting error messages into standard response dictionaries."""
        # Test with simple error message
        response = format_error_response("Something went wrong")
        assert response["status"] == "error"
        assert response["error"] == "Something went wrong"
        assert "data" not in response
        
        # Test with error object
        error = ValidationError("Invalid parameter", "search", "query")
        response = format_error_response(error)
        assert response["status"] == "error"
        assert "Invalid parameter" in response["error"]
        assert "search" in response["error"]
        assert "query" in response["error"]
        
        # Test with additional context
        context = {"request_id": str(uuid4()), "timestamp": "2023-01-01T00:00:00Z"}
        response = format_error_response("Error with context", context=context)
        assert response["status"] == "error"
        assert response["error"] == "Error with context"
        assert response["context"] == context
    
    def test_create_error_response(self):
        """Test creating standardized error responses."""
        # Basic error response
        response = create_error_response("Operation failed")
        assert response["status"] == "error"
        assert response["error"] == "Operation failed"
        
        # Error response with error code
        response = create_error_response("Not found", error_code="NOT_FOUND")
        assert response["status"] == "error"
        assert response["error"] == "Not found"
        assert response["error_code"] == "NOT_FOUND"
        
        # Error response with context
        details = {"field": "username", "problem": "already exists"}
        response = create_error_response("Validation failed", details=details)
        assert response["status"] == "error"
        assert response["error"] == "Validation failed"
        assert response["details"] == details
        
        # Full error response
        response = create_error_response(
            "Permission denied",
            error_code="FORBIDDEN",
            details={"resource": "document", "action": "edit"},
            request_id=str(uuid4())
        )
        assert response["status"] == "error"
        assert response["error"] == "Permission denied"
        assert response["error_code"] == "FORBIDDEN"
        assert response["details"]["resource"] == "document"
        assert "request_id" in response
