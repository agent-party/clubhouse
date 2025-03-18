"""
Tests for the enhanced BaseCapability class.

This module tests the enhanced BaseCapability implementation that uses the
centralized error framework and Pydantic validation.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch
from pydantic import ValidationError as PydanticValidationError

from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.schemas import BaseCapabilityParams
from clubhouse.agents.capability import BaseCapability


class TestCapabilityParams(BaseCapabilityParams):
    """Test parameter model for capability testing."""
    param1: str
    param2: int = 42


class TestBaseCapability(BaseCapability):
    """Test implementation of BaseCapability for testing."""
    
    def __init__(self, requires_human_approval: bool = False):
        super().__init__(requires_human_approval=requires_human_approval)
        self.execution_called = False
        self.validation_called = False
    
    @property
    def name(self) -> str:
        return "test_capability"
    
    @property
    def description(self) -> str:
        return "Test capability for unit testing"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "param1": {
                "type": "string",
                "description": "First parameter",
                "required": True
            },
            "param2": {
                "type": "integer",
                "description": "Second parameter",
                "required": False,
                "default": 42
            }
        }
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Custom parameter validation for testing."""
        self.validation_called = True
        # Use Pydantic model for validation
        try:
            params = TestCapabilityParams(**kwargs)
            return params.model_dump()
        except PydanticValidationError as e:
            # Convert Pydantic error to our ValidationError
            msg = str(e)
            raise ValidationError(msg, self.name)
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute implementation for testing."""
        self.execution_called = True
        # Simulate some processing
        validated_params = self.validate_parameters(**kwargs)
        return {
            "status": "success",
            "data": {
                "message": f"Executed with param1={validated_params['param1']} and param2={validated_params['param2']}"
            }
        }


class TestBaseCapabilityClass:
    """Tests for the BaseCapability class."""
    
    def test_initialization(self):
        """Test basic initialization of BaseCapability."""
        cap = TestBaseCapability()
        assert cap.name == "test_capability"
        assert cap.description == "Test capability for unit testing"
        assert not cap.requires_human_approval()
        assert cap._event_handlers == {}
        assert cap._operation_cost == {}
    
    def test_requires_approval(self):
        """Test the requires_approval property."""
        cap1 = TestBaseCapability(requires_human_approval=False)
        assert not cap1.requires_human_approval()
        
        cap2 = TestBaseCapability(requires_human_approval=True)
        assert cap2.requires_human_approval()
    
    def test_event_handlers(self):
        """Test event handler registration and triggering."""
        cap = TestBaseCapability()
        
        # Create mock event handlers
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        # Register handlers
        cap.register_event_handler("test_event", handler1)
        cap.register_event_handler("test_event", handler2)
        
        # Trigger the event
        event_data = {"param": "value"}
        cap.trigger_event("test_event", **event_data)
        
        # Check that handlers were called with correct arguments
        # Include event_type in expected args since trigger_event adds it
        expected_args = {
            "capability_name": cap.name, 
            "event_type": "test_event",
            **event_data
        }
        handler1.assert_called_once_with(**expected_args)
        handler2.assert_called_once_with(**expected_args)
    
    def test_operation_cost_tracking(self):
        """Test operation cost tracking."""
        cap = TestBaseCapability()
        
        # Record some costs
        cap.record_operation_cost("api_call", 0.01)
        cap.record_operation_cost("tokens", 0.05)
        cap.record_operation_cost("api_call", 0.02)  # Add to existing
        
        # Check the cost tracker
        costs = cap.get_operation_cost()
        assert costs["api_call"] == 0.03
        assert costs["tokens"] == 0.05
        assert costs["total"] == 0.08
    
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle_success(self):
        """Test successful execution with lifecycle events."""
        cap = TestBaseCapability()
        
        # Add mock event handlers to track lifecycle events
        start_handler = MagicMock()
        complete_handler = MagicMock()
        cap.register_event_handler("test_capability.started", start_handler)
        cap.register_event_handler("test_capability.completed", complete_handler)
        
        # Execute with valid parameters
        result = await cap.execute_with_lifecycle(param1="test", param2=123)
        
        # Check execution result
        assert result["status"] == "success"
        assert "Executed with param1=test and param2=123" in result["data"]["message"]
        
        # Verify lifecycle events were triggered
        start_handler.assert_called_once()
        start_kwargs = start_handler.call_args[1]
        assert start_kwargs["params"]["param1"] == "test"
        assert start_kwargs["params"]["param2"] == 123
        
        complete_handler.assert_called_once()
        complete_kwargs = complete_handler.call_args[1]
        assert complete_kwargs["params"]["param1"] == "test"
        assert complete_kwargs["params"]["param2"] == 123
        assert "message" in complete_kwargs["result"]["data"]
    
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle_validation_error(self):
        """Test execution with lifecycle events when validation fails."""
        cap = TestBaseCapability()
        
        # Add mock event handlers to track lifecycle events
        start_handler = MagicMock()
        error_handler = MagicMock()
        cap.register_event_handler("test_capability.started", start_handler)
        cap.register_event_handler("test_capability.error", error_handler)
        
        # Execute with invalid parameters (missing required param1)
        result = await cap.execute_with_lifecycle(param2=123)
        
        # Check error result
        assert result["status"] == "error"
        assert "param1" in result["error"]
        
        # Verify appropriate lifecycle events were triggered
        start_handler.assert_not_called()  # Should not trigger started event on validation error
        error_handler.assert_called_once()
        error_kwargs = error_handler.call_args[1]
        assert error_kwargs["params"]["param2"] == 123
        assert "error" in error_kwargs
    
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle_execution_error(self):
        """Test execution with lifecycle events when execution fails."""
        cap = TestBaseCapability()
        
        # Mock execute to raise an error
        async def mock_execute(**kwargs):
            raise ExecutionError("Execution failed", cap.name)
        
        cap.execute = mock_execute
        
        # Add mock event handlers to track lifecycle events
        start_handler = MagicMock()
        error_handler = MagicMock()
        cap.register_event_handler("test_capability.started", start_handler)
        cap.register_event_handler("test_capability.error", error_handler)
        
        # Execute with valid parameters but execution will fail
        result = await cap.execute_with_lifecycle(param1="test", param2=123)
        
        # Check error result
        assert result["status"] == "error"
        assert "Execution failed" in result["error"]
        
        # Verify appropriate lifecycle events were triggered
        start_handler.assert_called_once()
        error_handler.assert_called_once()
        error_kwargs = error_handler.call_args[1]
        assert "Execution failed" in str(error_kwargs["error"])
    
    def test_create_success_response(self):
        """Test creation of success response."""
        cap = TestBaseCapability()
        data = {"key": "value"}
        
        response = cap.create_success_response(data)
        assert response["status"] == "success"
        assert response["data"] == data
    
    def test_create_error_response(self):
        """Test creation of error response."""
        cap = TestBaseCapability()
        
        # Test with string error
        response = cap.create_error_response("Something went wrong")
        assert response["status"] == "error"
        assert response["error"] == "Something went wrong"
        
        # Test with exception
        error = ValueError("Invalid value")
        response = cap.create_error_response(error)
        assert response["status"] == "error"
        assert "Invalid value" in response["error"]
