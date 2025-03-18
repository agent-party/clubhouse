"""
Tests for the ReasoningCapability implementation.

This module tests the functionality of the ReasoningCapability, verifying that
it correctly handles parameter validation, execution, error handling,
and event triggering during the reasoning process lifecycle.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.capability import CapabilityResult

# This will be implemented later
pytest.importorskip("clubhouse.agents.capabilities.reasoning_capability")
from clubhouse.agents.capabilities.reasoning_capability import ReasoningCapability

class TestReasoningCapability:
    """Test suite for ReasoningCapability."""
    
    def test_initialization(self):
        """Test capability initialization."""
        capability = ReasoningCapability()
        assert capability.name == "reasoning"
        assert "reasoning" in capability.description.lower()
        assert isinstance(capability.parameters, dict)
        assert capability._operation_costs == {}
    
    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        capability = ReasoningCapability()
        
        # Test with minimal required parameters
        params = {
            "query": "What would happen if we doubled our marketing budget?"
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["query"] == params["query"]
        assert validated["reasoning_type"] == "general"  # Default value
        
        # Test with all parameters
        params = {
            "query": "What would happen if we doubled our marketing budget?",
            "reasoning_type": "financial",
            "context": "We're a small startup with limited runway.",
            "constraints": ["Must consider ROI", "Must account for market conditions"],
            "max_steps": 5,
            "step_by_step": True
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["query"] == params["query"]
        assert validated["reasoning_type"] == params["reasoning_type"]
        assert validated["context"] == params["context"]
        assert validated["constraints"] == params["constraints"]
        assert validated["max_steps"] == params["max_steps"]
        assert validated["step_by_step"] == params["step_by_step"]
    
    def test_parameter_validation_failure(self):
        """Test parameter validation with invalid parameters."""
        capability = ReasoningCapability()
        
        # Missing required parameter
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(reasoning_type="financial")
        assert "query" in str(exc_info.value)
        
        # Invalid reasoning_type
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                query="How should we proceed?", 
                reasoning_type="invalid_type"
            )
        assert "reasoning_type" in str(exc_info.value).lower()
        
        # Invalid max_steps (too high)
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                query="How should we proceed?", 
                max_steps=15
            )
        assert "max_steps" in str(exc_info.value).lower()
    
    def test_cost_tracking(self):
        """Test operation cost tracking."""
        capability = ReasoningCapability()
        
        # Add some costs
        capability.add_operation_cost("base", 0.02)
        capability.add_operation_cost("query_complexity", 0.05)
        capability.add_operation_cost("steps", 0.01)
        
        # Get the costs
        costs = capability.get_operation_cost()
        assert costs["base"] == 0.02
        assert costs["query_complexity"] == 0.05
        assert costs["steps"] == 0.01
        assert costs["total"] == 0.08
        
        # Reset the costs
        capability.reset_operation_cost()
        costs = capability.get_operation_cost()
        assert costs["total"] == 0
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and triggering."""
        capability = ReasoningCapability()
        
        # Mock event handlers
        before_handler = MagicMock()
        after_handler = MagicMock()
        
        # Register handlers
        capability.register_event_handler("before_execution", before_handler)
        capability.register_event_handler("after_execution", after_handler)
        
        # Execute with mocked reasoning function
        with patch.object(capability, '_perform_reasoning', return_value={"steps": ["Step 1", "Step 2"], "conclusion": "Test conclusion"}):
            result = await capability.execute_with_lifecycle(query="Test query")
        
        # Check that the handlers were called with correct arguments
        before_handler.assert_called_once()
        after_handler.assert_called_once()
        
        # Verify correct parameters were passed to before_execution
        before_args = before_handler.call_args[1]
        assert before_args["capability_name"] == "reasoning"
        assert "params" in before_args
        
        # Verify result was passed to after_execution
        after_args = after_handler.call_args[1]
        assert after_args["capability_name"] == "reasoning"
        assert "result" in after_args
        
    @pytest.mark.asyncio
    async def test_perform_reasoning(self):
        """Test the internal reasoning function."""
        capability = ReasoningCapability()
        
        query = "What would happen if we doubled our marketing budget?"
        reasoning_type = "financial"
        context = "We're a small startup with limited runway."
        
        # Call the internal method directly
        result = await capability._perform_reasoning(
            query=query,
            reasoning_type=reasoning_type,
            context=context,
            constraints=["Consider ROI"],
            max_steps=3,
            step_by_step=True
        )
        
        # Check the result structure
        assert isinstance(result, dict)
        assert "reasoning_type" in result
        assert result["reasoning_type"] == reasoning_type
        assert "steps" in result
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) <= 3  # Should respect max_steps
        assert "conclusion" in result
        assert isinstance(result["conclusion"], str)
    
    @pytest.mark.asyncio
    async def test_execution_success(self):
        """Test successful execution of the capability."""
        capability = ReasoningCapability()
        
        # Execute with actual parameters
        result = await capability.execute_with_lifecycle(
            query="What's the best approach to enter a new market?",
            reasoning_type="business",
            step_by_step=True
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        assert "metadata" in result
        
        # Check specific result content
        results = result["results"]
        assert "reasoning_type" in results
        assert results["reasoning_type"] == "business"
        assert "steps" in results
        assert isinstance(results["steps"], list)
        assert "conclusion" in results
        
        # Check metadata
        metadata = result["metadata"]
        assert "execution_time" in metadata
        assert "query_length" in metadata
    
    @pytest.mark.asyncio
    async def test_step_by_step_reasoning(self):
        """Test reasoning with step-by-step option enabled."""
        capability = ReasoningCapability()
        
        # Execute with step-by-step set to True
        result = await capability.execute_with_lifecycle(
            query="How should we allocate our resources?",
            reasoning_type="strategic",
            step_by_step=True
        )
        
        # Check result structure for step-by-step reasoning
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        
        # Check that steps are included
        results = result["results"]
        assert "steps" in results
        assert isinstance(results["steps"], list)
        assert len(results["steps"]) > 0
    
    @pytest.mark.asyncio
    async def test_execution_with_validation_error(self):
        """Test execution with validation error."""
        capability = ReasoningCapability()
        
        # Execute with missing required parameter
        result = await capability.execute_with_lifecycle(reasoning_type="financial")
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "query" in result["error"].lower()
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_execution_with_unexpected_error(self):
        """Test execution with unexpected runtime error."""
        capability = ReasoningCapability()
        
        # Mock the reasoning function to raise an exception
        error_message = "Reasoning service unavailable"
        with patch.object(capability, '_perform_reasoning', side_effect=Exception(error_message)):
            result = await capability.execute_with_lifecycle(
                query="This should fail during reasoning"
            )
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert error_message in result["error"]
        assert "error_type" in result
