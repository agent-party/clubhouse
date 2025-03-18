"""
Tests for the AnalyzeTextCapability implementation.

This module tests the functionality of the AnalyzeTextCapability, verifying that
it correctly handles parameter validation, execution, error handling,
and event triggering during the text analysis lifecycle.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.capability import CapabilityResult

# This will be implemented later
pytest.importorskip("clubhouse.agents.capabilities.analyze_text_capability")
from clubhouse.agents.capabilities.analyze_text_capability import AnalyzeTextCapability

class TestAnalyzeTextCapability:
    """Test suite for AnalyzeTextCapability."""
    
    def test_initialization(self):
        """Test capability initialization."""
        capability = AnalyzeTextCapability()
        assert capability.name == "analyze_text"
        assert "Analyze text" in capability.description
        assert isinstance(capability.parameters, dict)
        assert capability._operation_costs == {}
    
    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        capability = AnalyzeTextCapability()
        
        # Test with minimal required parameters
        params = {
            "text": "This is some text to analyze for testing purposes."
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["text"] == params["text"]
        assert validated["analysis_type"] == "general"  # Default value
        
        # Test with all parameters
        params = {
            "text": "This is some text to analyze for testing purposes.",
            "analysis_type": "sentiment",
            "additional_context": "Customer service interaction",
            "max_insights": 5
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["text"] == params["text"]
        assert validated["analysis_type"] == params["analysis_type"]
        assert validated["additional_context"] == params["additional_context"]
        assert validated["max_insights"] == params["max_insights"]
    
    def test_parameter_validation_failure(self):
        """Test parameter validation with invalid parameters."""
        capability = AnalyzeTextCapability()
        
        # Missing required parameter
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(analysis_type="sentiment")
        assert "text" in str(exc_info.value)
        
        # Invalid analysis type
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                text="Some text", 
                analysis_type="invalid_type"
            )
        assert "analysis_type" in str(exc_info.value).lower()
        
        # Invalid max_insights (not integer or negative)
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                text="Some text", 
                max_insights=-1
            )
        assert "max_insights" in str(exc_info.value).lower()
    
    def test_cost_tracking(self):
        """Test operation cost tracking."""
        capability = AnalyzeTextCapability()
        
        # Add some costs
        capability.add_operation_cost("base", 0.01)
        capability.add_operation_cost("text_length", 0.05)
        capability.add_operation_cost("analysis_type", 0.03)
        
        # Get the costs
        costs = capability.get_operation_cost()
        assert costs["base"] == 0.01
        assert costs["text_length"] == 0.05
        assert costs["analysis_type"] == 0.03
        assert costs["total"] == 0.09
        
        # Reset the costs
        capability.reset_operation_cost()
        costs = capability.get_operation_cost()
        assert costs["total"] == 0
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and triggering."""
        capability = AnalyzeTextCapability()
        
        # Mock event handlers
        before_handler = MagicMock()
        after_handler = MagicMock()
        
        # Register handlers
        capability.register_event_handler("before_execution", before_handler)
        capability.register_event_handler("after_execution", after_handler)
        
        # Execute with mocked analysis function
        with patch.object(capability, '_perform_analysis', return_value={"sentiment": "positive"}):
            result = await capability.execute_with_lifecycle(text="Sample text")
        
        # Check that the handlers were called with correct arguments
        before_handler.assert_called_once()
        after_handler.assert_called_once()
        
        # Verify correct parameters were passed to before_execution
        before_args = before_handler.call_args[1]
        assert before_args["capability_name"] == "analyze_text"
        assert "text" in before_args["params"]
        
        # Verify result was passed to after_execution
        after_args = after_handler.call_args[1]
        assert after_args["capability_name"] == "analyze_text"
        assert "result" in after_args
        
    @pytest.mark.asyncio
    async def test_perform_analysis(self):
        """Test the internal analysis function."""
        capability = AnalyzeTextCapability()
        
        text = "I'm really happy with the service provided."
        analysis_type = "sentiment"
        
        # Call the internal method directly
        result = await capability._perform_analysis(
            text=text, 
            analysis_type=analysis_type,
            additional_context=None,
            max_insights=3
        )
        
        # Check the result structure
        assert isinstance(result, dict)
        assert "analysis_type" in result
        assert result["analysis_type"] == analysis_type
        assert "insights" in result
        assert len(result["insights"]) <= 3
    
    @pytest.mark.asyncio
    async def test_execution_success(self):
        """Test successful execution of the capability."""
        capability = AnalyzeTextCapability()
        
        # Execute with actual parameters
        result = await capability.execute_with_lifecycle(
            text="The customer was satisfied with our quick response.",
            analysis_type="sentiment"
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        assert "metadata" in result
        
        # Check specific result content
        results = result["results"]
        assert "analysis_type" in results
        assert results["analysis_type"] == "sentiment"
        assert "insights" in results
        
        # Check metadata
        metadata = result["metadata"]
        assert "execution_time" in metadata
        assert "input_length" in metadata
    
    @pytest.mark.asyncio
    async def test_execution_with_validation_error(self):
        """Test execution with validation error."""
        capability = AnalyzeTextCapability()
        
        # Execute with missing required parameter
        result = await capability.execute_with_lifecycle(analysis_type="sentiment")
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "text" in result["error"].lower()
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_execution_with_unexpected_error(self):
        """Test execution with unexpected runtime error."""
        capability = AnalyzeTextCapability()
        
        # Mock the analysis function to raise an exception
        error_message = "Service unavailable"
        with patch.object(capability, '_perform_analysis', side_effect=Exception(error_message)):
            result = await capability.execute_with_lifecycle(
                text="This should fail during analysis"
            )
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert error_message in result["error"]
        assert "error_type" in result
