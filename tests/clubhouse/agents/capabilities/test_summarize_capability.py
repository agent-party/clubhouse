"""
Tests for the SummarizeCapability implementation.

This module tests the functionality of the SummarizeCapability, verifying that
it correctly handles parameter validation, execution, error handling,
and event triggering during the summarization lifecycle.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from clubhouse.agents.capabilities.summarize_capability import SummarizeCapability
from clubhouse.agents.errors import ValidationError

class TestSummarizeCapability:
    """Test suite for SummarizeCapability."""
    
    def test_initialization(self):
        """Test capability initialization."""
        capability = SummarizeCapability()
        assert capability.name == "summarize"
        assert "Generate summaries" in capability.description
        assert isinstance(capability.parameters, dict)
        assert capability._operation_costs == {}
    
    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        capability = SummarizeCapability()
        
        params = {
            "text": "This is some text to summarize for testing purposes.",
            "max_length": 100,
            "format": "concise"
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["text"] == params["text"]
        assert validated["max_length"] == params["max_length"]
        assert validated["format"] == params["format"]
    
    def test_parameter_validation_failure(self):
        """Test parameter validation with invalid parameters."""
        capability = SummarizeCapability()
        
        # Missing required parameter
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(max_length=100, format="concise")
        assert "text" in str(exc_info.value)
        
        # Invalid type for parameter that cannot be converted to integer
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(text="Some text", max_length="invalid_number")
        assert "max_length" in str(exc_info.value).lower()
    
    def test_cost_tracking(self):
        """Test operation cost tracking."""
        capability = SummarizeCapability()
        
        # Add some costs
        capability.add_operation_cost("base", 0.01)
        capability.add_operation_cost("text_length", 0.05)
        
        # Get the costs
        costs = capability.get_operation_cost()
        assert costs["base"] == 0.01
        assert costs["text_length"] == 0.05
        assert costs["total"] == 0.06
        
        # Reset the costs
        capability.reset_operation_cost()
        costs = capability.get_operation_cost()
        assert costs["total"] == 0
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and triggering."""
        capability = SummarizeCapability()
        
        # Mock event handlers
        started_handler = MagicMock()
        completed_handler = MagicMock()
        
        # Register handlers
        capability.register_event_handler("summarize_started", started_handler)
        capability.register_event_handler("summarize_completed", completed_handler)
        
        # Trigger events
        capability.trigger_event("summarize_started", text="Sample text", max_length=100, format="concise")
        
        # Check that the handler was called with correct arguments
        started_handler.assert_called_once()
        kwargs = started_handler.call_args[1]
        assert kwargs["capability_name"] == "summarize"
        assert kwargs["text"] == "Sample text"
        assert kwargs["max_length"] == 100
        assert kwargs["format"] == "concise"
    
    @pytest.mark.asyncio
    async def test_perform_summarization(self):
        """Test the summarization operation."""
        capability = SummarizeCapability()
        
        # Test with different formats
        
        # Concise format
        result = await capability._perform_summarization(
            "This is a test text that needs to be summarized. It contains multiple sentences with important information.",
            max_length=50,
            format="concise"
        )
        assert isinstance(result, dict)
        assert "summary" in result
        assert len(result["summary"]) <= 50
        
        # Detailed format
        result = await capability._perform_summarization(
            "This is a test text that needs to be summarized. It contains multiple sentences with important information.",
            max_length=100,
            format="detailed"
        )
        assert isinstance(result, dict)
        assert "summary" in result
        assert len(result["summary"]) <= 100
        
        # Bullet points format
        result = await capability._perform_summarization(
            "This is a test text that needs to be summarized. It contains multiple sentences with important information.",
            max_length=100,
            format="bullet_points"
        )
        assert isinstance(result, dict)
        assert "summary" in result
        assert len(result["summary"]) <= 100
        assert "•" in result["summary"]
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution of the capability."""
        capability = SummarizeCapability()
        
        # Mock event handlers
        started_handler = MagicMock()
        completed_handler = MagicMock()
        capability.register_event_handler("summarize_started", started_handler)
        capability.register_event_handler("summarize_completed", completed_handler)
        
        # Execute the capability
        result = await capability.execute(
            text="This is a test text that needs to be summarized for execution testing.",
            max_length=100,
            format="concise"
        )
        
        # Check the result
        assert result.result["summary"] is not None
        assert "metadata" in result.result
        assert result.metadata["cost"]["total"] > 0
        
        # Verify events were triggered
        started_handler.assert_called_once()
        completed_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle(self):
        """Test execution with lifecycle events."""
        capability = SummarizeCapability()
        
        # Mock event handlers
        before_execution_handler = MagicMock()
        after_execution_handler = MagicMock()
        capability.register_event_handler("before_execution", before_execution_handler)
        capability.register_event_handler("after_execution", after_execution_handler)
        
        # Execute with lifecycle
        result = await capability.execute_with_lifecycle(
            text="This is a test text for lifecycle execution testing.",
            max_length=100,
            format="concise"
        )
        
        # Check the result
        assert "summary" in result
        assert "metadata" in result
        
        # Verify lifecycle events were triggered
        before_execution_handler.assert_called_once()
        after_execution_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_validation_error(self):
        """Test execution with validation error."""
        capability = SummarizeCapability()
        
        # Mock event handlers
        started_handler = MagicMock()
        error_handler = MagicMock()
        capability.register_event_handler("summarize_started", started_handler)
        capability.register_event_handler("summarize.error", error_handler)
        
        # Execute with missing required parameter
        result = await capability.execute_with_lifecycle(
            max_length=100,
            format="concise"
        )
        
        # Check the result
        assert "status" in result
        assert result["status"] == "error"
        assert "text" in result["error"]
        
        # Verify events
        started_handler.assert_not_called()
        error_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_with_unexpected_error(self):
        """Test execution with unexpected error."""
        capability = SummarizeCapability()
        
        # Patch _perform_summarization to raise an exception
        with patch.object(capability, '_perform_summarization', side_effect=Exception("Test error")):
            # Execute the capability
            result = await capability.execute(
                text="This is a test text that will trigger an error.",
                max_length=100,
                format="concise"
            )
            
            # Check the result
            assert result.result["status"] == "error"
            assert "Test error" in result.result["error"]
            assert result.metadata["error_type"] == "Exception"
