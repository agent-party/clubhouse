"""
Tests for the ClassifyContentCapability implementation.

This module tests the functionality of the ClassifyContentCapability, verifying that
it correctly handles parameter validation, execution, error handling,
and event triggering during the content classification lifecycle.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.capability import CapabilityResult

# This will be implemented later
pytest.importorskip("clubhouse.agents.capabilities.classify_content_capability")
from clubhouse.agents.capabilities.classify_content_capability import ClassifyContentCapability

class TestClassifyContentCapability:
    """Test suite for ClassifyContentCapability."""
    
    def test_initialization(self):
        """Test capability initialization."""
        capability = ClassifyContentCapability()
        assert capability.name == "classify_content"
        assert "Classify content" in capability.description
        assert isinstance(capability.parameters, dict)
        assert capability._operation_costs == {}
    
    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        capability = ClassifyContentCapability()
        
        # Test with minimal required parameters
        params = {
            "content": "This is some content to classify for testing purposes."
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["content"] == params["content"]
        assert validated["classification_type"] == "general"  # Default value
        
        # Test with all parameters
        params = {
            "content": "This is some content to classify for testing purposes.",
            "classification_type": "sentiment",
            "categories": ["positive", "negative", "neutral"],
            "multi_label": True,
            "confidence_threshold": 0.6
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["content"] == params["content"]
        assert validated["classification_type"] == params["classification_type"]
        assert validated["categories"] == params["categories"]
        assert validated["multi_label"] == params["multi_label"]
        assert validated["confidence_threshold"] == params["confidence_threshold"]
    
    def test_parameter_validation_failure(self):
        """Test parameter validation with invalid parameters."""
        capability = ClassifyContentCapability()
        
        # Missing required parameter
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(classification_type="sentiment")
        assert "content" in str(exc_info.value)
        
        # Invalid classification_type
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                content="Some content", 
                classification_type="invalid_type"
            )
        assert "classification_type" in str(exc_info.value).lower()
        
        # Invalid confidence_threshold (outside valid range)
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                content="Some content", 
                confidence_threshold=1.5
            )
        assert "confidence_threshold" in str(exc_info.value).lower()
    
    def test_cost_tracking(self):
        """Test operation cost tracking."""
        capability = ClassifyContentCapability()
        
        # Add some costs
        capability.add_operation_cost("base", 0.01)
        capability.add_operation_cost("content_length", 0.05)
        capability.add_operation_cost("classification_type", 0.03)
        
        # Get the costs
        costs = capability.get_operation_cost()
        assert costs["base"] == 0.01
        assert costs["content_length"] == 0.05
        assert costs["classification_type"] == 0.03
        assert costs["total"] == 0.09
        
        # Reset the costs
        capability.reset_operation_cost()
        costs = capability.get_operation_cost()
        assert costs["total"] == 0
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and triggering."""
        capability = ClassifyContentCapability()
        
        # Mock event handlers
        before_handler = MagicMock()
        after_handler = MagicMock()
        
        # Register handlers
        capability.register_event_handler("before_execution", before_handler)
        capability.register_event_handler("after_execution", after_handler)
        
        # Execute with mocked classification function
        with patch.object(capability, '_perform_classification', return_value={"categories": ["positive"], "confidence": 0.9}):
            result = await capability.execute_with_lifecycle(content="Sample content")
        
        # Check that the handlers were called with correct arguments
        before_handler.assert_called_once()
        after_handler.assert_called_once()
        
        # Verify correct parameters were passed to before_execution
        before_args = before_handler.call_args[1]
        assert before_args["capability_name"] == "classify_content"
        assert "params" in before_args
        
        # Verify result was passed to after_execution
        after_args = after_handler.call_args[1]
        assert after_args["capability_name"] == "classify_content"
        assert "result" in after_args
        
    @pytest.mark.asyncio
    async def test_perform_classification(self):
        """Test the internal classification function."""
        capability = ClassifyContentCapability()
        
        content = "I'm really happy with the service provided."
        classification_type = "sentiment"
        categories = ["positive", "negative", "neutral"]
        
        # Call the internal method directly
        result = await capability._perform_classification(
            content=content, 
            classification_type=classification_type,
            categories=categories,
            multi_label=False,
            confidence_threshold=0.5
        )
        
        # Check the result structure
        assert isinstance(result, dict)
        assert "classification_type" in result
        assert result["classification_type"] == classification_type
        assert "categories" in result
        assert isinstance(result["categories"], list)
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_execution_success(self):
        """Test successful execution of the capability."""
        capability = ClassifyContentCapability()
        
        # Execute with actual parameters
        result = await capability.execute_with_lifecycle(
            content="The customer was satisfied with our quick response.",
            classification_type="sentiment"
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        assert "metadata" in result
        
        # Check specific result content
        results = result["results"]
        assert "classification_type" in results
        assert results["classification_type"] == "sentiment"
        assert "categories" in results
        assert isinstance(results["categories"], list)
        
        # Check metadata
        metadata = result["metadata"]
        assert "execution_time" in metadata
        assert "input_length" in metadata
    
    @pytest.mark.asyncio
    async def test_multi_label_classification(self):
        """Test classification with multi-label option enabled."""
        capability = ClassifyContentCapability()
        
        # Execute with multi-label set to True
        result = await capability.execute_with_lifecycle(
            content="This text could fit into multiple categories.",
            classification_type="topics",
            multi_label=True,
            categories=["technology", "business", "education", "entertainment"]
        )
        
        # Check result structure for multi-label classification
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        
        # Check that categories is a list with potentially multiple items
        results = result["results"]
        assert "categories" in results
        assert isinstance(results["categories"], list)
        assert "confidence" in results
    
    @pytest.mark.asyncio
    async def test_execution_with_validation_error(self):
        """Test execution with validation error."""
        capability = ClassifyContentCapability()
        
        # Execute with missing required parameter
        result = await capability.execute_with_lifecycle(classification_type="sentiment")
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "content" in result["error"].lower()
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_execution_with_unexpected_error(self):
        """Test execution with unexpected runtime error."""
        capability = ClassifyContentCapability()
        
        # Mock the classification function to raise an exception
        error_message = "Classification service unavailable"
        with patch.object(capability, '_perform_classification', side_effect=Exception(error_message)):
            result = await capability.execute_with_lifecycle(
                content="This should fail during classification"
            )
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert error_message in result["error"]
        assert "error_type" in result
