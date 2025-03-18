"""
Tests for the enhanced SearchCapability implementation.

This module tests the SearchCapability with the integrated error handling
framework and Pydantic validation system.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from clubhouse.agents.capability import BaseCapability
from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.schemas import SearchCapabilityParams
from clubhouse.agents.capabilities.search_capability import SearchCapability
from clubhouse.agents.capability import CapabilityResult


class TestSearchCapability:
    """Test suite for the SearchCapability implementation."""

    def test_initialization(self):
        """Test basic initialization of SearchCapability."""
        cap = SearchCapability()
        assert cap.name == "search"
        assert cap.description == "Search for information in knowledge bases and other sources"
        assert not cap.requires_human_approval()
        assert cap._event_handlers == {}
        assert isinstance(cap._operation_cost, dict)

    def test_parameters_schema(self):
        """Test that the parameters property defines the correct schema."""
        cap = SearchCapability()
        params = cap.parameters
        
        assert "query" in params
        assert params["query"]["required"] is True
        assert "max_results" in params
        assert "sources" in params
        assert params["max_results"]["default"] == 5
        assert params["sources"]["default"] == ["knowledge_base"]

    def test_validation_with_valid_params(self):
        """Test parameter validation with valid parameters."""
        cap = SearchCapability()
        
        # Test with minimal parameters
        valid_params = {"query": "test query"}
        validated = cap.validate_parameters(**valid_params)
        assert validated["query"] == "test query"
        assert validated["max_results"] == 5  # Default value
        assert validated["sources"] == ["knowledge_base"]  # Default value
        
        # Test with all parameters specified
        valid_params = {
            "query": "full test query",
            "max_results": 10,
            "sources": ["web", "documents"]
        }
        validated = cap.validate_parameters(**valid_params)
        assert validated["query"] == "full test query"
        assert validated["max_results"] == 10
        assert validated["sources"] == ["web", "documents"]

    def test_validation_with_invalid_params(self):
        """Test parameter validation with invalid parameters."""
        cap = SearchCapability()
        
        # Test with missing required parameter
        with pytest.raises(ValidationError) as exc_info:
            cap.validate_parameters()
        assert "query" in str(exc_info.value).lower()
        
        # Test with invalid max_results type
        with pytest.raises(ValidationError) as exc_info:
            cap.validate_parameters(query="test", max_results="invalid")
        assert "max_results" in str(exc_info.value).lower()
        
        # Test with invalid sources type
        with pytest.raises(ValidationError) as exc_info:
            cap.validate_parameters(query="test", sources="not_a_list")
        assert "sources" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execution_success(self):
        """Test successful execution of the search capability."""
        cap = SearchCapability()
        
        # Mock the internal search function
        with patch.object(cap, '_perform_search', return_value=["result1", "result2"]):
            result = await cap.execute_with_lifecycle(query="test query")
        
        # Check the result has the expected structure (dictionary format after refactoring)
        assert isinstance(result, dict)
        assert "results" in result
        assert "metadata" in result
        
        # Verify the content
        assert len(result["results"]) == 2
        assert "execution_time" in result["metadata"]

    @pytest.mark.asyncio
    async def test_execution_error(self):
        """Test handling of execution errors."""
        cap = SearchCapability()
        
        # Mock the internal search function to raise an exception
        error_message = "Search engine unavailable"
        with patch.object(cap, '_perform_search', side_effect=Exception(error_message)):
            result = await cap.execute_with_lifecycle(query="test query")
        
        # Check error details in the result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert error_message in result["error"]

    @pytest.mark.asyncio
    async def test_event_triggering(self):
        """Test that events are properly triggered during execution."""
        cap = SearchCapability()
        
        # Use a more explicit handler function that matches expected signature
        def before_handler(**kwargs):
            print(f"Before handler called with: {kwargs}")
        
        def after_handler(**kwargs):
            print(f"After handler called with: {kwargs}")
        
        before_mock = MagicMock(side_effect=before_handler)
        after_mock = MagicMock(side_effect=after_handler)
        
        # Register handlers
        cap.register_event_handler("before_execution", before_mock)
        cap.register_event_handler("after_execution", after_mock)
        
        # Execute the capability
        with patch.object(cap, '_perform_search', return_value=["result"]):
            await cap.execute_with_lifecycle(query="test query")
        
        # Verify the handlers were called with the right parameters
        before_mock.assert_called_once()
        after_mock.assert_called_once()
        
        # Check that the capability name was passed
        assert before_mock.call_args[1]["capability_name"] == "search"
        
        # Check that the parameters were passed to before_execution
        assert "query" in before_mock.call_args[1]["params"]
        
        # Check that the result was passed to after_execution
        assert "result" in after_mock.call_args[1]

    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test operation cost tracking during execution."""
        cap = SearchCapability()
        
        # Execute the capability with mocked search
        with patch.object(cap, '_perform_search', return_value=["result"]):
            await cap.execute_with_lifecycle(query="test query")
        
        # Check that costs were recorded
        costs = cap.get_operation_cost()
        assert costs["total"] > 0
        assert "query" in costs
