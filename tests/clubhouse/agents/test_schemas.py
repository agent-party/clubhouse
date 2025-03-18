"""
Tests for the Pydantic schema models for agent capabilities.

This module tests the parameter validation models and utility functions
defined in the clubhouse.agents.schemas module.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError
from typing import Dict, List, Any, Optional

from clubhouse.agents.schemas import (
    BaseCapabilityParams,
    SearchCapabilityParams,
    SummarizeCapabilityParams,
    validate_capability_params
)


class TestBaseCapabilityParams:
    """Tests for the base capability parameter model."""
    
    def test_base_capability_params_empty(self):
        """Test that BaseCapabilityParams can be created with no parameters."""
        params = BaseCapabilityParams()
        assert params.model_dump() == {}
    
    def test_base_capability_params_with_extra(self):
        """Test that BaseCapabilityParams allows extra fields."""
        params = BaseCapabilityParams(custom_field="value")
        assert params.custom_field == "value"
        assert "custom_field" in params.model_dump()


class TestSearchCapabilityParams:
    """Tests for the search capability parameter model."""
    
    def test_search_params_valid(self):
        """Test valid search parameters."""
        # Test with required parameters only
        params = SearchCapabilityParams(query="test query")
        assert params.query == "test query"
        assert params.max_results == 5  # Default value
        assert params.sources == ["knowledge_base"]  # Default value
        
        # Test with all parameters specified
        params = SearchCapabilityParams(
            query="another query",
            max_results=10,
            sources=["web", "documents"]
        )
        assert params.query == "another query"
        assert params.max_results == 10
        assert params.sources == ["web", "documents"]
    
    def test_search_params_required_validation(self):
        """Test validation of required parameters."""
        # Should raise ValidationError when query is missing
        with pytest.raises(PydanticValidationError) as excinfo:
            SearchCapabilityParams()
        
        # Check that the error mentions the missing field
        error_str = str(excinfo.value)
        assert "query" in error_str
        assert "Field required" in error_str
    
    def test_search_params_type_validation(self):
        """Test validation of parameter types."""
        # Invalid max_results type
        with pytest.raises(PydanticValidationError) as excinfo:
            SearchCapabilityParams(query="test", max_results="invalid")
        assert "max_results" in str(excinfo.value)
        
        # Invalid sources type
        with pytest.raises(PydanticValidationError) as excinfo:
            SearchCapabilityParams(query="test", sources="not_a_list")
        assert "sources" in str(excinfo.value)
    
    def test_search_params_constraints(self):
        """Test constraints on parameter values."""
        # max_results must be positive
        with pytest.raises(PydanticValidationError) as excinfo:
            SearchCapabilityParams(query="test", max_results=0)
        assert "max_results" in str(excinfo.value)
        
        # Sources can't be an empty list
        with pytest.raises(PydanticValidationError) as excinfo:
            SearchCapabilityParams(query="test", sources=[])
        assert "sources" in str(excinfo.value)


class TestSummarizeCapabilityParams:
    """Tests for the summarize capability parameter model."""
    
    def test_summarize_params_valid(self):
        """Test valid summarize parameters."""
        # Test with required parameters only
        params = SummarizeCapabilityParams(content="test content")
        assert params.content == "test content"
        assert params.max_length == 100  # Default value
        assert params.format == "paragraph"  # Default value
        
        # Test with all parameters specified
        params = SummarizeCapabilityParams(
            content="content to summarize",
            max_length=50,
            format="bullet_points"
        )
        assert params.content == "content to summarize"
        assert params.max_length == 50
        assert params.format == "bullet_points"
    
    def test_summarize_params_required_validation(self):
        """Test validation of required parameters."""
        # Should raise ValidationError when content is missing
        with pytest.raises(PydanticValidationError) as excinfo:
            SummarizeCapabilityParams()
        
        # Check that the error mentions the missing field
        error_str = str(excinfo.value)
        assert "content" in error_str
        assert "Field required" in error_str  # Updated for Pydantic V2 format
    
    def test_summarize_params_type_validation(self):
        """Test validation of parameter types."""
        # Invalid max_length type
        with pytest.raises(PydanticValidationError) as excinfo:
            SummarizeCapabilityParams(content="test", max_length="invalid")
        assert "max_length" in str(excinfo.value)
        
        # Invalid format type
        with pytest.raises(PydanticValidationError) as excinfo:
            SummarizeCapabilityParams(content="test", format=123)
        assert "format" in str(excinfo.value)
    
    def test_summarize_params_constraints(self):
        """Test constraints on parameter values."""
        # max_length must be at least 10
        with pytest.raises(PydanticValidationError) as excinfo:
            SummarizeCapabilityParams(content="test", max_length=5)
        assert "max_length" in str(excinfo.value)
        
        # format must be one of the allowed values
        with pytest.raises(PydanticValidationError) as excinfo:
            SummarizeCapabilityParams(content="test", format="invalid_format")
        assert "format" in str(excinfo.value)


class TestValidationUtility:
    """Tests for the validation utility functions."""
    
    def test_validate_capability_params_search(self):
        """Test validation utility with search parameters."""
        # Valid parameters
        params = {"query": "test query", "max_results": 10}
        validated = validate_capability_params("search", params)
        assert validated.query == "test query"
        assert validated.max_results == 10
        assert validated.sources == ["knowledge_base"]  # Default
        
        # Invalid parameters
        invalid_params = {"max_results": "not a number"}
        with pytest.raises(PydanticValidationError):
            validate_capability_params("search", invalid_params)
    
    def test_validate_capability_params_summarize(self):
        """Test validation utility with summarize parameters."""
        # Valid parameters
        params = {"content": "test content", "format": "bullet_points"}
        validated = validate_capability_params("summarize", params)
        assert validated.content == "test content"
        assert validated.format == "bullet_points"
        assert validated.max_length == 100  # Default
        
        # Invalid parameters
        invalid_params = {"format": "invalid_format"}
        with pytest.raises(PydanticValidationError):
            validate_capability_params("summarize", invalid_params)
    
    def test_validate_capability_params_unknown(self):
        """Test validation utility with unknown capability."""
        with pytest.raises(ValueError) as excinfo:
            validate_capability_params("unknown_capability", {})
        assert "Unknown capability" in str(excinfo.value)
        assert "unknown_capability" in str(excinfo.value)
