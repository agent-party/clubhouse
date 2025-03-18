"""
Tests for the TranslateCapability implementation.

This module tests the functionality of the TranslateCapability, verifying that
it correctly handles parameter validation, execution, error handling,
and event triggering during the translation lifecycle.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.capability import CapabilityResult

# This will be implemented later
pytest.importorskip("clubhouse.agents.capabilities.translate_capability")
from clubhouse.agents.capabilities.translate_capability import TranslateCapability

class TestTranslateCapability:
    """Test suite for TranslateCapability."""
    
    def test_initialization(self):
        """Test capability initialization."""
        capability = TranslateCapability()
        assert capability.name == "translate"
        assert "Translate text" in capability.description
        assert isinstance(capability.parameters, dict)
        assert capability._operation_costs == {}
    
    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        capability = TranslateCapability()
        
        # Test with minimal required parameters
        params = {
            "text": "This is some text to translate for testing purposes.",
            "target_language": "es"
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["text"] == params["text"]
        assert validated["target_language"] == params["target_language"]
        assert validated["source_language"] == "auto"  # Default value
        
        # Test with all parameters
        params = {
            "text": "This is some text to translate for testing purposes.",
            "target_language": "fr",
            "source_language": "en",
            "preserve_formatting": True,
            "formality": "formal"
        }
        
        validated = capability.validate_parameters(**params)
        assert validated["text"] == params["text"]
        assert validated["target_language"] == params["target_language"]
        assert validated["source_language"] == params["source_language"]
        assert validated["preserve_formatting"] == params["preserve_formatting"]
        assert validated["formality"] == params["formality"]
    
    def test_parameter_validation_failure(self):
        """Test parameter validation with invalid parameters."""
        capability = TranslateCapability()
        
        # Missing required parameter - text
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(target_language="es")
        assert "text" in str(exc_info.value)
        
        # Missing required parameter - target_language
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(text="Some text")
        assert "target_language" in str(exc_info.value)
        
        # Invalid target_language (not a valid language code)
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                text="Some text", 
                target_language="not_a_real_language"
            )
        assert "target_language" in str(exc_info.value).lower()
        
        # Invalid formality option
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(
                text="Some text", 
                target_language="es",
                formality="super_formal"
            )
        assert "formality" in str(exc_info.value).lower()
    
    def test_cost_tracking(self):
        """Test operation cost tracking."""
        capability = TranslateCapability()
        
        # Add some costs
        capability.add_operation_cost("base", 0.01)
        capability.add_operation_cost("text_length", 0.05)
        capability.add_operation_cost("language_pair", 0.03)
        
        # Get the costs
        costs = capability.get_operation_cost()
        assert costs["base"] == 0.01
        assert costs["text_length"] == 0.05
        assert costs["language_pair"] == 0.03
        assert costs["total"] == 0.09
        
        # Reset the costs
        capability.reset_operation_cost()
        costs = capability.get_operation_cost()
        assert costs["total"] == 0
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and triggering."""
        capability = TranslateCapability()
        
        # Mock event handlers
        before_handler = MagicMock()
        after_handler = MagicMock()
        
        # Register handlers
        capability.register_event_handler("before_execution", before_handler)
        capability.register_event_handler("after_execution", after_handler)
        
        # Execute with mocked translation function
        with patch.object(capability, '_perform_translation', return_value={"translated_text": "Hola mundo"}):
            result = await capability.execute_with_lifecycle(text="Hello world", target_language="es")
        
        # Check that the handlers were called with correct arguments
        before_handler.assert_called_once()
        after_handler.assert_called_once()
        
        # Verify correct parameters were passed to before_execution
        before_args = before_handler.call_args[1]
        assert before_args["capability_name"] == "translate"
        assert "params" in before_args
        
        # Verify result was passed to after_execution
        after_args = after_handler.call_args[1]
        assert after_args["capability_name"] == "translate"
        assert "result" in after_args
        
    @pytest.mark.asyncio
    async def test_perform_translation(self):
        """Test the internal translation function."""
        capability = TranslateCapability()
        
        text = "Hello world"
        target_language = "es"
        
        # Call the internal method directly
        result = await capability._perform_translation(
            text=text, 
            target_language=target_language,
            source_language="en",
            preserve_formatting=True,
            formality="formal"
        )
        
        # Check the result structure
        assert isinstance(result, dict)
        assert "translated_text" in result
        assert "source_language" in result
        assert "target_language" in result
        assert result["target_language"] == target_language
    
    @pytest.mark.asyncio
    async def test_execution_success(self):
        """Test successful execution of the capability."""
        capability = TranslateCapability()
        
        # Execute with actual parameters
        result = await capability.execute_with_lifecycle(
            text="Hello world",
            target_language="es"
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        assert "metadata" in result
        
        # Check specific result content
        results = result["results"]
        assert "translated_text" in results
        assert "source_language" in results
        assert results["target_language"] == "es"
        
        # Check metadata
        metadata = result["metadata"]
        assert "execution_time" in metadata
        assert "cost" in metadata
    
    @pytest.mark.asyncio
    async def test_execution_with_validation_error(self):
        """Test execution with validation error."""
        capability = TranslateCapability()
        
        # Execute with missing required parameter
        result = await capability.execute_with_lifecycle(text="Hello world")
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "target_language" in result["error"].lower()
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_execution_with_unexpected_error(self):
        """Test execution with unexpected runtime error."""
        capability = TranslateCapability()
        
        # Mock the translation function to raise an exception
        error_message = "Translation service unavailable"
        with patch.object(capability, '_perform_translation', side_effect=Exception(error_message)):
            result = await capability.execute_with_lifecycle(
                text="This should fail during translation",
                target_language="fr"
            )
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert error_message in result["error"]
        assert "error_type" in result
        assert "Error" in result["error_type"]
