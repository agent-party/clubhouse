"""
Tests for the refactored SummarizeCapability.

This test suite follows the test-driven development approach by defining
tests for the refactored SummarizeCapability before its implementation.
"""

import json
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

from pydantic import BaseModel, Field

# Import the existing test for reference patterns and compatibility
from agent_orchestration.capabilities.base_capability import BaseCapability, CapabilityStatus
from agent_orchestration.infrastructure.errors import ValidationError, ExecutionError


class SummarizeRequest(BaseModel):
    """Pydantic model for summarize capability parameters."""
    
    text: str = Field(
        ..., 
        description="The text to summarize",
        min_length=1
    )
    max_length: Optional[int] = Field(
        None, 
        description="Maximum length of the summary in characters",
        gt=0
    )
    format: Optional[str] = Field(
        "paragraph", 
        description="Format of the summary (paragraph, bullet_points, or key_points)"
    )
    
    class Config:
        """Pydantic model configuration."""
        
        extra = "forbid"


class SummarizeResponse(BaseModel):
    """Pydantic model for summarize capability response."""
    
    summary: str = Field(..., description="The generated summary")
    original_length: int = Field(..., description="Length of the original text")
    summary_length: int = Field(..., description="Length of the generated summary")
    compression_ratio: float = Field(..., description="Ratio of summary to original length")


class TestSummarizeCapability:
    """Test suite for the refactored SummarizeCapability."""
    
    @pytest.fixture
    def mock_event_publisher(self):
        """Create a mock event publisher."""
        publisher = MagicMock()
        publisher.publish = AsyncMock()
        return publisher
    
    @pytest.fixture
    def mock_mcp_service(self):
        """Create a mock MCP service."""
        service = MagicMock()
        service.generate_summary = AsyncMock(return_value="This is a test summary.")
        return service
    
    @pytest.fixture
    def summarize_capability(self, mock_event_publisher, mock_mcp_service):
        """Create an instance of SummarizeCapability with mocked dependencies."""
        from agent_orchestration.capabilities.summarize_capability import SummarizeCapability
        
        capability = SummarizeCapability(
            event_publisher=mock_event_publisher,
            mcp_service=mock_mcp_service
        )
        return capability
    
    @pytest.mark.asyncio
    async def test_parameter_validation(self, summarize_capability):
        """Test that parameters are properly validated using Pydantic."""
        # Valid parameters
        valid_params = {
            "text": "This is a test text that needs to be summarized.",
            "max_length": 50,
            "format": "paragraph"
        }
        
        # Should not raise an exception
        summarize_capability.validate_parameters(valid_params)
        
        # Invalid parameters - empty text
        invalid_params_empty = {
            "text": "",
            "max_length": 50
        }
        
        with pytest.raises(ValidationError):
            summarize_capability.validate_parameters(invalid_params_empty)
        
        # Invalid parameters - negative max_length
        invalid_params_negative = {
            "text": "This is a test.",
            "max_length": -10
        }
        
        with pytest.raises(ValidationError):
            summarize_capability.validate_parameters(invalid_params_negative)
        
        # Invalid parameters - unknown format
        invalid_params_format = {
            "text": "This is a test.",
            "format": "unknown_format"
        }
        
        with pytest.raises(ValidationError):
            summarize_capability.validate_parameters(invalid_params_format)
        
        # Invalid parameters - extra field
        invalid_params_extra = {
            "text": "This is a test.",
            "extra_field": "should not be allowed"
        }
        
        with pytest.raises(ValidationError):
            summarize_capability.validate_parameters(invalid_params_extra)
    
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle(self, summarize_capability, mock_event_publisher):
        """Test that execute_with_lifecycle properly handles the capability lifecycle."""
        # Arrange
        capability_id = "test-capability-id"
        conversation_id = "test-conversation-id"
        params = {
            "text": "This is a long text that needs to be summarized for testing purposes. It contains multiple sentences that provide context and information.",
            "max_length": 50
        }
        
        # Act
        result = await summarize_capability.execute_with_lifecycle(
            capability_id=capability_id,
            conversation_id=conversation_id,
            parameters=params
        )
        
        # Assert
        assert result is not None
        assert "summary" in result
        
        # Verify events were published
        events_published = [call.args[0]["type"] for call in mock_event_publisher.publish.call_args_list]
        
        # Should have these standard lifecycle events
        assert "capability_started" in events_published
        assert "capability_completed" in events_published
        
        # Should also have the specific capability events for backward compatibility
        assert "summarize_started" in events_published
        assert "summarize_completed" in events_published
    
    @pytest.mark.asyncio
    async def test_execute_summarize(self, summarize_capability, mock_mcp_service):
        """Test that execute_summarize correctly calls the MCP service and formats the response."""
        # Arrange
        mock_mcp_service.generate_summary.return_value = "This is a short summary."
        
        params = {
            "text": "This is a long text that needs to be summarized. It has multiple sentences and contains a lot of information that could be condensed.",
            "max_length": 30
        }
        
        # Act
        result = await summarize_capability.execute_summarize(params)
        
        # Assert
        assert result is not None
        assert result["summary"] == "This is a short summary."
        assert result["original_length"] > 0
        assert result["summary_length"] > 0
        assert 0 < result["compression_ratio"] < 1
        
        # Verify MCP service was called with correct parameters
        mock_mcp_service.generate_summary.assert_called_once()
        call_args = mock_mcp_service.generate_summary.call_args[0]
        assert params["text"] in call_args
        assert "max_length" in mock_mcp_service.generate_summary.call_args[1]
        assert mock_mcp_service.generate_summary.call_args[1]["max_length"] == params["max_length"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, summarize_capability, mock_mcp_service, mock_event_publisher):
        """Test that errors during summarization are properly handled and reported."""
        # Arrange
        mock_mcp_service.generate_summary.side_effect = Exception("Test error")
        
        capability_id = "test-capability-id"
        conversation_id = "test-conversation-id"
        params = {
            "text": "This is a test text."
        }
        
        # Act
        with pytest.raises(ExecutionError):
            await summarize_capability.execute_with_lifecycle(
                capability_id=capability_id,
                conversation_id=conversation_id,
                parameters=params
            )
        
        # Assert
        # Should have published capability_started and capability_error events
        events_published = [call.args[0]["type"] for call in mock_event_publisher.publish.call_args_list]
        assert "capability_started" in events_published
        assert "capability_error" in events_published
        
        # Should also have the specific capability events for backward compatibility
        assert "summarize_started" in events_published
        assert "summarize_error" in events_published
    
    @pytest.mark.asyncio
    async def test_backward_compatibility(self, summarize_capability, mock_event_publisher):
        """Test backward compatibility with existing code and tests."""
        # Arrange
        capability_id = "test-capability-id"
        conversation_id = "test-conversation-id"
        params = {
            "text": "This is a test text."
        }
        
        # Use the old execution method for backward compatibility
        result = await summarize_capability.execute(
            capability_id=capability_id,
            conversation_id=conversation_id,
            parameters=params
        )
        
        # Assert
        assert result is not None
        
        # Verify legacy events were published
        events_published = [call.args[0]["type"] for call in mock_event_publisher.publish.call_args_list]
        assert "summarize_started" in events_published
        assert "summarize_completed" in events_published


class TestSummarizeCapabilityIntegration:
    """Integration tests for SummarizeCapability with real dependencies."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_integration_with_event_system(self):
        """Test integration with the event system."""
        # This test would use the actual event publisher with a test Kafka instance
        # Skipping implementation details as it would require setting up Kafka
        pass
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_integration_with_mcp(self):
        """Test integration with the MCP service."""
        # This test would use a mocked MCP service that matches the real API
        # but doesn't actually make external calls
        pass


class TestCapabilityBaseClassIntegration:
    """Tests to ensure SummarizeCapability properly inherits from BaseCapability."""
    
    @pytest.fixture
    def summarize_capability_instance(self):
        """Create a SummarizeCapability instance for testing base class integration."""
        from agent_orchestration.capabilities.summarize_capability import SummarizeCapability
        
        capability = SummarizeCapability(
            event_publisher=MagicMock(),
            mcp_service=MagicMock()
        )
        return capability
    
    def test_is_base_capability_subclass(self, summarize_capability_instance):
        """Test that SummarizeCapability is a subclass of BaseCapability."""
        assert isinstance(summarize_capability_instance, BaseCapability)
    
    def test_implements_required_methods(self, summarize_capability_instance):
        """Test that SummarizeCapability implements all required methods."""
        # Check that all abstract methods are implemented
        assert hasattr(summarize_capability_instance, "validate_parameters")
        assert hasattr(summarize_capability_instance, "execute")
        
        # Check that it has access to base class methods
        assert hasattr(summarize_capability_instance, "execute_with_lifecycle")
        assert hasattr(summarize_capability_instance, "publish_event")


# Additional mocks and utilities for testing
class MockBaseSummarizeCapability(BaseCapability):
    """Mock implementation of BaseCapability for testing."""
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate capability parameters."""
        SummarizeRequest(**parameters)
    
    async def execute(self, capability_id: str, conversation_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the capability."""
        return await self.execute_with_lifecycle(capability_id, conversation_id, parameters)
    
    async def _execute_implementation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation-specific execution."""
        return {
            "summary": "Mock summary",
            "original_length": 100,
            "summary_length": 20,
            "compression_ratio": 0.2
        }
