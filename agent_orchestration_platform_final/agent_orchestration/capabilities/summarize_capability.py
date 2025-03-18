"""
SummarizeCapability implementation.

This module provides the SummarizeCapability for generating summaries of text content.
It has been refactored to follow best practices including:
1. Using Pydantic models for parameter validation
2. Leveraging BaseCapability's execute_with_lifecycle method
3. Implementing proper error handling
4. Maintaining backward compatibility with existing code
"""

import time
import logging
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, validator

from agent_orchestration.capabilities.base_capability import BaseCapability, CapabilityStatus
from agent_orchestration.infrastructure.errors import ValidationError, ExecutionError
from agent_orchestration.services.mcp_service import MCPService

logger = logging.getLogger(__name__)


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
    format: str = Field(
        "paragraph", 
        description="Format of the summary (paragraph, bullet_points, or key_points)"
    )
    
    @validator('format')
    def format_must_be_valid(cls, v):
        """Validate that format is one of the allowed values."""
        allowed_formats = ['paragraph', 'bullet_points', 'key_points']
        if v not in allowed_formats:
            raise ValueError(f"Format must be one of: {', '.join(allowed_formats)}")
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        extra = "forbid"


class SummarizeResponse(BaseModel):
    """Pydantic model for summarize capability response."""
    
    summary: str = Field(..., description="The generated summary")
    original_length: int = Field(..., description="Length of the original text")
    summary_length: int = Field(..., description="Length of the generated summary")
    compression_ratio: float = Field(..., description="Ratio of summary to original length")


class SummarizeCapability(BaseCapability):
    """
    Capability for generating summaries of text content.
    
    This capability uses the MCP service to generate summaries of text content.
    It has been refactored to use Pydantic models for validation and to follow
    the standard BaseCapability lifecycle.
    """
    
    def __init__(self, event_publisher, mcp_service: MCPService):
        """
        Initialize SummarizeCapability.
        
        Args:
            event_publisher: Event publisher for emitting events
            mcp_service: MCP service for generating summaries
        """
        super().__init__(event_publisher)
        self.mcp_service = mcp_service
        self.name = "summarize"
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate capability parameters using Pydantic.
        
        Args:
            parameters: Parameters to validate
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            SummarizeRequest(**parameters)
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            raise ValidationError(f"Invalid parameters: {str(e)}")
    
    async def execute(self, capability_id: str, conversation_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the summarize capability.
        
        This method maintains backward compatibility with existing code.
        It emits the custom events expected by existing tests.
        
        Args:
            capability_id: Unique identifier for this capability invocation
            conversation_id: Conversation ID
            parameters: Capability parameters
            
        Returns:
            Dictionary containing the summary result
            
        Raises:
            ValidationError: If parameters are invalid
            ExecutionError: If execution fails
        """
        self.validate_parameters(parameters)
        
        try:
            # Publish legacy start event for backward compatibility
            await self.publish_event(
                event_type="summarize_started",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.STARTED,
                metadata={
                    "parameters": parameters
                }
            )
            
            start_time = time.time()
            result = await self.execute_summarize(parameters)
            execution_time = time.time() - start_time
            
            # Publish legacy completion event for backward compatibility
            await self.publish_event(
                event_type="summarize_completed",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.COMPLETED,
                metadata={
                    "result": result,
                    "execution_time": execution_time
                }
            )
            
            return result
            
        except Exception as e:
            # Publish legacy error event for backward compatibility
            await self.publish_event(
                event_type="summarize_error",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.ERROR,
                metadata={
                    "error": str(e),
                    "parameters": parameters
                }
            )
            
            logger.error(f"Error executing summarize capability: {str(e)}", exc_info=e)
            raise ExecutionError(f"Error executing summarize capability: {str(e)}")
    
    async def _execute_implementation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation-specific execution for the summarize capability.
        
        This is called by execute_with_lifecycle from the BaseCapability.
        
        Args:
            parameters: Validated capability parameters
            
        Returns:
            Dictionary containing the summary result
        """
        return await self.execute_summarize(parameters)
    
    async def execute_summarize(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the summarize capability using MCP service.
        
        Args:
            parameters: Validated capability parameters
            
        Returns:
            Dictionary containing the summary result
        """
        try:
            # Prepare parameters for MCP service
            text = parameters["text"]
            max_length = parameters.get("max_length")
            format_type = parameters.get("format", "paragraph")
            
            # Generate summary using MCP service
            summary = await self.mcp_service.generate_summary(
                text=text,
                max_length=max_length,
                format=format_type
            )
            
            # Calculate statistics
            original_length = len(text)
            summary_length = len(summary)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            # Create and validate response
            response = SummarizeResponse(
                summary=summary,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio
            )
            
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in execute_summarize: {str(e)}", exc_info=e)
            raise ExecutionError(f"Failed to generate summary: {str(e)}")
    
    async def execute_with_lifecycle(
        self, capability_id: str, conversation_id: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute capability with standard lifecycle events.
        
        This overrides the BaseCapability method to add custom events
        for backward compatibility.
        
        Args:
            capability_id: Unique identifier for this capability invocation
            conversation_id: Conversation ID
            parameters: Capability parameters
            
        Returns:
            Capability execution result
        """
        # Validate parameters
        self.validate_parameters(parameters)
        
        # Start execution with standard lifecycle
        try:
            # Publish standard start event
            await self.publish_event(
                event_type="capability_started",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.STARTED,
                metadata={
                    "capability_name": self.name,
                    "parameters": parameters
                }
            )
            
            # Also publish legacy start event for backward compatibility
            await self.publish_event(
                event_type="summarize_started",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.STARTED,
                metadata={
                    "parameters": parameters
                }
            )
            
            # Execute the capability
            start_time = time.time()
            result = await self._execute_implementation(parameters)
            execution_time = time.time() - start_time
            
            # Publish standard completion event
            await self.publish_event(
                event_type="capability_completed",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.COMPLETED,
                metadata={
                    "capability_name": self.name,
                    "result": result,
                    "execution_time": execution_time
                }
            )
            
            # Also publish legacy completion event for backward compatibility
            await self.publish_event(
                event_type="summarize_completed",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.COMPLETED,
                metadata={
                    "result": result,
                    "execution_time": execution_time
                }
            )
            
            return result
            
        except Exception as e:
            # Publish standard error event
            await self.publish_event(
                event_type="capability_error",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.ERROR,
                metadata={
                    "capability_name": self.name,
                    "error": str(e),
                    "parameters": parameters
                }
            )
            
            # Also publish legacy error event for backward compatibility
            await self.publish_event(
                event_type="summarize_error",
                capability_id=capability_id,
                conversation_id=conversation_id,
                status=CapabilityStatus.ERROR,
                metadata={
                    "error": str(e),
                    "parameters": parameters
                }
            )
            
            logger.error(f"Error executing {self.name} capability: {str(e)}", exc_info=e)
            raise ExecutionError(f"Error executing {self.name} capability: {str(e)}")
