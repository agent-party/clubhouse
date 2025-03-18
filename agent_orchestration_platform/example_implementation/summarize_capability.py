"""
SummarizeCapability implementation following the MCP tools approach.

This file demonstrates the refactoring of the SummarizeCapability to:
1. Use Pydantic models for parameter validation
2. Leverage the BaseCapability's execute_with_lifecycle method
3. Use centralized error handling
4. Align with standard event patterns while maintaining backward compatibility
5. Follow consistent response formats
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, ClassVar
import uuid
from datetime import datetime
import logging
from enum import Enum

# Import abstract base classes and protocols
from agent_orchestration_platform.core.protocols import (
    AgentProtocol, 
    LLMServiceProtocol,
    ServiceRegistry
)
from agent_orchestration_platform.core.base_capability import (
    BaseCapability,
    CapabilityError,
    CapabilityResponse
)

# Configure logging
logger = logging.getLogger(__name__)

class SummarizeMode(str, Enum):
    """Enumeration of summary modes."""
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    EXECUTIVE = "executive"

class SummarizeParameters(BaseModel):
    """
    Parameters for text summarization.
    
    Validates and standardizes input parameters for the summarize capability.
    """
    text: str = Field(
        ..., 
        description="The text to summarize"
    )
    max_length: Optional[int] = Field(
        None, 
        description="Maximum length of the summary in characters",
        gt=0
    )
    focus_areas: Optional[List[str]] = Field(
        None, 
        description="Specific areas or topics to focus on in the summary"
    )
    mode: SummarizeMode = Field(
        SummarizeMode.CONCISE, 
        description="The type of summary to generate"
    )
    include_analytics: bool = Field(
        False, 
        description="Whether to include text analytics in the response"
    )
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        """Validate that text is not empty."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v

class SummarizeResponse(CapabilityResponse):
    """
    Response from the summarize capability.
    
    Contains the generated summary and optional analytics.
    """
    summary: str = Field(..., description="The generated summary")
    original_length: int = Field(..., description="Length of original text in characters")
    summary_length: int = Field(..., description="Length of summary in characters")
    compression_ratio: float = Field(..., description="Ratio of summary to original length")
    focus_coverage: Optional[Dict[str, float]] = Field(
        None, 
        description="Coverage score for each requested focus area (0.0-1.0)"
    )
    analytics: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional text analytics if requested"
    )

class SummarizeCapability(BaseCapability[SummarizeParameters, SummarizeResponse]):
    """
    Capability for generating summaries of text.
    
    Uses LLM services to create concise summaries with various options.
    """
    
    # Class variables for capability identification
    capability_name: ClassVar[str] = "summarize"
    capability_description: ClassVar[str] = "Generate a concise summary of provided text"
    capability_version: ClassVar[str] = "1.0.0"
    
    def __init__(
        self, 
        agent: AgentProtocol,
        service_registry: ServiceRegistry
    ):
        """
        Initialize the summarize capability.
        
        Args:
            agent: The agent this capability belongs to
            service_registry: Service registry for accessing required services
        """
        super().__init__(agent)
        self.service_registry = service_registry
        self.llm_service = service_registry.get_service(LLMServiceProtocol)
        
        if not self.llm_service:
            raise CapabilityError(
                "LLMServiceProtocol implementation required for SummarizeCapability"
            )
    
    async def execute(self, parameters: SummarizeParameters) -> SummarizeResponse:
        """
        Execute the capability with validated parameters.
        
        This is called by execute_with_lifecycle after parameter validation.
        
        Args:
            parameters: Validated parameters for summarization
            
        Returns:
            SummarizeResponse with the generated summary and metadata
            
        Raises:
            CapabilityError: If summarization fails
        """
        try:
            # For backward compatibility - emit the legacy event
            # In the future, this can be removed when all dependent code is updated
            await self.agent.emit_event("summarize_started", {
                "text_length": len(parameters.text)
            })
            
            # Generate the system prompt based on parameters
            system_prompt = self._build_system_prompt(parameters)
            
            # Calculate the appropriate max_tokens if needed
            max_tokens = self._calculate_max_tokens(parameters)
            
            # Generate summary using LLM
            try:
                summary_text = await self.llm_service.generate(
                    system_prompt=system_prompt,
                    user_prompt=parameters.text,
                    max_tokens=max_tokens,
                    temperature=0.5,  # Lower temperature for more factual summaries
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                raise CapabilityError(f"Failed to generate summary: {str(e)}")
            
            # Process focus areas if specified
            focus_coverage = None
            if parameters.focus_areas:
                focus_coverage = await self._analyze_focus_coverage(
                    summary_text, 
                    parameters.text, 
                    parameters.focus_areas
                )
            
            # Generate analytics if requested
            analytics = None
            if parameters.include_analytics:
                analytics = await self._generate_analytics(parameters.text, summary_text)
            
            # Calculate compression ratio
            original_length = len(parameters.text)
            summary_length = len(summary_text)
            compression_ratio = summary_length / original_length if original_length > 0 else 0
            
            # For backward compatibility - emit the legacy event
            # In the future, this can be removed when all dependent code is updated
            await self.agent.emit_event("summarize_completed", {
                "summary_length": summary_length
            })
            
            # Create and return the response
            return SummarizeResponse(
                summary=summary_text,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
                focus_coverage=focus_coverage,
                analytics=analytics,
                metadata={
                    "capability": self.capability_name,
                    "version": self.capability_version,
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": self.agent.id,
                    "request_id": str(uuid.uuid4())
                }
            )
            
        except Exception as e:
            # Log the error
            logger.error(f"Summarize capability execution error: {str(e)}")
            
            # Convert to CapabilityError and maintain the stack trace
            if not isinstance(e, CapabilityError):
                raise CapabilityError(f"Summarization failed: {str(e)}") from e
            raise
    
    def _build_system_prompt(self, parameters: SummarizeParameters) -> str:
        """
        Build a system prompt based on the summarization parameters.
        
        Args:
            parameters: The summarization parameters
            
        Returns:
            A system prompt string for the LLM
        """
        mode_instructions = {
            SummarizeMode.CONCISE: "Create a brief, concise summary capturing the key points.",
            SummarizeMode.DETAILED: "Create a detailed summary covering all important aspects.",
            SummarizeMode.BULLET_POINTS: "Create a bullet-point summary with each main point as a separate bullet.",
            SummarizeMode.EXECUTIVE: "Create an executive summary highlighting business impact and key decisions."
        }
        
        # Base prompt
        prompt = f"You are an expert summarizer. {mode_instructions[parameters.mode]}"
        
        # Add length constraint if specified
        if parameters.max_length:
            prompt += f" The summary should be no longer than {parameters.max_length} characters."
        
        # Add focus areas if specified
        if parameters.focus_areas:
            areas = ", ".join(parameters.focus_areas)
            prompt += f" Focus particularly on these aspects: {areas}."
        
        return prompt
    
    def _calculate_max_tokens(self, parameters: SummarizeParameters) -> int:
        """
        Calculate the appropriate max_tokens value for LLM generation.
        
        Args:
            parameters: The summarization parameters
            
        Returns:
            The max_tokens value to use
        """
        # If max_length is specified, convert to approximate token count (assuming ~4 chars per token)
        if parameters.max_length:
            return min(parameters.max_length // 4 + 10, 1000)  # Add padding, cap at 1000
        
        # Default values based on mode
        mode_token_limits = {
            SummarizeMode.CONCISE: 200,
            SummarizeMode.DETAILED: 800,
            SummarizeMode.BULLET_POINTS: 500,
            SummarizeMode.EXECUTIVE: 400
        }
        
        return mode_token_limits[parameters.mode]
    
    async def _analyze_focus_coverage(
        self,
        summary: str,
        original_text: str,
        focus_areas: List[str]
    ) -> Dict[str, float]:
        """
        Analyze how well the summary covers each focus area.
        
        Args:
            summary: The generated summary
            original_text: The original text
            focus_areas: List of focus areas to analyze
            
        Returns:
            Dictionary mapping focus areas to coverage scores (0.0-1.0)
        """
        # In a full implementation, this would use semantic analysis
        # For this example, we'll use a simple keyword-based approach
        
        coverage = {}
        
        for area in focus_areas:
            # Convert to lowercase for case-insensitive matching
            area_lower = area.lower()
            summary_lower = summary.lower()
            original_lower = original_text.lower()
            
            # Count occurrences in original and summary
            original_count = original_lower.count(area_lower)
            summary_count = summary_lower.count(area_lower)
            
            # Calculate normalized coverage
            if original_count > 0:
                # Normalize by the compression ratio to account for length difference
                compression = len(summary) / len(original_text)
                expected_count = original_count * compression
                
                # Calculate coverage score (capped at 1.0)
                if expected_count > 0:
                    coverage_score = min(summary_count / expected_count, 1.0)
                else:
                    coverage_score = 0.0
            else:
                # If focus area wasn't in original, score is 0
                coverage_score = 0.0
            
            coverage[area] = coverage_score
        
        return coverage
    
    async def _generate_analytics(
        self,
        original_text: str,
        summary: str
    ) -> Dict[str, Any]:
        """
        Generate analytics for the text and summary.
        
        Args:
            original_text: The original text
            summary: The generated summary
            
        Returns:
            Dictionary of analytics data
        """
        # In a full implementation, this would do more sophisticated analysis
        # For this example, we'll provide basic metrics
        
        # Calculate word counts
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        # Calculate sentence counts (rough approximation)
        original_sentences = original_text.count(". ") + original_text.count("! ") + original_text.count("? ")
        summary_sentences = summary.count(". ") + summary.count("! ") + summary.count("? ")
        
        # Calculate average words per sentence
        avg_words_original = original_words / max(original_sentences, 1)
        avg_words_summary = summary_words / max(summary_sentences, 1)
        
        return {
            "word_count": {
                "original": original_words,
                "summary": summary_words,
                "reduction_percent": (original_words - summary_words) / original_words * 100 if original_words > 0 else 0
            },
            "sentence_count": {
                "original": original_sentences,
                "summary": summary_sentences
            },
            "avg_words_per_sentence": {
                "original": avg_words_original,
                "summary": avg_words_summary
            },
            "summary_density": summary_words / max(original_words, 1)
        }
