"""
Analyze text capability for extracting insights from text content.

This module provides a capability for analyzing text data to extract
various insights such as sentiment, key concepts, entities, and other
text analytics. It follows the standardized capability pattern with
proper parameter validation, event triggering, and error handling.
"""

import time
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Literal, Union

from pydantic import BaseModel, Field, field_validator

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError

# Configure logging
logger = logging.getLogger(__name__)


class AnalyzeTextParameters(BaseModel):
    """Model for analyze text parameters validation."""
    
    text: str = Field(..., description="The text content to analyze")
    analysis_type: str = Field(
        default="general",
        description="Type of analysis to perform (general, sentiment, entities, concepts, etc.)"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Additional context to help guide the analysis"
    )
    max_insights: int = Field(
        default=10,
        description="Maximum number of insights to return"
    )
    
    @field_validator('analysis_type')
    def validate_analysis_type(cls, v: str) -> str:
        """Validate that analysis_type is one of the supported types."""
        valid_types = ["general", "sentiment", "entities", "concepts", "keywords", "syntax"]
        if v not in valid_types:
            raise ValueError(f"analysis_type must be one of: {', '.join(valid_types)}")
        return v
    
    @field_validator('max_insights')
    def validate_max_insights(cls, v: int) -> int:
        """Validate that max_insights is a positive integer."""
        if v <= 0:
            raise ValueError("max_insights must be a positive integer")
        return v


class AnalyzeTextCapability(BaseCapability):
    """Capability for analyzing text to extract insights."""
    
    name = "analyze_text"
    description = "Analyze text to extract insights such as sentiment, entities, concepts, and more"
    parameters_schema = AnalyzeTextParameters
    
    # Base cost factors
    _base_cost = 0.005
    _cost_per_character = 0.0001
    _cost_multipliers = {
        "general": 1.0,
        "sentiment": 1.2,
        "entities": 1.5,
        "concepts": 1.8,
        "keywords": 1.3,
        "syntax": 1.4
    }
    
    def __init__(self) -> None:
        """Initialize the AnalyzeTextCapability."""
        super().__init__()
        self._operation_costs: Dict[str, float] = {}
    
    def reset_operation_cost(self) -> None:
        """Reset the operation cost tracking."""
        self._operation_costs = {}
    
    def add_operation_cost(self, operation: str, cost: float) -> None:
        """
        Add a cost for a specific operation.
        
        Args:
            operation: The type of operation
            cost: The cost value to add
        """
        if operation in self._operation_costs:
            self._operation_costs[operation] += cost
        else:
            self._operation_costs[operation] = cost
    
    def get_operation_cost(self) -> Dict[str, float]:
        """
        Get the operation costs.
        
        Returns:
            Dictionary with operation types as keys and costs as values,
            plus a 'total' key with the sum of all costs.
        """
        costs = dict(self._operation_costs)
        costs["total"] = sum(costs.values())
        return costs
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters specification for this capability.
        
        Returns:
            Dictionary mapping parameter names to specifications
        """
        return {
            "text": {
                "type": "string",
                "description": "The text content to analyze",
                "required": True
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform (general, sentiment, entities, concepts, etc.)",
                "required": False,
                "default": "general"
            },
            "additional_context": {
                "type": "string",
                "description": "Additional context to help guide the analysis",
                "required": False
            },
            "max_insights": {
                "type": "integer",
                "description": "Maximum number of insights to return",
                "required": False,
                "default": 10
            }
        }
    
    def get_version(self) -> str:
        """
        Get the version of the capability.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        return "1.0.0"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this capability.
        
        For backwards compatibility with older capabilities.
        
        Returns:
            The parameters schema dictionary
        """
        return self.parameters
    
    def validate_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Validate the parameters for the analyze text capability.
        
        This method uses Pydantic for validation to ensure type safety
        and proper constraint checking. It overrides the base class's
        validate_parameters method to add capability-specific validation.
        
        Args:
            **kwargs: The parameters to validate
            
        Returns:
            Dictionary with validated parameters
            
        Raises:
            ValidationError: If parameters fail validation
        """
        try:
            # Use Pydantic model for validation
            params = AnalyzeTextParameters(**kwargs)
            return params.model_dump()
        except Exception as e:
            # Convert any Pydantic validation errors to our ValidationError
            error_msg = f"Parameter validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg, self.name)
    
    async def _perform_analysis(
        self, 
        text: str, 
        analysis_type: str,
        additional_context: Optional[str] = None,
        max_insights: int = 10
    ) -> Dict[str, Any]:
        """
        Perform the actual text analysis operation.
        
        This method would connect to actual text analysis services in a real implementation.
        For demonstration purposes, it's a placeholder that mocks analysis results
        and demonstrates proper cost tracking.
        
        Args:
            text: The text to analyze
            analysis_type: Type of analysis to perform
            additional_context: Additional context to guide analysis
            max_insights: Maximum number of insights to return
            
        Returns:
            Dictionary containing analysis results
        """
        # Simulate processing time based on text length and analysis type
        analysis_complexity = len(text) * 0.001 * self._cost_multipliers.get(analysis_type, 1.0)
        await asyncio.sleep(min(0.5, analysis_complexity))  # Cap at 0.5 seconds for testing
        
        # Add to operation costs
        self.add_operation_cost("text_length", len(text) * self._cost_per_character)
        self.add_operation_cost("analysis_type", self._base_cost * self._cost_multipliers.get(analysis_type, 1.0))
        
        # Generate mock insights based on analysis type
        insights = []
        
        if analysis_type == "sentiment":
            sentiment_options = ["positive", "negative", "neutral"]
            confidence = random.uniform(0.6, 0.95)
            
            # Simple sentiment heuristic for demo purposes
            sentiment = "neutral"
            if "happy" in text.lower() or "good" in text.lower() or "great" in text.lower():
                sentiment = "positive"
            elif "bad" in text.lower() or "unhappy" in text.lower() or "disappointed" in text.lower():
                sentiment = "negative"
                
            insights.append({
                "type": "sentiment",
                "value": sentiment,
                "confidence": confidence,
                "explanation": f"The text expresses a {sentiment} sentiment overall."
            })
            
        elif analysis_type == "entities":
            # Extract potential entities from the text (very simplistic for demo)
            words = text.split()
            capitalized_words = [word for word in words if word[0].isupper()]
            
            for i, word in enumerate(capitalized_words[:max_insights]):
                insights.append({
                    "type": "entity",
                    "value": word,
                    "entity_type": random.choice(["PERSON", "ORGANIZATION", "LOCATION"]),
                    "confidence": random.uniform(0.7, 0.9)
                })
                
        elif analysis_type == "keywords":
            # Simple extraction of potential keywords
            words = text.lower().split()
            # Filter out common stop words
            stop_words = ["the", "a", "an", "in", "to", "for", "with", "on", "at", "from", "by"]
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            for i, keyword in enumerate(list(set(keywords))[:max_insights]):
                insights.append({
                    "type": "keyword",
                    "value": keyword,
                    "relevance": random.uniform(0.6, 0.95)
                })
                
        else:  # general or other types
            # Generate generic insights
            insight_types = ["observation", "theme", "suggestion", "highlight"]
            
            for i in range(min(max_insights, 5)):
                insights.append({
                    "type": random.choice(insight_types),
                    "description": f"Insight {i+1} about the text content",
                    "confidence": random.uniform(0.6, 0.9)
                })
        
        # Create the result structure
        result = {
            "analysis_type": analysis_type,
            "text_length": len(text),
            "insights": insights[:max_insights],
            "processing_time": analysis_complexity
        }
        
        if additional_context:
            result["context_used"] = True
        
        return result
    
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the analyze text capability.
        
        For backwards compatibility, this method calls execute_with_lifecycle
        which provides the standardized execution flow.
        
        Args:
            **kwargs: The parameters for the analysis operation
            
        Returns:
            CapabilityResult containing the analysis results and metadata
        """
        # For backwards compatibility, delegate to execute_with_lifecycle
        result = await self.execute_with_lifecycle(**kwargs)
        
        # Return result in CapabilityResult format
        return CapabilityResult(
            result=result,
            metadata={"cost": self.get_operation_cost()}
        )
    
    async def execute_with_lifecycle(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the capability with parameters and standardized lifecycle events.
        
        This method handles parameter validation and triggers standard before/after events.
        
        Args:
            **kwargs: The parameters for the capability
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Reset cost tracking
            self.reset_operation_cost()
            self.add_operation_cost("base", self._base_cost)
            
            # Validate parameters
            validated_params = self.validate_parameters(**kwargs)
            
            # Trigger before_execution event
            self.trigger_event("before_execution", capability_name=self.name, params=validated_params)
            
            # Extract validated parameters
            text = validated_params["text"]
            analysis_type = validated_params.get("analysis_type", "general")
            additional_context = validated_params.get("additional_context")
            max_insights = validated_params.get("max_insights", 10)
            
            # Execute the analysis
            operation_start_time = time.time()
            analysis_results = await self._perform_analysis(
                text=text, 
                analysis_type=analysis_type,
                additional_context=additional_context,
                max_insights=max_insights
            )
            execution_time = time.time() - operation_start_time
            
            # Format the result
            result_data = {
                "status": "success",
                "results": analysis_results,
                "metadata": {
                    "execution_time": execution_time,
                    "input_length": len(text),
                    "cost": self.get_operation_cost()
                }
            }
            
            # Trigger after_execution event
            self.trigger_event("after_execution", 
                capability_name=self.name, 
                result=result_data,
                execution_time=execution_time
            )
            
            return result_data
        except ValidationError as ve:
            error_message = str(ve)
            logger.error(f"Validation error in {self.name}: {error_message}")
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type="ValidationError")
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": "ValidationError"
            }
        except Exception as e:
            # Handle any other exceptions using the ExecutionError framework
            execution_error = ExecutionError(
                f"Error in {self.name} execution: {str(e)}", 
                self.name
            )
            error_message = str(execution_error)
            logger.error(error_message)
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type=type(execution_error).__name__)
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": type(execution_error).__name__
            }
