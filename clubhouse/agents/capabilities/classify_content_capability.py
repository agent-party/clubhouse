"""
Content classification capability for categorizing text content.

This module provides a capability for classifying text content into categories,
supporting various classification types like sentiment, topics, or custom categories.
It follows the standardized capability pattern with proper parameter validation,
event triggering, and error handling.
"""

import time
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Literal, Union, Set

from pydantic import BaseModel, Field, field_validator, ValidationInfo

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError

# Configure logging
logger = logging.getLogger(__name__)


class ClassifyContentParameters(BaseModel):
    """Model for classify content parameters validation."""
    
    content: str = Field(..., description="The content to classify")
    classification_type: str = Field(
        default="general",
        description="Type of classification to perform (general, sentiment, topics, etc.)"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Optional list of target categories for classification"
    )
    multi_label: bool = Field(
        default=False,
        description="Whether to allow multiple category assignments (True) or only the best match (False)"
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence threshold for category assignment"
    )
    
    @field_validator('classification_type')
    def validate_classification_type(cls, v: str) -> str:
        """Validate that classification_type is one of the supported types."""
        valid_types = ["general", "sentiment", "topics", "categories", "content_moderation", "custom"]
        if v not in valid_types:
            raise ValueError(f"classification_type must be one of: {', '.join(valid_types)}")
        return v
    
    @field_validator('confidence_threshold')
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate that confidence_threshold is between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v
    
    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: Optional[List[str]], info: ValidationInfo) -> Optional[List[str]]:
        """Validate that categories are provided for classification."""
        # For "explicit" classification type, categories must be provided
        if info.data and "classification_type" in info.data and info.data["classification_type"] == "explicit" and (v is None or len(v) == 0):
            raise ValueError("Categories must be provided for explicit classification")
        return v


class ClassifyContentCapability(BaseCapability):
    """Capability for classifying content into categories."""
    
    name = "classify_content"
    description = "Classify content into categories based on various classification schemes"
    parameters_schema = ClassifyContentParameters
    
    # Base cost factors
    _base_cost = 0.004
    _cost_per_character = 0.0001
    _cost_multipliers = {
        "general": 1.0,
        "sentiment": 1.1,
        "topics": 1.3,
        "categories": 1.2,
        "content_moderation": 1.5,
        "custom": 1.4
    }
    
    # Default categories for built-in classification types
    _default_categories = {
        "sentiment": ["positive", "negative", "neutral"],
        "topics": ["technology", "business", "politics", "entertainment", "sports", "science", "health", "education"],
        "content_moderation": ["safe", "sensitive", "unsafe"]
    }
    
    def __init__(self) -> None:
        """Initialize the ClassifyContentCapability."""
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
            "content": {
                "type": "string",
                "description": "The content to classify",
                "required": True
            },
            "classification_type": {
                "type": "string",
                "description": "Type of classification to perform (general, sentiment, topics, etc.)",
                "required": False,
                "default": "general"
            },
            "categories": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Optional list of target categories for classification",
                "required": False
            },
            "multi_label": {
                "type": "boolean",
                "description": "Whether to allow multiple category assignments (True) or only the best match (False)",
                "required": False,
                "default": False
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence threshold for category assignment",
                "required": False,
                "default": 0.5
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
        Validate the parameters for the classify content capability.
        
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
            params = ClassifyContentParameters(**kwargs)
            
            # Additional validation for classification_type and categories
            classification_type = params.classification_type
            categories = params.categories
            
            # If classification_type is custom, ensure categories is provided
            if classification_type == "custom" and (categories is None or len(categories) == 0):
                raise ValueError("For custom classification type, categories must be provided")
            
            # If categories is not provided, use default categories for known classification types
            if categories is None and classification_type in self._default_categories:
                params.categories = self._default_categories[classification_type]
            
            return params.model_dump()
        except Exception as e:
            # Convert any Pydantic validation errors to our ValidationError
            error_msg = f"Parameter validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg, self.name)
    
    def _get_categories_for_type(self, classification_type: str, custom_categories: Optional[List[str]] = None) -> List[str]:
        """
        Get appropriate categories for the specified classification type.
        
        Args:
            classification_type: The type of classification
            custom_categories: Custom categories if provided
            
        Returns:
            List of categories to use for classification
        """
        if custom_categories is not None and len(custom_categories) > 0:
            return custom_categories
        
        # Use default categories for known classification types
        if classification_type in self._default_categories:
            return self._default_categories[classification_type]
        
        # For general classification
        return ["category_a", "category_b", "category_c", "other"]
    
    async def _perform_classification(
        self, 
        content: str, 
        classification_type: str,
        categories: Optional[List[str]] = None,
        multi_label: bool = False,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform the actual content classification operation.
        
        This method would connect to actual classification services in a real implementation.
        For demonstration purposes, it's a placeholder that mocks classification results
        and demonstrates proper cost tracking.
        
        Args:
            content: The content to classify
            classification_type: Type of classification to perform
            categories: Optional list of target categories
            multi_label: Whether to allow multiple category assignments
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary containing classification results
        """
        # Ensure we have appropriate categories
        actual_categories = self._get_categories_for_type(classification_type, categories)
        
        # Simulate processing time based on content length and classification complexity
        classification_complexity = len(content) * 0.001 * self._cost_multipliers.get(classification_type, 1.0)
        await asyncio.sleep(min(0.5, classification_complexity))  # Cap at 0.5 seconds for testing
        
        # Add to operation costs
        self.add_operation_cost("content_length", len(content) * self._cost_per_character)
        self.add_operation_cost(
            "classification_type", 
            self._base_cost * self._cost_multipliers.get(classification_type, 1.0)
        )
        
        # Generate mock classification results
        result_categories = []
        confidences = {}
        
        # Simplified classification logic based on type
        if classification_type == "sentiment":
            # Check for sentiment keywords in content
            positive_words = ["good", "great", "excellent", "happy", "pleased", "satisfied"]
            negative_words = ["bad", "poor", "terrible", "unhappy", "disappointed", "dissatisfied"]
            
            # Count occurrences of positive and negative words
            positive_count = sum(1 for word in positive_words if word in content.lower())
            negative_count = sum(1 for word in negative_words if word in content.lower())
            
            # Determine sentiment based on counts
            if positive_count > negative_count:
                primary_category = "positive"
                confidence = min(0.5 + (positive_count - negative_count) * 0.1, 0.95)
            elif negative_count > positive_count:
                primary_category = "negative"
                confidence = min(0.5 + (negative_count - positive_count) * 0.1, 0.95)
            else:
                primary_category = "neutral"
                confidence = 0.7
            
            # Add primary category
            result_categories.append(primary_category)
            confidences[primary_category] = confidence
            
            # For multi-label, potentially add secondary category
            if multi_label:
                if primary_category == "positive" and negative_count > 0:
                    result_categories.append("neutral")
                    confidences["neutral"] = 0.3
                elif primary_category == "negative" and positive_count > 0:
                    result_categories.append("neutral")
                    confidences["neutral"] = 0.3
        
        elif classification_type == "topics":
            # Create mock topic classification based on keywords
            topic_keywords = {
                "technology": ["computer", "software", "hardware", "tech", "digital", "internet", "app"],
                "business": ["company", "market", "finance", "economy", "investment", "trade", "business"],
                "politics": ["government", "policy", "election", "political", "democracy", "president"],
                "entertainment": ["movie", "music", "show", "celebrity", "film", "game", "entertainment"],
                "sports": ["team", "player", "game", "tournament", "championship", "win", "score"],
                "science": ["research", "discovery", "scientist", "study", "theory", "experiment"],
                "health": ["medical", "disease", "treatment", "doctor", "patient", "health", "drug"],
                "education": ["school", "student", "learn", "teacher", "course", "education", "training"]
            }
            
            # Count keyword matches for each topic
            matches = {}
            content_lower = content.lower()
            
            for topic, keywords in topic_keywords.items():
                count = sum(1 for keyword in keywords if keyword in content_lower)
                if count > 0:
                    matches[topic] = count
            
            # Sort topics by match count (descending)
            sorted_topics = sorted(matches.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_topics:
                # Add primary topic
                primary_topic = sorted_topics[0][0]
                primary_count = sorted_topics[0][1]
                
                result_categories.append(primary_topic)
                confidences[primary_topic] = min(0.5 + primary_count * 0.1, 0.95)
                
                # For multi-label, add additional topics above threshold
                if multi_label and len(sorted_topics) > 1:
                    for topic, count in sorted_topics[1:]:
                        confidence = min(0.4 + count * 0.08, 0.9)  # Slightly lower confidence for secondary topics
                        if confidence >= confidence_threshold:
                            result_categories.append(topic)
                            confidences[topic] = confidence
            else:
                # If no topic matches found
                result_categories.append("other")
                confidences["other"] = 0.6
        
        else:
            # For other classification types, generate random results
            # (In a real implementation, this would use appropriate models)
            if multi_label:
                # Select multiple categories with random confidences
                num_categories = random.randint(1, min(3, len(actual_categories)))
                selected_categories = random.sample(actual_categories, num_categories)
                
                for category in selected_categories:
                    confidence = random.uniform(0.5, 0.95)
                    if confidence >= confidence_threshold:
                        result_categories.append(category)
                        confidences[category] = confidence
            else:
                # Select single best category
                primary_category = random.choice(actual_categories)
                confidence = random.uniform(0.6, 0.95)
                
                result_categories.append(primary_category)
                confidences[primary_category] = confidence
        
        # Ensure we return at least one category if possible
        if not result_categories and actual_categories:
            fallback_category = actual_categories[0]
            result_categories.append(fallback_category)
            confidences[fallback_category] = confidence_threshold
        
        # Build detailed result
        result = {
            "classification_type": classification_type,
            "categories": result_categories,
            "confidence": confidences,
            "content_length": len(content),
            "multi_label": multi_label,
            "all_categories": actual_categories,
            "threshold": confidence_threshold
        }
        
        return result
    
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the classify content capability.
        
        For backwards compatibility, this method calls execute_with_lifecycle
        which provides the standardized execution flow.
        
        Args:
            **kwargs: The parameters for the classification operation
            
        Returns:
            CapabilityResult containing the classification results and metadata
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
            content = validated_params["content"]
            classification_type = validated_params.get("classification_type", "general")
            categories = validated_params.get("categories")
            multi_label = validated_params.get("multi_label", False)
            confidence_threshold = validated_params.get("confidence_threshold", 0.5)
            
            # Execute the classification
            operation_start_time = time.time()
            classification_results = await self._perform_classification(
                content=content, 
                classification_type=classification_type,
                categories=categories,
                multi_label=multi_label,
                confidence_threshold=confidence_threshold
            )
            execution_time = time.time() - operation_start_time
            
            # Format the result
            result_data = {
                "status": "success",
                "results": classification_results,
                "metadata": {
                    "execution_time": execution_time,
                    "input_length": len(content),
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
