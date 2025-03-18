"""
Pydantic schema models for capability parameters.

This module provides standardized parameter validation using Pydantic models
for all agent capabilities. It ensures consistent validation, type checking,
and error handling for capability parameters.
"""

from typing import Dict, List, Any, Optional, Union, Type, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError


class BaseCapabilityParams(BaseModel):
    """
    Base model for capability parameters.
    
    This model serves as the foundation for all capability parameter models
    and allows for arbitrary extra fields for flexibility.
    """
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields not defined in the model


class SearchCapabilityParams(BaseCapabilityParams):
    """
    Parameters for the search capability.
    
    This model defines and validates the parameters required for executing
    a search capability, with appropriate defaults and constraints.
    """
    
    query: str = Field(
        ...,  # ... means required
        description="Search query string"
    )
    
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1  # Must be greater than or equal to 1
    )
    
    sources: List[str] = Field(
        default=["knowledge_base"],
        description="Sources to search (e.g. knowledge_base, web, documents)",
        min_length=1  # Must have at least one source
    )
    
    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: Any) -> list[str]:
        """Ensure sources is a list with at least one element."""
        if not isinstance(v, list) or len(v) < 1:
            raise ValueError("At least one source must be specified")
        return v


class SummarizeCapabilityParams(BaseCapabilityParams):
    """
    Parameters for the summarize capability.
    
    This model defines and validates the parameters required for executing
    a summarization capability, with appropriate defaults and constraints.
    """
    
    content: str = Field(
        ...,  # ... means required
        description="Content to summarize"
    )
    
    max_length: int = Field(
        default=100,
        description="Maximum length of summary in words",
        ge=10  # Must be greater than or equal to 10
    )
    
    format: Literal["paragraph", "bullet_points", "key_points"] = Field(
        default="paragraph",
        description="Format of the summary (paragraph, bullet_points, key_points)"
    )


# Registry of capability parameter models
CAPABILITY_PARAM_MODELS: Dict[str, Type[BaseCapabilityParams]] = {
    "search": SearchCapabilityParams,
    "summarize": SummarizeCapabilityParams,
}


def validate_capability_params(
    capability_name: str,
    params: Dict[str, Any]
) -> BaseCapabilityParams:
    """
    Validate parameters for a specific capability using the appropriate Pydantic model.
    
    This function selects the correct parameter model based on the capability name
    and validates the provided parameters against that model.
    
    Args:
        capability_name: The name of the capability
        params: The parameters to validate as a dictionary
        
    Returns:
        Validated parameters as a Pydantic model instance
        
    Raises:
        ValueError: If the capability is unknown
        ValidationError: If the parameters fail validation
    """
    # Get the appropriate model for this capability
    model_class = CAPABILITY_PARAM_MODELS.get(capability_name)
    if not model_class:
        raise ValueError(f"Unknown capability: {capability_name}")
    
    # Validate the parameters using the model
    return model_class(**params)
