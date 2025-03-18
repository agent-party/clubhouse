"""
Observation Event Schema.

This module defines the event schemas for system observations that may
trigger evolution proposals, including user feedback, errors, and metrics.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List, Union, Literal
from uuid import UUID

from pydantic import Field, field_validator, ConfigDict, root_validator

from clubhouse.schemas.events.base import EventBase


class ObservationSource(str, Enum):
    """Sources of system observations."""
    
    USER_FEEDBACK = "user_feedback"
    AGENT_FEEDBACK = "agent_feedback"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_LOG = "error_log"
    USAGE_PATTERN = "usage_pattern"
    SECURITY_AUDIT = "security_audit"
    SYSTEM_HEALTH = "system_health"


class ObservationCategory(str, Enum):
    """Categories of system observations."""
    
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"


class ImportanceLevel(str, Enum):
    """Importance levels for observations."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ObservationEvent(EventBase):
    """
    Events representing system observations that may trigger evolution.
    
    These events capture feedback, metrics, errors, and patterns that could
    indicate opportunities for system improvement or evolution.
    """
    
    kafka_topic: ClassVar[str] = "agent.observations"
    
    # Core observation fields
    observation_id: UUID = Field(..., description="Unique identifier for this observation")
    source: ObservationSource = Field(..., description="Source of the observation")
    category: ObservationCategory = Field(..., description="Category of the observation")
    target_system: str = Field(..., description="System or component being observed")
    
    # Observation details
    title: str = Field(..., description="Short title describing the observation")
    description: str = Field(..., description="Detailed description of the observation")
    importance: ImportanceLevel = Field(..., description="Importance level of the observation")
    importance_score: float = Field(
        ..., 
        description="Numerical importance score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Context and data
    context: Optional[str] = Field(
        default=None, description="Context in which the observation was made"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Structured data associated with the observation"
    )
    references: List[str] = Field(
        default_factory=list, 
        description="References to related observations or artifacts"
    )
    
    # Source-specific fields
    user_id: Optional[str] = Field(
        default=None, description="ID of the user providing feedback (for user_feedback)"
    )
    agent_id: Optional[str] = Field(
        default=None, description="ID of the agent providing feedback (for agent_feedback)"
    )
    metric_name: Optional[str] = Field(
        default=None, description="Name of the metric (for performance_metric)"
    )
    metric_value: Optional[Union[float, int, str, bool]] = Field(
        default=None, description="Value of the metric (for performance_metric)"
    )
    error_type: Optional[str] = Field(
        default=None, description="Type of error (for error_log)"
    )
    
    # Evolution tracking
    triggered_evolution: bool = Field(
        default=False, description="Whether this observation triggered an evolution proposal"
    )
    evolution_proposal_id: Optional[UUID] = Field(
        default=None, description="ID of the evolution proposal triggered by this observation"
    )
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type for observation events."""
        return "observation"
    
    @root_validator
    @classmethod
    def validate_source_specific_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that source-specific fields are provided."""
        source = values.get("source")
        if source == ObservationSource.USER_FEEDBACK and not values.get("user_id"):
            raise ValueError("user_id is required for user_feedback observations")
        elif source == ObservationSource.AGENT_FEEDBACK and not values.get("agent_id"):
            raise ValueError("agent_id is required for agent_feedback observations")
        elif source == ObservationSource.PERFORMANCE_METRIC and (
            not values.get("metric_name") or values.get("metric_value") is None
        ):
            raise ValueError("metric_name and metric_value are required for performance_metric observations")
        elif source == ObservationSource.ERROR_LOG and not values.get("error_type"):
            raise ValueError("error_type is required for error_log observations")
        return values
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "123e4567-e89b-12d3-a456-426614174000",
                    "event_type": "observation",
                    "observation_id": "234f5678-e89b-12d3-a456-426614174000",
                    "source": "user_feedback",
                    "category": "usability",
                    "target_system": "agent_cli",
                    "title": "Difficult to understand agent responses",
                    "description": "Users report that agent responses are too technical and hard to understand.",
                    "importance": "high",
                    "importance_score": 0.8,
                    "user_id": "user-123",
                    "data": {
                        "feedback_rating": 2,
                        "session_id": "session-456"
                    },
                    "producer_id": "feedback_service",
                    "timestamp": "2025-03-16T15:45:00Z"
                }
            ]
        }
    )
