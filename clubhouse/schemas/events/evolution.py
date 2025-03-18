"""
Evolution Event Schema.

This module defines the event schemas for system evolution lifecycle,
tracking evolution proposals, approvals, and implementations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List, Union, Literal
from uuid import UUID

from pydantic import Field, field_validator, ConfigDict

from clubhouse.schemas.events.base import EventBase


class EvolutionStage(str, Enum):
    """Stages in the evolution process."""
    
    PROPOSAL_CREATED = "proposal_created"
    PROPOSAL_UPDATED = "proposal_updated"
    PROPOSAL_REVIEWED = "proposal_reviewed"
    PROPOSAL_APPROVED = "proposal_approved"
    PROPOSAL_REJECTED = "proposal_rejected"
    IMPLEMENTATION_STARTED = "implementation_started"
    IMPLEMENTATION_PROGRESS = "implementation_progress"
    IMPLEMENTATION_COMPLETED = "implementation_completed"
    IMPLEMENTATION_FAILED = "implementation_failed"
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_COMPLETED = "rollback_completed"
    EVOLUTION_COMPLETED = "evolution_completed"


class ImpactLevel(str, Enum):
    """Impact levels for evolution proposals."""
    
    CRITICAL = "critical"  # System-wide, fundamental changes
    MAJOR = "major"        # Significant component changes
    MODERATE = "moderate"  # Notable functionality changes
    MINOR = "minor"        # Small enhancements or optimizations
    PATCH = "patch"        # Bug fixes or minimal adjustments


class ComplexityLevel(str, Enum):
    """Complexity levels for implementation."""
    
    VERY_HIGH = "very_high"  # Requires major architectural changes
    HIGH = "high"            # Complex changes across multiple components
    MEDIUM = "medium"        # Moderate changes in specific components
    LOW = "low"              # Simple changes in well-defined areas
    VERY_LOW = "very_low"    # Trivial changes or configuration updates


class EvolutionEvent(EventBase):
    """
    Events tracking the evolution lifecycle.
    
    These events capture the full lifecycle of evolution proposals,
    from creation to implementation and validation.
    """
    
    kafka_topic: ClassVar[str] = "agent.evolution"
    
    # Core evolution fields
    evolution_id: UUID = Field(..., description="Unique identifier for this evolution")
    stage: EvolutionStage = Field(..., description="Current stage in the evolution process")
    target_system: str = Field(..., description="System or component being evolved")
    
    # Evolution details
    title: str = Field(..., description="Short title describing the evolution")
    description: str = Field(..., description="Detailed description of the evolution")
    impact_level: ImpactLevel = Field(..., description="Impact level of the evolution")
    complexity_level: ComplexityLevel = Field(..., description="Complexity level of implementation")
    
    # Tracking information
    proposed_by: str = Field(..., description="ID of the agent or user that proposed the evolution")
    approved_by: Optional[str] = Field(
        default=None, description="ID of the user that approved the evolution"
    )
    implemented_by: Optional[str] = Field(
        default=None, description="ID of the agent or user that implemented the evolution"
    )
    
    # Related observations
    trigger_observation_ids: List[UUID] = Field(
        default_factory=list, 
        description="IDs of observations that triggered this evolution"
    )
    
    # Implementation details
    implementation_plan: Optional[str] = Field(
        default=None, description="Plan for implementing the evolution"
    )
    implementation_progress: Optional[float] = Field(
        default=None, 
        description="Progress of implementation (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Changes and outcomes
    changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of specific changes made"
    )
    expected_outcomes: List[str] = Field(
        default_factory=list, description="Expected outcomes of the evolution"
    )
    actual_outcomes: Optional[List[str]] = Field(
        default=None, description="Actual outcomes observed after implementation"
    )
    
    # Validation and metrics
    validation_results: Optional[Dict[str, Any]] = Field(
        default=None, description="Results of validation tests"
    )
    performance_delta: Optional[Dict[str, float]] = Field(
        default=None, description="Performance changes after evolution"
    )
    
    # Human-in-the-loop
    requires_human_approval: bool = Field(
        default=True, description="Whether this evolution requires human approval"
    )
    human_feedback: Optional[str] = Field(
        default=None, description="Feedback from human reviewers"
    )
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type for evolution events."""
        return "evolution"
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "123e4567-e89b-12d3-a456-426614174000",
                    "event_type": "evolution",
                    "evolution_id": "345g6789-e89b-12d3-a456-426614174000",
                    "stage": "proposal_created",
                    "target_system": "search_capability",
                    "title": "Improve search relevance with semantic ranking",
                    "description": "Enhance search capability by adding semantic ranking to improve relevance of results.",
                    "impact_level": "moderate",
                    "complexity_level": "medium",
                    "proposed_by": "agent-456",
                    "trigger_observation_ids": ["234f5678-e89b-12d3-a456-426614174000"],
                    "expected_outcomes": [
                        "Improved search result relevance by 30%",
                        "Reduced need for query refinement"
                    ],
                    "requires_human_approval": True,
                    "producer_id": "evolution_service",
                    "timestamp": "2025-03-16T16:30:00Z"
                }
            ]
        }
    )
