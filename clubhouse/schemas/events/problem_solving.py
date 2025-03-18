"""
Problem Solving Event Schema.

This module defines the event schemas for collaborative problem-solving,
tracking the stages, participants, and outcomes of problem-solving sessions.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, ClassVar, List, Set
from uuid import UUID

from pydantic import Field, field_validator, ConfigDict

from clubhouse.schemas.events.base import EventBase


class ProblemStage(str, Enum):
    """Stages in the problem-solving process."""
    
    INITIATED = "initiated"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    DISCUSSION = "discussion"
    SOLUTION_DRAFTING = "solution_drafting"
    SOLUTION_REVIEW = "solution_review"
    SOLUTION_FINALIZED = "solution_finalized"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ProblemSolvingEvent(EventBase):
    """
    Events related to collaborative problem-solving.
    
    These events track the progression of problem-solving sessions,
    including the agents involved, stage transitions, and outcomes.
    """
    
    kafka_topic: ClassVar[str] = "agent.problems"
    
    # Core problem fields
    problem_id: UUID = Field(..., description="Unique identifier for this problem")
    session_id: UUID = Field(..., description="Session in which this problem is being solved")
    title: str = Field(..., description="Short title describing the problem")
    description: str = Field(..., description="Detailed description of the problem")
    
    # Stage and progress tracking
    current_stage: ProblemStage = Field(..., description="Current stage in the problem-solving process")
    previous_stage: Optional[ProblemStage] = Field(
        default=None, description="Previous stage in the problem-solving process"
    )
    stage_started_at: datetime = Field(
        default_factory=datetime.now, description="When the current stage started"
    )
    
    # Participants
    initiator_id: str = Field(..., description="ID of the user or agent that initiated the problem")
    participating_agent_ids: List[str] = Field(
        default_factory=list, description="IDs of agents participating in solving the problem"
    )
    participating_user_ids: List[str] = Field(
        default_factory=list, description="IDs of users participating in solving the problem"
    )
    
    # Context and artifacts
    context_references: List[str] = Field(
        default_factory=list, description="References to relevant context (e.g., URLs, document IDs)"
    )
    artifact_references: List[str] = Field(
        default_factory=list, description="References to artifacts created during problem-solving"
    )
    
    # Metrics and outcomes
    metrics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metrics associated with the problem-solving process"
    )
    solution_status: Optional[str] = Field(
        default=None, 
        description="Status of the solution (e.g., 'proposed', 'approved', 'implemented')"
    )
    solution_quality: Optional[float] = Field(
        default=None, 
        description="Quality score of the solution (0.0 to 1.0)"
    )
    
    # Human-in-the-loop tracking
    requires_human_input: bool = Field(
        default=False, 
        description="Whether this stage requires human input to proceed"
    )
    human_input_type: Optional[str] = Field(
        default=None,
        description="Type of human input required (e.g., 'approval', 'clarification', 'evaluation')"
    )
    
    @field_validator("event_type", mode="before")
    @classmethod
    def set_event_type(cls, v: Optional[str]) -> str:
        """Set the event_type for problem solving events."""
        return "problem_solving"
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "event_id": "123e4567-e89b-12d3-a456-426614174000",
                    "event_type": "problem_solving",
                    "problem_id": "789e0123-e89b-12d3-a456-426614174000",
                    "session_id": "456e7890-e89b-12d3-a456-426614174000",
                    "title": "Optimize Database Queries",
                    "description": "Improve performance of customer search queries",
                    "current_stage": "analysis",
                    "previous_stage": "initiated",
                    "stage_started_at": "2025-03-16T14:30:00Z",
                    "initiator_id": "user-789",
                    "participating_agent_ids": ["agent-123", "agent-456"],
                    "participating_user_ids": ["user-789"],
                    "context_references": ["doc-123", "repo-456"],
                    "producer_id": "problem_solver_service",
                    "timestamp": "2025-03-16T14:30:00Z",
                    "requires_human_input": False
                }
            ]
        }
    )
