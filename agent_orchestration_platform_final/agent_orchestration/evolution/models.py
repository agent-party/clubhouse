"""
Models for the Evolution Engine.

This module defines the Pydantic models used throughout the Evolution Engine.
These models ensure proper validation and type safety for all operations.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator, root_validator


class ImpactLevel(str, Enum):
    """Impact level of an evolution proposal."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplexityLevel(str, Enum):
    """Complexity level of an evolution proposal implementation."""
    
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ProposalStatus(str, Enum):
    """Status of an evolution proposal."""
    
    DRAFT = "draft"
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    FAILED = "failed"


class EvolutionTrigger(BaseModel):
    """Trigger that initiated an evolution proposal."""
    
    type: str = Field(..., description="Type of trigger (performance, user_feedback, error, opportunity)")
    source: str = Field(..., description="Source of the trigger (agent_id, service, user_id)")
    description: str = Field(..., description="Description of the trigger")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Metrics associated with the trigger")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the trigger occurred")


class CapabilityChange(BaseModel):
    """Change to a capability as part of an evolution proposal."""
    
    capability_name: str = Field(..., description="Name of the capability to modify")
    change_type: str = Field(..., description="Type of change (add, modify, remove)")
    description: str = Field(..., description="Description of the change")
    code_changes: Optional[Dict[str, Any]] = Field(None, description="Code changes required")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameter changes")
    test_changes: Optional[Dict[str, Any]] = Field(None, description="Test changes required")


class ServiceChange(BaseModel):
    """Change to a service as part of an evolution proposal."""
    
    service_name: str = Field(..., description="Name of the service to modify")
    change_type: str = Field(..., description="Type of change (add, modify, remove)")
    description: str = Field(..., description="Description of the change")
    api_changes: Optional[Dict[str, Any]] = Field(None, description="API changes required")
    implementation_changes: Optional[Dict[str, Any]] = Field(None, description="Implementation changes required")
    config_changes: Optional[Dict[str, Any]] = Field(None, description="Configuration changes required")
    test_changes: Optional[Dict[str, Any]] = Field(None, description="Test changes required")


class ModelChange(BaseModel):
    """Change to a model configuration as part of an evolution proposal."""
    
    model_name: str = Field(..., description="Name of the model to modify")
    change_type: str = Field(..., description="Type of change (add, modify, remove)")
    description: str = Field(..., description="Description of the change")
    parameter_changes: Optional[Dict[str, Any]] = Field(None, description="Parameter changes")
    prompt_changes: Optional[Dict[str, Any]] = Field(None, description="Prompt template changes")


class Change(BaseModel):
    """Generic change as part of an evolution proposal."""
    
    type: str = Field(..., description="Type of change (capability, service, model, other)")
    description: str = Field(..., description="Description of the change")
    details: Union[CapabilityChange, ServiceChange, ModelChange, Dict[str, Any]] = Field(
        ..., description="Details of the change"
    )
    rationale: str = Field(..., description="Rationale for the change")
    impact_assessment: str = Field(..., description="Assessment of the impact of the change")


class ExpectedOutcome(BaseModel):
    """Expected outcome of an evolution proposal."""
    
    description: str = Field(..., description="Description of the expected outcome")
    metrics: Dict[str, Any] = Field(..., description="Metrics to measure the outcome")
    validation_criteria: List[str] = Field(..., description="Criteria to validate the outcome")


class EvolutionProposal(BaseModel):
    """Evolution proposal model."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique proposal ID")
    title: str = Field(..., description="Title of the proposal")
    description: str = Field(..., description="Description of the proposal")
    trigger: EvolutionTrigger = Field(..., description="Trigger that initiated the proposal")
    target_system: str = Field(..., description="Target system for the evolution")
    changes: List[Change] = Field(..., description="Proposed changes")
    expected_outcomes: List[ExpectedOutcome] = Field(..., description="Expected outcomes")
    impact_level: ImpactLevel = Field(..., description="Level of impact")
    complexity: ComplexityLevel = Field(..., description="Implementation complexity")
    status: ProposalStatus = Field(default=ProposalStatus.DRAFT, description="Current status")
    agent_id: str = Field(..., description="ID of the agent that created the proposal")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator("changes")
    def validate_changes(cls, changes):
        """Validate that there is at least one change."""
        if not changes:
            raise ValueError("At least one change is required")
        return changes
    
    @root_validator
    def update_timestamps(cls, values):
        """Update timestamps on validation."""
        values["updated_at"] = datetime.now()
        return values


class ValidationCriterion(BaseModel):
    """Validation criterion for an evolution proposal."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique criterion ID")
    name: str = Field(..., description="Name of the criterion")
    description: str = Field(..., description="Description of the criterion")
    validation_type: str = Field(..., description="Type of validation (static, dynamic, expert)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for validation")


class ValidationResult(BaseModel):
    """Result of validating a criterion."""
    
    criterion_id: str = Field(..., description="ID of the criterion")
    passed: bool = Field(..., description="Whether the criterion passed")
    score: Optional[float] = Field(None, description="Score for the criterion (0-1)")
    details: str = Field(..., description="Details of the validation result")
    timestamp: datetime = Field(default_factory=datetime.now, description="Validation timestamp")


class ValidationStep(BaseModel):
    """Step in the validation process."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique step ID") 
    name: str = Field(..., description="Name of the step")
    description: str = Field(..., description="Description of the step")
    criteria: List[ValidationCriterion] = Field(..., description="Criteria to validate")
    order: int = Field(..., description="Order of execution")
    results: Optional[List[ValidationResult]] = Field(None, description="Results of validation")
    status: str = Field(default="pending", description="Status of the step")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class ValidationPlan(BaseModel):
    """Plan for validating an evolution proposal."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique plan ID")
    proposal_id: str = Field(..., description="ID of the proposal being validated")
    steps: List[ValidationStep] = Field(..., description="Validation steps")
    status: str = Field(default="pending", description="Status of the plan")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    results_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of validation results")
    
    @validator("steps")
    def validate_steps(cls, steps):
        """Validate that there is at least one step and steps have unique orders."""
        if not steps:
            raise ValueError("At least one validation step is required")
        
        orders = [step.order for step in steps]
        if len(orders) != len(set(orders)):
            raise ValueError("Validation steps must have unique order values")
        
        return sorted(steps, key=lambda step: step.order)


class ExecutionStep(BaseModel):
    """Step in the execution of an evolution proposal."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique step ID")
    name: str = Field(..., description="Name of the step")
    description: str = Field(..., description="Description of the step")
    action_type: str = Field(..., description="Type of action to perform")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the action")
    order: int = Field(..., description="Order of execution")
    status: str = Field(default="pending", description="Status of the step")
    result: Optional[Dict[str, Any]] = Field(None, description="Result of execution")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class ExecutionPlan(BaseModel):
    """Plan for executing an evolution proposal."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique plan ID")
    proposal_id: str = Field(..., description="ID of the proposal being executed")
    steps: List[ExecutionStep] = Field(..., description="Execution steps")
    status: str = Field(default="pending", description="Status of the plan")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    results_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of execution results")
    
    @validator("steps")
    def validate_steps(cls, steps):
        """Validate that there is at least one step and steps have unique orders."""
        if not steps:
            raise ValueError("At least one execution step is required")
        
        orders = [step.order for step in steps]
        if len(orders) != len(set(orders)):
            raise ValueError("Execution steps must have unique order values")
        
        return sorted(steps, key=lambda step: step.order)


class ObservationSource(str, Enum):
    """Source of an observation."""
    
    USER_FEEDBACK = "user_feedback"
    AGENT_FEEDBACK = "agent_feedback"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_LOG = "error_log"
    USAGE_PATTERN = "usage_pattern"


class Observation(BaseModel):
    """Observation that may trigger an evolution proposal."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique observation ID")
    source: ObservationSource = Field(..., description="Source of the observation")
    target_system: str = Field(..., description="System being observed")
    category: str = Field(..., description="Category of observation")
    description: str = Field(..., description="Description of the observation")
    data: Dict[str, Any] = Field(..., description="Observation data")
    importance: float = Field(..., description="Importance score (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Observation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
