"""
Evolution Validation Service Implementation.

This module provides the EvolutionValidationService which is responsible for
validating evolution proposals before they are approved for implementation.
It creates validation plans, executes validation criteria, and manages the
overall validation process.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from agent_orchestration.core.service_base import ServiceBase
from agent_orchestration.evolution.models import (
    EvolutionProposal,
    ValidationPlan,
    ValidationStep,
    ValidationCriterion,
    ValidationResult,
    ProposalStatus,
)
from agent_orchestration.evolution.evolution_proposal_service import EvolutionProposalService
from agent_orchestration.infrastructure.errors import (
    ValidationError,
    ResourceNotFoundError,
    DuplicateResourceError,
    ExecutionError,
)
from agent_orchestration.integration.event_publisher import EventPublisher

logger = logging.getLogger(__name__)


class ValidationCriteriaRegistry:
    """Registry for validation criteria implementations."""
    
    def __init__(self):
        """Initialize criteria registry."""
        self.criteria: Dict[str, Any] = {}
    
    def register_criterion(self, criterion_type: str, implementation: Any) -> None:
        """
        Register a criterion implementation.
        
        Args:
            criterion_type: Type identifier for the criterion
            implementation: Implementation function or class
        """
        self.criteria[criterion_type] = implementation
    
    def get_criterion(self, criterion_type: str) -> Any:
        """
        Get a criterion implementation by type.
        
        Args:
            criterion_type: Type identifier for the criterion
            
        Returns:
            Criterion implementation
            
        Raises:
            ResourceNotFoundError: If criterion type not found
        """
        if criterion_type not in self.criteria:
            raise ResourceNotFoundError(f"Validation criterion type '{criterion_type}' not found")
        
        return self.criteria[criterion_type]


class EvolutionValidationService(ServiceBase):
    """Service for validating evolution proposals."""
    
    def __init__(
        self,
        proposal_service: EvolutionProposalService,
        event_publisher: EventPublisher,
        criteria_registry: Optional[ValidationCriteriaRegistry] = None,
        mcp_service: Optional[Any] = None,
    ):
        """
        Initialize EvolutionValidationService.
        
        Args:
            proposal_service: Service for accessing evolution proposals
            event_publisher: Event publisher for emitting events
            criteria_registry: Optional registry for validation criteria
            mcp_service: Optional MCP service for validation
        """
        self.proposal_service = proposal_service
        self.event_publisher = event_publisher
        self.criteria_registry = criteria_registry or ValidationCriteriaRegistry()
        self.mcp_service = mcp_service
        self.validation_plans: Dict[str, ValidationPlan] = {}
    
    async def create_validation_plan(
        self, proposal_id: str, custom_criteria: Optional[List[ValidationCriterion]] = None
    ) -> ValidationPlan:
        """
        Create a validation plan for an evolution proposal.
        
        Args:
            proposal_id: ID of the proposal to validate
            custom_criteria: Optional custom validation criteria
            
        Returns:
            Created validation plan
            
        Raises:
            ResourceNotFoundError: If proposal not found
            ValidationError: If plan creation fails
            DuplicateResourceError: If plan already exists for proposal
        """
        try:
            # Check if proposal exists
            proposal = await self.proposal_service.get_proposal(proposal_id)
            
            # Check if plan already exists
            existing_plans = [p for p in self.validation_plans.values() if p.proposal_id == proposal_id]
            if existing_plans:
                raise DuplicateResourceError(f"Validation plan already exists for proposal {proposal_id}")
            
            # Generate validation steps
            steps = await self._generate_validation_steps(proposal, custom_criteria)
            
            # Create validation plan
            plan = ValidationPlan(
                proposal_id=proposal_id,
                steps=steps,
                status="pending",
            )
            
            # Store plan
            self.validation_plans[plan.id] = plan
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.validation",
                key=plan.id,
                value={
                    "type": "validation_plan_created",
                    "plan_id": plan.id,
                    "proposal_id": proposal_id,
                    "step_count": len(steps),
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Created validation plan {plan.id} for proposal {proposal_id}")
            return plan
            
        except ResourceNotFoundError:
            logger.error(f"Proposal {proposal_id} not found")
            raise
            
        except Exception as e:
            logger.error(f"Error creating validation plan: {str(e)}", exc_info=e)
            raise ValidationError(f"Failed to create validation plan: {str(e)}")
    
    async def get_validation_plan(self, plan_id: str) -> ValidationPlan:
        """
        Get a validation plan by ID.
        
        Args:
            plan_id: ID of the validation plan
            
        Returns:
            Validation plan
            
        Raises:
            ResourceNotFoundError: If plan not found
        """
        if plan_id not in self.validation_plans:
            raise ResourceNotFoundError(f"Validation plan {plan_id} not found")
        
        return self.validation_plans[plan_id]
    
    async def get_validation_plans_for_proposal(self, proposal_id: str) -> List[ValidationPlan]:
        """
        Get validation plans for a specific proposal.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            List of validation plans for the proposal
        """
        # Filter plans for the proposal
        plans = [p for p in self.validation_plans.values() if p.proposal_id == proposal_id]
        
        # Sort by created_at (descending)
        return sorted(plans, key=lambda p: -p.created_at.timestamp())
    
    async def start_validation(self, plan_id: str) -> ValidationPlan:
        """
        Start executing a validation plan.
        
        Args:
            plan_id: ID of the validation plan
            
        Returns:
            Updated validation plan
            
        Raises:
            ResourceNotFoundError: If plan not found
            ValidationError: If plan cannot be started
        """
        if plan_id not in self.validation_plans:
            raise ResourceNotFoundError(f"Validation plan {plan_id} not found")
        
        plan = self.validation_plans[plan_id]
        
        if plan.status != "pending":
            raise ValidationError(f"Cannot start validation plan with status {plan.status}")
        
        # Update plan status and start time
        plan.status = "in_progress"
        plan.started_at = datetime.now()
        
        # Update proposal status
        await self.proposal_service.update_proposal_status(
            proposal_id=plan.proposal_id,
            status=ProposalStatus.VALIDATING,
            reason=f"Validation started with plan {plan_id}"
        )
        
        # Publish event
        await self.event_publisher.publish(
            topic="evolution.validation",
            key=plan.id,
            value={
                "type": "validation_started",
                "plan_id": plan.id,
                "proposal_id": plan.proposal_id,
                "timestamp": int(time.time() * 1000)
            }
        )
        
        logger.info(f"Started validation plan {plan.id}")
        
        # Execute validation asynchronously
        # In a real implementation, we would use a background task or queue
        # For now, we'll execute synchronously
        try:
            await self._execute_validation_plan(plan)
        except Exception as e:
            logger.error(f"Error executing validation plan: {str(e)}", exc_info=e)
            plan.status = "failed"
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.validation",
                key=plan.id,
                value={
                    "type": "validation_failed",
                    "plan_id": plan.id,
                    "proposal_id": plan.proposal_id,
                    "error": str(e),
                    "timestamp": int(time.time() * 1000)
                }
            )
        
        return plan
    
    async def _execute_validation_plan(self, plan: ValidationPlan) -> None:
        """
        Execute a validation plan by running all validation steps.
        
        Args:
            plan: Validation plan to execute
            
        Raises:
            ExecutionError: If validation execution fails
        """
        try:
            # Get proposal
            proposal = await self.proposal_service.get_proposal(plan.proposal_id)
            
            # Process each step in order
            for step in plan.steps:
                await self._execute_validation_step(step, proposal, plan)
            
            # Calculate overall results
            passed_criteria = 0
            total_criteria = 0
            
            for step in plan.steps:
                if step.results:
                    total_criteria += len(step.results)
                    passed_criteria += len([r for r in step.results if r.passed])
            
            pass_rate = passed_criteria / total_criteria if total_criteria > 0 else 0
            
            # Update plan status and completion time
            plan.status = "completed"
            plan.completed_at = datetime.now()
            plan.results_summary = {
                "total_criteria": total_criteria,
                "passed_criteria": passed_criteria,
                "pass_rate": pass_rate,
                "overall_result": "passed" if pass_rate >= 0.8 else "failed"
            }
            
            # Update proposal status based on validation results
            new_status = ProposalStatus.APPROVED if pass_rate >= 0.8 else ProposalStatus.REJECTED
            reason = f"Validation {plan.results_summary['overall_result']} with pass rate {pass_rate:.2f}"
            
            await self.proposal_service.update_proposal_status(
                proposal_id=plan.proposal_id,
                status=new_status,
                reason=reason
            )
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.validation",
                key=plan.id,
                value={
                    "type": "validation_completed",
                    "plan_id": plan.id,
                    "proposal_id": plan.proposal_id,
                    "results": plan.results_summary,
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Completed validation plan {plan.id} with result: {plan.results_summary['overall_result']}")
            
        except Exception as e:
            logger.error(f"Error executing validation plan: {str(e)}", exc_info=e)
            raise ExecutionError(f"Validation execution failed: {str(e)}")
    
    async def _execute_validation_step(
        self, step: ValidationStep, proposal: EvolutionProposal, plan: ValidationPlan
    ) -> None:
        """
        Execute a validation step by running all criteria.
        
        Args:
            step: Validation step to execute
            proposal: Proposal being validated
            plan: Validation plan
            
        Raises:
            ExecutionError: If step execution fails
        """
        try:
            # Update step status and start time
            step.status = "in_progress"
            step.started_at = datetime.now()
            
            # Initialize results list
            step.results = []
            
            # Execute each criterion
            for criterion in step.criteria:
                try:
                    # Get criterion implementation
                    criterion_impl = self.criteria_registry.get_criterion(criterion.validation_type)
                    
                    # Execute criterion
                    result = await criterion_impl(
                        criterion=criterion,
                        proposal=proposal,
                        step=step,
                        plan=plan,
                        mcp_service=self.mcp_service
                    )
                    
                    # Store result
                    step.results.append(ValidationResult(
                        criterion_id=criterion.id,
                        passed=result["passed"],
                        score=result.get("score"),
                        details=result["details"],
                    ))
                    
                except Exception as e:
                    logger.error(f"Error executing criterion {criterion.id}: {str(e)}", exc_info=e)
                    
                    # Add failed result
                    step.results.append(ValidationResult(
                        criterion_id=criterion.id,
                        passed=False,
                        score=0.0,
                        details=f"Execution error: {str(e)}",
                    ))
            
            # Update step status and completion time
            step.status = "completed"
            step.completed_at = datetime.now()
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.validation",
                key=plan.id,
                value={
                    "type": "validation_step_completed",
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "step_name": step.name,
                    "passed_criteria": len([r for r in step.results if r.passed]),
                    "total_criteria": len(step.results),
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Completed validation step {step.id} in plan {plan.id}")
            
        except Exception as e:
            logger.error(f"Error executing validation step: {str(e)}", exc_info=e)
            step.status = "failed"
            step.completed_at = datetime.now()
            raise ExecutionError(f"Validation step execution failed: {str(e)}")
    
    async def _generate_validation_steps(
        self, proposal: EvolutionProposal, custom_criteria: Optional[List[ValidationCriterion]] = None
    ) -> List[ValidationStep]:
        """
        Generate validation steps for a proposal.
        
        Args:
            proposal: Proposal to validate
            custom_criteria: Optional custom validation criteria
            
        Returns:
            List of validation steps
        """
        # Define default criteria
        default_steps = [
            ValidationStep(
                name="Static Analysis",
                description="Static analysis of the proposal structure and content",
                criteria=[
                    ValidationCriterion(
                        name="Proposal Structure",
                        description="Validate that the proposal has all required fields",
                        validation_type="structure_validator",
                        parameters={}
                    ),
                    ValidationCriterion(
                        name="Change Validation",
                        description="Validate that changes are well-defined",
                        validation_type="change_validator",
                        parameters={}
                    )
                ],
                order=1
            ),
            ValidationStep(
                name="Impact Assessment",
                description="Assessment of the potential impact of the proposal",
                criteria=[
                    ValidationCriterion(
                        name="Impact Level",
                        description="Validate the impact level is appropriate",
                        validation_type="impact_validator",
                        parameters={}
                    ),
                    ValidationCriterion(
                        name="Risk Assessment",
                        description="Assess risks associated with the proposal",
                        validation_type="risk_validator",
                        parameters={}
                    )
                ],
                order=2
            ),
            ValidationStep(
                name="Outcome Validation",
                description="Validation of expected outcomes",
                criteria=[
                    ValidationCriterion(
                        name="Outcome Measurability",
                        description="Validate that outcomes are measurable",
                        validation_type="outcome_validator",
                        parameters={}
                    ),
                    ValidationCriterion(
                        name="Success Criteria",
                        description="Validate success criteria are well-defined",
                        validation_type="criteria_validator",
                        parameters={}
                    )
                ],
                order=3
            )
        ]
        
        # Add custom criteria if provided
        if custom_criteria:
            # Create a new step for custom criteria
            custom_step = ValidationStep(
                name="Custom Validation",
                description="Custom validation criteria",
                criteria=custom_criteria,
                order=len(default_steps) + 1
            )
            default_steps.append(custom_step)
        
        # Add a system-specific step if MCP service is available
        if self.mcp_service:
            system_step = ValidationStep(
                name=f"{proposal.target_system.capitalize()} Validation",
                description=f"System-specific validation for {proposal.target_system}",
                criteria=[
                    ValidationCriterion(
                        name="System Compatibility",
                        description=f"Validate compatibility with {proposal.target_system}",
                        validation_type="system_validator",
                        parameters={"system": proposal.target_system}
                    ),
                    ValidationCriterion(
                        name="Implementation Feasibility",
                        description="Assess the feasibility of implementation",
                        validation_type="feasibility_validator",
                        parameters={}
                    )
                ],
                order=len(default_steps) + 1
            )
            default_steps.append(system_step)
        
        return default_steps
    
    # Example validator implementations
    async def structure_validator(
        self, criterion: ValidationCriterion, proposal: EvolutionProposal, **kwargs
    ) -> Dict[str, Any]:
        """
        Validate proposal structure.
        
        Args:
            criterion: Validation criterion
            proposal: Proposal to validate
            
        Returns:
            Validation result
        """
        # Check required fields
        required_fields = ["title", "description", "trigger", "changes", "expected_outcomes"]
        missing_fields = [field for field in required_fields if not getattr(proposal, field, None)]
        
        if missing_fields:
            return {
                "passed": False,
                "score": 0.0,
                "details": f"Missing required fields: {', '.join(missing_fields)}"
            }
        
        # Check non-empty lists
        list_fields = ["changes", "expected_outcomes"]
        empty_lists = [field for field in list_fields if not getattr(proposal, field, [])]
        
        if empty_lists:
            return {
                "passed": False,
                "score": 0.5,
                "details": f"Empty lists for fields: {', '.join(empty_lists)}"
            }
        
        return {
            "passed": True,
            "score": 1.0,
            "details": "Proposal structure is valid"
        }
    
    async def change_validator(
        self, criterion: ValidationCriterion, proposal: EvolutionProposal, **kwargs
    ) -> Dict[str, Any]:
        """
        Validate proposal changes.
        
        Args:
            criterion: Validation criterion
            proposal: Proposal to validate
            
        Returns:
            Validation result
        """
        # Check that change details are appropriate for change type
        issues = []
        
        for i, change in enumerate(proposal.changes):
            if change.type == "capability":
                if not isinstance(change.details, CapabilityChange):
                    issues.append(f"Change {i} has type 'capability' but invalid details")
            elif change.type == "service":
                if not isinstance(change.details, ServiceChange):
                    issues.append(f"Change {i} has type 'service' but invalid details")
            elif change.type == "model":
                if not isinstance(change.details, ModelChange):
                    issues.append(f"Change {i} has type 'model' but invalid details")
            
            if not change.rationale:
                issues.append(f"Change {i} is missing a rationale")
            if not change.impact_assessment:
                issues.append(f"Change {i} is missing an impact assessment")
        
        if issues:
            return {
                "passed": False,
                "score": 0.5,
                "details": "Change validation issues: " + "; ".join(issues)
            }
        
        return {
            "passed": True,
            "score": 1.0,
            "details": "All changes are valid"
        }


# Register built-in validators
def register_default_validators(service: EvolutionValidationService) -> None:
    """
    Register default validator implementations.
    
    Args:
        service: Validation service to register validators with
    """
    service.criteria_registry.register_criterion(
        "structure_validator", service.structure_validator
    )
    service.criteria_registry.register_criterion(
        "change_validator", service.change_validator
    )
