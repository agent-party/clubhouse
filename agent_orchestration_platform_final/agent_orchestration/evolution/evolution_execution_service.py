"""
Evolution Execution Service Implementation.

This module provides the EvolutionExecutionService which is responsible for
executing approved evolution proposals by implementing the proposed changes.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

from agent_orchestration.core.service_base import ServiceBase
from agent_orchestration.evolution.models import (
    EvolutionProposal,
    ExecutionPlan,
    ExecutionStep,
    ProposalStatus,
    Change,
    CapabilityChange,
    ServiceChange,
    ModelChange
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


class ActionRegistry:
    """Registry for execution action implementations."""
    
    def __init__(self):
        """Initialize action registry."""
        self.actions: Dict[str, Any] = {}
    
    def register_action(self, action_type: str, implementation: Any) -> None:
        """
        Register an action implementation.
        
        Args:
            action_type: Type identifier for the action
            implementation: Implementation function or class
        """
        self.actions[action_type] = implementation
    
    def get_action(self, action_type: str) -> Any:
        """
        Get an action implementation by type.
        
        Args:
            action_type: Type identifier for the action
            
        Returns:
            Action implementation
            
        Raises:
            ResourceNotFoundError: If action type not found
        """
        if action_type not in self.actions:
            raise ResourceNotFoundError(f"Execution action type '{action_type}' not found")
        
        return self.actions[action_type]


class EvolutionExecutionService(ServiceBase):
    """Service for executing approved evolution proposals."""
    
    def __init__(
        self,
        proposal_service: EvolutionProposalService,
        event_publisher: EventPublisher,
        action_registry: Optional[ActionRegistry] = None,
        mcp_service: Optional[Any] = None,
    ):
        """
        Initialize EvolutionExecutionService.
        
        Args:
            proposal_service: Service for accessing evolution proposals
            event_publisher: Event publisher for emitting events
            action_registry: Optional registry for execution actions
            mcp_service: Optional MCP service for execution
        """
        self.proposal_service = proposal_service
        self.event_publisher = event_publisher
        self.action_registry = action_registry or ActionRegistry()
        self.mcp_service = mcp_service
        self.execution_plans: Dict[str, ExecutionPlan] = {}
    
    async def create_execution_plan(self, proposal_id: str) -> ExecutionPlan:
        """
        Create an execution plan for an approved evolution proposal.
        
        Args:
            proposal_id: ID of the proposal to execute
            
        Returns:
            Created execution plan
            
        Raises:
            ResourceNotFoundError: If proposal not found
            ValidationError: If proposal is not approved or plan creation fails
            DuplicateResourceError: If plan already exists for proposal
        """
        try:
            # Check if proposal exists and is approved
            proposal = await self.proposal_service.get_proposal(proposal_id)
            
            if proposal.status != ProposalStatus.APPROVED:
                raise ValidationError(f"Cannot execute proposal with status {proposal.status}")
            
            # Check if plan already exists
            existing_plans = [p for p in self.execution_plans.values() if p.proposal_id == proposal_id]
            if existing_plans:
                raise DuplicateResourceError(f"Execution plan already exists for proposal {proposal_id}")
            
            # Generate execution steps
            steps = await self._generate_execution_steps(proposal)
            
            # Create execution plan
            plan = ExecutionPlan(
                proposal_id=proposal_id,
                steps=steps,
                status="pending",
            )
            
            # Store plan
            self.execution_plans[plan.id] = plan
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.execution",
                key=plan.id,
                value={
                    "type": "execution_plan_created",
                    "plan_id": plan.id,
                    "proposal_id": proposal_id,
                    "step_count": len(steps),
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Created execution plan {plan.id} for proposal {proposal_id}")
            return plan
            
        except ResourceNotFoundError:
            logger.error(f"Proposal {proposal_id} not found")
            raise
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {str(e)}", exc_info=e)
            raise ValidationError(f"Failed to create execution plan: {str(e)}")
    
    async def get_execution_plan(self, plan_id: str) -> ExecutionPlan:
        """
        Get an execution plan by ID.
        
        Args:
            plan_id: ID of the execution plan
            
        Returns:
            Execution plan
            
        Raises:
            ResourceNotFoundError: If plan not found
        """
        if plan_id not in self.execution_plans:
            raise ResourceNotFoundError(f"Execution plan {plan_id} not found")
        
        return self.execution_plans[plan_id]
    
    async def get_execution_plans_for_proposal(self, proposal_id: str) -> List[ExecutionPlan]:
        """
        Get execution plans for a specific proposal.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            List of execution plans for the proposal
        """
        # Filter plans for the proposal
        plans = [p for p in self.execution_plans.values() if p.proposal_id == proposal_id]
        
        # Sort by created_at (descending)
        return sorted(plans, key=lambda p: -p.created_at.timestamp())
    
    async def start_execution(self, plan_id: str) -> ExecutionPlan:
        """
        Start executing an approved plan.
        
        Args:
            plan_id: ID of the execution plan
            
        Returns:
            Updated execution plan
            
        Raises:
            ResourceNotFoundError: If plan not found
            ValidationError: If plan cannot be started
        """
        if plan_id not in self.execution_plans:
            raise ResourceNotFoundError(f"Execution plan {plan_id} not found")
        
        plan = self.execution_plans[plan_id]
        
        if plan.status != "pending":
            raise ValidationError(f"Cannot start execution plan with status {plan.status}")
        
        # Update plan status and start time
        plan.status = "in_progress"
        plan.started_at = datetime.now()
        
        # Update proposal status
        await self.proposal_service.update_proposal_status(
            proposal_id=plan.proposal_id,
            status=ProposalStatus.IMPLEMENTED,
            reason=f"Execution started with plan {plan_id}"
        )
        
        # Publish event
        await self.event_publisher.publish(
            topic="evolution.execution",
            key=plan.id,
            value={
                "type": "execution_started",
                "plan_id": plan.id,
                "proposal_id": plan.proposal_id,
                "timestamp": int(time.time() * 1000)
            }
        )
        
        logger.info(f"Started execution plan {plan.id}")
        
        # Execute plan asynchronously
        # In a real implementation, we would use a background task or queue
        # For now, we'll execute synchronously
        try:
            await self._execute_plan(plan)
        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}", exc_info=e)
            plan.status = "failed"
            
            # Update proposal status
            await self.proposal_service.update_proposal_status(
                proposal_id=plan.proposal_id,
                status=ProposalStatus.FAILED,
                reason=f"Execution failed: {str(e)}"
            )
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.execution",
                key=plan.id,
                value={
                    "type": "execution_failed",
                    "plan_id": plan.id,
                    "proposal_id": plan.proposal_id,
                    "error": str(e),
                    "timestamp": int(time.time() * 1000)
                }
            )
        
        return plan
    
    async def _execute_plan(self, plan: ExecutionPlan) -> None:
        """
        Execute a plan by running all execution steps.
        
        Args:
            plan: Execution plan to execute
            
        Raises:
            ExecutionError: If execution fails
        """
        try:
            # Get proposal
            proposal = await self.proposal_service.get_proposal(plan.proposal_id)
            
            # Process each step in order
            for step in plan.steps:
                await self._execute_step(step, proposal, plan)
            
            # Calculate overall results
            successful_steps = len([step for step in plan.steps if step.status == "completed"])
            total_steps = len(plan.steps)
            
            # Update plan status and completion time
            plan.status = "completed"
            plan.completed_at = datetime.now()
            plan.results_summary = {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
                "overall_result": "success" if successful_steps == total_steps else "partial"
            }
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.execution",
                key=plan.id,
                value={
                    "type": "execution_completed",
                    "plan_id": plan.id,
                    "proposal_id": plan.proposal_id,
                    "results": plan.results_summary,
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Completed execution plan {plan.id} with result: {plan.results_summary['overall_result']}")
            
        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}", exc_info=e)
            raise ExecutionError(f"Execution failed: {str(e)}")
    
    async def _execute_step(
        self, step: ExecutionStep, proposal: EvolutionProposal, plan: ExecutionPlan
    ) -> None:
        """
        Execute a step of the execution plan.
        
        Args:
            step: Execution step to execute
            proposal: Proposal being executed
            plan: Execution plan
            
        Raises:
            ExecutionError: If step execution fails
        """
        try:
            # Update step status and start time
            step.status = "in_progress"
            step.started_at = datetime.now()
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.execution",
                key=plan.id,
                value={
                    "type": "execution_step_started",
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "step_name": step.name,
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            # Get action implementation
            action_impl = self.action_registry.get_action(step.action_type)
            
            # Execute action
            result = await action_impl(
                step=step,
                proposal=proposal,
                plan=plan,
                mcp_service=self.mcp_service
            )
            
            # Store result
            step.result = result
            
            # Update step status and completion time
            step.status = "completed"
            step.completed_at = datetime.now()
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.execution",
                key=plan.id,
                value={
                    "type": "execution_step_completed",
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "step_name": step.name,
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            logger.info(f"Completed execution step {step.id} in plan {plan.id}")
            
        except Exception as e:
            logger.error(f"Error executing step: {str(e)}", exc_info=e)
            step.status = "failed"
            step.completed_at = datetime.now()
            step.result = {"error": str(e)}
            
            # Publish event
            await self.event_publisher.publish(
                topic="evolution.execution",
                key=plan.id,
                value={
                    "type": "execution_step_failed",
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "step_name": step.name,
                    "error": str(e),
                    "timestamp": int(time.time() * 1000)
                }
            )
            
            raise ExecutionError(f"Execution step failed: {str(e)}")
    
    async def _generate_execution_steps(self, proposal: EvolutionProposal) -> List[ExecutionStep]:
        """
        Generate execution steps for a proposal.
        
        Args:
            proposal: Proposal to execute
            
        Returns:
            List of execution steps
        """
        steps = []
        order = 1
        
        # Create backup step
        backup_step = ExecutionStep(
            name="Create System Backup",
            description="Create a backup of the current system state before making changes",
            action_type="create_backup",
            parameters={
                "target_system": proposal.target_system,
                "backup_type": "full"
            },
            order=order
        )
        steps.append(backup_step)
        order += 1
        
        # Create steps for each change
        for i, change in enumerate(proposal.changes):
            change_steps = await self._create_steps_for_change(change, order + i)
            steps.extend(change_steps)
            order += len(change_steps)
        
        # Create verification step
        verification_step = ExecutionStep(
            name="Verify Changes",
            description="Verify that all changes have been applied correctly",
            action_type="verify_changes",
            parameters={
                "target_system": proposal.target_system
            },
            order=order
        )
        steps.append(verification_step)
        order += 1
        
        # Create documentation step
        documentation_step = ExecutionStep(
            name="Update Documentation",
            description="Update system documentation to reflect the changes",
            action_type="update_documentation",
            parameters={
                "target_system": proposal.target_system,
                "changes": [c.dict() for c in proposal.changes]
            },
            order=order
        )
        steps.append(documentation_step)
        
        return steps
    
    async def _create_steps_for_change(self, change: Change, start_order: int) -> List[ExecutionStep]:
        """
        Create execution steps for a specific change.
        
        Args:
            change: Change to create steps for
            start_order: Starting order for steps
            
        Returns:
            List of execution steps for the change
        """
        steps = []
        order = start_order
        
        if change.type == "capability":
            if isinstance(change.details, CapabilityChange):
                if change.details.change_type == "add":
                    steps.append(ExecutionStep(
                        name=f"Add Capability: {change.details.capability_name}",
                        description=f"Add new capability: {change.details.description}",
                        action_type="add_capability",
                        parameters={
                            "capability_name": change.details.capability_name,
                            "code_changes": change.details.code_changes or {},
                            "parameters": change.details.parameters or {}
                        },
                        order=order
                    ))
                    order += 1
                    
                    steps.append(ExecutionStep(
                        name=f"Test Capability: {change.details.capability_name}",
                        description=f"Run tests for the new capability",
                        action_type="test_capability",
                        parameters={
                            "capability_name": change.details.capability_name,
                            "test_changes": change.details.test_changes or {}
                        },
                        order=order
                    ))
                    
                elif change.details.change_type == "modify":
                    steps.append(ExecutionStep(
                        name=f"Modify Capability: {change.details.capability_name}",
                        description=f"Modify existing capability: {change.details.description}",
                        action_type="modify_capability",
                        parameters={
                            "capability_name": change.details.capability_name,
                            "code_changes": change.details.code_changes or {},
                            "parameters": change.details.parameters or {}
                        },
                        order=order
                    ))
                    order += 1
                    
                    steps.append(ExecutionStep(
                        name=f"Test Modified Capability: {change.details.capability_name}",
                        description=f"Run tests for the modified capability",
                        action_type="test_capability",
                        parameters={
                            "capability_name": change.details.capability_name,
                            "test_changes": change.details.test_changes or {}
                        },
                        order=order
                    ))
                    
                elif change.details.change_type == "remove":
                    steps.append(ExecutionStep(
                        name=f"Remove Capability: {change.details.capability_name}",
                        description=f"Remove capability: {change.details.description}",
                        action_type="remove_capability",
                        parameters={
                            "capability_name": change.details.capability_name
                        },
                        order=order
                    ))
        
        elif change.type == "service":
            if isinstance(change.details, ServiceChange):
                if change.details.change_type == "add":
                    steps.append(ExecutionStep(
                        name=f"Add Service: {change.details.service_name}",
                        description=f"Add new service: {change.details.description}",
                        action_type="add_service",
                        parameters={
                            "service_name": change.details.service_name,
                            "implementation_changes": change.details.implementation_changes or {},
                            "api_changes": change.details.api_changes or {},
                            "config_changes": change.details.config_changes or {}
                        },
                        order=order
                    ))
                    order += 1
                    
                    steps.append(ExecutionStep(
                        name=f"Test Service: {change.details.service_name}",
                        description=f"Run tests for the new service",
                        action_type="test_service",
                        parameters={
                            "service_name": change.details.service_name,
                            "test_changes": change.details.test_changes or {}
                        },
                        order=order
                    ))
                    
                elif change.details.change_type == "modify":
                    steps.append(ExecutionStep(
                        name=f"Modify Service: {change.details.service_name}",
                        description=f"Modify existing service: {change.details.description}",
                        action_type="modify_service",
                        parameters={
                            "service_name": change.details.service_name,
                            "implementation_changes": change.details.implementation_changes or {},
                            "api_changes": change.details.api_changes or {},
                            "config_changes": change.details.config_changes or {}
                        },
                        order=order
                    ))
                    order += 1
                    
                    steps.append(ExecutionStep(
                        name=f"Test Modified Service: {change.details.service_name}",
                        description=f"Run tests for the modified service",
                        action_type="test_service",
                        parameters={
                            "service_name": change.details.service_name,
                            "test_changes": change.details.test_changes or {}
                        },
                        order=order
                    ))
                    
                elif change.details.change_type == "remove":
                    steps.append(ExecutionStep(
                        name=f"Remove Service: {change.details.service_name}",
                        description=f"Remove service: {change.details.description}",
                        action_type="remove_service",
                        parameters={
                            "service_name": change.details.service_name
                        },
                        order=order
                    ))
        
        elif change.type == "model":
            if isinstance(change.details, ModelChange):
                if change.details.change_type == "add":
                    steps.append(ExecutionStep(
                        name=f"Add Model: {change.details.model_name}",
                        description=f"Add new model: {change.details.description}",
                        action_type="add_model",
                        parameters={
                            "model_name": change.details.model_name,
                            "parameter_changes": change.details.parameter_changes or {},
                            "prompt_changes": change.details.prompt_changes or {}
                        },
                        order=order
                    ))
                    
                elif change.details.change_type == "modify":
                    steps.append(ExecutionStep(
                        name=f"Modify Model: {change.details.model_name}",
                        description=f"Modify existing model: {change.details.description}",
                        action_type="modify_model",
                        parameters={
                            "model_name": change.details.model_name,
                            "parameter_changes": change.details.parameter_changes or {},
                            "prompt_changes": change.details.prompt_changes or {}
                        },
                        order=order
                    ))
                    
                elif change.details.change_type == "remove":
                    steps.append(ExecutionStep(
                        name=f"Remove Model: {change.details.model_name}",
                        description=f"Remove model: {change.details.description}",
                        action_type="remove_model",
                        parameters={
                            "model_name": change.details.model_name
                        },
                        order=order
                    ))
        
        return steps


# Example action implementations
async def create_backup_action(
    step: ExecutionStep, proposal: EvolutionProposal, **kwargs
) -> Dict[str, Any]:
    """
    Create a backup of the system.
    
    Args:
        step: Execution step
        proposal: Proposal being executed
        
    Returns:
        Action result
    """
    # In a real implementation, this would create an actual backup
    return {
        "backup_id": str(uuid4()),
        "backup_time": datetime.now().isoformat(),
        "target_system": proposal.target_system,
        "backup_type": step.parameters.get("backup_type", "full"),
        "status": "success"
    }


async def verify_changes_action(
    step: ExecutionStep, proposal: EvolutionProposal, **kwargs
) -> Dict[str, Any]:
    """
    Verify that changes have been applied correctly.
    
    Args:
        step: Execution step
        proposal: Proposal being executed
        
    Returns:
        Action result
    """
    # In a real implementation, this would perform actual verification
    return {
        "verification_time": datetime.now().isoformat(),
        "target_system": proposal.target_system,
        "verification_result": "success",
        "verified_changes": len(proposal.changes)
    }


# Register default actions
def register_default_actions(service: EvolutionExecutionService) -> None:
    """
    Register default action implementations.
    
    Args:
        service: Execution service to register actions with
    """
    service.action_registry.register_action(
        "create_backup", create_backup_action
    )
    service.action_registry.register_action(
        "verify_changes", verify_changes_action
    )
