"""
Tests for the Evolution Execution Service.

This module contains tests for the EvolutionExecutionService which is responsible
for executing approved evolution proposals by implementing the proposed changes.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from agent_orchestration.evolution.evolution_execution_service import (
    EvolutionExecutionService,
    ActionRegistry,
    register_default_actions,
    create_backup_action,
    verify_changes_action,
)
from agent_orchestration.evolution.evolution_proposal_service import (
    EvolutionProposalService,
    ObservationService,
)
from agent_orchestration.evolution.models import (
    EvolutionProposal,
    EvolutionTrigger,
    Observation,
    ObservationSource,
    Change,
    ExpectedOutcome,
    ImpactLevel,
    ComplexityLevel,
    ProposalStatus,
    CapabilityChange,
    ServiceChange,
    ModelChange,
    ExecutionPlan,
    ExecutionStep,
)
from agent_orchestration.infrastructure.errors import (
    ValidationError,
    ResourceNotFoundError,
    DuplicateResourceError,
    ExecutionError,
)


@pytest.fixture
def mock_event_publisher():
    """Create a mock event publisher."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    return publisher


@pytest.fixture
def mock_mcp_service():
    """Create a mock MCP service."""
    service = AsyncMock()
    service.generate_evolution_proposal = AsyncMock()
    return service


@pytest.fixture
def observation_service(mock_event_publisher):
    """Create an ObservationService instance with mock publisher."""
    return ObservationService(event_publisher=mock_event_publisher)


@pytest.fixture
def proposal_service(observation_service, mock_event_publisher, mock_mcp_service):
    """Create an EvolutionProposalService instance with mock dependencies."""
    return EvolutionProposalService(
        observation_service=observation_service,
        event_publisher=mock_event_publisher,
        mcp_service=mock_mcp_service,
    )


@pytest.fixture
def action_registry():
    """Create an ActionRegistry instance with test actions."""
    registry = ActionRegistry()
    registry.register_action("mock_action", AsyncMock(return_value={"status": "success"}))
    registry.register_action("mock_failing_action", AsyncMock(side_effect=Exception("Action failed")))
    return registry


@pytest.fixture
def execution_service(proposal_service, mock_event_publisher, action_registry, mock_mcp_service):
    """Create an EvolutionExecutionService instance with mock dependencies."""
    service = EvolutionExecutionService(
        proposal_service=proposal_service,
        event_publisher=mock_event_publisher,
        action_registry=action_registry,
        mcp_service=mock_mcp_service,
    )
    
    # Register default actions
    register_default_actions(service)
    
    return service


@pytest.fixture
async def approved_proposal(proposal_service, observation_service):
    """Create an approved proposal for testing."""
    # Create observation
    observation = Observation(
        source=ObservationSource.PERFORMANCE_METRIC,
        target_system="search_service",
        category="performance",
        description="Search service response time degraded",
        data={"avg_response_time_ms": 250, "threshold_ms": 100},
        importance=0.8,
    )
    await observation_service.add_observation(observation)
    
    # Generate proposal
    proposal = await proposal_service.generate_proposal(
        target_system="search_service",
        agent_id="agent-123",
        trigger_observation_id=observation.id,
        use_mcp=False
    )
    
    # Set proposal to approved
    proposal.status = ProposalStatus.APPROVED
    proposal_service.proposals[proposal.id] = proposal
    
    return proposal


class TestActionRegistry:
    """Tests for the ActionRegistry."""
    
    def test_register_action(self, action_registry):
        """Test registering an action."""
        # Arrange
        mock_impl = AsyncMock()
        
        # Act
        action_registry.register_action("test_action", mock_impl)
        
        # Assert
        assert "test_action" in action_registry.actions
        assert action_registry.actions["test_action"] == mock_impl
    
    def test_get_action(self, action_registry):
        """Test retrieving an action by type."""
        # Act
        action = action_registry.get_action("mock_action")
        
        # Assert
        assert action is not None
        assert isinstance(action, AsyncMock)
    
    def test_get_action_not_found(self, action_registry):
        """Test retrieving a non-existent action."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            action_registry.get_action("non_existent_action")


class TestEvolutionExecutionService:
    """Tests for the EvolutionExecutionService."""
    
    async def test_create_execution_plan(self, execution_service, approved_proposal, mock_event_publisher):
        """Test creating an execution plan."""
        # Act
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        
        # Assert
        assert plan.id in execution_service.execution_plans
        assert plan.proposal_id == approved_proposal.id
        assert plan.status == "pending"
        assert len(plan.steps) > 0
        
        # Verify steps include backup, changes, verification and documentation
        step_names = [step.name for step in plan.steps]
        assert any("Backup" in name for name in step_names)
        assert any("Verify" in name for name in step_names)
        assert any("Documentation" in name for name in step_names)
        
        # Verify event
        mock_event_publisher.publish.assert_called()
        call_args = mock_event_publisher.publish.call_args[1]
        assert call_args["topic"] == "evolution.execution"
        assert call_args["value"]["type"] == "execution_plan_created"
    
    async def test_create_execution_plan_proposal_not_found(self, execution_service):
        """Test error when proposal not found."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await execution_service.create_execution_plan("non_existent_id")
    
    async def test_create_execution_plan_not_approved(self, execution_service, proposal_service, observation_service):
        """Test error when proposal not approved."""
        # Arrange
        # Create non-approved proposal
        observation = Observation(
            source=ObservationSource.PERFORMANCE_METRIC,
            target_system="search_service",
            category="performance",
            description="Test observation",
            data={},
            importance=0.5,
        )
        await observation_service.add_observation(observation)
        
        proposal = await proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=observation.id,
            use_mcp=False
        )
        # Status is SUBMITTED by default
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await execution_service.create_execution_plan(proposal.id)
    
    async def test_create_execution_plan_duplicate(self, execution_service, approved_proposal):
        """Test error when plan already exists for proposal."""
        # Arrange
        await execution_service.create_execution_plan(approved_proposal.id)
        
        # Act & Assert
        with pytest.raises(DuplicateResourceError):
            await execution_service.create_execution_plan(approved_proposal.id)
    
    async def test_get_execution_plan(self, execution_service, approved_proposal):
        """Test retrieving an execution plan by ID."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        
        # Act
        retrieved_plan = await execution_service.get_execution_plan(plan.id)
        
        # Assert
        assert retrieved_plan.id == plan.id
        assert retrieved_plan.proposal_id == plan.proposal_id
    
    async def test_get_execution_plan_not_found(self, execution_service):
        """Test retrieving a non-existent execution plan."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await execution_service.get_execution_plan("non_existent_id")
    
    async def test_get_execution_plans_for_proposal(self, execution_service, proposal_service, observation_service):
        """Test retrieving execution plans for a proposal."""
        # Arrange
        # Create multiple proposals
        proposals = []
        for i in range(2):
            observation = Observation(
                source=ObservationSource.PERFORMANCE_METRIC,
                target_system=f"service_{i}",
                category="performance",
                description=f"Observation {i}",
                data={"value": i},
                importance=0.8,
            )
            await observation_service.add_observation(observation)
            
            proposal = await proposal_service.generate_proposal(
                target_system=f"service_{i}",
                agent_id=f"agent-{i}",
                trigger_observation_id=observation.id,
                use_mcp=False
            )
            proposal.status = ProposalStatus.APPROVED
            proposal_service.proposals[proposal.id] = proposal
            proposals.append(proposal)
        
        # Create plans for proposals
        await execution_service.create_execution_plan(proposals[0].id)
        await execution_service.create_execution_plan(proposals[1].id)
        
        # Act
        plans = await execution_service.get_execution_plans_for_proposal(proposals[0].id)
        
        # Assert
        assert len(plans) == 1
        assert plans[0].proposal_id == proposals[0].id
    
    async def test_start_execution(self, execution_service, approved_proposal, mock_event_publisher):
        """Test starting execution for a plan."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        mock_event_publisher.reset_mock()
        
        # Mock _execute_plan to prevent actual execution
        execution_service._execute_plan = AsyncMock()
        
        # Act
        updated_plan = await execution_service.start_execution(plan.id)
        
        # Assert
        assert updated_plan.status == "in_progress"
        assert updated_plan.started_at is not None
        
        # Verify proposal status update
        proposal_new = await execution_service.proposal_service.get_proposal(approved_proposal.id)
        assert proposal_new.status == ProposalStatus.IMPLEMENTED
        
        # Verify event
        mock_event_publisher.publish.assert_called()
        call_args = mock_event_publisher.publish.call_args[1]
        assert call_args["topic"] == "evolution.execution"
        assert call_args["value"]["type"] == "execution_started"
    
    async def test_start_execution_plan_not_found(self, execution_service):
        """Test error when starting a non-existent execution plan."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await execution_service.start_execution("non_existent_id")
    
    async def test_start_execution_invalid_status(self, execution_service, approved_proposal):
        """Test error when starting a plan with invalid status."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        plan.status = "in_progress"
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await execution_service.start_execution(plan.id)
    
    async def test_execute_plan_success(self, execution_service, approved_proposal):
        """Test executing a plan successfully."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        
        # Replace all action types with mock_action
        for step in plan.steps:
            step.action_type = "mock_action"
        
        # Act
        await execution_service._execute_plan(plan)
        
        # Assert
        assert plan.status == "completed"
        assert plan.completed_at is not None
        assert plan.results_summary is not None
        assert plan.results_summary["overall_result"] == "success"
        
        # Verify all steps were executed
        assert all(step.status == "completed" for step in plan.steps)
        assert all(step.result is not None for step in plan.steps)
    
    async def test_execute_plan_failure(self, execution_service, approved_proposal):
        """Test executing a plan with a failing step."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        
        # Make one step fail
        plan.steps[1].action_type = "mock_failing_action"
        
        # Act & Assert
        with pytest.raises(ExecutionError):
            await execution_service._execute_plan(plan)
        
        # Verify the failing step
        assert plan.steps[1].status == "failed"
        assert "error" in plan.steps[1].result
        assert "Action failed" in plan.steps[1].result["error"]
    
    async def test_execute_step_success(self, execution_service, approved_proposal):
        """Test executing a step successfully."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        step = plan.steps[0]
        step.action_type = "mock_action"
        
        # Act
        await execution_service._execute_step(step, approved_proposal, plan)
        
        # Assert
        assert step.status == "completed"
        assert step.completed_at is not None
        assert step.result is not None
        assert step.result["status"] == "success"
    
    async def test_execute_step_failure(self, execution_service, approved_proposal):
        """Test executing a step with failure."""
        # Arrange
        plan = await execution_service.create_execution_plan(approved_proposal.id)
        step = plan.steps[0]
        step.action_type = "mock_failing_action"
        
        # Act & Assert
        with pytest.raises(ExecutionError):
            await execution_service._execute_step(step, approved_proposal, plan)
        
        # Verify step status and result
        assert step.status == "failed"
        assert step.completed_at is not None
        assert step.result is not None
        assert "error" in step.result
    
    async def test_generate_execution_steps(self, execution_service, approved_proposal):
        """Test generating execution steps for a proposal."""
        # Act
        steps = await execution_service._generate_execution_steps(approved_proposal)
        
        # Assert
        assert len(steps) >= 2  # At least backup and verification steps
        assert all(isinstance(step, ExecutionStep) for step in steps)
        
        # Verify first step is backup
        assert "Backup" in steps[0].name
        
        # Verify last steps include verification and documentation
        last_steps = steps[-2:]
        assert any("Verify" in step.name for step in last_steps)
        assert any("Documentation" in step.name for step in last_steps)
        
        # Verify steps are ordered
        for i in range(1, len(steps)):
            assert steps[i].order > steps[i-1].order
    
    async def test_create_steps_for_change_capability_add(self, execution_service):
        """Test creating steps for capability addition."""
        # Arrange
        change = Change(
            type="capability",
            target_component="search_service",
            rationale="Improve search performance",
            details=CapabilityChange(
                change_type="add",
                capability_name="AdvancedSearch",
                description="Advanced search capability with improved performance",
                code_changes={"implementation": "code here"},
                test_changes={"test_implementation": "test code here"}
            )
        )
        
        # Act
        steps = await execution_service._create_steps_for_change(change, 1)
        
        # Assert
        assert len(steps) == 2
        assert "Add Capability" in steps[0].name
        assert "Test Capability" in steps[1].name
        assert steps[0].parameters["capability_name"] == "AdvancedSearch"
        assert steps[1].parameters["capability_name"] == "AdvancedSearch"
    
    async def test_create_steps_for_change_service_modify(self, execution_service):
        """Test creating steps for service modification."""
        # Arrange
        change = Change(
            type="service",
            target_component="search_service",
            rationale="Improve search service",
            details=ServiceChange(
                change_type="modify",
                service_name="SearchService",
                description="Modify search service for better performance",
                implementation_changes={"implementation": "new code"},
                api_changes={"api": "updated api"},
                test_changes={"tests": "updated tests"}
            )
        )
        
        # Act
        steps = await execution_service._create_steps_for_change(change, 1)
        
        # Assert
        assert len(steps) == 2
        assert "Modify Service" in steps[0].name
        assert "Test Modified Service" in steps[1].name
        assert steps[0].parameters["service_name"] == "SearchService"
        assert steps[1].parameters["service_name"] == "SearchService"
    
    async def test_create_steps_for_change_model_add(self, execution_service):
        """Test creating steps for model addition."""
        # Arrange
        change = Change(
            type="model",
            target_component="search_service",
            rationale="Add new model",
            details=ModelChange(
                change_type="add",
                model_name="SearchModel",
                description="New search model",
                parameter_changes={"parameters": "new params"},
                prompt_changes={"prompts": "new prompts"}
            )
        )
        
        # Act
        steps = await execution_service._create_steps_for_change(change, 1)
        
        # Assert
        assert len(steps) == 1
        assert "Add Model" in steps[0].name
        assert steps[0].parameters["model_name"] == "SearchModel"


class TestDefaultActions:
    """Tests for the default action implementations."""
    
    async def test_create_backup_action(self, approved_proposal):
        """Test the create_backup_action."""
        # Arrange
        step = ExecutionStep(
            name="Create Backup",
            description="Create system backup",
            action_type="create_backup",
            parameters={"backup_type": "full"}
        )
        
        # Act
        result = await create_backup_action(step, approved_proposal)
        
        # Assert
        assert result is not None
        assert "backup_id" in result
        assert result["backup_type"] == "full"
        assert result["status"] == "success"
    
    async def test_verify_changes_action(self, approved_proposal):
        """Test the verify_changes_action."""
        # Arrange
        step = ExecutionStep(
            name="Verify Changes",
            description="Verify all changes",
            action_type="verify_changes",
            parameters={}
        )
        
        # Act
        result = await verify_changes_action(step, approved_proposal)
        
        # Assert
        assert result is not None
        assert "verification_result" in result
        assert result["verification_result"] == "success"
        assert result["verified_changes"] == len(approved_proposal.changes)
