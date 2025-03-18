"""
Tests for the Evolution Validation Service.

This module contains tests for the EvolutionValidationService which is responsible
for validating evolution proposals before they are approved for implementation.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from agent_orchestration.evolution.evolution_validation_service import (
    EvolutionValidationService,
    ValidationCriteriaRegistry,
    register_default_validators,
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
    ValidationPlan,
    ValidationStep,
    ValidationCriterion,
    ValidationResult,
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
def criteria_registry():
    """Create a ValidationCriteriaRegistry instance."""
    registry = ValidationCriteriaRegistry()
    registry.register_criterion("mock_criterion", AsyncMock(return_value={"passed": True, "details": "Passed"}))
    registry.register_criterion("mock_failing_criterion", AsyncMock(return_value={"passed": False, "details": "Failed"}))
    return registry


@pytest.fixture
def validation_service(proposal_service, mock_event_publisher, criteria_registry, mock_mcp_service):
    """Create an EvolutionValidationService instance with mock dependencies."""
    service = EvolutionValidationService(
        proposal_service=proposal_service,
        event_publisher=mock_event_publisher,
        criteria_registry=criteria_registry,
        mcp_service=mock_mcp_service,
    )
    
    # Register default validators
    register_default_validators(service)
    
    return service


@pytest.fixture
async def sample_proposal(proposal_service, observation_service):
    """Create a sample proposal for testing."""
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
    return await proposal_service.generate_proposal(
        target_system="search_service",
        agent_id="agent-123",
        trigger_observation_id=observation.id,
        use_mcp=False
    )


class TestValidationCriteriaRegistry:
    """Tests for the ValidationCriteriaRegistry."""
    
    def test_register_criterion(self, criteria_registry):
        """Test registering a criterion."""
        # Arrange
        mock_impl = AsyncMock()
        
        # Act
        criteria_registry.register_criterion("test_criterion", mock_impl)
        
        # Assert
        assert "test_criterion" in criteria_registry.criteria
        assert criteria_registry.criteria["test_criterion"] == mock_impl
    
    def test_get_criterion(self, criteria_registry):
        """Test retrieving a criterion by type."""
        # Act
        criterion = criteria_registry.get_criterion("mock_criterion")
        
        # Assert
        assert criterion is not None
        assert isinstance(criterion, AsyncMock)
    
    def test_get_criterion_not_found(self, criteria_registry):
        """Test retrieving a non-existent criterion."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            criteria_registry.get_criterion("non_existent_criterion")


class TestEvolutionValidationService:
    """Tests for the EvolutionValidationService."""
    
    async def test_create_validation_plan(self, validation_service, sample_proposal, mock_event_publisher):
        """Test creating a validation plan."""
        # Act
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        
        # Assert
        assert plan.id in validation_service.validation_plans
        assert plan.proposal_id == sample_proposal.id
        assert plan.status == "pending"
        assert len(plan.steps) > 0
        
        # Verify event
        mock_event_publisher.publish.assert_called()
        call_args = mock_event_publisher.publish.call_args[1]
        assert call_args["topic"] == "evolution.validation"
        assert call_args["value"]["type"] == "validation_plan_created"
    
    async def test_create_validation_plan_proposal_not_found(self, validation_service):
        """Test error when proposal not found."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await validation_service.create_validation_plan("non_existent_id")
    
    async def test_create_validation_plan_duplicate(self, validation_service, sample_proposal):
        """Test error when plan already exists for proposal."""
        # Arrange
        await validation_service.create_validation_plan(sample_proposal.id)
        
        # Act & Assert
        with pytest.raises(DuplicateResourceError):
            await validation_service.create_validation_plan(sample_proposal.id)
    
    async def test_create_validation_plan_with_custom_criteria(self, validation_service, sample_proposal):
        """Test creating a validation plan with custom criteria."""
        # Arrange
        custom_criteria = [
            ValidationCriterion(
                name="Custom Criterion",
                description="A custom validation criterion",
                validation_type="mock_criterion",
                parameters={"custom": True}
            )
        ]
        
        # Act
        plan = await validation_service.create_validation_plan(
            sample_proposal.id,
            custom_criteria=custom_criteria
        )
        
        # Assert
        assert plan.id in validation_service.validation_plans
        assert any(
            criterion.name == "Custom Criterion"
            for step in plan.steps
            for criterion in step.criteria
        )
    
    async def test_get_validation_plan(self, validation_service, sample_proposal):
        """Test retrieving a validation plan by ID."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        
        # Act
        retrieved_plan = await validation_service.get_validation_plan(plan.id)
        
        # Assert
        assert retrieved_plan.id == plan.id
        assert retrieved_plan.proposal_id == plan.proposal_id
    
    async def test_get_validation_plan_not_found(self, validation_service):
        """Test retrieving a non-existent validation plan."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await validation_service.get_validation_plan("non_existent_id")
    
    async def test_get_validation_plans_for_proposal(self, validation_service, proposal_service, observation_service):
        """Test retrieving validation plans for a proposal."""
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
            proposals.append(proposal)
        
        # Create plans for proposals
        await validation_service.create_validation_plan(proposals[0].id)
        await validation_service.create_validation_plan(proposals[1].id)
        
        # Act
        plans = await validation_service.get_validation_plans_for_proposal(proposals[0].id)
        
        # Assert
        assert len(plans) == 1
        assert plans[0].proposal_id == proposals[0].id
    
    async def test_start_validation(self, validation_service, sample_proposal, mock_event_publisher):
        """Test starting validation for a plan."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        mock_event_publisher.reset_mock()
        
        # Mock _execute_validation_plan to prevent actual execution
        validation_service._execute_validation_plan = AsyncMock()
        
        # Act
        updated_plan = await validation_service.start_validation(plan.id)
        
        # Assert
        assert updated_plan.status == "in_progress"
        assert updated_plan.started_at is not None
        
        # Verify proposal status update
        sample_proposal_new = await validation_service.proposal_service.get_proposal(sample_proposal.id)
        assert sample_proposal_new.status == ProposalStatus.VALIDATING
        
        # Verify event
        mock_event_publisher.publish.assert_called()
        call_args = mock_event_publisher.publish.call_args[1]
        assert call_args["topic"] == "evolution.validation"
        assert call_args["value"]["type"] == "validation_started"
    
    async def test_start_validation_plan_not_found(self, validation_service):
        """Test error when starting a non-existent validation plan."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await validation_service.start_validation("non_existent_id")
    
    async def test_start_validation_invalid_status(self, validation_service, sample_proposal):
        """Test error when starting a plan with invalid status."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        plan.status = "in_progress"
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await validation_service.start_validation(plan.id)
    
    @pytest.mark.parametrize("all_pass,expected_status", [
        (True, ProposalStatus.APPROVED),
        (False, ProposalStatus.REJECTED)
    ])
    async def test_execute_validation_plan(self, validation_service, sample_proposal, all_pass, expected_status):
        """Test executing a validation plan."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        
        # Replace criteria with mocks (all pass or all fail)
        criterion_type = "mock_criterion" if all_pass else "mock_failing_criterion"
        for step in plan.steps:
            for criterion in step.criteria:
                criterion.validation_type = criterion_type
        
        # Act
        await validation_service._execute_validation_plan(plan)
        
        # Assert
        assert plan.status == "completed"
        assert plan.completed_at is not None
        assert plan.results_summary is not None
        assert "pass_rate" in plan.results_summary
        
        # Verify all steps were executed
        assert all(step.status == "completed" for step in plan.steps)
        assert all(step.results is not None for step in plan.steps)
        
        # Verify proposal status was updated
        proposal = await validation_service.proposal_service.get_proposal(sample_proposal.id)
        assert proposal.status == expected_status
    
    async def test_execute_validation_step(self, validation_service, sample_proposal):
        """Test executing a validation step."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        step = plan.steps[0]
        
        # Replace criteria with mocks
        for criterion in step.criteria:
            criterion.validation_type = "mock_criterion"
        
        # Act
        await validation_service._execute_validation_step(step, sample_proposal, plan)
        
        # Assert
        assert step.status == "completed"
        assert step.completed_at is not None
        assert step.results is not None
        assert len(step.results) == len(step.criteria)
        
        # Verify all criteria were executed and passed
        assert all(result.passed for result in step.results)
    
    async def test_execute_validation_step_with_failing_criteria(self, validation_service, sample_proposal):
        """Test executing a validation step with failing criteria."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        step = plan.steps[0]
        
        # Replace some criteria with failing mocks
        for i, criterion in enumerate(step.criteria):
            criterion.validation_type = "mock_criterion" if i % 2 == 0 else "mock_failing_criterion"
        
        # Act
        await validation_service._execute_validation_step(step, sample_proposal, plan)
        
        # Assert
        assert step.status == "completed"
        assert step.results is not None
        
        # Verify mixed results
        passing_results = [result for result in step.results if result.passed]
        failing_results = [result for result in step.results if not result.passed]
        assert len(passing_results) > 0
        assert len(failing_results) > 0
    
    async def test_execute_validation_step_with_error(self, validation_service, sample_proposal):
        """Test handling errors during validation step execution."""
        # Arrange
        plan = await validation_service.create_validation_plan(sample_proposal.id)
        step = plan.steps[0]
        
        # Create a criterion that raises an exception
        error_criterion = ValidationCriterion(
            name="Error Criterion",
            description="A criterion that raises an error",
            validation_type="mock_criterion",
            parameters={}
        )
        step.criteria = [error_criterion]
        
        # Make the criterion implementation raise an exception
        validation_service.criteria_registry.get_criterion = MagicMock(side_effect=Exception("Test error"))
        
        # Act
        await validation_service._execute_validation_step(step, sample_proposal, plan)
        
        # Assert
        assert step.status == "completed"  # It should still complete
        assert step.results is not None
        assert len(step.results) == 1
        assert not step.results[0].passed
        assert "Test error" in step.results[0].details
    
    async def test_generate_validation_steps(self, validation_service, sample_proposal):
        """Test generating validation steps for a proposal."""
        # Act
        steps = await validation_service._generate_validation_steps(sample_proposal)
        
        # Assert
        assert len(steps) >= 3  # At least the default steps
        assert all(isinstance(step, ValidationStep) for step in steps)
        assert all(step.criteria for step in steps)
        
        # Verify steps are ordered
        for i in range(1, len(steps)):
            assert steps[i].order > steps[i-1].order
    
    async def test_generate_validation_steps_with_custom_criteria(self, validation_service, sample_proposal):
        """Test generating validation steps with custom criteria."""
        # Arrange
        custom_criteria = [
            ValidationCriterion(
                name="Custom Criterion 1",
                description="First custom criterion",
                validation_type="mock_criterion",
                parameters={}
            ),
            ValidationCriterion(
                name="Custom Criterion 2",
                description="Second custom criterion",
                validation_type="mock_criterion",
                parameters={}
            )
        ]
        
        # Act
        steps = await validation_service._generate_validation_steps(
            sample_proposal,
            custom_criteria=custom_criteria
        )
        
        # Assert
        # Find the custom step
        custom_step = next((step for step in steps if step.name == "Custom Validation"), None)
        assert custom_step is not None
        assert len(custom_step.criteria) == len(custom_criteria)
        assert all(criterion.name in [c.name for c in custom_criteria] for criterion in custom_step.criteria)
    
    async def test_built_in_validators(self, validation_service, sample_proposal):
        """Test the built-in validators."""
        # Test structure validator
        result = await validation_service.structure_validator(None, sample_proposal)
        assert result["passed"] is True
        
        # Test with missing field
        broken_proposal = sample_proposal.copy()
        broken_proposal.changes = []
        result = await validation_service.structure_validator(None, broken_proposal)
        assert result["passed"] is False
        
        # Test change validator
        result = await validation_service.change_validator(None, sample_proposal)
        assert result["passed"] is True
        
        # Test with invalid change
        broken_proposal = sample_proposal.copy()
        invalid_change = broken_proposal.changes[0].copy()
        invalid_change.rationale = ""
        broken_proposal.changes = [invalid_change]
        result = await validation_service.change_validator(None, broken_proposal)
        assert result["passed"] is False
