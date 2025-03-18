"""
Tests for the Evolution Proposal Service.

This module contains tests for the EvolutionProposalService which is responsible
for generating evolution proposals based on observations.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

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
)
from agent_orchestration.infrastructure.errors import (
    ValidationError, 
    ResourceNotFoundError,
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
def evolution_proposal_service(observation_service, mock_event_publisher, mock_mcp_service):
    """Create an EvolutionProposalService instance with mock dependencies."""
    return EvolutionProposalService(
        observation_service=observation_service,
        event_publisher=mock_event_publisher,
        mcp_service=mock_mcp_service,
    )


@pytest.fixture
def sample_observation():
    """Create a sample observation."""
    return Observation(
        source=ObservationSource.PERFORMANCE_METRIC,
        target_system="search_service",
        category="performance",
        description="Search service response time degraded",
        data={"avg_response_time_ms": 250, "threshold_ms": 100},
        importance=0.8,
    )


class TestObservationService:
    """Tests for the ObservationService."""
    
    async def test_add_observation(self, observation_service, sample_observation, mock_event_publisher):
        """Test adding an observation."""
        # Act
        observation_id = await observation_service.add_observation(sample_observation)
        
        # Assert
        assert observation_id == sample_observation.id
        assert observation_id in observation_service.observations
        mock_event_publisher.publish.assert_called_once()
        
        # Verify event data
        call_args = mock_event_publisher.publish.call_args[1]
        assert call_args["topic"] == "evolution.observations"
        assert call_args["key"] == observation_id
        assert call_args["value"]["type"] == "observation_added"
        assert call_args["value"]["observation_id"] == observation_id
    
    async def test_add_observation_from_dict(self, observation_service, mock_event_publisher):
        """Test adding an observation from a dictionary."""
        # Arrange
        observation_dict = {
            "source": "performance_metric",
            "target_system": "search_service",
            "category": "performance",
            "description": "Search service response time degraded",
            "data": {"avg_response_time_ms": 250, "threshold_ms": 100},
            "importance": 0.8,
        }
        
        # Act
        observation_id = await observation_service.add_observation(observation_dict)
        
        # Assert
        assert observation_id in observation_service.observations
        mock_event_publisher.publish.assert_called_once()
    
    async def test_add_observation_validation_error(self, observation_service):
        """Test validation error when adding an invalid observation."""
        # Arrange
        invalid_observation = {
            "source": "invalid_source",
            "target_system": "search_service",
            "category": "performance",
            "description": "Search service response time degraded",
            "data": {"avg_response_time_ms": 250, "threshold_ms": 100},
            "importance": 0.8,
        }
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await observation_service.add_observation(invalid_observation)
    
    async def test_get_observation(self, observation_service, sample_observation):
        """Test retrieving an observation by ID."""
        # Arrange
        await observation_service.add_observation(sample_observation)
        
        # Act
        retrieved_observation = await observation_service.get_observation(sample_observation.id)
        
        # Assert
        assert retrieved_observation.id == sample_observation.id
        assert retrieved_observation.description == sample_observation.description
    
    async def test_get_observation_not_found(self, observation_service):
        """Test retrieving a non-existent observation."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await observation_service.get_observation("non_existent_id")
    
    async def test_get_observations_for_system(self, observation_service):
        """Test retrieving observations for a specific system."""
        # Arrange
        observations = [
            Observation(
                source=ObservationSource.PERFORMANCE_METRIC,
                target_system="search_service",
                category="performance",
                description=f"Observation {i}",
                data={"value": i},
                importance=0.5 + i * 0.1,
            )
            for i in range(5)
        ]
        
        # Add observations for search_service
        for obs in observations:
            await observation_service.add_observation(obs)
        
        # Add observation for different system
        other_obs = Observation(
            source=ObservationSource.ERROR_LOG,
            target_system="other_service",
            category="error",
            description="Error in other service",
            data={"error_count": 10},
            importance=0.9,
        )
        await observation_service.add_observation(other_obs)
        
        # Act
        result = await observation_service.get_observations_for_system("search_service")
        
        # Assert
        assert len(result) == 5
        assert all(obs.target_system == "search_service" for obs in result)
        
        # Check sorting by importance
        assert result[0].importance > result[-1].importance
    
    async def test_get_observations_for_system_with_importance_threshold(self, observation_service):
        """Test retrieving observations with importance threshold."""
        # Arrange
        observations = [
            Observation(
                source=ObservationSource.PERFORMANCE_METRIC,
                target_system="search_service",
                category="performance",
                description=f"Observation {i}",
                data={"value": i},
                importance=0.2 * i,  # 0.0, 0.2, 0.4, 0.6, 0.8
            )
            for i in range(5)
        ]
        
        for obs in observations:
            await observation_service.add_observation(obs)
        
        # Act
        result = await observation_service.get_observations_for_system(
            "search_service",
            importance_threshold=0.5
        )
        
        # Assert
        assert len(result) == 2  # Only observations with importance >= 0.5
        assert all(obs.importance >= 0.5 for obs in result)


class TestEvolutionProposalService:
    """Tests for the EvolutionProposalService."""
    
    async def test_generate_proposal_with_mcp(
        self, evolution_proposal_service, observation_service, sample_observation, mock_mcp_service
    ):
        """Test generating a proposal using MCP service."""
        # Arrange
        await observation_service.add_observation(sample_observation)
        
        # Mock MCP response
        mock_mcp_service.generate_evolution_proposal.return_value = {
            "proposal": {
                "title": "Improve Search Service Performance",
                "description": "A proposal to optimize search service response times",
                "changes": [
                    {
                        "type": "service",
                        "description": "Optimize search service",
                        "details": {
                            "service_name": "search_service",
                            "change_type": "modify",
                            "description": "Optimize query performance",
                            "implementation_changes": {"focus_areas": ["indexing", "caching"]}
                        },
                        "rationale": "Improve response times",
                        "impact_assessment": "Medium impact, should improve user experience"
                    }
                ],
                "expected_outcomes": [
                    {
                        "description": "Faster search responses",
                        "metrics": {"response_time_ms": "< 100ms"},
                        "validation_criteria": ["Response time below 100ms for 95% of queries"]
                    }
                ],
                "impact_level": "medium",
                "complexity": "moderate"
            }
        }
        
        # Act
        proposal = await evolution_proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=sample_observation.id,
            use_mcp=True
        )
        
        # Assert
        assert proposal.id in evolution_proposal_service.proposals
        assert proposal.title == "Improve Search Service Performance"
        assert proposal.target_system == "search_service"
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.agent_id == "agent-123"
        assert len(proposal.changes) == 1
        assert isinstance(proposal.changes[0].details, ServiceChange)
        assert len(proposal.expected_outcomes) == 1
        
        # Verify MCP call
        mock_mcp_service.generate_evolution_proposal.assert_called_once()
        
        # Verify event
        evolution_proposal_service.event_publisher.publish.assert_called_once()
        call_args = evolution_proposal_service.event_publisher.publish.call_args[1]
        assert call_args["topic"] == "evolution.proposals"
        assert call_args["value"]["type"] == "evolution_proposal_created"
    
    async def test_generate_proposal_without_mcp(
        self, evolution_proposal_service, observation_service, sample_observation
    ):
        """Test generating a proposal without using MCP service."""
        # Arrange
        await observation_service.add_observation(sample_observation)
        
        # Act
        proposal = await evolution_proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=sample_observation.id,
            use_mcp=False
        )
        
        # Assert
        assert proposal.id in evolution_proposal_service.proposals
        assert "search_service" in proposal.title
        assert proposal.target_system == "search_service"
        assert proposal.status == ProposalStatus.DRAFT
        assert proposal.agent_id == "agent-123"
        assert len(proposal.changes) > 0
        assert len(proposal.expected_outcomes) > 0
        
        # Verify event
        evolution_proposal_service.event_publisher.publish.assert_called_once()
    
    async def test_generate_proposal_no_observations(self, evolution_proposal_service):
        """Test error when generating a proposal without observations."""
        # Act & Assert
        with pytest.raises(ValidationError):
            await evolution_proposal_service.generate_proposal(
                target_system="empty_service",
                agent_id="agent-123",
                use_mcp=False
            )
    
    async def test_generate_proposal_trigger_not_found(self, evolution_proposal_service, observation_service):
        """Test error when trigger observation is not found."""
        # Arrange
        sample_observation = Observation(
            source=ObservationSource.PERFORMANCE_METRIC,
            target_system="search_service",
            category="performance",
            description="Search service response time degraded",
            data={"avg_response_time_ms": 250, "threshold_ms": 100},
            importance=0.8,
        )
        await observation_service.add_observation(sample_observation)
        
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await evolution_proposal_service.generate_proposal(
                target_system="search_service",
                agent_id="agent-123",
                trigger_observation_id="non_existent_id",
                use_mcp=False
            )
    
    async def test_submit_proposal(self, evolution_proposal_service):
        """Test submitting a proposal for validation."""
        # Arrange
        # Create a proposal first
        sample_observation = Observation(
            source=ObservationSource.PERFORMANCE_METRIC,
            target_system="search_service",
            category="performance",
            description="Search service response time degraded",
            data={"avg_response_time_ms": 250, "threshold_ms": 100},
            importance=0.8,
        )
        await evolution_proposal_service.observation_service.add_observation(sample_observation)
        
        proposal = await evolution_proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=sample_observation.id,
            use_mcp=False
        )
        
        # Reset mock for event publisher
        evolution_proposal_service.event_publisher.publish.reset_mock()
        
        # Act
        updated_proposal = await evolution_proposal_service.submit_proposal(proposal.id)
        
        # Assert
        assert updated_proposal.status == ProposalStatus.SUBMITTED
        assert updated_proposal.id == proposal.id
        
        # Verify event
        evolution_proposal_service.event_publisher.publish.assert_called_once()
        call_args = evolution_proposal_service.event_publisher.publish.call_args[1]
        assert call_args["value"]["type"] == "evolution_proposal_submitted"
    
    async def test_submit_proposal_not_found(self, evolution_proposal_service):
        """Test error when submitting a non-existent proposal."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await evolution_proposal_service.submit_proposal("non_existent_id")
    
    async def test_submit_proposal_invalid_status(self, evolution_proposal_service):
        """Test error when submitting a proposal with invalid status."""
        # Arrange
        # Create a proposal first
        sample_observation = Observation(
            source=ObservationSource.PERFORMANCE_METRIC,
            target_system="search_service",
            category="performance",
            description="Search service response time degraded",
            data={"avg_response_time_ms": 250, "threshold_ms": 100},
            importance=0.8,
        )
        await evolution_proposal_service.observation_service.add_observation(sample_observation)
        
        proposal = await evolution_proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=sample_observation.id,
            use_mcp=False
        )
        
        # Change status to something other than DRAFT
        proposal.status = ProposalStatus.APPROVED
        
        # Act & Assert
        with pytest.raises(ValidationError):
            await evolution_proposal_service.submit_proposal(proposal.id)
    
    async def test_get_proposal(self, evolution_proposal_service):
        """Test retrieving a proposal by ID."""
        # Arrange
        # Create a proposal first
        sample_observation = Observation(
            source=ObservationSource.PERFORMANCE_METRIC,
            target_system="search_service",
            category="performance",
            description="Search service response time degraded",
            data={"avg_response_time_ms": 250, "threshold_ms": 100},
            importance=0.8,
        )
        await evolution_proposal_service.observation_service.add_observation(sample_observation)
        
        proposal = await evolution_proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=sample_observation.id,
            use_mcp=False
        )
        
        # Act
        retrieved_proposal = await evolution_proposal_service.get_proposal(proposal.id)
        
        # Assert
        assert retrieved_proposal.id == proposal.id
        assert retrieved_proposal.title == proposal.title
    
    async def test_get_proposal_not_found(self, evolution_proposal_service):
        """Test error when retrieving a non-existent proposal."""
        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            await evolution_proposal_service.get_proposal("non_existent_id")
    
    async def test_get_proposals_for_system(self, evolution_proposal_service):
        """Test retrieving proposals for a specific system."""
        # Arrange
        # Create observations for two different systems
        systems = ["search_service", "recommendation_service"]
        proposals_per_system = 3
        
        for system in systems:
            for i in range(proposals_per_system):
                # Create observation
                observation = Observation(
                    source=ObservationSource.PERFORMANCE_METRIC,
                    target_system=system,
                    category="performance",
                    description=f"{system} observation {i}",
                    data={"value": i},
                    importance=0.5 + i * 0.1,
                )
                await evolution_proposal_service.observation_service.add_observation(observation)
                
                # Generate proposal
                await evolution_proposal_service.generate_proposal(
                    target_system=system,
                    agent_id=f"agent-{system}-{i}",
                    trigger_observation_id=observation.id,
                    use_mcp=False
                )
        
        # Act
        search_proposals = await evolution_proposal_service.get_proposals_for_system("search_service")
        recommendation_proposals = await evolution_proposal_service.get_proposals_for_system("recommendation_service")
        
        # Assert
        assert len(search_proposals) == proposals_per_system
        assert len(recommendation_proposals) == proposals_per_system
        assert all(p.target_system == "search_service" for p in search_proposals)
        assert all(p.target_system == "recommendation_service" for p in recommendation_proposals)
    
    async def test_get_proposals_for_system_with_status(self, evolution_proposal_service):
        """Test retrieving proposals for a system with status filter."""
        # Arrange
        # Create proposals
        system = "search_service"
        for i in range(3):
            # Create observation
            observation = Observation(
                source=ObservationSource.PERFORMANCE_METRIC,
                target_system=system,
                category="performance",
                description=f"{system} observation {i}",
                data={"value": i},
                importance=0.5 + i * 0.1,
            )
            await evolution_proposal_service.observation_service.add_observation(observation)
            
            # Generate proposal
            proposal = await evolution_proposal_service.generate_proposal(
                target_system=system,
                agent_id=f"agent-{i}",
                trigger_observation_id=observation.id,
                use_mcp=False
            )
            
            # Submit one proposal
            if i == 0:
                await evolution_proposal_service.submit_proposal(proposal.id)
        
        # Act
        draft_proposals = await evolution_proposal_service.get_proposals_for_system(
            system, status=ProposalStatus.DRAFT
        )
        submitted_proposals = await evolution_proposal_service.get_proposals_for_system(
            system, status=ProposalStatus.SUBMITTED
        )
        
        # Assert
        assert len(draft_proposals) == 2
        assert len(submitted_proposals) == 1
        assert all(p.status == ProposalStatus.DRAFT for p in draft_proposals)
        assert all(p.status == ProposalStatus.SUBMITTED for p in submitted_proposals)
    
    async def test_update_proposal_status(self, evolution_proposal_service):
        """Test updating the status of a proposal."""
        # Arrange
        # Create a proposal first
        sample_observation = Observation(
            source=ObservationSource.PERFORMANCE_METRIC,
            target_system="search_service",
            category="performance",
            description="Search service response time degraded",
            data={"avg_response_time_ms": 250, "threshold_ms": 100},
            importance=0.8,
        )
        await evolution_proposal_service.observation_service.add_observation(sample_observation)
        
        proposal = await evolution_proposal_service.generate_proposal(
            target_system="search_service",
            agent_id="agent-123",
            trigger_observation_id=sample_observation.id,
            use_mcp=False
        )
        
        # Reset mock for event publisher
        evolution_proposal_service.event_publisher.publish.reset_mock()
        
        # Act
        reason = "Approved by evolution validation service"
        updated_proposal = await evolution_proposal_service.update_proposal_status(
            proposal.id, ProposalStatus.APPROVED, reason
        )
        
        # Assert
        assert updated_proposal.status == ProposalStatus.APPROVED
        assert updated_proposal.metadata.get("status_change_reason") == reason
        
        # Verify event
        evolution_proposal_service.event_publisher.publish.assert_called_once()
        call_args = evolution_proposal_service.event_publisher.publish.call_args[1]
        assert call_args["value"]["type"] == "evolution_proposal_status_changed"
        assert call_args["value"]["old_status"] == ProposalStatus.DRAFT
        assert call_args["value"]["new_status"] == ProposalStatus.APPROVED
        assert call_args["value"]["reason"] == reason
