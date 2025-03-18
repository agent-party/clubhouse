"""
Tests for the agent state management functionality.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from uuid import UUID

from clubhouse.agents.state import (
    AgentState,
    AgentStateManager,
    InvalidStateTransitionError
)
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.core.utils.datetime_utils import utc_now


class TestAgentState:
    """Test cases for AgentState enum."""
    
    def test_agent_state_values(self):
        """Test that the AgentState enum has the expected values."""
        assert AgentState.CREATED.value == "created"
        assert AgentState.INITIALIZING.value == "initializing"
        assert AgentState.READY.value == "ready"
        assert AgentState.PROCESSING.value == "processing"
        assert AgentState.PAUSED.value == "paused"
        assert AgentState.ERROR.value == "error"
        assert AgentState.TERMINATED.value == "terminated"


class TestAgentStateManager:
    """Test cases for AgentStateManager."""
    
    @pytest.fixture
    def mock_neo4j_service(self):
        """Create a mock Neo4j service."""
        mock_service = MagicMock(spec=Neo4jServiceProtocol)
        
        # Set up return values for common methods
        test_node_id = UUID("12345678-1234-5678-1234-567812345678")
        mock_service.create_node.return_value = test_node_id
        
        # Mock find_nodes to return a valid agent node
        mock_service.find_nodes.return_value = [{
            "id": str(test_node_id),
            "agent_id": "test-agent",
            "state": AgentState.CREATED.value
        }]
        
        # Mock update_node to return success
        mock_service.update_node.return_value = True
        
        # Mock run_query to return empty list by default
        mock_service.run_query.return_value = []
        
        return mock_service
    
    @pytest.fixture
    def initialized_state_manager(self, mock_neo4j_service):
        """Create an agent state manager with mock service that returns INITIALIZING state."""
        # Override the find_nodes to return INITIALIZING state
        test_node_id = UUID("12345678-1234-5678-1234-567812345678")
        mock_neo4j_service.find_nodes.return_value = [{
            "id": str(test_node_id),
            "agent_id": "test-agent",
            "state": AgentState.INITIALIZING.value
        }]
        return AgentStateManager(mock_neo4j_service)
    
    @pytest.fixture
    def state_manager(self, mock_neo4j_service):
        """Create an agent state manager with mock service."""
        return AgentStateManager(mock_neo4j_service)
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.agent_id = "test-agent"
        agent.name = "Test Agent"
        agent.description = "A test agent"
        return agent
    
    def test_initialize_agent_state(self, state_manager, mock_agent, mock_neo4j_service):
        """Test initializing agent state in Neo4j."""
        # Call the method
        node_id = state_manager.initialize_agent_state(mock_agent)
        
        # Verify the Neo4j service was called correctly
        mock_neo4j_service.create_node.assert_called_once()
        
        # Verify the created node has the right properties
        args, kwargs = mock_neo4j_service.create_node.call_args
        assert args[0] == "Agent"
        assert args[1]["agent_id"] == "test-agent"
        assert args[1]["name"] == "Test Agent"
        assert args[1]["description"] == "A test agent"
        assert args[1]["state"] == AgentState.CREATED.value
        
        # Verify the returned node ID
        assert node_id == mock_neo4j_service.create_node.return_value
    
    def test_get_agent_state(self, state_manager, mock_neo4j_service):
        """Test getting an agent's state."""
        # Call the method
        state = state_manager.get_agent_state("test-agent")
        
        # Verify the Neo4j service was called correctly
        mock_neo4j_service.find_nodes.assert_called_once_with(
            "Agent", {"agent_id": "test-agent"}, limit=1
        )
        
        # Verify the returned state
        assert state == AgentState.CREATED
    
    def test_get_agent_state_not_found(self, state_manager, mock_neo4j_service):
        """Test getting state for a non-existent agent."""
        # Set up the mock to return no nodes
        mock_neo4j_service.find_nodes.return_value = []
        
        # Call the method
        state = state_manager.get_agent_state("nonexistent-agent")
        
        # Verify the Neo4j service was called correctly
        mock_neo4j_service.find_nodes.assert_called_once()
        
        # Verify the returned state
        assert state is None
    
    def test_update_agent_state(self, state_manager, mock_neo4j_service):
        """Test updating an agent's state."""
        # Call the method
        updated = state_manager.update_agent_state(
            "test-agent", AgentState.INITIALIZING
        )
        
        # Verify the Neo4j service was called correctly
        mock_neo4j_service.find_nodes.assert_called_once()
        mock_neo4j_service.update_node.assert_called_once()
        
        # Verify the properties being updated
        args, kwargs = mock_neo4j_service.update_node.call_args
        assert args[1]["state"] == AgentState.INITIALIZING.value
        
        # Verify the result
        assert updated is True
    
    def test_update_agent_state_with_metadata(self, initialized_state_manager, mock_neo4j_service):
        """Test updating an agent's state with metadata."""
        # Call the method with metadata
        metadata = {"reason": "test update", "user": "tester"}
        updated = initialized_state_manager.update_agent_state(
            "test-agent", AgentState.READY, metadata
        )
        
        # Verify the Neo4j service was called correctly
        mock_neo4j_service.update_node.assert_called_once()
        
        # Verify the properties being updated
        args, kwargs = mock_neo4j_service.update_node.call_args
        assert args[1]["state"] == AgentState.READY.value
        assert args[1]["meta_reason"] == "test update"
        assert args[1]["meta_user"] == "tester"
        
        # Verify the result
        assert updated is True
    
    def test_update_agent_state_invalid_transition(self, state_manager):
        """Test that invalid state transitions raise exceptions."""
        # Try an invalid transition
        with pytest.raises(InvalidStateTransitionError):
            state_manager.update_agent_state(
                "test-agent", AgentState.TERMINATED
            )
    
    def test_get_agent_history(self, state_manager, mock_neo4j_service):
        """Test getting an agent's history."""
        # Set up the mock to return history entries
        mock_neo4j_service.run_query.return_value = [
            {"e": {"event_type": "state_transition", "from_state": "created", "to_state": "initializing"}},
            {"e": {"event_type": "state_transition", "from_state": "initializing", "to_state": "ready"}}
        ]
        
        # Call the method
        history = state_manager.get_agent_history("test-agent")
        
        # Verify the Neo4j service was called correctly
        mock_neo4j_service.find_nodes.assert_called_once()
        mock_neo4j_service.run_query.assert_called_once()
        
        # Verify the history
        assert len(history) == 2
        assert history[0]["event_type"] == "state_transition"
        assert history[0]["from_state"] == "created"
        assert history[0]["to_state"] == "initializing"
    
    def test_is_valid_transition(self, state_manager):
        """Test the transition validation logic."""
        # Valid transitions
        assert state_manager._is_valid_transition(AgentState.CREATED, AgentState.INITIALIZING)
        assert state_manager._is_valid_transition(AgentState.INITIALIZING, AgentState.READY)
        assert state_manager._is_valid_transition(AgentState.READY, AgentState.PROCESSING)
        
        # Invalid transitions
        assert not state_manager._is_valid_transition(AgentState.CREATED, AgentState.READY)
        assert not state_manager._is_valid_transition(AgentState.CREATED, AgentState.TERMINATED)
        assert not state_manager._is_valid_transition(AgentState.TERMINATED, AgentState.READY)
        
        # Same state (no-op) is always valid
        assert state_manager._is_valid_transition(AgentState.READY, AgentState.READY)
        assert state_manager._is_valid_transition(AgentState.ERROR, AgentState.ERROR)
    
    def test_record_state_transition(self, state_manager, mock_neo4j_service):
        """Test recording a state transition event."""
        # Test UUID
        agent_node_id = UUID("12345678-1234-5678-1234-567812345678")
        event_node_id = UUID("87654321-4321-8765-4321-876543210987")
        
        # Set up the mock to return proper event UUID
        mock_neo4j_service.create_node.side_effect = [agent_node_id, event_node_id]
        
        # Call the method
        event_id = state_manager._record_state_transition(
            agent_node_id,
            AgentState.CREATED,
            AgentState.INITIALIZING,
            {"reason": "test transition"}
        )
        
        # Verify the Neo4j service was called correctly
        assert mock_neo4j_service.create_node.call_count >= 1
        mock_neo4j_service.create_relationship.assert_called_once()
        
        # Verify the created event node properties
        create_node_args = mock_neo4j_service.create_node.call_args[0]
        assert create_node_args[0] == "StateTransitionEvent"
        assert create_node_args[1]["event_type"] == "state_transition"
        assert create_node_args[1]["from_state"] == "created"
        assert create_node_args[1]["to_state"] == "initializing"
        assert create_node_args[1]["metadata"] == {"reason": "test transition"}
        
        # Verify the relationship was created
        mock_neo4j_service.create_relationship.assert_called_once_with(
            agent_node_id, event_id, "STATE_TRANSITION"
        )
