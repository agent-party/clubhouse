"""
Tests for the clubhouse agent manager service.

This module contains tests for the agent manager service, which is responsible
for creating, retrieving, and managing agent lifecycles.
"""

import pytest
from unittest.mock import MagicMock, patch

from clubhouse.agents.agent_protocol import AgentProtocol
from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.services.agent_manager import AgentManager


@pytest.fixture
def mock_agent():
    """Fixture for a mock agent."""
    agent = MagicMock(spec=AgentProtocol)
    agent.agent_id.return_value = "test-agent"
    agent.name.return_value = "Test Agent"
    agent.description.return_value = "A test agent"
    return agent


@pytest.fixture
def mock_agent_factory():
    """Fixture for a mock agent factory."""
    with patch('clubhouse.agents.factory.AgentFactory') as mock_factory:
        instance = mock_factory.return_value
        yield instance


@pytest.fixture
def service_registry():
    """Fixture for a service registry."""
    return MagicMock(spec=ServiceRegistry)


@pytest.fixture
def agent_manager(service_registry, mock_agent_factory):
    """Fixture for an agent manager with mocked dependencies."""
    manager = AgentManager(service_registry)
    manager._agent_factory = mock_agent_factory
    return manager


def test_agent_manager_initialization(agent_manager, service_registry):
    """Test that the agent manager initializes correctly."""
    assert agent_manager._service_registry == service_registry
    assert agent_manager._agents == {}


def test_create_agent(agent_manager, mock_agent_factory, mock_agent):
    """Test creating a new agent."""
    # Set up the mock factory to return our mock agent
    mock_agent_factory.create_agent.return_value = mock_agent
    
    # Create an agent
    agent = agent_manager.create_agent(
        agent_id="test-agent",
        personality_type="researcher",
        metadata={"key": "value"}
    )
    
    # Verify the agent was created correctly
    assert agent == mock_agent
    
    # Verify the factory was called with the right parameters
    # We need to match the parameters that AgentManager now passes to the factory
    mock_agent_factory.create_agent.assert_called_once()
    call_args = mock_agent_factory.create_agent.call_args[1]
    
    # Check the expected parameters
    from clubhouse.agents.simple_agent import SimpleAgent
    assert call_args["agent_class"] == SimpleAgent
    assert call_args["agent_id"] == "test-agent"
    assert call_args["personality_type"] == "researcher"
    assert call_args["name"] == "Researcher Agent"  # Default name based on personality type
    assert call_args["description"] == "A researcher agent that can assist users."  # Default description
    assert call_args["capabilities"] == []  # Empty capabilities list
    assert call_args["metadata"] == {"key": "value"}
    
    # Verify agent was initialized (this is now done in the factory)
    # mock_agent.initialize.assert_called_once()
    
    # Verify the agent was stored
    assert "test-agent" in agent_manager._agents
    assert agent_manager._agents["test-agent"] == mock_agent


def test_create_agent_duplicate_id(agent_manager, mock_agent):
    """Test creating an agent with a duplicate ID."""
    # Add an agent to the manager
    agent_manager._agents["test-agent"] = mock_agent
    
    # Attempt to create an agent with the same ID
    with pytest.raises(ValueError):
        agent_manager.create_agent(
            agent_id="test-agent",
            personality_type="researcher",
            metadata={}
        )


def test_get_agent(agent_manager, mock_agent):
    """Test retrieving an agent by ID."""
    # Add an agent to the manager
    agent_manager._agents["test-agent"] = mock_agent
    
    # Get the agent
    agent = agent_manager.get_agent("test-agent")
    
    # Verify the correct agent was returned
    assert agent == mock_agent


def test_get_agent_not_found(agent_manager):
    """Test retrieving an agent that doesn't exist."""
    # Attempt to get a non-existent agent
    with pytest.raises(ValueError):
        agent_manager.get_agent("non-existent-agent")


def test_delete_agent(agent_manager, mock_agent):
    """Test deleting an agent."""
    # Add an agent to the manager
    agent_manager._agents["test-agent"] = mock_agent
    
    # Delete the agent
    agent_manager.delete_agent("test-agent")
    
    # Verify the agent was shut down
    mock_agent.shutdown.assert_called_once()
    
    # Verify the agent was removed
    assert "test-agent" not in agent_manager._agents


def test_delete_agent_not_found(agent_manager):
    """Test deleting an agent that doesn't exist."""
    # Attempt to delete a non-existent agent
    with pytest.raises(ValueError):
        agent_manager.delete_agent("non-existent-agent")


def test_list_agents(agent_manager, mock_agent):
    """Test listing all agents."""
    # Add agents to the manager
    agent1 = mock_agent
    agent2 = MagicMock(spec=AgentProtocol)
    agent2.agent_id.return_value = "test-agent-2"
    
    agent_manager._agents = {
        "test-agent": agent1,
        "test-agent-2": agent2
    }
    
    # List the agents
    agents = agent_manager.list_agents()
    
    # Verify the correct agents were returned
    assert len(agents) == 2
    assert agent1 in agents
    assert agent2 in agents
