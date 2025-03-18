"""
Agent manager service for the clubhouse.

This module provides a service for managing agent lifecycle, including
creation, retrieval, and deletion of agents.
"""

import logging
from typing import Dict, Optional, Protocol, List, Any, cast

from clubhouse.agents.agent_protocol import AgentProtocol, AgentState
from clubhouse.agents.simple_agent import SimpleAgent
from clubhouse.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


class AgentManagerProtocol(Protocol):
    """Protocol for agent manager service."""
    
    def create_agent(
        self, 
        agent_id: str, 
        personality_type: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentProtocol:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique ID for the agent
            personality_type: Type of personality for the agent (e.g., "assistant")
            metadata: Optional metadata for the agent
            
        Returns:
            The created agent
            
        Raises:
            ValueError: If an agent with the given ID already exists
        """
        ...
    
    def get_agent(self, agent_id: str) -> AgentProtocol:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The requested agent
            
        Raises:
            ValueError: If the agent does not exist
        """
        ...
    
    def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Raises:
            ValueError: If the agent does not exist
        """
        ...
    
    def list_agents(self) -> List[AgentProtocol]:
        """
        List all agents.
        
        Returns:
            List of all agents
        """
        ...


class AgentManager(AgentManagerProtocol):
    """
    Service for managing agent lifecycle.
    
    This service is responsible for creating, retrieving, and deleting
    agents within the system.
    """
    
    def __init__(self, service_registry: ServiceRegistry, agent_factory=None) -> None:
        """
        Initialize the agent manager.
        
        Args:
            service_registry: Registry for accessing required services
            agent_factory: Optional agent factory to use for creating agents
        """
        self._service_registry = service_registry
        self._agents: Dict[str, AgentProtocol] = {}
        
        # Use provided agent_factory if available
        if agent_factory is not None:
            self._agent_factory = agent_factory
        else:
            # Import here to avoid circular imports
            try:
                from clubhouse.agents.factory import AgentFactory
                self._agent_factory = self._service_registry.get(AgentFactory)
                if self._agent_factory is None:
                    self._agent_factory = AgentFactory(service_registry)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Agent factory not available: {e}")
                self._agent_factory = None
    
    def create_agent(
        self, 
        agent_id: str, 
        personality_type: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentProtocol:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique ID for the agent
            personality_type: Type of personality for the agent (e.g., "assistant")
            metadata: Optional metadata for the agent
            
        Returns:
            The created agent
            
        Raises:
            ValueError: If an agent with the given ID already exists
        """
        # Check if agent already exists
        if agent_id in self._agents:
            raise ValueError(f"Agent with ID {agent_id} already exists")
        
        # Create agent
        logger.info(f"Creating agent: {agent_id} (type: {personality_type})")
        
        # Ensure metadata is a dictionary
        metadata = metadata or {}
        
        # Extract or generate name and description
        name = metadata.get("name", f"{personality_type.capitalize()} Agent")
        description = metadata.get("description", f"A {personality_type} agent that can assist users.")
        
        # Use factory if available, otherwise fall back to SimpleAgent
        if self._agent_factory is not None:
            # Import here to avoid circular imports
            from clubhouse.agents.simple_agent import SimpleAgent
            
            # Create the agent using the factory with all required parameters
            agent = self._agent_factory.create_agent(
                agent_class=SimpleAgent,
                agent_id=agent_id,
                personality_type=personality_type,
                name=name,
                description=description,
                capabilities=[],  # Empty list as default
                metadata=metadata
            )
        else:
            # Fallback to SimpleAgent for our Kafka CLI integration
            agent = SimpleAgent(
                agent_id=agent_id,
                personality_type=personality_type,
                name=name,
                description=description,
                metadata=metadata
            )
        
        # Initialize the agent
        agent.initialize()
        
        # Store the agent
        self._agents[agent_id] = agent
        
        return agent
    
    def get_agent(self, agent_id: str) -> AgentProtocol:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            The requested agent
            
        Raises:
            ValueError: If the agent does not exist
        """
        # Check if agent exists
        if agent_id not in self._agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist")
        
        return self._agents[agent_id]
    
    def delete_agent(self, agent_id: str) -> None:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent to delete
            
        Raises:
            ValueError: If the agent does not exist
        """
        # Check if agent exists
        if agent_id not in self._agents:
            raise ValueError(f"Agent with ID {agent_id} does not exist")
        
        # Get the agent
        agent = self._agents[agent_id]
        
        # Shut down the agent
        logger.info(f"Shutting down agent: {agent_id}")
        agent.shutdown()
        
        # Remove the agent
        del self._agents[agent_id]
    
    def list_agents(self) -> List[AgentProtocol]:
        """
        List all agents.
        
        Returns:
            List of all agents
        """
        return list(self._agents.values())
