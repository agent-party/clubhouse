"""
Agent factory for creating and managing agent instances.

This module provides factory methods for creating agent instances,
simplifying the process of agent creation and configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, cast
from uuid import UUID

from clubhouse.agents.base import BaseAgent
from clubhouse.agents.protocols import AgentCapability, AgentProtocol
from clubhouse.core.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

# Type variable for agent types
T = TypeVar('T', bound=BaseAgent)


class AgentFactory:
    """
    Factory for creating and configuring agent instances.
    
    This factory provides methods for creating agent instances with
    standard configurations, and for persisting agents to storage.
    """
    
    def __init__(self, service_registry: ServiceRegistry) -> None:
        """
        Initialize the agent factory.
        
        Args:
            service_registry: Service registry for accessing services
        """
        self._service_registry = service_registry
    
    def create_agent(
        self,
        agent_class: Type[T],
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        agent_id: Optional[str] = None,
        personality_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> T:
        """
        Create an agent of the specified type with standard configuration.
        
        Args:
            agent_class: Class of the agent to create
            name: Name for the agent
            description: Description of the agent
            capabilities: List of agent capabilities
            agent_id: Optional ID for the agent (will be auto-generated if not provided)
            personality_type: Optional personality type for the agent
            metadata: Optional metadata for the agent
            **kwargs: Additional arguments to pass to the agent constructor
            
        Returns:
            Configured agent instance
        """
        # Combine all kwargs for agent creation
        agent_kwargs = kwargs.copy()
        
        # Add the required parameters that SimpleAgent expects
        if agent_id is not None:
            agent_kwargs["agent_id"] = agent_id
        if personality_type is not None:
            agent_kwargs["personality_type"] = personality_type
        if metadata is not None:
            agent_kwargs["metadata"] = metadata
            
        # Create the agent
        agent = agent_class(
            name=name,
            description=description,
            capabilities=capabilities,
            **agent_kwargs
        )
        
        # Initialize the agent
        agent.initialize()
        
        logger.info(f"Created agent: {name} ({agent.agent_id()})")
        return agent
    
    def persist_agent(self, agent: AgentProtocol) -> UUID:
        """
        Persist an agent to the database.
        
        This method saves the agent's metadata and state to the
        graph database for later retrieval.
        
        Args:
            agent: Agent to persist
            
        Returns:
            UUID of the persisted agent
        """
        try:
            # Get the Neo4j service if available
            from clubhouse.services.neo4j_protocol import Neo4jServiceProtocol  # type: ignore[import-not-found]
            neo4j_service: Optional[Neo4jServiceProtocol] = self._service_registry.get_protocol(Neo4jServiceProtocol)
            
            # Prepare agent data for persistence
            agent_data = {
                "uuid": str(agent.metadata.id),
                "name": agent.metadata.name,
                "description": agent.metadata.description,
                "version": agent.metadata.version,
                "capabilities": [cap.name for cap in agent.metadata.capabilities],
                "state": agent.metadata.state.name,
                "created_at": agent.metadata.created_at.isoformat(),
                "last_active": agent.metadata.last_active.isoformat() if agent.metadata.last_active else None,
                "owner_id": str(agent.metadata.owner_id) if agent.metadata.owner_id else None,
                "model_id": agent.metadata.model_id,
                "tags": agent.metadata.tags,
                "custom_properties": agent.metadata.custom_properties,
                "agent_state": agent.get_state()
            }
            
            # Check if neo4j_service is available
            if neo4j_service is None:
                logger.warning(f"Neo4j service unavailable, skipping database operations for agent: {agent.metadata.id}")
                return agent.metadata.id
                
            # Check if agent already exists
            existing_agents = neo4j_service.find_nodes(
                labels=["Agent"],
                properties={"uuid": str(agent.metadata.id)}
            )
            
            if existing_agents:
                # Update existing agent
                neo4j_service.update_node(
                    node_id=agent.metadata.id,
                    properties=agent_data
                )
                logger.info(f"Updated agent in database: {agent.metadata.name} ({agent.metadata.id})")
            else:
                # Create new agent node
                neo4j_service.create_node(
                    labels=["Agent"],
                    properties=agent_data
                )
                logger.info(f"Persisted agent to database: {agent.metadata.name} ({agent.metadata.id})")
            
            return agent.metadata.id
        
        except Exception as e:
            logger.error(f"Failed to persist agent: {str(e)}")
            raise
    
    def load_agent(self, agent_id: UUID, agent_class: Type[T]) -> Optional[T]:
        """
        Load an agent from the database.
        
        This method retrieves an agent's metadata and state from the
        graph database and reconstructs the agent instance.
        
        Args:
            agent_id: UUID of the agent to load
            agent_class: Class to use for the agent instance
            
        Returns:
            Loaded agent instance, or None if not found
        """
        try:
            # Get the Neo4j service if available
            from clubhouse.services.neo4j_protocol import Neo4jServiceProtocol  # type: ignore[import-not-found]
            neo4j_service: Optional[Neo4jServiceProtocol] = self._service_registry.get_protocol(Neo4jServiceProtocol)
            
            # Check if neo4j_service is available
            if neo4j_service is None:
                logger.warning(f"Neo4j service unavailable, skipping database operations for agent: {agent_id}")
                return None
            
            # Get the agent node
            agent_node = neo4j_service.get_node(agent_id)
            
            if not agent_node:
                logger.warning(f"Agent with ID {agent_id} not found in database")
                return None
            
            # Create a new agent instance
            agent = agent_class(
                name=agent_node["name"],
                description=agent_node["description"],
                capabilities=[AgentCapability[cap] for cap in agent_node["capabilities"]],
                version=agent_node["version"],
                owner_id=UUID(agent_node["owner_id"]) if agent_node.get("owner_id") else None,
                model_id=agent_node.get("model_id"),
                tags=agent_node.get("tags", []),
                custom_properties=agent_node.get("custom_properties", {})
            )
            
            # Set the agent's state
            if "agent_state" in agent_node:
                agent.set_state(agent_node["agent_state"])
            
            # Initialize the agent
            agent.initialize()
            
            logger.info(f"Loaded agent from database: {agent.metadata.name} ({agent.metadata.id})")
            return agent
        
        except Exception as e:
            logger.error(f"Failed to load agent: {str(e)}")
            raise
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all agents in the database.
        
        Returns:
            List of agent metadata dictionaries
        """
        try:
            # Get the Neo4j service if available
            from clubhouse.services.neo4j_protocol import Neo4jServiceProtocol  # type: ignore[import-not-found]
            neo4j_service: Optional[Neo4jServiceProtocol] = self._service_registry.get_protocol(Neo4jServiceProtocol)
            
            # Check if neo4j_service is available
            if neo4j_service is None:
                logger.warning(f"Neo4j service unavailable, skipping database operations")
                return []
            
            # Get all agent nodes
            agent_nodes = neo4j_service.find_nodes(
                labels=["Agent"],
                limit=1000
            )
            
            # Explicitly return as List[Dict[str, Any]] to satisfy mypy
            return [dict(node) for node in agent_nodes]
        
        except Exception as e:
            logger.error(f"Failed to get all agents: {str(e)}")
            raise
