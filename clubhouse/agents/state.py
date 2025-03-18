"""
Agent state management models and utilities.

This module contains models and utilities for managing agent state persistence,
lifecycle events, and state transitions within the Neo4j graph database.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
from uuid import UUID, uuid4
import logging

from clubhouse.agents.agent_protocol import AgentProtocol, AgentMessage, AgentResponse
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from typing import cast, List, Dict, Any, Type
from clubhouse.core.utils.datetime_utils import utc_now

# Configure logger
logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Enumeration of possible agent states in their lifecycle."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class InvalidStateTransitionError(Exception):
    """Exception raised when an invalid state transition is attempted."""
    
    def __init__(self, current_state: AgentState, target_state: AgentState) -> None:
        self.current_state = current_state
        self.target_state = target_state
        message = f"Invalid state transition from {current_state} to {target_state}"
        super().__init__(message)


class AgentStateManager:
    """
    Manages agent state persistence and transitions in Neo4j.
    
    This class handles:
    1. Storing and retrieving agent state in Neo4j
    2. Validating state transitions
    3. Recording state transition events
    4. Tracking agent history
    """
    
    # Valid state transitions dictionary
    # Keys are current states, values are sets of valid target states
    VALID_TRANSITIONS = {
        AgentState.CREATED: {AgentState.INITIALIZING},
        AgentState.INITIALIZING: {AgentState.READY, AgentState.ERROR},
        AgentState.READY: {AgentState.PROCESSING, AgentState.PAUSED, AgentState.ERROR, AgentState.TERMINATED},
        AgentState.PROCESSING: {AgentState.READY, AgentState.PAUSED, AgentState.ERROR, AgentState.TERMINATED},
        AgentState.PAUSED: {AgentState.READY, AgentState.ERROR, AgentState.TERMINATED},
        AgentState.ERROR: {AgentState.INITIALIZING, AgentState.READY, AgentState.TERMINATED},
        AgentState.TERMINATED: set()  # Terminal state - no valid transitions out
    }
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol) -> None:
        """
        Initialize an agent state manager with Neo4j service.
        
        Args:
            neo4j_service: Neo4j service for state persistence
        """
        self.neo4j_service = neo4j_service
    
    def initialize_agent_state(self, agent: AgentProtocol) -> UUID:
        """
        Initialize an agent's state in the graph database.
        
        Args:
            agent: Agent protocol implementation
            
        Returns:
            UUID of the created agent node
        """
        # Create agent node with initial state
        properties = {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "state": AgentState.CREATED.value,
            "created_at": utc_now().isoformat(),
            "updated_at": utc_now().isoformat(),
        }
        
        # Create the agent node in Neo4j
        node_id = self.neo4j_service.create_node("Agent", properties)
        logger.info(f"Initialized agent state for {agent.agent_id} with node ID {node_id}")
        
        return node_id
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """
        Get the current state of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Current state of the agent, or None if the agent doesn't exist
        """
        # Find the agent node by agent_id property
        nodes = self.neo4j_service.find_nodes("Agent", {"agent_id": agent_id}, limit=1)
        
        if not nodes:
            logger.warning(f"Agent {agent_id} not found in database")
            return None
        
        # Return the agent's state
        state_value = nodes[0].get("state")
        return AgentState(state_value) if state_value else None
    
    def update_agent_state(self, agent_id: str, new_state: AgentState, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an agent's state after validating the transition.
        
        Args:
            agent_id: ID of the agent
            new_state: Target state to transition to
            metadata: Optional metadata about the transition
            
        Returns:
            True if the state was updated, False otherwise
            
        Raises:
            InvalidStateTransitionError: If the transition is not valid
        """
        # Find the agent node
        nodes = self.neo4j_service.find_nodes("Agent", {"agent_id": agent_id}, limit=1)
        
        if not nodes:
            logger.warning(f"Agent {agent_id} not found in database")
            return False
        
        node_id = UUID(nodes[0].get("id"))
        current_state = AgentState(nodes[0].get("state"))
        
        # Validate the state transition
        if not self._is_valid_transition(current_state, new_state):
            raise InvalidStateTransitionError(current_state, new_state)
        
        # Update agent state
        properties = {
            "state": new_state.value,
            "updated_at": utc_now().isoformat()
        }
        
        # Add optional metadata
        if metadata:
            for key, value in metadata.items():
                # Prefix metadata keys to avoid collision with standard properties
                properties[f"meta_{key}"] = value
        
        # Update the node
        updated = self.neo4j_service.update_node(node_id, properties)
        
        if updated:
            # Record state transition event
            self._record_state_transition(node_id, current_state, new_state, metadata)
            logger.info(f"Updated agent {agent_id} state from {current_state} to {new_state}")
        
        return updated
    
    def _is_valid_transition(self, current_state: AgentState, target_state: AgentState) -> bool:
        """
        Check if a state transition is valid.
        
        Args:
            current_state: Current state of the agent
            target_state: Target state for transition
            
        Returns:
            True if the transition is valid, False otherwise
        """
        # Same state is always allowed (no-op)
        if current_state == target_state:
            return True
            
        # Check transition validity
        return target_state in self.VALID_TRANSITIONS.get(current_state, set())
    
    def _record_state_transition(self, agent_node_id: UUID, from_state: AgentState, 
                               to_state: AgentState, metadata: Optional[Dict[str, Any]]) -> UUID:
        """
        Record a state transition event in the graph database.
        
        Args:
            agent_node_id: UUID of the agent node
            from_state: Previous state
            to_state: New state
            metadata: Optional metadata about the transition
            
        Returns:
            UUID of the created transition event node
        """
        # Create transition event node
        event_properties = {
            "event_type": "state_transition",
            "from_state": from_state.value,
            "to_state": to_state.value,
            "timestamp": utc_now().isoformat()
        }
        
        # Add metadata
        if metadata:
            event_properties["metadata"] = metadata  # type: ignore[type_assignment]
        
        # Create the event node
        result = self.neo4j_service.create_node("StateTransitionEvent", event_properties)
        
        # Extract event ID from result if it's a dict, otherwise use the result directly
        if isinstance(result, dict) and "id" in result:
            event_id = result["id"]
        else:
            event_id = str(result)
        
        # Connect event to agent
        self.neo4j_service.create_relationship(
            agent_node_id, 
            event_id, 
            "STATE_TRANSITION"
        )
        
        return event_id  # type: ignore[return_type]
    
    def get_agent_history(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the state transition history for an agent.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of history entries to retrieve
            
        Returns:
            List of state transition events, ordered by timestamp (newest first)
        """
        # Find the agent node
        nodes = self.neo4j_service.find_nodes("Agent", {"agent_id": agent_id}, limit=1)
        
        if not nodes:
            logger.warning(f"Agent {agent_id} not found in database")
            return []
        
        node_id = UUID(nodes[0].get("id"))
        
        # Query for the agent's state transition events
        query = """
        MATCH (a:Agent {id: $agent_id})-[:STATE_TRANSITION]->(e:StateTransitionEvent)
        RETURN e
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        
        parameters = {
            "agent_id": str(node_id),
            "limit": limit
        }
        
        results = self.neo4j_service.run_query(query, parameters)
        
        # Extract event data
        return [result.get("e", {}) for result in results]