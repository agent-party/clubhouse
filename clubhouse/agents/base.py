"""
Base Agent implementation for the Clubhouse platform.

This module provides base implementations of the agent protocols defined
in the protocols module, following the "Implementation Inheritance" pattern
where common functionality is implemented in base classes.
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Type, cast, Set
from uuid import uuid4, UUID

logger = logging.getLogger(__name__)

from clubhouse.agents.protocols import (
    AgentCapability,
    AgentInput,
    AgentInputType,
    AgentMetadata,
    AgentOutput,
    AgentOutputType,
    AgentProtocol,
    AgentState,
)


class BaseAgentMetadata:
    """Base implementation of the AgentMetadata protocol."""

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        version: str = "0.1.0",
        owner_id: Optional[UUID] = None,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize agent metadata.

        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            capabilities: List of capabilities this agent possesses
            version: Version of the agent implementation
            owner_id: Identifier of the owner of this agent
            model_id: Identifier of the underlying model, if applicable
            tags: Tags associated with this agent
            custom_properties: Custom properties specific to this agent implementation
        """
        self._id = uuid4()
        self._name = name
        self._description = description
        self._version = version
        self._capabilities = capabilities
        self._state = AgentState.INITIALIZING
        self._created_at = datetime.now()
        self._last_active: Optional[datetime] = None
        self._owner_id = owner_id
        self._model_id = model_id
        self._tags = tags or []
        self._custom_properties = custom_properties or {}

    @property
    def id(self) -> UUID:
        """Unique identifier for the agent."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable name for the agent."""
        return self._name

    @property
    def description(self) -> str:
        """Description of the agent's purpose and capabilities."""
        return self._description

    @property
    def version(self) -> str:
        """Version of the agent implementation."""
        return self._version

    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent possesses."""
        return self._capabilities.copy()

    @property
    def state(self) -> AgentState:
        """Current state of the agent."""
        return self._state

    @state.setter
    def state(self, value: AgentState) -> None:
        """Set the current state of the agent."""
        logger.info(f"Agent {self._id} state changing from {self._state} to {value}")
        self._state = value

    @property
    def created_at(self) -> datetime:
        """Timestamp when the agent was created."""
        return self._created_at

    @property
    def last_active(self) -> Optional[datetime]:
        """Timestamp of the agent's last activity."""
        return self._last_active

    @last_active.setter
    def last_active(self, value: Optional[datetime]) -> None:
        """Set the timestamp of the agent's last activity."""
        self._last_active = value

    @property
    def owner_id(self) -> Optional[UUID]:
        """Identifier of the owner of this agent."""
        return self._owner_id

    @property
    def model_id(self) -> Optional[str]:
        """Identifier of the underlying model, if applicable."""
        return self._model_id

    @property
    def tags(self) -> List[str]:
        """Tags associated with this agent."""
        return self._tags.copy()

    @property
    def custom_properties(self) -> Dict[str, Any]:
        """Custom properties specific to this agent implementation."""
        return self._custom_properties.copy()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the metadata to a dictionary.

        Returns:
            Dictionary representation of the metadata
        """
        return {
            "id": str(self._id),
            "name": self._name,
            "description": self._description,
            "version": self._version,
            "capabilities": [cap.name for cap in self._capabilities],
            "state": self._state.name,
            "created_at": self._created_at.isoformat(),
            "last_active": self._last_active.isoformat() if self._last_active else None,
            "owner_id": str(self._owner_id) if self._owner_id else None,
            "model_id": self._model_id,
            "tags": self._tags,
            "custom_properties": self._custom_properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseAgentMetadata":
        """
        Create a BaseAgentMetadata instance from a dictionary.

        Args:
            data: Dictionary representation of the metadata

        Returns:
            A new BaseAgentMetadata instance
        """
        capabilities = [
            AgentCapability[cap] for cap in data.get("capabilities", [])
        ]
        
        instance = cls(
            name=data["name"],
            description=data["description"],
            capabilities=capabilities,
            version=data.get("version", "0.1.0"),
            owner_id=UUID(data["owner_id"]) if data.get("owner_id") else None,
            model_id=data.get("model_id"),
            tags=data.get("tags", []),
            custom_properties=data.get("custom_properties", {}),
        )
        
        # Set properties that aren't part of the constructor
        instance._id = UUID(data["id"])
        instance._state = AgentState[data["state"]]
        instance._created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_active"):
            instance._last_active = datetime.fromisoformat(data["last_active"])
            
        return instance


class BaseAgentInput:
    """Base implementation of the AgentInput protocol."""

    def __init__(
        self,
        input_type: AgentInputType,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize agent input.

        Args:
            input_type: Type of the input data
            content: Content of the input data
            metadata: Metadata associated with the input
        """
        self._type = input_type
        self._content = content
        self._metadata = metadata or {}
        
    @property
    def type(self) -> AgentInputType:
        """Type of the input data."""
        return self._type
    
    @property
    def content(self) -> Any:
        """Content of the input data."""
        return self._content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata associated with the input."""
        return self._metadata.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the input to a dictionary.
        
        Returns:
            Dictionary representation of the input
        """
        return {
            "type": self._type.name,
            "content": self._content,
            "metadata": self._metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseAgentInput":
        """
        Create a BaseAgentInput instance from a dictionary.
        
        Args:
            data: Dictionary representation of the input
            
        Returns:
            A new BaseAgentInput instance
        """
        return cls(
            input_type=AgentInputType[data["type"]],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class BaseAgentOutput:
    """Base implementation of the AgentOutput protocol."""

    def __init__(
        self,
        output_type: AgentOutputType,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize agent output.
        
        Args:
            output_type: Type of the output data
            content: Content of the output data
            metadata: Metadata associated with the output
        """
        self._type = output_type
        self._content = content
        self._metadata = metadata or {}
        
    @property
    def type(self) -> AgentOutputType:
        """Type of the output data."""
        return self._type
    
    @property
    def content(self) -> Any:
        """Content of the output data."""
        return self._content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata associated with the output."""
        return self._metadata.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the output to a dictionary.
        
        Returns:
            Dictionary representation of the output
        """
        return {
            "type": self._type.name,
            "content": self._content,
            "metadata": self._metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseAgentOutput":
        """
        Create a BaseAgentOutput instance from a dictionary.
        
        Args:
            data: Dictionary representation of the output
            
        Returns:
            A new BaseAgentOutput instance
        """
        return cls(
            output_type=AgentOutputType[data["type"]],
            content=data["content"],
            metadata=data.get("metadata", {}),
        )


class BaseAgent:
    """
    Base implementation of the AgentProtocol.
    
    This class provides a foundation for building agents with common
    functionality implemented, allowing derived classes to focus on
    implementing their specific capabilities.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        version: str = "0.1.0",
        owner_id: Optional[UUID] = None,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize an agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose and capabilities
            capabilities: List of capabilities this agent possesses
            version: Version of the agent implementation
            owner_id: Identifier of the owner of this agent
            model_id: Identifier of the underlying model, if applicable
            tags: Tags associated with this agent
            custom_properties: Custom properties specific to this agent implementation
        """
        self._metadata = BaseAgentMetadata(
            name=name,
            description=description,
            capabilities=capabilities,
            version=version,
            owner_id=owner_id,
            model_id=model_id,
            tags=tags,
            custom_properties=custom_properties,
        )
        self._state: Dict[str, Any] = {}
        
    @property
    def metadata(self) -> AgentMetadata:
        """Get the agent's metadata."""
        return self._metadata
    
    def initialize(self) -> None:
        """
        Initialize the agent with any required setup.
        
        This method is called when the agent is first created or registered
        with the system. It should perform any necessary setup operations.
        """
        logger.info(f"Initializing agent: {self._metadata.name} ({self._metadata.id})")
        self._metadata.state = AgentState.READY
    
    def shutdown(self) -> None:
        """
        Shutdown the agent and perform cleanup.
        
        This method is called when the agent is being removed from the system
        or when the system is shutting down. It should perform any necessary
        cleanup operations.
        """
        logger.info(f"Shutting down agent: {self._metadata.name} ({self._metadata.id})")
        self._metadata.state = AgentState.TERMINATED
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process input data and produce output.
        
        This is the main method that agents implement to process input
        and produce output based on their capabilities.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The output data produced by the agent
            
        Raises:
            NotImplementedError: This method must be implemented by derived classes
        """
        self._metadata.last_active = datetime.now()
        self._metadata.state = AgentState.ACTIVE
        
        # This is a base implementation that should be overridden by derived classes
        raise NotImplementedError("The process method must be implemented by derived classes")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        This method returns a dictionary representing the agent's current
        internal state, which can be used for persistence or debugging.
        
        Returns:
            A dictionary representing the agent's state
        """
        # Create a copy of the state to prevent external modification
        state_copy = self._state.copy()
        
        # Add metadata to the state
        state_copy["metadata"] = self._metadata.to_dict()
        
        return state_copy
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the agent's state.
        
        This method allows the agent's state to be restored from
        a previously saved state.
        
        Args:
            state: A dictionary representing the agent's state
        """
        # Restore metadata if present
        if "metadata" in state:
            self._metadata = BaseAgentMetadata.from_dict(state["metadata"])
            
            # Remove metadata from state to avoid duplication
            state_copy = state.copy()
            del state_copy["metadata"]
            
            # Update internal state
            self._state = state_copy
        else:
            # Just update internal state
            self._state = state.copy()
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent to a dictionary.
        
        Returns:
            Dictionary representation of the agent
        """
        return {
            "metadata": self._metadata.to_dict(),
            "state": self._state,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseAgent":
        """
        Create a BaseAgent instance from a dictionary.
        
        Args:
            data: Dictionary representation of the agent
            
        Returns:
            A new BaseAgent instance
        """
        metadata = data["metadata"]
        capabilities = [
            AgentCapability[cap] for cap in metadata.get("capabilities", [])
        ]
        
        instance = cls(
            name=metadata["name"],
            description=metadata["description"],
            capabilities=capabilities,
            version=metadata.get("version", "0.1.0"),
            owner_id=UUID(metadata["owner_id"]) if metadata.get("owner_id") else None,
            model_id=metadata.get("model_id"),
            tags=metadata.get("tags", []),
            custom_properties=metadata.get("custom_properties", {}),
        )
        
        # Set the state
        if "state" in data:
            instance._state = data["state"]
            
        return instance
