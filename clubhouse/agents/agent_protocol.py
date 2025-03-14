"""
Agent protocol definitions for the MCP demo.

This module defines the protocols that agents must implement to participate
in the MCP integration.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar, runtime_checkable


@runtime_checkable
class CapabilityProtocol(Protocol):
    """
    Protocol defining the capability interface for agents.
    
    Capabilities represent specific functionalities that an agent can perform.
    """
    
    @property
    def name(self) -> str:
        """Get the unique name of the capability."""
        ...
    
    @property
    def description(self) -> str:
        """Get a human-readable description of what this capability does."""
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get the parameters required by this capability."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol defining the core agent interface.
    
    Agents are responsible for processing messages and executing capabilities.
    """
    
    @property
    def agent_id(self) -> str:
        """Get the unique identifier for this agent."""
        ...
    
    @property
    def name(self) -> str:
        """Get the human-readable name of this agent."""
        ...
    
    @property
    def description(self) -> str:
        """Get a human-readable description of what this agent does."""
        ...
    
    def get_capabilities(self) -> List[CapabilityProtocol]:
        """
        Get all capabilities supported by this agent.
        
        Returns:
            A list of capabilities supported by this agent.
        """
        ...
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message to process.
            
        Returns:
            The response message.
        """
        ...


class BaseCapability(ABC):
    """
    Base class for implementing capabilities.
    
    This provides a common implementation for the capability protocol.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the unique name of the capability."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a human-readable description of what this capability does."""
        pass
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the parameters required by this capability.
        
        Override this in subclasses to define capability-specific parameters.
        
        Returns:
            A dictionary mapping parameter names to their schema definitions.
        """
        return {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute this capability with the given parameters.
        
        Args:
            **kwargs: The parameters to use when executing this capability.
            
        Returns:
            The result of executing this capability.
        """
        pass


class BaseAgent(ABC):
    """
    Base class for implementing agents.
    
    This provides a common implementation for the agent protocol.
    """
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Get the unique identifier for this agent."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the human-readable name of this agent."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a human-readable description of what this agent does."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[CapabilityProtocol]:
        """
        Get all capabilities supported by this agent.
        
        Returns:
            A list of capabilities supported by this agent.
        """
        pass
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and return a response.
        
        Args:
            message: The message to process.
            
        Returns:
            The response message.
        """
        pass
