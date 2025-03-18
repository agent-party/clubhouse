"""
Agent Protocol definitions for the Clubhouse platform.

This module defines the Protocol interfaces for agents, establishing the contract
that all agent implementations must fulfill. Following the API-First Development
principle, these interfaces are defined before any implementations.
"""

from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union
from uuid import UUID
from typing import cast, List, Dict, Any, Type

T = TypeVar("T")


class AgentState(Enum):
    """
    Possible states for an agent in its lifecycle.
    
    These states represent the different phases an agent can be in,
    from creation to termination.
    """
    INITIALIZING = auto()
    READY = auto()
    ACTIVE = auto()
    PAUSED = auto()
    ERROR = auto()
    TERMINATED = auto()


class AgentCapability(Enum):
    """
    Capabilities that an agent may possess.
    
    These capabilities define what types of tasks an agent can perform
    and how it can interact with other agents and the system.
    """
    TEXT_PROCESSING = auto()
    IMAGE_PROCESSING = auto()
    AUDIO_PROCESSING = auto()
    VIDEO_PROCESSING = auto()
    CODE_GENERATION = auto()
    REASONING = auto()
    SEARCH = auto()
    TOOL_USE = auto()
    MEMORY = auto()
    LEARNING = auto()
    PLANNING = auto()
    COLLABORATION = auto()
    ORCHESTRATION = auto()


class CapabilityProtocol(Protocol):
    """Protocol that all agent capabilities must implement.
    
    This protocol defines the minimum interface that all capabilities
    must adhere to in order to be usable by agents in the platform.
    """
    
    def get_name(self) -> str:
        """Get the name of the capability.
        
        Returns:
            The name of the capability
        """
        ...
        
    def get_description(self) -> str:
        """Get a description of what the capability does.
        
        Returns:
            A description of the capability
        """
        ...
        
    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the capability with the given parameters.
        
        Args:
            parameters: Parameters for the capability execution
            
        Returns:
            The result of executing the capability
        """
        ...


class AgentMetadata(Protocol):
    """
    Metadata associated with an agent.
    
    This metadata provides information about the agent's identity,
    capabilities, and current state.
    """
    
    @property
    def id(self) -> UUID:
        """Unique identifier for the agent."""
        ...
    
    @property
    def name(self) -> str:
        """Human-readable name for the agent."""
        ...
    
    @property
    def description(self) -> str:
        """Description of the agent's purpose and capabilities."""
        ...
    
    @property
    def version(self) -> str:
        """Version of the agent implementation."""
        ...
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent possesses."""
        ...
    
    @property
    def state(self) -> AgentState:
        """Current state of the agent."""
        ...
    
    @property
    def created_at(self) -> datetime:
        """Timestamp when the agent was created."""
        ...
    
    @property
    def last_active(self) -> Optional[datetime]:
        """Timestamp of the agent's last activity."""
        ...
    
    @property
    def owner_id(self) -> Optional[UUID]:
        """Identifier of the owner of this agent."""
        ...
    
    @property
    def model_id(self) -> Optional[str]:
        """Identifier of the underlying model, if applicable."""
        ...
    
    @property
    def tags(self) -> List[str]:
        """Tags associated with this agent."""
        ...
    
    @property
    def custom_properties(self) -> Dict[str, Any]:
        """Custom properties specific to this agent implementation."""
        ...


class AgentInputType(Enum):
    """Types of input that an agent can process."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    STRUCTURED_DATA = auto()
    BINARY = auto()


class AgentOutputType(Enum):
    """Types of output that an agent can produce."""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    STRUCTURED_DATA = auto()
    BINARY = auto()


class AgentInput(Protocol):
    """
    Input data for an agent to process.
    
    This protocol defines the structure of input data that can be
    sent to an agent for processing.
    """
    
    @property
    def type(self) -> AgentInputType:
        """Type of the input data."""
        ...
    
    @property
    def content(self) -> Any:
        """Content of the input data."""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata associated with the input."""
        ...


class AgentOutput(Protocol):
    """
    Output data produced by an agent.
    
    This protocol defines the structure of output data that an
    agent produces after processing an input.
    """
    
    @property
    def type(self) -> AgentOutputType:
        """Type of the output data."""
        ...
    
    @property
    def content(self) -> Any:
        """Content of the output data."""
        ...
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata associated with the output."""
        ...


class AgentProtocol(Protocol):
    """
    Core protocol that all agents must implement.
    
    This protocol defines the minimum interface that all agents
    must adhere to in order to be usable in the Clubhouse platform.
    """
    
    @property
    def metadata(self) -> AgentMetadata:
        """Get the agent's metadata."""
        ...
    
    def initialize(self) -> None:
        """
        Initialize the agent with any required setup.
        
        This method is called when the agent is first created or registered
        with the system. It should perform any necessary setup operations.
        """
        ...
    
    def shutdown(self) -> None:
        """
        Shutdown the agent and perform cleanup.
        
        This method is called when the agent is being removed from the system
        or when the system is shutting down. It should perform any necessary
        cleanup operations.
        """
        ...
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process input data and produce output.
        
        This is the main method that agents implement to process input
        and produce output based on their capabilities.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The output data produced by the agent
        """
        ...
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the agent.
        
        This method returns a dictionary representing the agent's current
        internal state, which can be used for persistence or debugging.
        
        Returns:
            A dictionary representing the agent's state
        """
        ...
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the agent's state.
        
        This method allows the agent's state to be restored from
        a previously saved state.
        
        Args:
            state: A dictionary representing the agent's state
        """
        ...


class CollaborativeAgentProtocol(AgentProtocol, Protocol):
    """
    Protocol for agents that can collaborate with other agents.
    
    This protocol extends the basic AgentProtocol with methods
    for collaborating with other agents.
    """
    
    def can_collaborate_with(self, agent: AgentProtocol) -> bool:
        """
        Check if this agent can collaborate with another agent.
        
        Args:
            agent: The agent to check for collaboration
            
        Returns:
            True if this agent can collaborate with the given agent, False otherwise
        """
        ...
    
    def get_collaboration_requirements(self) -> Dict[str, Any]:
        """
        Get the requirements for collaborating with this agent.
        
        Returns:
            A dictionary of requirements for collaboration
        """
        ...


class LearningAgentProtocol(AgentProtocol, Protocol):
    """
    Protocol for agents that can learn and adapt over time.
    
    This protocol extends the basic AgentProtocol with methods
    for learning and adaptation.
    """
    
    def learn(self, training_data: List[AgentInput], expected_outputs: List[AgentOutput]) -> None:
        """
        Update the agent's knowledge or behavior based on training data.
        
        Args:
            training_data: List of input data for training
            expected_outputs: List of expected outputs for the training data
        """
        ...
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get information about the agent's learning progress.
        
        Returns:
            A dictionary with information about the agent's learning progress
        """
        ...


class ToolUsingAgentProtocol(AgentProtocol, Protocol):
    """
    Protocol for agents that can use tools.
    
    This protocol extends the basic AgentProtocol with methods
    for using tools to accomplish tasks.
    """
    
    def get_available_tools(self) -> List[str]:
        """
        Get the list of tools available to this agent.
        
        Returns:
            A list of tool identifiers
        """
        ...
    
    def use_tool(self, tool_id: str, parameters: Dict[str, Any]) -> Any:
        """
        Use a tool with the given parameters.
        
        Args:
            tool_id: Identifier of the tool to use
            parameters: Parameters to pass to the tool
            
        Returns:
            The result of using the tool
        """
        ...


class OrchestratorAgentProtocol(AgentProtocol, Protocol):
    """
    Protocol for agents that can orchestrate other agents.
    
    This protocol extends the basic AgentProtocol with methods
    for orchestrating other agents to accomplish complex tasks.
    """
    
    def add_agent(self, agent: AgentProtocol) -> None:
        """
        Add an agent to be orchestrated.
        
        Args:
            agent: The agent to add
        """
        ...
    
    def remove_agent(self, agent_id: UUID) -> None:
        """
        Remove an agent from orchestration.
        
        Args:
            agent_id: The ID of the agent to remove
        """
        ...
    
    def get_orchestrated_agents(self) -> List[UUID]:
        """
        Get the IDs of all agents being orchestrated.
        
        Returns:
            A list of agent IDs
        """
        ...
    
    def create_workflow(self, workflow_definition: Dict[str, Any]) -> UUID:
        """
        Create a workflow involving multiple agents.
        
        Args:
            workflow_definition: Definition of the workflow
            
        Returns:
            ID of the created workflow
        """
        ...
    
    def execute_workflow(self, workflow_id: UUID, input_data: AgentInput) -> AgentOutput:
        """
        Execute a workflow with the given input.
        
        Args:
            workflow_id: ID of the workflow to execute
            input_data: Input data for the workflow
            
        Returns:
            Output of the workflow execution
        """
        ...
