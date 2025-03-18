"""
Agent protocol definitions for the MCP demo.

This module defines the protocols that agents must implement to participate
in the MCP integration. It provides both Protocol interfaces for type checking
and abstract base classes for implementation inheritance.

The design follows the capability-based security model, where agents expose
specific capabilities that can be invoked through a standard interface.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    Any, 
    ClassVar, 
    Dict, 
    Final, 
    Generic, 
    List, 
    Literal, 
    Optional, 
    Protocol, 
    Self,
    Type, 
    TypedDict, 
    TypeVar, 
    Union, 
    NotRequired,
    cast,
    runtime_checkable,
    Callable,
    overload
)
from typing_extensions import Annotated
from uuid import UUID, uuid4
import traceback
from dataclasses import dataclass
import sys
import json
import logging
from typing import cast, List, Dict, Any, Type
from typing import AsyncIterator

# Configure logger
logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar("T")
R = TypeVar("R")


class ApprovalStatus(Enum):
    """
    Approval status for capability execution requiring human intervention.
    
    This enum represents the different states of human approval for agent actions.
    """
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    CANCELLED = auto()


class CapabilityError(Exception):
    """Base exception for capability-related errors."""
    pass


class ExecutionError(CapabilityError):
    """Exception raised when a capability execution fails."""
    def __init__(self, message: str, capability_name: str) -> None:
        self.capability_name = capability_name
        super().__init__(f"Error executing capability '{capability_name}': {message}")


class ValidationError(CapabilityError):
    """Exception raised when capability parameters fail validation."""
    def __init__(self, message: str, capability_name: str, parameter_name: Optional[str] = None) -> None:
        self.capability_name = capability_name
        self.parameter_name = parameter_name
        param_info = f" (parameter: {parameter_name})" if parameter_name else ""
        super().__init__(f"Validation error for capability '{capability_name}'{param_info}: {message}")


class CapabilityNotFoundError(CapabilityError):
    """Exception raised when a requested capability is not found."""
    def __init__(self, capability_name: str) -> None:
        self.capability_name = capability_name
        super().__init__(f"Capability '{capability_name}' not found")


class ParameterSchema(TypedDict):
    """Type definition for capability parameter schema."""
    type: Literal["string", "integer", "number", "boolean", "object", "array"]
    description: str
    required: NotRequired[bool]
    default: NotRequired[Any]
    enum: NotRequired[List[Any]]


class CapabilityParameters(TypedDict):
    """Type definition for capability parameters dictionary."""
    pass


class CapabilityResult(TypedDict):
    """Type definition for capability execution result."""
    status: Literal["success", "error"]
    data: NotRequired[Any]
    error: NotRequired[str]


class AgentState(Enum):
    """Agent state enum."""
    CREATED = auto()
    INITIALIZING = auto()
    READY = auto()
    PROCESSING = auto()
    PAUSED = auto()
    ERROR = auto()
    TERMINATED = auto()


class MessageType(Enum):
    """Message type enum for agent communication."""
    COMMAND = auto()
    RESPONSE = auto()
    EVENT = auto()
    ERROR = auto()


@dataclass
class AgentResponse:
    """Standard response format for agent operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentMessage:
    """Standard message format for agent communication."""
    message_id: UUID
    sender: str
    message_type: MessageType
    content: Dict[str, Any]
    recipient: Optional[str] = None


@runtime_checkable
class CapabilityEventHandler(Protocol):
    """Protocol for capability event handlers."""
    
    def __call__(self, **kwargs: Any) -> None:
        """
        Handler function for capability events.
        
        Args:
            **kwargs: Event-specific parameters
        """
        ...


@runtime_checkable
class CapabilityProtocol(Protocol):
    """
    Protocol defining the interface for agent capabilities.
    
    A capability is a discrete piece of functionality that an agent can perform.
    It includes metadata about what the capability does, what parameters it requires,
    and methods for executing the capability.
    """
    
    @property
    def name(self) -> str:
        """
        Get the unique identifier for this capability.
        
        This name should be unique within an agent and is used for capability lookup
        and routing.
        
        Returns:
            The capability name as a string
        """
        ...
    
    @property
    def description(self) -> str:
        """
        Get a human-readable description of what this capability does.
        
        This description should clearly explain the purpose and function of the
        capability to both developers and end-users.
        
        Returns:
            Description string
        """
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get the parameters required by this capability.
        
        This metadata is used for validation and documentation.
        
        Returns:
            Dictionary mapping parameter names to descriptions or schemas
        """
        ...
    
    @property
    def version(self) -> str:
        """
        Get the version of this capability.
        
        Version information is important for tracking API changes over time.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        ...
    
    def requires_human_approval(self) -> bool:
        """
        Determine if this capability requires human approval before execution.
        
        This is a critical security and safety feature that allows certain
        capabilities to be gated behind human review.
        
        Returns:
            True if human approval is required, False otherwise
        """
        ...
    
    def register_event_handler(self, event_type: str, handler: Callable[..., Any]) -> None:
        """
        Register an event handler for capability lifecycle events.
        
        Event handlers allow for extending capability behavior without modifying
        the core implementation.
        
        Args:
            event_type: The event type to register for (e.g., "started", "completed")
            handler: The handler function to call when the event occurs
        """
        ...
    
    def get_operation_cost(self) -> Dict[str, float]:
        """
        Get the cost details for the last executed operation.
        
        This is used for accounting and monitoring resource usage.
        
        Returns:
            Dictionary with cost details (tokens, API calls, etc.)
        """
        ...
    
    async def execute(self, parameters: Dict[str, Any]) -> CapabilityResult:
        """
        Execute the capability with the provided parameters.
        
        This is the core method that implements the capability's functionality.
        
        Args:
            parameters: The parameters for the capability execution
            
        Returns:
            Dictionary with the execution results
        """
        ...
    
    async def execute_with_lifecycle(self, parameters: Dict[str, Any]) -> CapabilityResult:
        """
        Execute the capability with lifecycle event handling.
        
        This method handles the entire capability lifecycle, including:
        - Parameter validation
        - Event notifications (before_execution, after_execution, etc.)
        - Error handling
        - Cost tracking
        
        Args:
            parameters: The parameters for the capability execution
            
        Returns:
            The result of executing the capability
            
        Raises:
            ValidationError: If the parameters are invalid
            ExecutionError: If execution fails
        """
        ...
    
    async def execute_with_streaming(self, parameters: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute the capability with streaming results.
        
        This allows for incremental updates during long-running operations.
        
        Args:
            parameters: The parameters for the capability execution
            
        Yields:
            Dictionaries with partial results as they become available
        """
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol defining the interface for agents.
    
    This protocol defines the methods and properties that all agents must
    implement to be usable in the system.
    """
    
    @property
    def agent_id(self) -> str:
        """
        Get the unique identifier for this agent.
        
        Returns:
            The unique identifier for this agent.
        """
        ...
        
    @property
    def name(self) -> str:
        """
        Get the human-readable name for this agent.
        
        Returns:
            The human-readable name for this agent.
        """
        ...
        
    @property
    def description(self) -> str:
        """
        Get a human-readable description of this agent.
        
        Returns:
            A string describing what this agent does.
        """
        ...
        
    def get_capabilities(self) -> List[CapabilityProtocol]:
        """
        Get the capabilities provided by this agent.
        
        Returns:
            A list of capabilities that this agent provides.
        """
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
        
    def health_check(self) -> bool:
        """
        Perform a health check on the agent.
        
        Returns:
            True if the agent is healthy, False otherwise.
        """
        ...
        
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        
        This is useful for clearing any accumulated state or errors.
        """
        ...
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process a message and return a response.
        
        This is the main entry point for agent communication. Messages can include
        capability invocations, queries, or other requests.
        
        Args:
            message: The message to process.
            
        Returns:
            The agent's response to the message.
        """
        ...
        
    def record_operation_cost(self, cost: float, operation: str, context: Any = None) -> None:
        """
        Record the cost of an operation.
        
        This method is used to track costs associated with agent operations,
        particularly for model usage and external API calls.
        
        Args:
            cost: The cost of the operation.
            operation: The name or type of operation.
            context: Additional context about the operation.
        """
        ...
        
    def get_total_cost(self) -> float:
        """
        Get the total cost incurred by this agent.
        
        Returns:
            The total cost incurred by this agent across all operations.
        """
        ...
        
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """
        Get a detailed breakdown of costs.
        
        Returns:
            A dictionary containing detailed cost information.
        """
        ...
        
    def register_event_handler(self, event_type: str, handler_func: Callable[..., Any]) -> None:
        """
        Register a handler for a specific event type.
        
        This allows external systems to be notified when specific events
        occur within the agent.
        
        Args:
            event_type: The type of event to register for.
            handler_func: A function to call when events of this type occur.
        """
        ...
        
    def emit_event(self, event_type: str, event_data: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        This is used to notify external systems of events that occur within
        the agent.
        
        Args:
            event_type: The type of event that occurred.
            event_data: Data associated with the event.
        """
        ...


class BaseAgent(ABC):
    """
    Abstract base class implementing the AgentProtocol.
    
    This class provides a foundation for building agents with common
    functionality implemented, allowing derived classes to focus on
    implementing their specific capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        capabilities: Optional[List[CapabilityProtocol]] = None,
    ) -> None:
        """
        Initialize a new agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for this agent
            description: Description of this agent's purpose
            capabilities: List of capabilities this agent provides
        """
        self._agent_id = agent_id
        self._name = name
        self._description = description
        self._capabilities = capabilities or []
        self._event_handlers: Dict[str, List[Callable[..., Any]]] = {}
        self._cost_tracker: Dict[str, Any] = {
            "total": 0.0,
            "operations": []
        }
        
    @property
    def agent_id(self) -> str:
        """Get the unique identifier for this agent."""
        return self._agent_id
        
    @property
    def name(self) -> str:
        """Get the human-readable name of this agent."""
        return self._name
        
    @property
    def description(self) -> str:
        """Get a human-readable description of what this agent does."""
        return self._description
        
    def get_capabilities(self) -> List[CapabilityProtocol]:
        """
        Get all capabilities supported by this agent.
        
        Returns:
            A list of capabilities supported by this agent.
        """
        return self._capabilities
        
    def initialize(self) -> None:
        """
        Initialize the agent with any required setup.
        
        This method is called when the agent is first created or registered
        with the system. It should perform any necessary setup operations.
        
        Subclasses should override this method to perform their specific
        initialization.
        """
        self.emit_event("agent_initialized", {"agent_id": self.agent_id})
        
    def shutdown(self) -> None:
        """
        Shutdown the agent and perform cleanup.
        
        This method is called when the agent is being removed from the system
        or when the system is shutting down. It should perform any necessary
        cleanup operations.
        
        Subclasses should override this method to perform their specific
        cleanup.
        """
        self.emit_event("agent_shutdown", {"agent_id": self.agent_id})
        
    def health_check(self) -> bool:
        """
        Perform a health check on the agent.
        
        Returns:
            True if the agent is healthy, False otherwise.
        """
        return True  # Default implementation assumes a healthy agent
        
    def reset(self) -> None:
        """
        Reset the agent to its initial state.
        
        This is useful for clearing any accumulated state or errors.
        """
        self._cost_tracker = {
            "total": 0.0,
            "operations": []
        }
        self.emit_event("agent_reset", {"agent_id": self.agent_id})
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """
        Process a message and return a response.
        
        This is the main entry point for agent communication. Messages can include
        capability invocations, queries, or other requests.
        
        Args:
            message: The message to process.
            
        Returns:
            The agent's response to the message.
        """
        pass
        
    def record_operation_cost(self, cost: float, operation: str, context: Any = None) -> None:
        """
        Record the cost of an operation.
        
        This method is used to track costs associated with agent operations,
        particularly for model usage and external API calls.
        
        Args:
            cost: The cost of the operation.
            operation: The name or type of operation.
            context: Additional context about the operation.
        """
        import datetime
        
        self._cost_tracker["total"] += cost
        self._cost_tracker["operations"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "operation": operation,
            "cost": cost,
            "context": context
        })
        
        # Emit cost event for monitoring
        self.emit_event("cost_incurred", {
            "agent_id": self.agent_id,
            "operation": operation,
            "cost": cost
        })
        
    def get_total_cost(self) -> float:
        """
        Get the total cost incurred by this agent.
        
        Returns:
            The total cost incurred by this agent across all operations.
        """
        return float(self._cost_tracker["total"])
        
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """
        Get a detailed breakdown of costs.
        
        Returns:
            A dictionary containing detailed cost information.
        """
        return self._cost_tracker
        
    def register_event_handler(self, event_type: str, handler_func: Callable[..., Any]) -> None:
        """
        Register a handler for a specific event type.
        
        This allows external systems to be notified when specific events
        occur within the agent.
        
        Args:
            event_type: The type of event to register for.
            handler_func: A function to call when events of this type occur.
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler_func)
        
    def emit_event(self, event_type: str, event_data: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        This is used to notify external systems of events that occur within
        the agent.
        
        Args:
            event_type: The type of event that occurred.
            event_data: Data associated with the event.
        """
        # Add standard fields to all events
        if isinstance(event_data, dict):
            if "agent_id" not in event_data:
                event_data["agent_id"] = self.agent_id
            if "timestamp" not in event_data:
                import datetime
                event_data["timestamp"] = datetime.datetime.now().isoformat()
                
        # Call all registered handlers for this event type
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    # Log error but don't crash the agent
                    import logging
                    logging.error(f"Error in event handler for '{event_type}': {str(e)}")
                    
    def _find_capability(self, capability_name: str) -> Optional[CapabilityProtocol]:
        """
        Find a capability by name.
        
        Args:
            capability_name: The name of the capability to find
            
        Returns:
            The capability if found, otherwise None
        """
        for capability in self._capabilities:
            if capability.name == capability_name:
                return capability
        return None

    def request_approval(self, capability_name: str, parameters: Dict[str, Any]) -> ApprovalStatus:
        """
        Request approval to execute a capability.
        
        This method is called before executing capabilities that require
        human approval. It should implement the appropriate approval workflow
        based on the agent's configuration.
        
        Args:
            capability_name: The name of the capability to execute
            parameters: The parameters to pass to the capability
            
        Returns:
            The approval status (APPROVED, REJECTED, or PENDING)
        """
        # In the base implementation, we'll just log the request
        # and return a default response (implementations should override this)
        logger.info(f"Approval requested for capability: {capability_name}")
        logger.info(f"Parameters: {parameters}")
        
        # For now, just return PENDING
        return ApprovalStatus.PENDING
        
    async def execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> CapabilityResult:
        """
        Execute a capability with the given parameters.
        
        This method finds the specified capability and executes it with the given
        parameters, handling approval if required.
        
        Args:
            capability_name: The name of the capability to execute.
            parameters: The parameters to pass to the capability.
            
        Returns:
            The result of executing the capability.
            
        Raises:
            CapabilityNotFoundError: If the capability is not found.
            ExecutionError: If the capability execution fails.
            ValidationError: If the parameters are invalid.
        """
        # Find the capability
        capability = self._find_capability(capability_name)
        if not capability:
            raise CapabilityNotFoundError(capability_name)
        
        # Check if approval is required
        if capability.requires_human_approval():
            # Request approval
            approval_status = self.request_approval(capability_name, parameters)
            
            # If not approved, return an error
            if approval_status != ApprovalStatus.APPROVED:
                return {
                    "status": "error",
                    "error": f"Capability execution not approved. Status: {approval_status.name}"
                }
        
        # Register event handlers for the capability
        capability.register_event_handler("before_execution", 
            lambda **event_data: self.emit_event(f"{capability_name}.before_execution", event_data))
        
        capability.register_event_handler("after_execution", 
            lambda **event_data: self.emit_event(f"{capability_name}.after_execution", event_data))
        
        capability.register_event_handler("validation_error", 
            lambda **event_data: self.emit_event(f"{capability_name}.validation_error", event_data))
        
        capability.register_event_handler("execution_error", 
            lambda **event_data: self.emit_event(f"{capability_name}.execution_error", event_data))
        
        # Execute the capability with lifecycle management
        try:
            result = await capability.execute_with_lifecycle(parameters)
            
            # Record any costs from the capability
            operation_cost = capability.get_operation_cost()
            if "total" in operation_cost and operation_cost["total"] > 0:
                self.record_operation_cost(
                    operation_cost["total"], 
                    f"capability.{capability_name}", 
                    {"parameters": parameters}
                )
                
            return result
        except Exception as e:
            # Handle any unexpected errors
            error_message = f"Error executing capability: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            
            return {
                "status": "error",
                "error": error_message
            }

class BaseCapability(ABC):
    """
    Base class for agent capabilities.
    
    This class provides a foundation for implementing capabilities with
    common functionality.
    """
    
    def __init__(self, requires_human_approval: bool = False) -> None:
        """
        Initialize a new capability.
        
        Args:
            requires_human_approval: Whether this capability requires human approval
                before execution.
        """
        self._requires_human_approval = requires_human_approval
        self._event_handlers: Dict[str, List[Callable[..., Any]]] = {}
        
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the unique name of the capability.
        
        Returns:
            The unique name of this capability.
        """
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get a human-readable description of what this capability does.
        
        Returns:
            A string describing what this capability does.
        """
        pass
        
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, ParameterSchema]:
        """
        Get the parameters required by this capability.
        
        Returns:
            A dictionary mapping parameter names to their schema definitions.
        """
        pass
        
    def requires_approval(self) -> bool:
        """
        Determine if this capability requires human approval before execution.
        
        Returns:
            True if human approval is required, False otherwise.
        """
        return self._requires_human_approval
        
    def register_event_handler(self, event_type: str, handler_func: Callable[..., Any]) -> None:
        """
        Register a handler for a specific event type.
        
        This allows external systems to be notified when specific events
        occur within the capability.
        
        Args:
            event_type: The type of event to register for.
            handler_func: A function to call when events of this type occur.
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler_func)
        
    def emit_event(self, event_type: str, event_data: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        This is used to notify external systems of events that occur within
        the capability.
        
        Args:
            event_type: The type of event that occurred.
            event_data: Data associated with the event.
        """
        # Add standard fields to all events
        if isinstance(event_data, dict):
            if "capability" not in event_data:
                event_data["capability"] = self.name
            if "timestamp" not in event_data:
                import datetime
                event_data["timestamp"] = datetime.datetime.now().isoformat()
                
        # Call all registered handlers for this event type
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    # Log error but don't crash the capability
                    import logging
                    logging.error(f"Error in event handler for '{event_type}': {str(e)}")
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate the parameters for this capability.
        
        This method checks that all required parameters are present and of the correct type.
        
        Args:
            parameters: The parameters to validate.
            
        Raises:
            ValidationError: If the parameters are invalid.
        """
        schema = self.parameters
        
        # Check for required parameters
        for param_name, param_schema in schema.items():
            if param_schema.get("required", False) and param_name not in parameters:
                raise ValidationError(
                    f"Missing required parameter: {param_name}",
                    self.name,
                    param_name
                )
                
        # Check parameter types
        for param_name, param_value in parameters.items():
            if param_name in schema:
                param_schema = schema[param_name]
                
                # Type checking
                if param_schema["type"] == "string" and not isinstance(param_value, str):
                    raise ValidationError(
                        f"Parameter {param_name} must be a string",
                        self.name,
                        param_name
                    )
                elif param_schema["type"] == "integer" and not isinstance(param_value, int):
                    raise ValidationError(
                        f"Parameter {param_name} must be an integer",
                        self.name,
                        param_name
                    )
                elif param_schema["type"] == "number" and not isinstance(param_value, (int, float)):
                    raise ValidationError(
                        f"Parameter {param_name} must be a number",
                        self.name,
                        param_name
                    )
                elif param_schema["type"] == "boolean" and not isinstance(param_value, bool):
                    raise ValidationError(
                        f"Parameter {param_name} must be a boolean",
                        self.name,
                        param_name
                    )
                elif param_schema["type"] == "object" and not isinstance(param_value, dict):
                    raise ValidationError(
                        f"Parameter {param_name} must be an object",
                        self.name,
                        param_name
                    )
                elif param_schema["type"] == "array" and not isinstance(param_value, list):
                    raise ValidationError(
                        f"Parameter {param_name} must be an array",
                        self.name,
                        param_name
                    )
                    
                # Enum validation
                if "enum" in param_schema and param_value not in param_schema["enum"]:
                    raise ValidationError(
                        f"Parameter {param_name} must be one of: {', '.join(map(str, param_schema['enum']))}",
                        self.name,
                        param_name
                    )
            else:
                # Unknown parameter - could warn but we'll allow it
                pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> CapabilityResult:
        """
        Execute this capability with the given parameters.
        
        Args:
            parameters: The parameters to use when executing this capability.
            
        Returns:
            The result of executing this capability.
            
        Raises:
            ValidationError: If the parameters fail validation.
            ExecutionError: If the execution fails.
        """
        pass

    async def execute_with_lifecycle(self, parameters: Dict[str, Any]) -> CapabilityResult:
        """
        Execute the capability with lifecycle event handling.
        
        This method handles the entire capability lifecycle, including:
        - Parameter validation
        - Event notifications (before_execution, after_execution, etc.)
        - Error handling
        - Cost tracking
        
        Args:
            parameters: The parameters for the capability execution
            
        Returns:
            The result of executing the capability
            
        Raises:
            ValidationError: If the parameters are invalid
            ExecutionError: If execution fails
        """
        try:
            # Validate parameters
            await self.validate_parameters(parameters)
            
            # Emit before_execution event
            self.emit_event("before_execution", {"parameters": parameters})
            
            # Execute the capability
            result = await self.execute(parameters)
            
            # Emit after_execution event
            self.emit_event("after_execution", {"result": result})
            
            return result
        except ValidationError as e:
            # Emit validation_error event
            self.emit_event("validation_error", {"error": str(e)})
            raise
        except ExecutionError as e:
            # Emit execution_error event
            self.emit_event("execution_error", {"error": str(e)})
            raise
        except Exception as e:
            # Emit execution_error event for unexpected errors
            self.emit_event("execution_error", {"error": str(e)})
            raise ExecutionError(str(e), self.name)
