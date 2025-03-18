"""
Simple agent implementation for demonstration purposes.

This module provides a simple agent implementation that can be used
as a template for creating more complex agents.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from clubhouse.agents.base import BaseAgent, BaseAgentOutput
from clubhouse.agents.protocols import AgentCapability, AgentInput, AgentOutput, AgentOutputType, AgentState
from typing import cast, List, Dict, Any, Type

logger = logging.getLogger(__name__)


class SimpleAgent(BaseAgent):
    """
    A simple agent implementation for demonstration purposes.
    
    This agent can perform basic text processing operations and
    serves as an example of how to implement the agent interfaces.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[AgentCapability],
        version: str = "1.0.0",
        owner_id: Optional[UUID] = None,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a SimpleAgent.
        
        Args:
            name: Agent name
            description: Agent description
            capabilities: List of agent capabilities
            version: Agent version
            owner_id: Optional ID of the agent owner
            model_id: Optional model ID for AI-based agents
            tags: Optional list of tags for categorization
            custom_properties: Optional dictionary of custom properties
        """
        super().__init__(
            name=name,
            description=description,
            capabilities=capabilities,
            version=version,
            owner_id=owner_id,
            model_id=model_id,
            tags=tags,
            custom_properties=custom_properties
        )
        self._processed_count = 0
        self._history: List[Dict[str, Any]] = []
    
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process input data and return output.
        
        Args:
            input_data: Agent input to process
            
        Returns:
            Agent output with processing results
        """
        # Update last_active using the update_last_active method 
        self._update_last_active(datetime.now())
        self._processed_count += 1
        
        # Create a history entry for this request
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data.content,  # Use content instead of data
            "metadata": input_data.metadata  # Use metadata instead of context
        }
        
        # Check the operation to perform (from metadata)
        operation = input_data.metadata.get("operation", "echo").lower()
        input_text = str(input_data.content)  # Use content instead of data
        result: Any = None
        
        # Route to appropriate operation
        if operation == "echo":
            # Echo capability - just return the input as output
            result = self._echo(input_text)
        elif operation == "transform":
            # Text transform capability - various text transformations
            transform_type = input_data.metadata.get("transform_type", "uppercase")
            result = self._transform_text(input_text, transform_type)
        elif operation == "analyze":
            # Text analysis capability
            analysis_type = input_data.metadata.get("analysis_type", "length")
            result = self._analyze_text(input_text, analysis_type)
        else:
            # Unknown operation
            return self._create_error_output(f"Unknown operation: {operation}")
        
        # Store result in history
        history_entry["result"] = result
        self._history.append(history_entry)
        
        # Create and return output using helper method
        return self._create_success_output(result)
    
    def _echo(self, text: str) -> str:
        """Simple echo operation - returns the input text unchanged."""
        return text
        
    def _transform_text(self, text: str, transform_type: str) -> str:
        """
        Transform text according to specified transformation.
        
        Args:
            text: Text to transform
            transform_type: Type of transformation to apply
            
        Returns:
            Transformed text
        """
        if transform_type == "uppercase":
            return text.upper()
        elif transform_type == "lowercase":
            return text.lower()
        elif transform_type == "capitalize":
            return text.capitalize()
        elif transform_type == "reverse":
            return text[::-1]
        else:
            return f"Unknown transform type: {transform_type}"
    
    def _analyze_text(self, text: str, analysis_type: str) -> str:
        """
        Analyze text according to specified analysis type.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis to apply
            
        Returns:
            Analysis result
        """
        if analysis_type == "length":
            return str(len(text))
        elif analysis_type == "word_count":
            return str(len(text.split()))
        else:
            return f"Unknown analysis type: {analysis_type}"
    
    def _create_success_output(self, data: Any) -> AgentOutput:
        """Create a successful output response."""
        # Create a proper AgentOutput implementation
        return BaseAgentOutput(
            output_type=AgentOutputType.TEXT,
            content=data,
            metadata={"success": True}
        )
        
    def _create_error_output(self, error_message: str) -> AgentOutput:
        """Create an error output response."""
        # Create a proper AgentOutput implementation
        return BaseAgentOutput(
            output_type=AgentOutputType.TEXT,
            content={"error": error_message},
            metadata={"success": False}
        )
        
    def start(self) -> bool:
        """
        Start the agent.
        
        Returns:
            True if started successfully, False otherwise
        """
        # Use the update_state method instead of directly assigning to state
        self._update_state(AgentState.ACTIVE)
        logger.info(f"Agent {self.metadata.name} started")
        return True
        
    def stop(self) -> bool:
        """
        Stop the agent.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        # Use the update_state method instead of directly assigning to state
        self._update_state(AgentState.PAUSED)
        logger.info(f"Agent {self.metadata.name} stopped")
        return True
        
    # Helper methods for updating read-only properties
    def _update_last_active(self, timestamp: datetime) -> None:
        """Update the last_active timestamp through the underlying storage."""
        self._metadata._last_active = timestamp
        
    def _update_state(self, state: AgentState) -> None:
        """Update the agent state through the underlying storage."""
        self._metadata._state = state
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the agent's current state.
        
        Returns:
            Dictionary representing agent state
        """
        return {
            "message_history": self._history,
            "processed_count": self._processed_count
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the agent's current state.
        
        Args:
            state: Dictionary containing state to restore
        """
        if "message_history" in state:
            self._history = state["message_history"]
        
        if "processed_count" in state:
            self._processed_count = state["processed_count"]
    
    def initialize(self) -> None:
        """
        Initialize the agent.
        
        This method is called when the agent is first created or
        loaded from storage. It should set up any resources needed
        by the agent.
        """
        logger.info(f"Initializing SimpleAgent: {self.metadata.name} ({self.metadata.id})")
        self._update_state(AgentState.READY)
    
    def shutdown(self) -> None:
        """
        Shutdown the agent.
        
        It should clean up any resources used by the agent.
        """
        logger.info(f"Shutting down SimpleAgent: {self.metadata.name} ({self.metadata.id})")
        self._update_state(AgentState.TERMINATED)
