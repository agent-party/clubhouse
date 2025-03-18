"""
Simple agent implementation for the Kafka CLI integration.

This module provides a simplified agent implementation that's tailored
for use with the Kafka CLI, focusing on basic message processing
without requiring complex persistence or capabilities.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from clubhouse.agents.agent_protocol import AgentProtocol

logger = logging.getLogger(__name__)


class SimpleAgent(AgentProtocol):
    """
    Simple agent implementation for Kafka CLI integration.
    
    This agent provides basic message processing functionality
    without requiring complex persistence or capabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        personality_type: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        capabilities: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize a simple agent.
        
        Args:
            agent_id: Unique identifier for the agent
            personality_type: Type of personality (e.g., "assistant", "researcher")
            name: Optional display name for the agent
            description: Optional description of the agent
            metadata: Optional additional metadata
            capabilities: Optional list of agent capabilities
            **kwargs: Additional arguments for extensibility
        """
        self._agent_id = agent_id
        self._personality_type = personality_type
        self._name = name or f"{personality_type.capitalize()} Agent"
        self._description = description or f"A {personality_type} agent that can answer questions and assist users."
        self._metadata = metadata or {}
        self._capabilities = capabilities or []
        self._created_at = datetime.now(timezone.utc)
        self._last_active = None
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self._is_initialized = False
    
    def agent_id(self) -> str:
        """Get the agent's ID."""
        return self._agent_id
    
    def name(self) -> str:
        """Get the agent's name."""
        return self._name
    
    def description(self) -> str:
        """Get the agent's description."""
        return self._description
    
    def personality_type(self) -> str:
        """Get the agent's personality type."""
        return self._personality_type
    
    def initialize(self) -> None:
        """Initialize the agent."""
        if not self._is_initialized:
            logger.info(f"Initializing agent: {self._name} ({self._agent_id})")
            self._is_initialized = True
    
    def shutdown(self) -> None:
        """Shut down the agent and clean up resources."""
        logger.info(f"Shutting down agent: {self._name} ({self._agent_id})")
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from a user.
        
        Args:
            message: The message to process, containing at least:
                - content: The content of the message
                - conversation_id: ID of the conversation
                
        Returns:
            Response message containing at least:
                - content: The response content
                - conversation_id: ID of the conversation
        """
        # Update last active timestamp
        self._last_active = datetime.now(timezone.utc)
        
        # Get conversation ID or generate one if not provided
        conversation_id = message.get("conversation_id", str(uuid.uuid4()))
        
        # Add message to conversation history
        if conversation_id not in self._conversation_history:
            self._conversation_history[conversation_id] = []
        
        self._conversation_history[conversation_id].append({
            "role": "user",
            "content": message["content"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Generate a simple response based on personality type
        content = self._generate_response(message["content"], conversation_id)
        
        # Add response to conversation history
        self._conversation_history[conversation_id].append({
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create response
        response = {
            "content": content,
            "conversation_id": conversation_id,
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {}
        }
        
        return response
    
    def _generate_response(self, user_message: str, conversation_id: str) -> str:
        """
        Generate a response based on the agent's personality type.
        
        Args:
            user_message: The user's message
            conversation_id: ID of the conversation
            
        Returns:
            Generated response
        """
        # Simple responses based on personality type
        if self._personality_type == "assistant":
            return f"I'm your helpful assistant. Regarding '{user_message}', I would be happy to assist you with that."
        
        elif self._personality_type == "researcher":
            return f"As a researcher, I find your query about '{user_message}' fascinating. Here's what I would investigate..."
        
        elif self._personality_type == "teacher":
            return f"Let me explain '{user_message}' in a way that's easy to understand..."
        
        else:
            return f"I received your message: '{user_message}'. How can I help you further?"
    
    def get_conversation_history(
        self, 
        conversation_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Optional limit on the number of messages to return
            
        Returns:
            List of messages in the conversation
        """
        history = self._conversation_history.get(conversation_id, [])
        
        if limit is not None and limit > 0:
            return history[-limit:]
        
        return history
