"""
Message adapter layer for standardizing agent message handling.

This module provides adapters to handle conversion between different message formats
and ensure consistent handling of messages across the agent system. It implements the
adapter pattern to shield agent implementations from the details of message formats.
"""

import logging
import json
import uuid
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Union, TypedDict, cast, List, Type
from uuid import uuid4

from clubhouse.agents.agent_protocol import AgentMessage, AgentResponse
from clubhouse.agents.communication import (
    EnhancedAgentMessage, 
    RoutingStrategy, 
    MessageType, 
    MessagePriority
)

# Configure logger
logger = logging.getLogger(__name__)


class MessageParsingError(Exception):
    """Exception raised when a message cannot be parsed correctly."""
    pass


class ResponseCreationError(Exception):
    """Exception raised when a response cannot be created correctly."""
    pass


class StandardizedMessageContent(TypedDict, total=False):
    """Standard format for parsed message content."""
    command: str
    parameters: Dict[str, Any]
    message_id: str
    sender: str
    recipient: Optional[str]
    routing_strategy: Union[str, RoutingStrategy]
    original_message: Dict[str, Any]


class StandardizedResponse(TypedDict, total=False):
    """Standard format for response messages."""
    status: str
    result: Dict[str, Any]
    error: Optional[str]


class MessageAdapterProtocol(Protocol):
    """Protocol defining the interface for message adapters."""
    
    def parse_message(self, message: Any) -> StandardizedMessageContent:
        """
        Parse and standardize an incoming message.
        
        Args:
            message: The message to parse, which could be in various formats
            
        Returns:
            A standardized message content dictionary
            
        Raises:
            MessageParsingError: If the message cannot be parsed
        """
        ...
    
    def create_response(
        self, 
        original_message: StandardizedMessageContent,
        response_content: StandardizedResponse,
        sender: str
    ) -> Dict[str, Any]:
        """
        Create a standardized response to a message.
        
        Args:
            original_message: The parsed original message
            response_content: The content of the response
            sender: The ID of the agent sending the response
            
        Returns:
            A properly formatted response message
            
        Raises:
            ResponseCreationError: If the response cannot be created
        """
        ...


class StandardMessageAdapter(MessageAdapterProtocol):
    """
    Standard implementation of the MessageAdapterProtocol.
    
    This adapter handles standardized message format parsing and response creation
    to ensure consistent message handling across the agent system.
    """

    def parse_message(self, message: Any) -> StandardizedMessageContent:
        """Parse an incoming message into a standardized format.
        
        Args:
            message: The message to parse, can be a dict, AgentMessage, or EnhancedAgentMessage.
            
        Returns:
            StandardizedMessageContent: The parsed message in a standardized format.
            
        Raises:
            MessageParsingError: If the message cannot be parsed properly.
        """
        try:
            # Extract message fields based on message type
            if isinstance(message, dict):
                # Handle raw dictionary messages
                message_id = message.get("message_id", str(uuid4()))
                sender = message.get("sender", "unknown")
                recipient = message.get("recipient", "unknown")
                
                # Get routing information
                routing = message.get("routing", {})
                if isinstance(routing, dict) and "strategy" in routing:
                    routing_strategy = routing["strategy"]
                else:
                    routing_strategy = message.get("routing_strategy", "direct")
                
                # Extract command and parameters
                content = message.get("content", {})
                if not isinstance(content, dict):
                    content = {"raw_content": content}
                
                command = content.get("command", "unknown")
                parameters = {k: v for k, v in content.items() if k != "command"}
                
                # Store original message
                original_message = message
                
            elif isinstance(message, EnhancedAgentMessage):
                # Handle EnhancedAgentMessage objects - access as dictionary since it inherits from dict
                message_id = message["message_id"]
                sender = message["sender"]
                recipient = message["routing"].get("recipient") if "routing" in message else None
                
                # Get routing strategy - safely handle both enum and string values
                if "routing" in message and "strategy" in message["routing"]:
                    routing_strategy = message["routing"]["strategy"]
                else:
                    routing_strategy = "direct"
                
                # Extract command and parameters from content
                content = message["content"] if "content" in message else {"raw_content": str(message)}
                command = content.get("command", "unknown")
                parameters = {k: v for k, v in content.items() if k != "command"}
                
                # Store original message
                original_message = dict(message)  # Convert to standard dict to avoid any method resolution issues
                
            elif isinstance(message, AgentMessage):
                # Handle base AgentMessage objects
                message_id = getattr(message, "message_id", str(uuid4()))
                sender = getattr(message, "sender", "unknown")
                recipient = getattr(message, "recipient", "unknown")
                routing_strategy = "direct"  # Default for base AgentMessage
                
                # Extract command and parameters assuming content is a dict
                content = message.content if isinstance(message.content, dict) else {"raw_content": message.content}
                command = content.get("command", "unknown")
                parameters = {k: v for k, v in content.items() if k != "command"}
                
                # Store original message as a dict representation
                original_message = {
                    "message_id": message_id,
                    "sender": sender,
                    "recipient": recipient,
                    "content": content
                }
                
            else:
                # Handle unknown message types by creating a minimal standardized content
                message_id = str(uuid4())
                sender = "unknown"
                recipient = "unknown"
                routing_strategy = "direct"
                command = "unknown"
                parameters = {"raw_content": str(message)}
                original_message = {"raw_content": str(message)}
            
            return StandardizedMessageContent(
                command=command,
                parameters=parameters,
                message_id=message_id,
                sender=sender,
                recipient=recipient,
                routing_strategy=routing_strategy,
                original_message=original_message
            )
            
        except Exception as e:
            # Log the error and raise a MessageParsingError
            logging.error(f"Error parsing message: {e}")
            raise MessageParsingError(f"Failed to parse message: {e}")
    
    def create_response(
        self,
        original_message: StandardizedMessageContent,
        response_content: StandardizedResponse,
        sender: str
    ) -> Dict[str, Any]:
        """Create a standardized response message.
        
        Args:
            original_message: The original message that is being responded to.
            response_content: The content of the response.
            sender: The sender of the response.
            
        Returns:
            Dict[str, Any]: The response message in a standardized format.
        """
        # Create a response with appropriate routing information
        response = {
            "in_response_to": original_message["message_id"],
            "sender": sender,
            "status": response_content["status"],
            "result": response_content["result"],
            "routing": {
                "strategy": original_message["routing_strategy"],
                "recipient": original_message["sender"]
            }
        }
        
        return response


# Create a singleton instance
standard_message_adapter = StandardMessageAdapter()
"""Default instance of the StandardMessageAdapter for global use."""
