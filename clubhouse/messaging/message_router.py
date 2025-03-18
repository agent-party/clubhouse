"""
Message routing for the clubhouse.

This module provides a router for handling incoming messages from Kafka
based on their message type, dispatching them to the appropriate handlers.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, Type, cast

from clubhouse.core.service_registry import ServiceRegistry
from scripts.kafka_cli.message_schemas import (
    BaseMessage,
    MessageType,
    CreateAgentCommand,
    DeleteAgentCommand,
    ProcessMessageCommand,
)

logger = logging.getLogger(__name__)


class MessageHandlerProtocol(Protocol):
    """Protocol for message handlers."""

    def can_handle(self, message_type: MessageType) -> bool:
        """
        Determine if this handler can handle the given message type.

        Args:
            message_type: The type of message to check

        Returns:
            True if this handler can handle the message type, False otherwise
        """
        ...

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the message.

        Args:
            message: The message to handle

        Returns:
            The response message to send back
        """
        ...


class MessageRouter:
    """
    Routes messages to the appropriate handler based on message type.
    
    This class maintains a registry of message handlers and routes incoming
    messages to the appropriate handler based on their message_type field.
    """

    def __init__(self, service_registry: ServiceRegistry) -> None:
        """
        Initialize the message router.

        Args:
            service_registry: The service registry to use for dependency resolution
        """
        self._service_registry = service_registry
        self._handlers: List[MessageHandlerProtocol] = []

    def register_handler(self, handler: MessageHandlerProtocol) -> None:
        """
        Register a message handler.

        Args:
            handler: The handler to register
        """
        self._handlers.append(handler)
        logger.info(f"Registered message handler: {handler.__class__.__name__}")

    def route_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Route a message to the appropriate handler.

        Args:
            message: The message to route

        Returns:
            The response message from the handler, or None if no handler was found

        Raises:
            ValueError: If the message does not have a valid message_type field
        """
        if "message_type" not in message:
            logger.error("Message missing message_type field")
            raise ValueError("Message missing message_type field")

        message_type_str = message["message_type"]
        
        # Normalize message type if needed - standardize on enum values
        if message_type_str not in [e.value for e in MessageType]:
            # Log error for unexpected format and suggest proper format
            logger.error(f"Invalid message_type format: {message_type_str}. "
                         f"Expected one of: {[e.value for e in MessageType]}")
            raise ValueError(f"Invalid message_type: {message_type_str}. "
                            f"Message types must be one of the MessageType enum values.")
        
        # Convert to enum
        message_type = MessageType(message_type_str)
        
        # Find handler
        for handler in self._handlers:
            if handler.can_handle(message_type):
                logger.debug(f"Routing message of type {message_type} to {handler.__class__.__name__}")
                return handler.handle(message)
        
        logger.warning(f"No handler found for message type: {message_type}")
        return None
