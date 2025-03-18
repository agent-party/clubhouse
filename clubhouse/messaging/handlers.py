"""
Handlers for processing messages in the clubhouse.

This module contains handlers for different types of messages
received from the Kafka-based clients.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from clubhouse.agents.factory import AgentFactory
from clubhouse.core.service_registry import ServiceRegistry
from scripts.kafka_cli.message_schemas import (
    AgentCreatedResponse,
    AgentDeletedResponse,
    AgentErrorEvent,
    AgentStateChangedEvent,
    AgentThinkingEvent,
    BaseMessage,
    CreateAgentCommand,
    DeleteAgentCommand,
    MessageProcessedResponse,
    MessageType,
    ProcessMessageCommand
)

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Base class for message handlers."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        """
        Initialize the handler with access to services.

        Args:
            service_registry: Registry for accessing required services
        """
        self._service_registry = service_registry
        self.handled_types: List[MessageType] = []

    def can_handle(self, message_type: Any) -> bool:
        """
        Determine if this handler can handle the given message type.

        Args:
            message_type: Type of message to handle (can be string or MessageType enum)

        Returns:
            True if this handler can handle the message type, False otherwise
        """
        # Handle the case where message_type is already an enum
        if isinstance(message_type, MessageType):
            return message_type in self.handled_types
        
        # Handle the case where message_type is a string
        return message_type in [msg_type.value for msg_type in self.handled_types]

    @abstractmethod
    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle the message and return a response.

        Args:
            message: Message to handle

        Returns:
            Response message
        """
        pass


class CreateAgentHandler(BaseHandler):
    """Handler for creating agents."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        """
        Initialize the handler.

        Args:
            service_registry: Registry for accessing required services
        """
        super().__init__(service_registry)
        self.handled_types = [MessageType.COMMAND_CREATE_AGENT]
        self._agent_manager = self._service_registry.get("agent_manager")
        self._event_publisher = self._service_registry.get("event_publisher")

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a create agent command.

        Args:
            message: The command message

        Returns:
            Response message
        """
        try:
            # Parse command
            logger.debug(f"CreateAgentHandler received message: {message}")
            logger.debug(f"Creating CreateAgentCommand from message")
            
            # Log the agent manager instance
            logger.debug(f"Agent manager: {self._agent_manager}")
            
            command = CreateAgentCommand(**message)
            logger.debug(f"Command parsed successfully: {command.model_dump()}")
            
            # Create the agent
            logger.info(f"Creating agent with ID: {command.agent_id}")
            agent = self._agent_manager.create_agent(
                agent_id=command.agent_id,
                personality_type=command.personality_type,
                metadata=command.metadata
            )
            logger.debug(f"Agent created successfully: {agent.agent_id()}")
            
            # Create response
            response_data = AgentCreatedResponse(
                message_id=str(uuid.uuid4()),
                agent_id=agent.agent_id(),
                agent_name=agent.name(),
                agent_role=agent.description(),
                personality_type=command.personality_type
            ).model_dump()
            
            # Add agent_description for test compatibility
            response_data["agent_description"] = agent.description()
            
            # Create agent state changed event
            event_data = {
                "message_id": str(uuid.uuid4()),
                "message_type": MessageType.EVENT_AGENT_STATE_CHANGED.value,
                "agent_id": agent.agent_id(),
                "current_state": "ready",
                "previous_state": None,
                "state": "ready"  # This field is specifically checked in tests
            }
            
            # Publish event
            if self._event_publisher:
                self._event_publisher.publish_event(event_data)
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}", exc_info=True)
            # Re-raise the exception to be handled by the caller
            raise


class DeleteAgentHandler(BaseHandler):
    """Handler for deleting agents."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        """
        Initialize the handler.

        Args:
            service_registry: Registry for accessing required services
        """
        super().__init__(service_registry)
        self.handled_types = [MessageType.COMMAND_DELETE_AGENT]
        self._agent_manager = self._service_registry.get("agent_manager")
        self._event_publisher = self._service_registry.get("event_publisher")

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a delete agent command.

        Args:
            message: The command message

        Returns:
            Response message
        """
        try:
            # Parse command
            command = DeleteAgentCommand(**message)
            
            # Delete the agent
            logger.info(f"Deleting agent with ID: {command.agent_id}")
            self._agent_manager.delete_agent(command.agent_id)
            
            # Create response
            response = AgentDeletedResponse(
                message_id=str(uuid.uuid4()),
                agent_id=command.agent_id
            )
            
            # Create state changed event to reflect deletion
            event_data = {
                "message_id": str(uuid.uuid4()),
                "message_type": MessageType.EVENT_AGENT_STATE_CHANGED.value,
                "agent_id": command.agent_id,
                "current_state": "deleted",
                "previous_state": "ready",
                "state": "deleted"  # This field is specifically checked in tests
            }
            
            # Publish event
            if self._event_publisher:
                self._event_publisher.publish_event(event_data)
            
            return response.model_dump()
            
        except ValueError as e:
            # Re-raise the exception to be handled by the caller
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting agent: {e}", exc_info=True)
            # Re-raise the exception to be handled by the caller
            raise


class ProcessMessageHandler(BaseHandler):
    """Handler for processing messages."""

    def __init__(self, service_registry: ServiceRegistry) -> None:
        """
        Initialize the handler.

        Args:
            service_registry: Registry for accessing required services
        """
        super().__init__(service_registry)
        self.handled_types = [MessageType.COMMAND_PROCESS_MESSAGE]
        self._agent_manager = self._service_registry.get("agent_manager")
        self._conversation_manager = self._service_registry.get("conversation_manager")
        self._event_publisher = self._service_registry.get("event_publisher")

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a process message command.

        Args:
            message: The command message

        Returns:
            Response message
        """
        try:
            # Parse command
            command = ProcessMessageCommand(**message)
            
            # Get the agent
            logger.info(f"Processing message for agent: {command.agent_id}")
            agent = self._agent_manager.get_agent(command.agent_id)
            
            if not agent:
                raise ValueError(f"Agent not found: {command.agent_id}")
            
            # Get or create conversation
            conversation = None
            conversation_id = command.conversation_id
            
            if conversation_id:
                conversation = self._conversation_manager.get_conversation(conversation_id)
                if not conversation:
                    logger.info(f"Conversation not found, creating new: {conversation_id}")
                    conversation = self._conversation_manager.create_conversation(
                        conversation_id=conversation_id,
                        agent_id=command.agent_id
                    )
            else:
                # Create a new conversation if one wasn't specified
                conversation = self._conversation_manager.create_conversation(
                    agent_id=command.agent_id
                )
                # For test compatibility, use the conversation ID from the returned conversation
                conversation_id = conversation.id if hasattr(conversation, 'id') else getattr(conversation, 'conversation_id', None)
            
            # Create thinking event
            thinking_event = AgentThinkingEvent(
                message_id=str(uuid.uuid4()),
                agent_id=command.agent_id
            )
            
            # Publish event
            if self._event_publisher:
                self._event_publisher.publish_event(thinking_event.model_dump())
            
            # Prepare message for agent processing
            agent_message = {
                "content": command.content,
                "conversation_id": conversation_id
            }
            
            # Process the message
            response_data = agent.process_message(agent_message)
            
            # For test compatibility, use the conversation ID from response_data if available
            # or fallback to the one we already determined
            if "conversation_id" in response_data:
                conversation_id = response_data["conversation_id"]
            
            # Create response
            response = MessageProcessedResponse(
                message_id=str(uuid.uuid4()),
                agent_id=command.agent_id,
                content=response_data.get("content", ""),
                conversation_id=conversation_id,
                metadata=response_data.get("metadata", {})
            )
            
            return response.model_dump()
            
        except ValueError as e:
            # Re-raise the exception to be handled by the caller
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing message: {e}", exc_info=True)
            # Re-raise the exception to be handled by the caller
            raise
