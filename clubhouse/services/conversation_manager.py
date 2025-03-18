"""
Conversation Manager Service for the clubhouse.

This module provides a service for managing conversations, including
tracking conversation history, context, and state.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Protocol, ClassVar, Set

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.event_publisher import EventPublisherProtocol
from clubhouse.schemas.events.agent_interaction import (
    ConversationCreatedEvent,
    MessageAddedEvent,
    ConversationDeletedEvent
)

logger = logging.getLogger(__name__)


class Message:
    """Represents a message in a conversation."""
    
    def __init__(
        self,
        content: str,
        sender: str,
        timestamp: Optional[datetime] = None,
        message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a message.
        
        Args:
            content: The content of the message
            sender: The sender of the message (user ID or agent ID)
            timestamp: When the message was sent (defaults to now)
            message_id: Unique ID for the message (defaults to a new UUID)
            metadata: Additional metadata for the message
        """
        self.content = content
        self.sender = sender
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.message_id = message_id or str(uuid.uuid4())
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "message_id": self.message_id,
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create a message from a dictionary.
        
        Args:
            data: Dictionary representation of a message
            
        Returns:
            Message instance
        """
        timestamp = None
        if "timestamp" in data:
            timestamp = datetime.fromisoformat(data["timestamp"])
            
        return cls(
            content=data["content"],
            sender=data["sender"],
            message_id=data.get("message_id"),
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


class Conversation:
    """Represents a conversation with history and context."""
    
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a conversation.
        
        Args:
            conversation_id: Unique ID for the conversation (defaults to a new UUID)
            title: Title for the conversation
            metadata: Additional metadata for the conversation
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.title = title or f"Conversation {self.conversation_id}"
        self.metadata = metadata or {}
        self.messages: List[Message] = []
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        # Track context for this conversation
        self.context: Dict[str, Any] = {}
        # Set of referenced entities (agents, tasks, etc.)
        self.references: Set[str] = set()
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the conversation.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        
        # Extract any references from metadata (for entity linking)
        if message.metadata and "references" in message.metadata:
            for ref in message.metadata["references"]:
                self.references.add(ref)
    
    def get_messages(self) -> List[Message]:
        """
        Get all messages in the conversation.
        
        Returns:
            List of messages in chronological order
        """
        return sorted(self.messages, key=lambda m: m.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "metadata": self.metadata,
            "messages": [message.to_dict() for message in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "context": self.context,
            "references": list(self.references)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """
        Create a conversation from a dictionary.
        
        Args:
            data: Dictionary representation of a conversation
            
        Returns:
            Conversation instance
        """
        conversation = cls(
            conversation_id=data.get("conversation_id"),
            title=data.get("title"),
            metadata=data.get("metadata", {})
        )
        
        if "created_at" in data:
            conversation.created_at = datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            conversation.updated_at = datetime.fromisoformat(data["updated_at"])
        
        if "messages" in data:
            for message_data in data["messages"]:
                conversation.messages.append(Message.from_dict(message_data))
        
        # Add context and references if present
        if "context" in data:
            conversation.context = data["context"]
            
        if "references" in data:
            conversation.references = set(data["references"])
            
        return conversation


class ConversationManagerProtocol(Protocol):
    """Protocol for conversation manager service."""
    
    def create_conversation(
        self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Optional title for the conversation
            metadata: Optional metadata for the conversation
            
        Returns:
            The created conversation
        """
        ...
    
    def get_conversation(self, conversation_id: str) -> Conversation:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            The requested conversation
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        ...
    
    def add_message(
        self, conversation_id: str, content: str, sender: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation to add to
            content: Content of the message
            sender: Sender of the message
            metadata: Optional metadata for the message
            
        Returns:
            The created message
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        ...
    
    def list_conversations(self) -> List[Conversation]:
        """
        List all conversations.
        
        Returns:
            List of all conversations
        """
        ...
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        ...


class ConversationManager(ConversationManagerProtocol):
    """
    Service for managing conversations.
    
    This service is responsible for creating, retrieving, and tracking
    conversations and their messages.
    """
    
    # Topic names for conversation events
    CONVERSATION_CREATED_TOPIC: ClassVar[str] = "conversation.created"
    MESSAGE_ADDED_TOPIC: ClassVar[str] = "conversation.message.added"
    CONVERSATION_DELETED_TOPIC: ClassVar[str] = "conversation.deleted"
    
    def __init__(self) -> None:
        """Initialize the conversation manager."""
        self._conversations: Dict[str, Conversation] = {}
        self._event_publisher: Optional[EventPublisherProtocol] = None
        
        # Try to get event publisher from service registry if available
        try:
            registry = ServiceRegistry()
            self._event_publisher = registry.get_protocol(EventPublisherProtocol)
        except Exception as e:
            logger.warning(f"Could not get event publisher: {e}")
    
    def create_conversation(
        self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Optional title for the conversation
            metadata: Optional metadata for the conversation
            
        Returns:
            The created conversation
        """
        conversation = Conversation(title=title, metadata=metadata)
        self._conversations[conversation.conversation_id] = conversation
        logger.info(f"Created conversation with ID {conversation.conversation_id}")
        
        # Publish event if publisher is available
        if self._event_publisher:
            try:
                event = ConversationCreatedEvent(
                    conversation_id=conversation.conversation_id,
                    title=conversation.title,
                    metadata=conversation.metadata
                )
                self._event_publisher.publish_event(
                    event.model_dump(),
                    self.CONVERSATION_CREATED_TOPIC
                )
            except Exception as e:
                logger.error(f"Failed to publish conversation created event: {e}")
        
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Conversation:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            The requested conversation
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        if conversation_id not in self._conversations:
            raise ValueError(f"No conversation found with ID {conversation_id}")
        
        return self._conversations[conversation_id]
    
    def add_message(
        self, conversation_id: str, content: str, sender: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: ID of the conversation to add to
            content: Content of the message
            sender: Sender of the message
            metadata: Optional metadata for the message
            
        Returns:
            The created message
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        # Get the conversation
        conversation = self.get_conversation(conversation_id)
        
        # Create the message
        message = Message(content=content, sender=sender, metadata=metadata)
        
        # Add the message to the conversation
        conversation.add_message(message)
        
        logger.debug(f"Added message {message.message_id} to conversation {conversation_id}")
        
        # Publish event if publisher is available
        if self._event_publisher:
            try:
                event = MessageAddedEvent(
                    conversation_id=conversation_id,
                    message_id=message.message_id,
                    content=content,
                    sender=sender,
                    metadata=message.metadata or {}
                )
                self._event_publisher.publish_event(
                    event.model_dump(),
                    self.MESSAGE_ADDED_TOPIC
                )
            except Exception as e:
                logger.error(f"Failed to publish message added event: {e}")
        
        return message
    
    def list_conversations(self) -> List[Conversation]:
        """
        List all conversations.
        
        Returns:
            List of all conversations
        """
        return list(self._conversations.values())
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        if conversation_id not in self._conversations:
            raise ValueError(f"No conversation found with ID {conversation_id}")
        
        # Get conversation before deleting for event
        conversation = self._conversations[conversation_id]
        
        del self._conversations[conversation_id]
        logger.info(f"Deleted conversation with ID {conversation_id}")
        
        # Publish event if publisher is available
        if self._event_publisher:
            try:
                event = ConversationDeletedEvent(
                    conversation_id=conversation_id,
                    metadata=conversation.metadata
                )
                self._event_publisher.publish_event(
                    event.model_dump(),
                    self.CONVERSATION_DELETED_TOPIC
                )
            except Exception as e:
                logger.error(f"Failed to publish conversation deleted event: {e}")
        
    def get_conversation_history(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get the message history for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Optional maximum number of messages to return (newest first)
            
        Returns:
            List of messages in the conversation
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        conversation = self.get_conversation(conversation_id)
        messages = conversation.get_messages()
        
        if limit is not None:
            messages = messages[-limit:]
            
        return messages

    def update_context(
        self, conversation_id: str, context_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the context for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            context_updates: Dictionary of context updates to apply
            
        Returns:
            The updated context dictionary
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        conversation = self.get_conversation(conversation_id)
        
        # Validate context updates
        if not isinstance(context_updates, dict):
            raise ValueError("Context updates must be a dictionary")
            
        # Validate values in context (could be expanded with more specific validation)
        for key, value in context_updates.items():
            if not isinstance(key, str):
                raise ValueError(f"Context keys must be strings, got {type(key)}")
        
        # Apply updates to context
        conversation.context.update(context_updates)
        conversation.updated_at = datetime.now(timezone.utc)
        
        logger.debug(f"Updated context for conversation {conversation_id}")
        
        # No event for context updates currently, but could be added
        # if tracking context changes becomes important
        
        return conversation.context

    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get the context for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            The context dictionary
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        conversation = self.get_conversation(conversation_id)
        return conversation.context
    
    def add_reference(
        self, conversation_id: str, reference_id: str
    ) -> Set[str]:
        """
        Add a reference to a conversation.
        
        Args:
            conversation_id: ID of the conversation
            reference_id: ID of the entity being referenced
            
        Returns:
            The updated set of references
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        conversation = self.get_conversation(conversation_id)
        conversation.references.add(reference_id)
        conversation.updated_at = datetime.now(timezone.utc)
        
        logger.debug(f"Added reference {reference_id} to conversation {conversation_id}")
        return conversation.references
    
    def get_references(self, conversation_id: str) -> Set[str]:
        """
        Get all references for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Set of reference IDs
            
        Raises:
            ValueError: If no conversation exists with the given ID
        """
        conversation = self.get_conversation(conversation_id)
        return conversation.references
