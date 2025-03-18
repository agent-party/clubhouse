"""
Conversation repository for Neo4j persistence and retrieval.

This module provides a repository implementation for working with conversations,
messages, and context data in the Neo4j graph database.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple, cast
from uuid import UUID, uuid4

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository for conversation persistence and retrieval.
    
    This class provides methods for creating, updating, and retrieving
    conversations, messages, and their associated context data from
    the Neo4j graph database.
    """
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol) -> None:
        """
        Initialize the repository with a Neo4j service.
        
        Args:
            neo4j_service: Service for interacting with Neo4j
        """
        self.neo4j_service = neo4j_service
    
    def create_conversation(
        self, 
        title: str, 
        description: str = "", 
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Create a new conversation node in the graph database.
        
        Args:
            title: Title of the conversation
            description: Optional description of the conversation
            metadata: Optional metadata associated with the conversation
            
        Returns:
            UUID of the created conversation node
        """
        conversation_id = uuid4()
        properties = {
            "conversation_id": str(conversation_id),
            "title": title,
            "description": description,
            "status": "ACTIVE",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        node_id = self.neo4j_service.create_node("Conversation", properties)
        logger.debug(f"Created conversation node with ID: {node_id}")
        return node_id
    
    def get_conversation(self, conversation_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by its ID.
        
        Args:
            conversation_id: UUID of the conversation to retrieve
            
        Returns:
            Dictionary representation of the conversation, or None if not found
        """
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        RETURN c {.*} AS conversation
        """
        
        result = self.neo4j_service.run_query(
            query,
            {"conversation_id": str(conversation_id)}
        )
        
        if not result:
            return None
        
        conversation = result[0].get("conversation")
        
        # Parse metadata from JSON string to dict
        if conversation and "metadata" in conversation:
            try:
                conversation["metadata"] = json.loads(conversation["metadata"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse metadata for conversation {conversation_id}")
                conversation["metadata"] = {}
        
        return conversation
    
    def update_conversation(
        self, 
        conversation_id: UUID, 
        properties: Dict[str, Any]
    ) -> bool:
        """
        Update a conversation's properties.
        
        Args:
            conversation_id: UUID of the conversation to update
            properties: Properties to update on the conversation
            
        Returns:
            True if the conversation was updated, False if it doesn't exist
        """
        # Add updated timestamp
        properties["updated_at"] = datetime.utcnow().isoformat()
        
        # Handle metadata serialization
        if "metadata" in properties and not isinstance(properties["metadata"], str):
            properties["metadata"] = json.dumps(properties["metadata"])
        
        # Find the node by conversation_id and update it
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        SET c += $properties
        RETURN count(c) > 0 AS updated
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "conversation_id": str(conversation_id),
                "properties": properties
            }
        )
        
        if not result:
            return False
        
        return result[0].get("updated", False)
    
    def add_message_to_conversation(
        self,
        conversation_id: UUID,
        content: str,
        role: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            content: Content of the message
            role: Role of the message sender (e.g., "user", "assistant")
            metadata: Optional metadata associated with the message
            
        Returns:
            UUID of the created message node
            
        Raises:
            ValueError: If the conversation does not exist
        """
        message_id = uuid4()
        
        # Prepare message properties
        message_properties = {
            "message_id": str(message_id),
            "content": content,
            "role": role,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        # Create message and relationship in a single transaction
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        CREATE (m:Message $message_properties)
        CREATE (c)-[r:CONTAINS {order: $order}]->(m)
        RETURN m.message_id AS message_id
        """
        
        # Get the current message count to set the order
        count_query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})-[r:CONTAINS]->(m:Message)
        RETURN count(m) AS message_count
        """
        
        count_result = self.neo4j_service.run_query(
            count_query,
            {"conversation_id": str(conversation_id)}
        )
        
        if not count_result:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        message_count = count_result[0].get("message_count", 0)
        
        result = self.neo4j_service.run_query(
            query,
            {
                "conversation_id": str(conversation_id),
                "message_properties": message_properties,
                "order": message_count
            }
        )
        
        if not result:
            raise ValueError(f"Failed to add message to conversation {conversation_id}")
        
        return message_id
    
    def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0,
        reverse: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            limit: Optional maximum number of messages to return
            offset: Optional number of messages to skip
            reverse: If True, return messages in reverse chronological order
            
        Returns:
            List of message dictionaries
        """
        # Build the query with optional limit and ordering
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})-[r:CONTAINS]->(m:Message)
        RETURN m {.*} AS message
        ORDER BY r.order """ + ("DESC" if reverse else "ASC")
        
        params: Dict[str, Any] = {"conversation_id": str(conversation_id)}
        
        # Apply pagination if specified
        if limit is not None:
            query += " SKIP $offset LIMIT $limit"
            params["offset"] = offset
            params["limit"] = limit
        
        result = self.neo4j_service.run_query(query, params)
        
        messages = []
        for record in result:
            message = record.get("message", {})
            
            # Parse metadata from JSON string to dict
            if message and "metadata" in message:
                try:
                    message["metadata"] = json.loads(message["metadata"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse metadata for message {message.get('message_id')}")
                    message["metadata"] = {}
            
            messages.append(message)
        
        return messages
    
    def add_context_to_conversation(
        self,
        conversation_id: UUID,
        context_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        relevance: float = 1.0
    ) -> UUID:
        """
        Add a context node to a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            context_type: Type of context (e.g., "document", "memory", "knowledge")
            content: Content of the context
            metadata: Optional metadata associated with the context
            relevance: Relevance score of the context to the conversation (0.0 to 1.0)
            
        Returns:
            UUID of the created context node
            
        Raises:
            ValueError: If the conversation does not exist or relevance is out of bounds
        """
        if relevance < 0.0 or relevance > 1.0:
            raise ValueError("Relevance must be between 0.0 and 1.0")
        
        context_id = uuid4()
        
        # Prepare context properties
        context_properties = {
            "context_id": str(context_id),
            "type": context_type,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {})
        }
        
        # Create context and relationship in a single transaction
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        CREATE (ctx:Context $context_properties)
        CREATE (c)-[r:HAS_CONTEXT {relevance: $relevance}]->(ctx)
        RETURN ctx.context_id AS context_id
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "conversation_id": str(conversation_id),
                "context_properties": context_properties,
                "relevance": relevance
            }
        )
        
        if not result:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        return context_id
    
    def get_conversation_contexts(
        self,
        conversation_id: UUID,
        context_types: Optional[List[str]] = None,
        min_relevance: float = 0.0,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get contexts for a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            context_types: Optional list of context types to filter by
            min_relevance: Minimum relevance score to include
            limit: Optional maximum number of contexts to return
            offset: Optional number of contexts to skip
            
        Returns:
            List of context dictionaries with relevance scores
        """
        # Build the query with optional type filter and pagination
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})-[r:HAS_CONTEXT]->(ctx:Context)
        WHERE r.relevance >= $min_relevance
        """
        
        params: Dict[str, Any] = {
            "conversation_id": str(conversation_id),
            "min_relevance": min_relevance
        }
        
        # Add context type filter if specified
        if context_types:
            query += " AND ctx.type IN $context_types"
            params["context_types"] = context_types
        
        query += """
        RETURN ctx {.*} AS context, r.relevance AS relevance
        ORDER BY r.relevance DESC
        """
        
        # Apply pagination if specified
        if limit is not None:
            query += " SKIP $offset LIMIT $limit"
            params["offset"] = offset
            params["limit"] = limit
        
        result = self.neo4j_service.run_query(query, params)
        
        contexts = []
        for record in result:
            context = record.get("context", {})
            relevance = record.get("relevance", 0.0)
            
            # Parse metadata from JSON string to dict
            if context and "metadata" in context:
                try:
                    context["metadata"] = json.loads(context["metadata"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse metadata for context {context.get('context_id')}")
                    context["metadata"] = {}
            
            # Add relevance to the context dictionary
            context["relevance"] = relevance
            contexts.append(context)
        
        return contexts
    
    def delete_conversation(self, conversation_id: UUID) -> bool:
        """
        Delete a conversation and all its related messages and contexts.
        
        Args:
            conversation_id: UUID of the conversation to delete
            
        Returns:
            True if the conversation was deleted, False if it doesn't exist
        """
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        OPTIONAL MATCH (c)-[:CONTAINS]->(m:Message)
        OPTIONAL MATCH (c)-[:HAS_CONTEXT]->(ctx:Context)
        DETACH DELETE c, m, ctx
        RETURN count(c) > 0 AS deleted
        """
        
        result = self.neo4j_service.run_query(
            query,
            {"conversation_id": str(conversation_id)}
        )
        
        if not result:
            return False
        
        return result[0].get("deleted", False)
    
    def get_recent_conversations(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversations ordered by creation date.
        
        Args:
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            
        Returns:
            List of conversation dictionaries
        """
        query = """
        MATCH (c:Conversation)
        RETURN c {.*} AS conversation
        ORDER BY c.created_at DESC
        SKIP $offset LIMIT $limit
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "offset": offset,
                "limit": limit
            }
        )
        
        conversations = []
        for record in result:
            conversation = record.get("conversation", {})
            
            # Parse metadata from JSON string to dict
            if conversation and "metadata" in conversation:
                try:
                    conversation["metadata"] = json.loads(conversation["metadata"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse metadata for conversation {conversation.get('conversation_id')}")
                    conversation["metadata"] = {}
            
            conversations.append(conversation)
        
        return conversations
