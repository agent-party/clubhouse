"""
Neo4j context utilities for managing conversation contexts.

This module provides utilities for working with conversation context
in the Neo4j graph database, including context association, relevance
scoring, and efficient context retrieval operations.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from uuid import UUID, uuid4

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol


logger = logging.getLogger(__name__)


class ContextType:
    """Constants for context types."""
    DOCUMENT = "document"
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    TASK = "task"
    CONVERSATION = "conversation"
    USER_PREFERENCE = "user_preference"
    CAPABILITY = "capability"
    METADATA = "metadata"


class ContextUtils:
    """
    Utilities for working with conversation context in Neo4j.
    
    This class provides methods for creating, linking, and retrieving
    conversation contexts efficiently, including relevance scoring and
    context relationship management.
    """
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol) -> None:
        """
        Initialize the context utilities.
        
        Args:
            neo4j_service: Neo4j service implementation
        """
        self._neo4j = neo4j_service
    
    def create_context(
        self, 
        context_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_uri: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> UUID:
        """
        Create a new context node.
        
        Args:
            context_type: Type of context (e.g., document, memory, knowledge)
            content: Context content text
            metadata: Optional metadata associated with the context
            source_uri: Optional URI indicating the source of the context
            tags: Optional list of tags for categorizing the context
            
        Returns:
            UUID of the created context node
            
        Raises:
            ValueError: If context_type is invalid or content is empty
        """
        if not content:
            raise ValueError("Context content cannot be empty")
        
        # Validate context type
        valid_types = {
            ContextType.DOCUMENT,
            ContextType.MEMORY,
            ContextType.KNOWLEDGE,
            ContextType.TASK,
            ContextType.CONVERSATION,
            ContextType.USER_PREFERENCE,
            ContextType.CAPABILITY,
            ContextType.METADATA
        }
        
        if context_type not in valid_types:
            raise ValueError(f"Invalid context type '{context_type}'. Must be one of: {', '.join(valid_types)}")
        
        # Create context properties
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        properties = {
            "context_id": str(context_id),
            "type": context_type,
            "content": content,
            "created_at": created_at,
            "metadata": json.dumps(metadata) if metadata else None,
            "source_uri": source_uri,
            "tags": json.dumps(tags) if tags else None
        }
        
        # Create context node in Neo4j
        self._neo4j.create_node(labels=["Context"], properties=properties)
        
        logger.debug(f"Created context node: {context_id} (type: {context_type})")
        return context_id
    
    def link_context_to_conversation(
        self,
        context_id: UUID,
        conversation_id: UUID,
        relevance: float = 1.0,
        relationship_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Link a context node to a conversation.
        
        Args:
            context_id: UUID of the context node
            conversation_id: UUID of the conversation node
            relevance: Relevance score (0.0 to 1.0) of this context to the conversation
            relationship_metadata: Optional metadata for the relationship
            
        Raises:
            ValueError: If relevance is not between 0.0 and 1.0
            Neo4jError: If the conversation or context does not exist
        """
        if not 0.0 <= relevance <= 1.0:
            raise ValueError("Relevance must be between 0.0 and 1.0")
        
        # Create relationship properties
        properties = {
            "relevance": relevance,
            "linked_at": datetime.utcnow().isoformat()
        }
        
        if relationship_metadata:
            properties["metadata"] = json.dumps(relationship_metadata)
        
        # Create the relationship
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        MATCH (ctx:Context {context_id: $context_id})
        MERGE (c)-[r:HAS_CONTEXT]->(ctx)
        SET r.relevance = $relevance,
            r.linked_at = $linked_at,
            r.metadata = $metadata
        RETURN r
        """
        
        params = {
            "conversation_id": str(conversation_id),
            "context_id": str(context_id),
            "relevance": relevance,
            "linked_at": properties["linked_at"],
            "metadata": properties.get("metadata")
        }
        
        self._neo4j.run_query(query, params)
        logger.debug(f"Linked context {context_id} to conversation {conversation_id} with relevance {relevance}")
    
    def update_context_relevance(
        self,
        context_id: UUID,
        conversation_id: UUID,
        relevance: float
    ) -> bool:
        """
        Update the relevance score of a context for a conversation.
        
        Args:
            context_id: UUID of the context node
            conversation_id: UUID of the conversation node
            relevance: New relevance score (0.0 to 1.0)
            
        Returns:
            True if the relationship was updated, False if it doesn't exist
            
        Raises:
            ValueError: If relevance is not between 0.0 and 1.0
        """
        if not 0.0 <= relevance <= 1.0:
            raise ValueError("Relevance must be between 0.0 and 1.0")
        
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        -[r:HAS_CONTEXT]->(ctx:Context {context_id: $context_id})
        SET r.relevance = $relevance,
            r.updated_at = $updated_at
        RETURN r
        """
        
        params = {
            "conversation_id": str(conversation_id),
            "context_id": str(context_id),
            "relevance": relevance,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = self._neo4j.run_query(query, params)
        return len(result) > 0
    
    def get_conversation_contexts(
        self,
        conversation_id: UUID,
        min_relevance: float = 0.0,
        context_types: Optional[List[str]] = None,
        limit: int = 100,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get contexts associated with a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            min_relevance: Minimum relevance score to include (0.0 to 1.0)
            context_types: Optional list of context types to filter by
            limit: Maximum number of contexts to return
            include_metadata: Whether to parse and include metadata
            
        Returns:
            List of context dictionaries with content and relevance
        """
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("Minimum relevance must be between 0.0 and 1.0")
        
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})
        -[r:HAS_CONTEXT]->(ctx:Context)
        WHERE r.relevance >= $min_relevance
        """
        
        params = {
            "conversation_id": str(conversation_id),
            "min_relevance": min_relevance
        }
        
        # Add context type filter if provided
        if context_types:
            query += " AND ctx.type IN $context_types"
            params["context_types"] = context_types
        
        # Add ordering and limit
        query += """
        RETURN ctx, r.relevance AS relevance
        ORDER BY relevance DESC, ctx.created_at DESC
        LIMIT $limit
        """
        params["limit"] = limit
        
        result = self._neo4j.run_query(query, params)
        contexts = []
        
        for record in result:
            ctx = record["ctx"]
            context_dict = {
                "context_id": UUID(ctx["context_id"]),
                "type": ctx["type"],
                "content": ctx["content"],
                "created_at": ctx["created_at"],
                "relevance": record["relevance"],
                "source_uri": ctx.get("source_uri")
            }
            
            # Parse and include tags if available
            if ctx.get("tags"):
                try:
                    context_dict["tags"] = json.loads(ctx["tags"])
                except (json.JSONDecodeError, TypeError):
                    context_dict["tags"] = []
            
            # Parse and include metadata if requested
            if include_metadata and ctx.get("metadata"):
                try:
                    context_dict["metadata"] = json.loads(ctx["metadata"])
                except (json.JSONDecodeError, TypeError):
                    context_dict["metadata"] = {}
            
            contexts.append(context_dict)
        
        return contexts
    
    def get_shared_contexts(
        self,
        entity_ids: List[UUID],
        entity_label: str = "Conversation",
        min_relevance: float = 0.0,
        context_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find contexts shared across multiple entities (conversations, agents, etc.).
        
        Args:
            entity_ids: List of entity UUIDs to find shared contexts for
            entity_label: Neo4j label for the entities (e.g., "Conversation", "Agent")
            min_relevance: Minimum relevance score to include
            context_types: Optional list of context types to filter by
            
        Returns:
            List of shared contexts with occurrence count and average relevance
        """
        if not entity_ids:
            return []
        
        id_field = "conversation_id" if entity_label == "Conversation" else "uuid"
        
        query = f"""
        MATCH (e:{entity_label})-[r:HAS_CONTEXT]->(ctx:Context)
        WHERE e.{id_field} IN $entity_ids
        AND r.relevance >= $min_relevance
        """
        
        params = {
            "entity_ids": [str(eid) for eid in entity_ids],
            "min_relevance": min_relevance
        }
        
        # Add context type filter if provided
        if context_types:
            query += " AND ctx.type IN $context_types"
            params["context_types"] = context_types
        
        # Group to find shared contexts
        query += """
        WITH ctx, count(e) AS occurrence_count, 
             avg(r.relevance) AS avg_relevance,
             collect(e.uuid) AS entity_uuids
        WHERE occurrence_count > 1
        RETURN ctx, occurrence_count, avg_relevance, entity_uuids
        ORDER BY occurrence_count DESC, avg_relevance DESC
        """
        
        result = self._neo4j.run_query(query, params)
        shared_contexts = []
        
        for record in result:
            ctx = record["ctx"]
            context_dict = {
                "context_id": UUID(ctx["context_id"]),
                "type": ctx["type"],
                "content": ctx["content"],
                "created_at": ctx["created_at"],
                "occurrence_count": record["occurrence_count"],
                "average_relevance": record["avg_relevance"],
                "entity_ids": [UUID(eid) for eid in record["entity_uuids"]]
            }
            
            # Parse and include metadata if available
            if ctx.get("metadata"):
                try:
                    context_dict["metadata"] = json.loads(ctx["metadata"])
                except (json.JSONDecodeError, TypeError):
                    context_dict["metadata"] = {}
            
            shared_contexts.append(context_dict)
        
        return shared_contexts
    
    def find_similar_contexts(
        self,
        content: str,
        context_types: Optional[List[str]] = None,
        min_similarity: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find contexts similar to the provided content using text similarity.
        
        Note: This requires the APOC library to be installed and configured
        in the Neo4j database.
        
        Args:
            content: Content to find similar contexts for
            context_types: Optional list of context types to filter by
            min_similarity: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of similar contexts with similarity scores
        """
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("Minimum similarity must be between 0.0 and 1.0")
        
        # This query uses APOC's text similarity function
        query = """
        MATCH (ctx:Context)
        """
        
        params = {
            "content": content,
            "min_similarity": min_similarity,
            "limit": limit
        }
        
        # Add context type filter if provided
        if context_types:
            query += " WHERE ctx.type IN $context_types"
            params["context_types"] = context_types
        
        # Calculate similarity and return results
        query += """
        WITH ctx, apoc.text.similarity(ctx.content, $content) AS similarity
        WHERE similarity >= $min_similarity
        RETURN ctx, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        try:
            result = self._neo4j.run_query(query, params)
            similar_contexts = []
            
            for record in result:
                ctx = record["ctx"]
                context_dict = {
                    "context_id": UUID(ctx["context_id"]),
                    "type": ctx["type"],
                    "content": ctx["content"],
                    "created_at": ctx["created_at"],
                    "similarity": record["similarity"]
                }
                
                similar_contexts.append(context_dict)
            
            return similar_contexts
            
        except Exception as e:
            logger.warning(f"Error finding similar contexts: {str(e)}")
            # This might happen if APOC is not available
            return []
    
    def associate_context_with_message(
        self,
        context_id: UUID,
        message_id: UUID,
        relationship_type: str = "REFERENCED_IN",
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Associate a context with a specific message.
        
        Args:
            context_id: UUID of the context
            message_id: UUID of the message
            relationship_type: Type of relationship
            properties: Optional properties for the relationship
            
        Raises:
            Neo4jError: If the message or context does not exist
        """
        rel_properties = {
            "created_at": datetime.utcnow().isoformat()
        }
        
        if properties:
            rel_properties.update(properties)
        
        query = f"""
        MATCH (m:Message {{message_id: $message_id}})
        MATCH (ctx:Context {{context_id: $context_id}})
        MERGE (ctx)-[r:{relationship_type}]->(m)
        SET r += $properties
        RETURN r
        """
        
        params = {
            "message_id": str(message_id),
            "context_id": str(context_id),
            "properties": rel_properties
        }
        
        self._neo4j.run_query(query, params)
        logger.debug(f"Associated context {context_id} with message {message_id}")
    
    def calculate_conversation_context_similarity(
        self,
        conversation_id1: UUID,
        conversation_id2: UUID
    ) -> float:
        """
        Calculate the context similarity between two conversations.
        
        This measures how similar the contexts of two conversations are,
        based on shared contexts and their relevance.
        
        Args:
            conversation_id1: UUID of the first conversation
            conversation_id2: UUID of the second conversation
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        query = """
        // Contexts for first conversation
        MATCH (c1:Conversation {conversation_id: $conversation_id1})-[r1:HAS_CONTEXT]->(ctx)
        WITH collect({context: ctx.context_id, relevance: r1.relevance}) AS c1_contexts
        
        // Contexts for second conversation
        MATCH (c2:Conversation {conversation_id: $conversation_id2})-[r2:HAS_CONTEXT]->(ctx)
        WITH c1_contexts, collect({context: ctx.context_id, relevance: r2.relevance}) AS c2_contexts
        
        // Calculate similarity using Jaccard similarity with relevance weights
        CALL {
            WITH c1_contexts, c2_contexts
            UNWIND c1_contexts AS c1_ctx
            UNWIND c2_contexts AS c2_ctx
            WITH c1_ctx, c2_ctx
            WHERE c1_ctx.context = c2_ctx.context
            RETURN sum(c1_ctx.relevance * c2_ctx.relevance) AS intersection
        }
        
        // Total context weights
        WITH intersection,
             reduce(s = 0.0, c IN c1_contexts | s + c.relevance) AS c1_total,
             reduce(s = 0.0, c IN c2_contexts | s + c.relevance) AS c2_total
        
        // Compute similarity
        RETURN CASE
               WHEN c1_total + c2_total = 0 THEN 0
               ELSE intersection / (c1_total + c2_total - intersection)
               END AS similarity
        """
        
        params = {
            "conversation_id1": str(conversation_id1),
            "conversation_id2": str(conversation_id2)
        }
        
        result = self._neo4j.run_query(query, params)
        if result and len(result) > 0:
            return float(result[0]["similarity"])
        
        return 0.0
    
    def build_context_summary(
        self,
        conversation_id: UUID,
        max_contexts: int = 10,
        include_content: bool = True,
        max_content_length: Optional[int] = 500
    ) -> Dict[str, Any]:
        """
        Build a summary of contexts for a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            max_contexts: Maximum number of contexts to include
            include_content: Whether to include the full content text
            max_content_length: Maximum length for each content entry
            
        Returns:
            Summary dictionary with context information
        """
        # Get the contexts
        contexts = self.get_conversation_contexts(
            conversation_id=conversation_id,
            min_relevance=0.1,
            limit=max_contexts,
            include_metadata=True
        )
        
        # Group contexts by type
        context_by_type: Dict[str, List[Dict[str, Any]]] = {}
        
        for ctx in contexts:
            ctx_type = ctx["type"]
            
            if ctx_type not in context_by_type:
                context_by_type[ctx_type] = []
            
            # Copy the context with optional content truncation
            context_entry = {
                "context_id": str(ctx["context_id"]),
                "relevance": ctx["relevance"],
                "created_at": ctx["created_at"]
            }
            
            if include_content:
                content = ctx["content"]
                if max_content_length and len(content) > max_content_length:
                    content = content[:max_content_length] + "..."
                context_entry["content"] = content
            
            # Include metadata if available
            if "metadata" in ctx:
                context_entry["metadata"] = ctx["metadata"]
            
            # Include tags if available
            if "tags" in ctx:
                context_entry["tags"] = ctx["tags"]
            
            context_by_type[ctx_type].append(context_entry)
        
        # Build the summary
        summary = {
            "conversation_id": str(conversation_id),
            "total_contexts": len(contexts),
            "context_types": list(context_by_type.keys()),
            "contexts_by_type": context_by_type,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return summary
