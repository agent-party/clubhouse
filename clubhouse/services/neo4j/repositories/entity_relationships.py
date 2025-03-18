"""
Entity Relationship Repository for Neo4j.

This module provides a repository implementation for managing relationships between
core entities in the Neo4j graph database, including:
1. Agent to Conversation relationships
2. Message to Context relationships 
3. Agent to Knowledge relationships

These relationships form the backbone of the agent collaboration platform,
enabling rich context tracking and knowledge sharing between agents.
"""

import logging
from typing import Dict, List, Optional, Any, Union, cast
from uuid import UUID
from datetime import datetime

from neo4j.exceptions import Neo4jError

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol

logger = logging.getLogger(__name__)


class EntityRelationshipRepository:
    """
    Repository for managing entity relationships in Neo4j.
    
    This class provides methods for creating, retrieving, and deleting
    relationships between core entities in the system, enabling the 
    graph-based representation of agent collaboration and knowledge.
    """
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol) -> None:
        """
        Initialize the repository with a Neo4j service.
        
        Args:
            neo4j_service: Service for interacting with Neo4j
        """
        self.neo4j_service = neo4j_service
    
    def link_agent_to_conversation(
        self,
        agent_id: Union[str, UUID],
        conversation_id: Union[str, UUID],
        relationship_type: str = "PARTICIPATES_IN",
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between an Agent and a Conversation.
        
        This relationship represents an agent's participation in a conversation,
        which can include properties like the agent's role or when they joined.
        
        Args:
            agent_id: Unique identifier for the agent
            conversation_id: Unique identifier for the conversation
            relationship_type: Type of relationship (default: PARTICIPATES_IN)
            properties: Optional properties to add to the relationship
            
        Returns:
            True if successful, False otherwise
        """
        # Convert UUID objects to strings if necessary
        agent_id_str = str(agent_id)
        conversation_id_str = str(conversation_id)
        
        # Set default properties if none provided
        if properties is None:
            properties = {"joined_at": datetime.now().isoformat()}
        
        # Build Cypher query to create the relationship
        query = f"""
        MATCH (a:Agent), (c:Conversation)
        WHERE a.id = $agent_id AND c.id = $conversation_id
        CREATE (a)-[r:{relationship_type} $properties]->(c)
        RETURN r
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(
                        query,
                        agent_id=agent_id_str,
                        conversation_id=conversation_id_str,
                        properties=properties
                    )
                    return True
        except Neo4jError as e:
            logger.error(f"Failed to link agent to conversation: {str(e)}")
            return False
    
    def link_message_to_context(
        self,
        message_id: Union[str, UUID],
        context_id: Union[str, UUID],
        relationship_type: str = "HAS_CONTEXT",
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between a Message and a Context item.
        
        This relationship represents that a message contains or references
        a specific context item, with properties like relevance score.
        
        Args:
            message_id: Unique identifier for the message
            context_id: Unique identifier for the context item
            relationship_type: Type of relationship (default: HAS_CONTEXT)
            properties: Optional properties to add to the relationship
            
        Returns:
            True if successful, False otherwise
        """
        # Convert UUID objects to strings if necessary
        message_id_str = str(message_id)
        context_id_str = str(context_id)
        
        # Set default properties if none provided
        if properties is None:
            properties = {"created_at": datetime.now().isoformat()}
        
        # Build Cypher query to create the relationship
        query = f"""
        MATCH (m:Message), (ctx:Context)
        WHERE m.id = $message_id AND ctx.id = $context_id
        CREATE (m)-[r:{relationship_type} $properties]->(ctx)
        RETURN r
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(
                        query,
                        message_id=message_id_str,
                        context_id=context_id_str,
                        properties=properties
                    )
                    return True
        except Neo4jError as e:
            logger.error(f"Failed to link message to context: {str(e)}")
            return False
    
    def link_agent_to_knowledge(
        self,
        agent_id: Union[str, UUID],
        knowledge_id: Union[str, UUID],
        relationship_type: str = "KNOWS_ABOUT",
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between an Agent and a Knowledge item.
        
        This relationship represents that an agent has knowledge about
        a specific topic or concept, with properties like confidence.
        
        Args:
            agent_id: Unique identifier for the agent
            knowledge_id: Unique identifier for the knowledge item
            relationship_type: Type of relationship (default: KNOWS_ABOUT)
            properties: Optional properties to add to the relationship
            
        Returns:
            True if successful, False otherwise
        """
        # Convert UUID objects to strings if necessary
        agent_id_str = str(agent_id)
        knowledge_id_str = str(knowledge_id)
        
        # Set default properties if none provided
        if properties is None:
            properties = {"confidence": 1.0, "last_updated": datetime.now().isoformat()}
        
        # Build Cypher query to create the relationship
        query = f"""
        MATCH (a:Agent), (k:Knowledge)
        WHERE a.id = $agent_id AND k.id = $knowledge_id
        CREATE (a)-[r:{relationship_type} $properties]->(k)
        RETURN r
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(
                        query,
                        agent_id=agent_id_str,
                        knowledge_id=knowledge_id_str,
                        properties=properties
                    )
                    return True
        except Neo4jError as e:
            logger.error(f"Failed to link agent to knowledge: {str(e)}")
            return False
    
    def get_conversation_agents(
        self,
        conversation_id: Union[str, UUID],
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all agents linked to a specific conversation.
        
        This method retrieves all agents that are participants in a conversation,
        along with the properties of their relationship to the conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            relationship_type: Optional specific relationship type to filter by
            
        Returns:
            List of dicts containing agent data and relationship properties
        """
        # Convert UUID objects to strings if necessary
        conversation_id_str = str(conversation_id)
        
        # Build relationship type filter if provided
        rel_filter = f":{relationship_type}" if relationship_type else ""
        
        # Build Cypher query to retrieve the agents
        query = f"""
        MATCH (a:Agent)-[r{rel_filter}]->(c:Conversation)
        WHERE c.id = $conversation_id
        RETURN a as agent, r as relationship
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    result = tx.run(query, conversation_id=conversation_id_str)
                    return result.data()
        except Neo4jError as e:
            logger.error(f"Failed to get conversation agents: {str(e)}")
            return []
    
    def get_message_context(
        self,
        message_id: Union[str, UUID],
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all context items linked to a specific message.
        
        This method retrieves all context items that are associated with a message,
        along with the properties of their relationship to the message.
        
        Args:
            message_id: Unique identifier for the message
            relationship_type: Optional specific relationship type to filter by
            
        Returns:
            List of dicts containing context data and relationship properties
        """
        # Convert UUID objects to strings if necessary
        message_id_str = str(message_id)
        
        # Build relationship type filter if provided
        rel_filter = f":{relationship_type}" if relationship_type else ""
        
        # Build Cypher query to retrieve the context items
        query = f"""
        MATCH (m:Message)-[r{rel_filter}]->(ctx:Context)
        WHERE m.id = $message_id
        RETURN ctx as context, r as relationship
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    result = tx.run(query, message_id=message_id_str)
                    return result.data()
        except Neo4jError as e:
            logger.error(f"Failed to get message context: {str(e)}")
            return []
    
    def get_agent_knowledge(
        self,
        agent_id: Union[str, UUID],
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all knowledge items linked to a specific agent.
        
        This method retrieves all knowledge items that an agent knows about,
        along with the properties of their relationship to the knowledge.
        
        Args:
            agent_id: Unique identifier for the agent
            relationship_type: Optional specific relationship type to filter by
            
        Returns:
            List of dicts containing knowledge data and relationship properties
        """
        # Convert UUID objects to strings if necessary
        agent_id_str = str(agent_id)
        
        # Build relationship type filter if provided
        rel_filter = f":{relationship_type}" if relationship_type else ""
        
        # Build Cypher query to retrieve the knowledge items
        query = f"""
        MATCH (a:Agent)-[r{rel_filter}]->(k:Knowledge)
        WHERE a.id = $agent_id
        RETURN k as knowledge, r as relationship
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    result = tx.run(query, agent_id=agent_id_str)
                    return result.data()
        except Neo4jError as e:
            logger.error(f"Failed to get agent knowledge: {str(e)}")
            return []
    
    def delete_relationship(
        self,
        source_id: Union[str, UUID],
        target_id: Union[str, UUID],
        relationship_type: str
    ) -> bool:
        """
        Delete a relationship between two entities.
        
        This method removes a specific relationship between source and target entities.
        
        Args:
            source_id: Unique identifier for the source entity
            target_id: Unique identifier for the target entity
            relationship_type: Type of relationship to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Convert UUID objects to strings if necessary
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Build Cypher query to delete the relationship
        query = f"""
        MATCH (source)-[r:{relationship_type}]->(target)
        WHERE source.id = $source_id AND target.id = $target_id
        DELETE r
        """
        
        try:
            with self.neo4j_service.session() as session:
                with session.begin_transaction() as tx:
                    tx.run(
                        query,
                        source_id=source_id_str,
                        target_id=target_id_str
                    )
                    return True
        except Neo4jError as e:
            logger.error(f"Failed to delete relationship: {str(e)}")
            return False
