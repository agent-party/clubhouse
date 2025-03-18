"""
Graph traversal utilities for the Neo4j database.

This module provides specialized utilities for traversing the Neo4j graph
to efficiently retrieve context and analyze relationships between entities.
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union, cast
from uuid import UUID

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol

logger = logging.getLogger(__name__)


class GraphTraversalUtils:
    """
    Utilities for traversing the graph to retrieve and analyze context.
    
    This class provides specialized methods for traversing the Neo4j graph
    database to efficiently retrieve context and analyze relationships
    between entities like agents, conversations, and knowledge nodes.
    """
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol) -> None:
        """
        Initialize with Neo4j service.
        
        Args:
            neo4j_service: Service for interacting with Neo4j
        """
        self.neo4j_service = neo4j_service
    
    def find_conversation_context(
        self, 
        conversation_id: UUID, 
        relevance_threshold: float = 0.5,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find context items for a conversation above relevance threshold.
        
        Args:
            conversation_id: UUID of the conversation
            relevance_threshold: Minimum relevance score to include (0.0 to 1.0)
            max_results: Maximum number of results to return
            
        Returns:
            List of context dictionaries with relevance scores
            
        Raises:
            ValueError: If relevance_threshold is out of bounds
        """
        if relevance_threshold < 0.0 or relevance_threshold > 1.0:
            raise ValueError("Relevance threshold must be between 0.0 and 1.0")
        
        query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})-[r:HAS_CONTEXT]->(ctx:Context)
        WHERE r.relevance >= $relevance_threshold
        RETURN ctx {.*} AS context, r.relevance AS relevance
        ORDER BY r.relevance DESC
        LIMIT $max_results
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "conversation_id": str(conversation_id),
                "relevance_threshold": relevance_threshold,
                "max_results": max_results
            }
        )
        
        contexts = []
        for record in result:
            context = record.get("context", {})
            relevance = record.get("relevance", 0.0)
            
            # Add relevance to the context dictionary
            context["relevance"] = relevance
            contexts.append(context)
        
        return contexts
    
    def find_related_conversations(
        self, 
        agent_id: UUID, 
        limit: int = 5,
        min_messages: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find conversations that an agent has participated in.
        
        Args:
            agent_id: UUID of the agent
            limit: Maximum number of conversations to return
            min_messages: Minimum number of messages in conversation to include
            
        Returns:
            List of conversation dictionaries with message counts
        """
        query = """
        MATCH (a:Agent {uuid: $agent_id})-[:PARTICIPATED_IN]->(c:Conversation)
        MATCH (c)-[:CONTAINS]->(m:Message)
        WITH c, count(m) AS message_count
        WHERE message_count >= $min_messages
        RETURN c {.*} AS conversation, message_count
        ORDER BY c.created_at DESC
        LIMIT $limit
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "agent_id": str(agent_id),
                "min_messages": min_messages,
                "limit": limit
            }
        )
        
        conversations = []
        for record in result:
            conversation = record.get("conversation", {})
            message_count = record.get("message_count", 0)
            
            # Add message count to the conversation dictionary
            conversation["message_count"] = message_count
            conversations.append(conversation)
        
        return conversations
    
    def find_common_contexts(
        self, 
        conversation_ids: List[UUID], 
        min_occurrence: int = 2,
        relevance_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find context nodes that are shared across multiple conversations.
        
        Args:
            conversation_ids: List of conversation UUIDs to analyze
            min_occurrence: Minimum number of conversations a context must appear in
            relevance_threshold: Minimum relevance score to consider
            
        Returns:
            List of context dictionaries with occurrence counts and average relevance
        """
        if not conversation_ids:
            return []
        
        if relevance_threshold < 0.0 or relevance_threshold > 1.0:
            raise ValueError("Relevance threshold must be between 0.0 and 1.0")
        
        # Convert UUIDs to strings for query
        conversation_id_strings = [str(cid) for cid in conversation_ids]
        
        query = """
        MATCH (c:Conversation)-[r:HAS_CONTEXT]->(ctx:Context)
        WHERE c.conversation_id IN $conversation_ids AND r.relevance >= $relevance_threshold
        WITH ctx, count(c) AS occurrence_count, avg(r.relevance) AS avg_relevance
        WHERE occurrence_count >= $min_occurrence
        RETURN ctx {.*} AS context, occurrence_count, avg_relevance
        ORDER BY occurrence_count DESC, avg_relevance DESC
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "conversation_ids": conversation_id_strings,
                "min_occurrence": min_occurrence,
                "relevance_threshold": relevance_threshold
            }
        )
        
        contexts = []
        for record in result:
            context = record.get("context", {})
            occurrence_count = record.get("occurrence_count", 0)
            avg_relevance = record.get("avg_relevance", 0.0)
            
            # Add statistics to the context dictionary
            context["occurrence_count"] = occurrence_count
            context["average_relevance"] = avg_relevance
            contexts.append(context)
        
        return contexts
    
    def find_context_path(
        self, 
        source_node_id: UUID, 
        target_node_id: UUID, 
        max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find the shortest path between two nodes through context relationships.
        
        Args:
            source_node_id: UUID of the source node
            target_node_id: UUID of the target node
            max_depth: Maximum path length to consider
            
        Returns:
            List of nodes and relationships in the path, or None if no path found
        """
        query = """
        MATCH path = shortestPath(
            (source)-[*1..{max_depth}]-(target)
        )
        WHERE source.uuid = $source_id AND target.uuid = $target_id
        RETURN [node IN nodes(path) | node {.*}] AS path_nodes,
               [rel IN relationships(path) | type(rel)] AS path_rels
        """
        
        result = self.neo4j_service.run_query(
            query,
            {
                "source_id": str(source_node_id),
                "target_id": str(target_node_id),
                "max_depth": max_depth
            }
        )
        
        if not result:
            return None
        
        path_nodes = result[0].get("path_nodes", [])
        path_rels = result[0].get("path_rels", [])
        
        # Combine nodes and relationships into a path representation
        path = []
        for i in range(len(path_nodes)):
            path.append({"node": path_nodes[i]})
            if i < len(path_rels):
                path.append({"relationship": path_rels[i]})
        
        return path
    
    def find_agents_by_context_similarity(
        self, 
        context_id: UUID, 
        min_similarity: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find agents that have used similar contexts.
        
        Args:
            context_id: UUID of the context node
            min_similarity: Minimum similarity score to include
            limit: Maximum number of agents to return
            
        Returns:
            List of agent dictionaries with similarity scores
        """
        if min_similarity < 0.0 or min_similarity > 1.0:
            raise ValueError("Minimum similarity must be between 0.0 and 1.0")
        
        query = """
        MATCH (ctx1:Context {context_id: $context_id})
        MATCH (a:Agent)-[:PARTICIPATED_IN]->(:Conversation)-[:HAS_CONTEXT]->(ctx2:Context)
        WHERE ctx1 <> ctx2
        WITH a, ctx1, ctx2,
             apoc.text.similarity(ctx1.content, ctx2.content) AS similarity
        WHERE similarity >= $min_similarity
        RETURN a {.*} AS agent, max(similarity) AS max_similarity
        ORDER BY max_similarity DESC
        LIMIT $limit
        """
        
        try:
            result = self.neo4j_service.run_query(
                query,
                {
                    "context_id": str(context_id),
                    "min_similarity": min_similarity,
                    "limit": limit
                }
            )
        except Exception as e:
            # Handle case where APOC extension might not be available
            logger.warning(f"Error executing similarity query: {str(e)}")
            return []
        
        agents = []
        for record in result:
            agent = record.get("agent", {})
            similarity = record.get("max_similarity", 0.0)
            
            # Add similarity to the agent dictionary
            agent["similarity"] = similarity
            agents.append(agent)
        
        return agents
    
    def analyze_conversation_context_graph(
        self, 
        conversation_id: UUID
    ) -> Dict[str, Any]:
        """
        Analyze the context graph for a conversation.
        
        Args:
            conversation_id: UUID of the conversation
            
        Returns:
            Dictionary containing graph metrics and analysis
        """
        # Get basic context count
        context_query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})-[:HAS_CONTEXT]->(ctx:Context)
        RETURN count(ctx) AS context_count, 
               count(DISTINCT ctx.type) AS unique_context_types
        """
        
        # Get message statistics
        message_query = """
        MATCH (c:Conversation {conversation_id: $conversation_id})-[:CONTAINS]->(m:Message)
        RETURN count(m) AS message_count,
               sum(size(m.content)) AS total_content_length
        """
        
        # Get participants
        participant_query = """
        MATCH (a:Agent)-[:PARTICIPATED_IN]->(c:Conversation {conversation_id: $conversation_id})
        RETURN count(a) AS participant_count,
               collect(a.agent_id) AS participant_ids
        """
        
        # Execute queries
        context_result = self.neo4j_service.run_query(
            context_query,
            {"conversation_id": str(conversation_id)}
        )
        
        message_result = self.neo4j_service.run_query(
            message_query,
            {"conversation_id": str(conversation_id)}
        )
        
        participant_result = self.neo4j_service.run_query(
            participant_query,
            {"conversation_id": str(conversation_id)}
        )
        
        # Extract metrics
        context_metrics = context_result[0] if context_result else {}
        message_metrics = message_result[0] if message_result else {}
        participant_metrics = participant_result[0] if participant_result else {}
        
        # Combine into analysis result
        analysis = {
            "conversation_id": str(conversation_id),
            "context_metrics": {
                "context_count": context_metrics.get("context_count", 0),
                "unique_context_types": context_metrics.get("unique_context_types", 0)
            },
            "message_metrics": {
                "message_count": message_metrics.get("message_count", 0),
                "total_content_length": message_metrics.get("total_content_length", 0),
                "avg_message_length": message_metrics.get("total_content_length", 0) / 
                                      message_metrics.get("message_count", 1) 
                                      if message_metrics.get("message_count", 0) > 0 else 0
            },
            "participant_metrics": {
                "participant_count": participant_metrics.get("participant_count", 0),
                "participant_ids": participant_metrics.get("participant_ids", [])
            }
        }
        
        return analysis
