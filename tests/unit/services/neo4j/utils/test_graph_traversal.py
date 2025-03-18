"""
Unit tests for the GraphTraversalUtils.

This module contains tests for the GraphTraversalUtils class,
which provides efficient graph traversal methods for the Neo4j database.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.utils.graph_traversal import GraphTraversalUtils


@pytest.fixture
def mock_neo4j_service() -> Mock:
    """Create a mock Neo4j service for testing."""
    service = Mock(spec=Neo4jServiceProtocol)
    return service


@pytest.fixture
def graph_utils(mock_neo4j_service: Mock) -> GraphTraversalUtils:
    """Create a GraphTraversalUtils with a mock Neo4j service."""
    return GraphTraversalUtils(mock_neo4j_service)


class TestGraphTraversalUtils:
    """Tests for the GraphTraversalUtils class."""

    def test_find_conversation_context(self, graph_utils: GraphTraversalUtils, 
                                     mock_neo4j_service: Mock) -> None:
        """Test finding context for a conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_contexts = [
            {
                "context": {
                    "context_id": str(uuid4()),
                    "type": "document",
                    "content": "Document content"
                },
                "relevance": 0.9
            },
            {
                "context": {
                    "context_id": str(uuid4()),
                    "type": "memory",
                    "content": "Memory content"
                },
                "relevance": 0.7
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_contexts
        
        # Act
        result = graph_utils.find_conversation_context(
            conversation_id, 
            relevance_threshold=0.5,
            max_results=10
        )
        
        # Assert
        assert len(result) == 2
        assert result[0]["type"] == "document"
        assert result[0]["relevance"] == 0.9
        assert result[1]["type"] == "memory"
        assert result[1]["relevance"] == 0.7
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation" in query
        assert "WHERE r.relevance >=" in query
        assert params["conversation_id"] == str(conversation_id)
        assert params["relevance_threshold"] == 0.5
        assert params["max_results"] == 10

    def test_find_conversation_context_invalid_threshold(self, graph_utils: GraphTraversalUtils) -> None:
        """Test finding context with an invalid relevance threshold."""
        # Arrange
        conversation_id = uuid4()
        
        # Act/Assert
        with pytest.raises(ValueError, match="Relevance threshold must be between 0.0 and 1.0"):
            graph_utils.find_conversation_context(conversation_id, relevance_threshold=1.5)

    def test_find_related_conversations(self, graph_utils: GraphTraversalUtils, 
                                      mock_neo4j_service: Mock) -> None:
        """Test finding conversations an agent has participated in."""
        # Arrange
        agent_id = uuid4()
        mock_conversations = [
            {
                "conversation": {
                    "conversation_id": str(uuid4()),
                    "title": "First Conversation",
                    "created_at": "2023-01-01T00:00:00Z"
                },
                "message_count": 10
            },
            {
                "conversation": {
                    "conversation_id": str(uuid4()),
                    "title": "Second Conversation",
                    "created_at": "2023-01-02T00:00:00Z"
                },
                "message_count": 5
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_conversations
        
        # Act
        result = graph_utils.find_related_conversations(
            agent_id, 
            limit=5,
            min_messages=1
        )
        
        # Assert
        assert len(result) == 2
        assert result[0]["title"] == "First Conversation"
        assert result[0]["message_count"] == 10
        assert result[1]["title"] == "Second Conversation"
        assert result[1]["message_count"] == 5
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (a:Agent" in query
        assert params["agent_id"] == str(agent_id)
        assert params["limit"] == 5
        assert params["min_messages"] == 1

    def test_find_common_contexts(self, graph_utils: GraphTraversalUtils, 
                               mock_neo4j_service: Mock) -> None:
        """Test finding contexts shared across multiple conversations."""
        # Arrange
        conversation_ids = [uuid4(), uuid4(), uuid4()]
        mock_contexts = [
            {
                "context": {
                    "context_id": str(uuid4()),
                    "type": "document",
                    "content": "Common document"
                },
                "occurrence_count": 3,
                "avg_relevance": 0.8
            },
            {
                "context": {
                    "context_id": str(uuid4()),
                    "type": "memory",
                    "content": "Common memory"
                },
                "occurrence_count": 2,
                "avg_relevance": 0.6
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_contexts
        
        # Act
        result = graph_utils.find_common_contexts(
            conversation_ids, 
            min_occurrence=2,
            relevance_threshold=0.3
        )
        
        # Assert
        assert len(result) == 2
        assert result[0]["type"] == "document"
        assert result[0]["occurrence_count"] == 3
        assert result[0]["average_relevance"] == 0.8
        assert result[1]["type"] == "memory"
        assert result[1]["occurrence_count"] == 2
        assert result[1]["average_relevance"] == 0.6
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation)-[r:HAS_CONTEXT]->(ctx:Context)" in query
        assert "WHERE c.conversation_id IN $conversation_ids" in query
        assert params["min_occurrence"] == 2
        assert params["relevance_threshold"] == 0.3
        assert params["conversation_ids"] == [str(cid) for cid in conversation_ids]

    def test_find_common_contexts_no_conversations(self, graph_utils: GraphTraversalUtils,
                                               mock_neo4j_service: Mock) -> None:
        """Test finding common contexts with no conversation IDs."""
        # Act
        result = graph_utils.find_common_contexts([])
        
        # Assert
        assert result == []
        mock_neo4j_service.run_query.assert_not_called()

    def test_find_common_contexts_invalid_threshold(self, graph_utils: GraphTraversalUtils) -> None:
        """Test finding common contexts with an invalid relevance threshold."""
        # Arrange
        conversation_ids = [uuid4()]
        
        # Act/Assert
        with pytest.raises(ValueError, match="Relevance threshold must be between 0.0 and 1.0"):
            graph_utils.find_common_contexts(conversation_ids, relevance_threshold=1.5)

    def test_find_context_path(self, graph_utils: GraphTraversalUtils, 
                             mock_neo4j_service: Mock) -> None:
        """Test finding a path between two nodes."""
        # Arrange
        source_id = uuid4()
        target_id = uuid4()
        
        mock_path = [
            {
                "path_nodes": [
                    {"uuid": str(source_id), "label": "Agent", "name": "Agent A"},
                    {"uuid": str(uuid4()), "label": "Conversation", "title": "Conversation X"},
                    {"uuid": str(target_id), "label": "Agent", "name": "Agent B"}
                ],
                "path_rels": [
                    "PARTICIPATED_IN",
                    "PARTICIPATED_IN"
                ]
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_path
        
        # Act
        result = graph_utils.find_context_path(source_id, target_id, max_depth=3)
        
        # Assert
        assert result is not None
        assert len(result) == 5  # 3 nodes and 2 relationships
        assert result[0]["node"]["uuid"] == str(source_id)
        assert result[1]["relationship"] == "PARTICIPATED_IN"
        assert result[2]["node"]["title"] == "Conversation X"
        assert result[3]["relationship"] == "PARTICIPATED_IN"
        assert result[4]["node"]["uuid"] == str(target_id)
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH path = shortestPath" in query
        assert params["source_id"] == str(source_id)
        assert params["target_id"] == str(target_id)
        assert params["max_depth"] == 3

    def test_find_context_path_not_found(self, graph_utils: GraphTraversalUtils, 
                                       mock_neo4j_service: Mock) -> None:
        """Test finding a path between two nodes when no path exists."""
        # Arrange
        source_id = uuid4()
        target_id = uuid4()
        
        mock_neo4j_service.run_query.return_value = []
        
        # Act
        result = graph_utils.find_context_path(source_id, target_id)
        
        # Assert
        assert result is None

    def test_find_agents_by_context_similarity(self, graph_utils: GraphTraversalUtils, 
                                            mock_neo4j_service: Mock) -> None:
        """Test finding agents with similar contexts."""
        # Arrange
        context_id = uuid4()
        mock_agents = [
            {
                "agent": {
                    "uuid": str(uuid4()),
                    "agent_id": "agent1",
                    "name": "Agent 1"
                },
                "max_similarity": 0.85
            },
            {
                "agent": {
                    "uuid": str(uuid4()),
                    "agent_id": "agent2",
                    "name": "Agent 2"
                },
                "max_similarity": 0.7
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_agents
        
        # Act
        result = graph_utils.find_agents_by_context_similarity(
            context_id, 
            min_similarity=0.5,
            limit=10
        )
        
        # Assert
        assert len(result) == 2
        assert result[0]["agent_id"] == "agent1"
        assert result[0]["similarity"] == 0.85
        assert result[1]["agent_id"] == "agent2"
        assert result[1]["similarity"] == 0.7
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (ctx1:Context {context_id: $context_id})" in query
        assert "apoc.text.similarity" in query
        assert params["context_id"] == str(context_id)
        assert params["min_similarity"] == 0.5
        assert params["limit"] == 10

    def test_find_agents_by_context_similarity_query_error(self, graph_utils: GraphTraversalUtils, 
                                                        mock_neo4j_service: Mock) -> None:
        """Test handling errors when querying for agent similarity."""
        # Arrange
        context_id = uuid4()
        mock_neo4j_service.run_query.side_effect = Exception("APOC procedures not available")
        
        # Act
        result = graph_utils.find_agents_by_context_similarity(context_id)
        
        # Assert
        assert result == []

    def test_find_agents_by_context_similarity_invalid_threshold(self, graph_utils: GraphTraversalUtils) -> None:
        """Test finding agents with an invalid similarity threshold."""
        # Arrange
        context_id = uuid4()
        
        # Act/Assert
        with pytest.raises(ValueError, match="Minimum similarity must be between 0.0 and 1.0"):
            graph_utils.find_agents_by_context_similarity(context_id, min_similarity=1.5)

    def test_analyze_conversation_context_graph(self, graph_utils: GraphTraversalUtils, 
                                              mock_neo4j_service: Mock) -> None:
        """Test analyzing the context graph for a conversation."""
        # Arrange
        conversation_id = uuid4()
        
        # Mock the three query results
        mock_neo4j_service.run_query.side_effect = [
            [{"context_count": 5, "unique_context_types": 3}],  # Context query
            [{"message_count": 20, "total_content_length": 2000}],  # Message query
            [{"participant_count": 2, "participant_ids": ["agent1", "agent2"]}]  # Participant query
        ]
        
        # Act
        result = graph_utils.analyze_conversation_context_graph(conversation_id)
        
        # Assert
        assert result["conversation_id"] == str(conversation_id)
        assert result["context_metrics"]["context_count"] == 5
        assert result["context_metrics"]["unique_context_types"] == 3
        assert result["message_metrics"]["message_count"] == 20
        assert result["message_metrics"]["total_content_length"] == 2000
        assert result["message_metrics"]["avg_message_length"] == 100  # 2000 / 20
        assert result["participant_metrics"]["participant_count"] == 2
        assert result["participant_metrics"]["participant_ids"] == ["agent1", "agent2"]
        
        # Verify query execution
        assert mock_neo4j_service.run_query.call_count == 3
