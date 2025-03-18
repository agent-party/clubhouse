"""
Unit tests for the Neo4j context utilities.

This module contains tests for the ContextUtils class, which provides
methods for managing conversation contexts in the Neo4j database.
"""

import json
import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.utils.context_utils import ContextUtils, ContextType


@pytest.fixture
def mock_neo4j_service() -> Mock:
    """Create a mock Neo4j service for testing."""
    service = Mock(spec=Neo4jServiceProtocol)
    return service


@pytest.fixture
def context_utils(mock_neo4j_service: Mock) -> ContextUtils:
    """Create a ContextUtils instance with a mock Neo4j service."""
    return ContextUtils(mock_neo4j_service)


class TestContextUtils:
    """Tests for the ContextUtils class."""

    def test_create_context(self, context_utils: ContextUtils, 
                          mock_neo4j_service: Mock) -> None:
        """Test creating a context node."""
        # Arrange
        expected_uuid = UUID('12345678-1234-5678-1234-567812345678')
        
        # Act
        with patch('clubhouse.services.neo4j.utils.context_utils.uuid4', return_value=expected_uuid):
            context_id = context_utils.create_context(
                context_type=ContextType.DOCUMENT,
                content="Test document content",
                metadata={"source": "test"},
                source_uri="http://example.com/doc",
                tags=["test", "document"]
            )
        
        # Assert
        assert context_id == expected_uuid
        mock_neo4j_service.create_node.assert_called_once()
        
        # Check the arguments using call_args - a tuple of (args, kwargs)
        call_args = mock_neo4j_service.create_node.call_args
        args, kwargs = call_args
        
        # Unpack and verify arguments
        assert kwargs.get('labels') == ["Context"]
        properties = kwargs.get('properties')
        assert properties["context_id"] == '12345678-1234-5678-1234-567812345678'
        assert properties["type"] == ContextType.DOCUMENT
        assert properties["content"] == "Test document content"
        assert properties["source_uri"] == "http://example.com/doc"
        assert json.loads(properties["metadata"]) == {"source": "test"}
        assert json.loads(properties["tags"]) == ["test", "document"]

    def test_create_context_invalid_type(self, context_utils: ContextUtils) -> None:
        """Test creating a context with an invalid type."""
        # Act/Assert
        with pytest.raises(ValueError, match="Invalid context type"):
            context_utils.create_context(
                context_type="invalid_type",
                content="Test content"
            )

    def test_create_context_empty_content(self, context_utils: ContextUtils) -> None:
        """Test creating a context with empty content."""
        # Act/Assert
        with pytest.raises(ValueError, match="Context content cannot be empty"):
            context_utils.create_context(
                context_type=ContextType.DOCUMENT,
                content=""
            )

    def test_link_context_to_conversation(self, context_utils: ContextUtils, 
                                        mock_neo4j_service: Mock) -> None:
        """Test linking a context to a conversation."""
        # Arrange
        context_id = uuid4()
        conversation_id = uuid4()
        mock_neo4j_service.run_query.return_value = [{"r": {}}]
        
        # Act
        context_utils.link_context_to_conversation(
            context_id=context_id,
            conversation_id=conversation_id,
            relevance=0.8,
            relationship_metadata={"added_by": "test"}
        )
        
        # Assert
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (c:Conversation" in query
        assert "MATCH (ctx:Context" in query
        assert "MERGE (c)-[r:HAS_CONTEXT]->(ctx)" in query
        
        assert params["conversation_id"] == str(conversation_id)
        assert params["context_id"] == str(context_id)
        assert params["relevance"] == 0.8
        assert json.loads(params["metadata"]) == {"added_by": "test"}

    def test_link_context_invalid_relevance(self, context_utils: ContextUtils) -> None:
        """Test linking a context with invalid relevance."""
        # Arrange
        context_id = uuid4()
        conversation_id = uuid4()
        
        # Act/Assert
        with pytest.raises(ValueError, match="Relevance must be between 0.0 and 1.0"):
            context_utils.link_context_to_conversation(
                context_id=context_id,
                conversation_id=conversation_id,
                relevance=1.5
            )

    def test_update_context_relevance(self, context_utils: ContextUtils, 
                                    mock_neo4j_service: Mock) -> None:
        """Test updating context relevance."""
        # Arrange
        context_id = uuid4()
        conversation_id = uuid4()
        mock_neo4j_service.run_query.return_value = [{"r": {}}]
        
        # Act
        result = context_utils.update_context_relevance(
            context_id=context_id,
            conversation_id=conversation_id,
            relevance=0.9
        )
        
        # Assert
        assert result is True
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (c:Conversation" in query
        assert "-[r:HAS_CONTEXT]->" in query
        assert "SET r.relevance = $relevance" in query
        
        assert params["conversation_id"] == str(conversation_id)
        assert params["context_id"] == str(context_id)
        assert params["relevance"] == 0.9

    def test_update_context_relevance_not_found(self, context_utils: ContextUtils, 
                                              mock_neo4j_service: Mock) -> None:
        """Test updating relevance for a non-existent relationship."""
        # Arrange
        context_id = uuid4()
        conversation_id = uuid4()
        mock_neo4j_service.run_query.return_value = []  # No relationship found
        
        # Act
        result = context_utils.update_context_relevance(
            context_id=context_id,
            conversation_id=conversation_id,
            relevance=0.9
        )
        
        # Assert
        assert result is False

    def test_update_context_relevance_invalid(self, context_utils: ContextUtils) -> None:
        """Test updating relevance with an invalid value."""
        # Arrange
        context_id = uuid4()
        conversation_id = uuid4()
        
        # Act/Assert
        with pytest.raises(ValueError, match="Relevance must be between 0.0 and 1.0"):
            context_utils.update_context_relevance(
                context_id=context_id,
                conversation_id=conversation_id,
                relevance=-0.1
            )

    def test_get_conversation_contexts(self, context_utils: ContextUtils, 
                                     mock_neo4j_service: Mock) -> None:
        """Test getting contexts for a conversation."""
        # Arrange
        conversation_id = uuid4()
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        mock_neo4j_service.run_query.return_value = [
            {
                "ctx": {
                    "context_id": str(context_id),
                    "type": ContextType.DOCUMENT,
                    "content": "Test content",
                    "created_at": created_at,
                    "source_uri": "http://example.com",
                    "tags": json.dumps(["test", "document"]),
                    "metadata": json.dumps({"source": "test"})
                },
                "relevance": 0.8
            }
        ]
        
        # Act
        contexts = context_utils.get_conversation_contexts(
            conversation_id=conversation_id,
            min_relevance=0.5,
            context_types=[ContextType.DOCUMENT],
            limit=10,
            include_metadata=True
        )
        
        # Assert
        assert len(contexts) == 1
        assert contexts[0]["context_id"] == context_id
        assert contexts[0]["type"] == ContextType.DOCUMENT
        assert contexts[0]["content"] == "Test content"
        assert contexts[0]["created_at"] == created_at
        assert contexts[0]["relevance"] == 0.8
        assert contexts[0]["source_uri"] == "http://example.com"
        assert contexts[0]["tags"] == ["test", "document"]
        assert contexts[0]["metadata"] == {"source": "test"}
        
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (c:Conversation" in query
        assert "WHERE r.relevance >=" in query
        assert "AND ctx.type IN" in query
        assert "ORDER BY relevance DESC" in query
        assert "LIMIT" in query
        
        assert params["conversation_id"] == str(conversation_id)
        assert params["min_relevance"] == 0.5
        assert params["context_types"] == [ContextType.DOCUMENT]
        assert params["limit"] == 10

    def test_get_conversation_contexts_invalid_relevance(self, context_utils: ContextUtils) -> None:
        """Test getting contexts with invalid relevance."""
        # Arrange
        conversation_id = uuid4()
        
        # Act/Assert
        with pytest.raises(ValueError, match="Minimum relevance must be between 0.0 and 1.0"):
            context_utils.get_conversation_contexts(
                conversation_id=conversation_id,
                min_relevance=2.0
            )

    def test_get_conversation_contexts_json_error(self, context_utils: ContextUtils, 
                                               mock_neo4j_service: Mock) -> None:
        """Test handling JSON decode errors in context metadata."""
        # Arrange
        conversation_id = uuid4()
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        # Invalid JSON in metadata and tags
        mock_neo4j_service.run_query.return_value = [
            {
                "ctx": {
                    "context_id": str(context_id),
                    "type": ContextType.DOCUMENT,
                    "content": "Test content",
                    "created_at": created_at,
                    "metadata": "{invalid_json",
                    "tags": "{invalid_json"
                },
                "relevance": 0.8
            }
        ]
        
        # Act
        contexts = context_utils.get_conversation_contexts(
            conversation_id=conversation_id,
            include_metadata=True
        )
        
        # Assert
        assert len(contexts) == 1
        assert contexts[0]["context_id"] == context_id
        assert contexts[0]["tags"] == []  # Default empty list on JSON error
        # Metadata not included when include_metadata=False

    def test_get_shared_contexts(self, context_utils: ContextUtils, 
                               mock_neo4j_service: Mock) -> None:
        """Test finding shared contexts."""
        # Arrange
        entity_ids = [uuid4(), uuid4()]
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        mock_neo4j_service.run_query.return_value = [
            {
                "ctx": {
                    "context_id": str(context_id),
                    "type": ContextType.DOCUMENT,
                    "content": "Shared content",
                    "created_at": created_at,
                    "metadata": json.dumps({"source": "test"})
                },
                "occurrence_count": 2,
                "avg_relevance": 0.75,
                "entity_uuids": [str(entity_ids[0]), str(entity_ids[1])]
            }
        ]
        
        # Act
        shared_contexts = context_utils.get_shared_contexts(
            entity_ids=entity_ids,
            entity_label="Conversation",
            min_relevance=0.5,
            context_types=[ContextType.DOCUMENT]
        )
        
        # Assert
        assert len(shared_contexts) == 1
        assert shared_contexts[0]["context_id"] == context_id
        assert shared_contexts[0]["type"] == ContextType.DOCUMENT
        assert shared_contexts[0]["content"] == "Shared content"
        assert shared_contexts[0]["occurrence_count"] == 2
        assert shared_contexts[0]["average_relevance"] == 0.75
        assert shared_contexts[0]["entity_ids"] == entity_ids
        assert shared_contexts[0]["metadata"] == {"source": "test"}
        
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (e:Conversation)-[r:HAS_CONTEXT]->(ctx:Context)" in query
        assert "WHERE e.conversation_id IN" in query
        assert "AND r.relevance >=" in query
        assert "WHERE occurrence_count > 1" in query
        
        assert params["entity_ids"] == [str(eid) for eid in entity_ids]
        assert params["min_relevance"] == 0.5
        assert params["context_types"] == [ContextType.DOCUMENT]

    def test_get_shared_contexts_empty(self, context_utils: ContextUtils, 
                                     mock_neo4j_service: Mock) -> None:
        """Test finding shared contexts with empty input."""
        # Act
        shared_contexts = context_utils.get_shared_contexts(entity_ids=[])
        
        # Assert
        assert shared_contexts == []
        mock_neo4j_service.run_query.assert_not_called()

    def test_get_shared_contexts_agent_label(self, context_utils: ContextUtils, 
                                          mock_neo4j_service: Mock) -> None:
        """Test finding shared contexts with Agent label."""
        # Arrange
        entity_ids = [uuid4()]
        mock_neo4j_service.run_query.return_value = []
        
        # Act
        context_utils.get_shared_contexts(
            entity_ids=entity_ids,
            entity_label="Agent"
        )
        
        # Assert
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (e:Agent)-[r:HAS_CONTEXT]->(ctx:Context)" in query
        assert "WHERE e.uuid IN" in query  # Should use uuid field for Agent label

    def test_find_similar_contexts(self, context_utils: ContextUtils, 
                                 mock_neo4j_service: Mock) -> None:
        """Test finding similar contexts."""
        # Arrange
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        mock_neo4j_service.run_query.return_value = [
            {
                "ctx": {
                    "context_id": str(context_id),
                    "type": ContextType.DOCUMENT,
                    "content": "Similar content",
                    "created_at": created_at
                },
                "similarity": 0.8
            }
        ]
        
        # Act
        similar_contexts = context_utils.find_similar_contexts(
            content="Test content",
            context_types=[ContextType.DOCUMENT],
            min_similarity=0.7,
            limit=5
        )
        
        # Assert
        assert len(similar_contexts) == 1
        assert similar_contexts[0]["context_id"] == context_id
        assert similar_contexts[0]["type"] == ContextType.DOCUMENT
        assert similar_contexts[0]["content"] == "Similar content"
        assert similar_contexts[0]["similarity"] == 0.8
        
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "apoc.text.similarity" in query
        assert "WHERE similarity >=" in query
        assert "WHERE ctx.type IN" in query
        
        assert params["content"] == "Test content"
        assert params["min_similarity"] == 0.7
        assert params["context_types"] == [ContextType.DOCUMENT]
        assert params["limit"] == 5

    def test_find_similar_contexts_error(self, context_utils: ContextUtils, 
                                      mock_neo4j_service: Mock) -> None:
        """Test handling errors in find_similar_contexts."""
        # Arrange
        mock_neo4j_service.run_query.side_effect = Exception("APOC not available")
        
        # Act
        similar_contexts = context_utils.find_similar_contexts(
            content="Test content"
        )
        
        # Assert
        assert similar_contexts == []  # Should return empty list on error

    def test_find_similar_contexts_invalid_similarity(self, context_utils: ContextUtils) -> None:
        """Test finding similar contexts with invalid similarity."""
        # Act/Assert
        with pytest.raises(ValueError, match="Minimum similarity must be between 0.0 and 1.0"):
            context_utils.find_similar_contexts(
                content="Test content",
                min_similarity=1.5
            )

    def test_associate_context_with_message(self, context_utils: ContextUtils, 
                                         mock_neo4j_service: Mock) -> None:
        """Test associating a context with a message."""
        # Arrange
        context_id = uuid4()
        message_id = uuid4()
        
        # Act
        context_utils.associate_context_with_message(
            context_id=context_id,
            message_id=message_id,
            relationship_type="REFERENCED_IN",
            properties={"position": 1}
        )
        
        # Assert
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (m:Message" in query
        assert "MATCH (ctx:Context" in query
        assert "MERGE (ctx)-[r:REFERENCED_IN]->(m)" in query
        
        assert params["message_id"] == str(message_id)
        assert params["context_id"] == str(context_id)
        assert params["properties"]["position"] == 1
        assert "created_at" in params["properties"]

    def test_calculate_conversation_context_similarity(self, context_utils: ContextUtils, 
                                                    mock_neo4j_service: Mock) -> None:
        """Test calculating context similarity between conversations."""
        # Arrange
        conversation_id1 = uuid4()
        conversation_id2 = uuid4()
        
        mock_neo4j_service.run_query.return_value = [{"similarity": 0.6}]
        
        # Act
        similarity = context_utils.calculate_conversation_context_similarity(
            conversation_id1=conversation_id1,
            conversation_id2=conversation_id2
        )
        
        # Assert
        assert similarity == 0.6
        
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        
        assert "MATCH (c1:Conversation" in query
        assert "MATCH (c2:Conversation" in query
        assert "similarity" in query
        
        assert params["conversation_id1"] == str(conversation_id1)
        assert params["conversation_id2"] == str(conversation_id2)

    def test_calculate_conversation_context_similarity_no_result(self, context_utils: ContextUtils, 
                                                             mock_neo4j_service: Mock) -> None:
        """Test similarity calculation with no result."""
        # Arrange
        conversation_id1 = uuid4()
        conversation_id2 = uuid4()
        
        mock_neo4j_service.run_query.return_value = []  # Empty result
        
        # Act
        similarity = context_utils.calculate_conversation_context_similarity(
            conversation_id1=conversation_id1,
            conversation_id2=conversation_id2
        )
        
        # Assert
        assert similarity == 0.0  # Default when no result

    def test_build_context_summary(self, context_utils: ContextUtils) -> None:
        """Test building a context summary."""
        # Arrange
        conversation_id = uuid4()
        context_id1 = uuid4()
        context_id2 = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        # Mock the get_conversation_contexts method
        context_utils.get_conversation_contexts = Mock(return_value=[
            {
                "context_id": context_id1,
                "type": ContextType.DOCUMENT,
                "content": "Document content",
                "created_at": created_at,
                "relevance": 0.9,
                "metadata": {"source": "test"},
                "tags": ["document", "test"]
            },
            {
                "context_id": context_id2,
                "type": ContextType.MEMORY,
                "content": "Memory content",
                "created_at": created_at,
                "relevance": 0.7,
                "metadata": {"origin": "user"},
                "tags": ["memory"]
            }
        ])
        
        # Act
        summary = context_utils.build_context_summary(
            conversation_id=conversation_id,
            max_contexts=10,
            include_content=True,
            max_content_length=100
        )
        
        # Assert
        assert summary["conversation_id"] == str(conversation_id)
        assert summary["total_contexts"] == 2
        assert set(summary["context_types"]) == {ContextType.DOCUMENT, ContextType.MEMORY}
        
        # Check document context
        doc_contexts = summary["contexts_by_type"][ContextType.DOCUMENT]
        assert len(doc_contexts) == 1
        assert doc_contexts[0]["context_id"] == str(context_id1)
        assert doc_contexts[0]["relevance"] == 0.9
        assert doc_contexts[0]["content"] == "Document content"
        assert doc_contexts[0]["metadata"] == {"source": "test"}
        assert doc_contexts[0]["tags"] == ["document", "test"]
        
        # Check memory context
        memory_contexts = summary["contexts_by_type"][ContextType.MEMORY]
        assert len(memory_contexts) == 1
        assert memory_contexts[0]["context_id"] == str(context_id2)
        assert memory_contexts[0]["relevance"] == 0.7
        assert memory_contexts[0]["content"] == "Memory content"
        assert memory_contexts[0]["metadata"] == {"origin": "user"}
        assert memory_contexts[0]["tags"] == ["memory"]
        
        # Verify method call
        context_utils.get_conversation_contexts.assert_called_once_with(
            conversation_id=conversation_id,
            min_relevance=0.1,
            limit=10,
            include_metadata=True
        )

    def test_build_context_summary_content_truncation(self, context_utils: ContextUtils) -> None:
        """Test content truncation in context summary."""
        # Arrange
        conversation_id = uuid4()
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        # Mock the get_conversation_contexts method with long content
        context_utils.get_conversation_contexts = Mock(return_value=[
            {
                "context_id": context_id,
                "type": ContextType.DOCUMENT,
                "content": "X" * 200,  # Long content
                "created_at": created_at,
                "relevance": 0.9
            }
        ])
        
        # Act
        summary = context_utils.build_context_summary(
            conversation_id=conversation_id,
            max_content_length=50  # Shorter than content
        )
        
        # Assert
        doc_contexts = summary["contexts_by_type"][ContextType.DOCUMENT]
        assert len(doc_contexts[0]["content"]) == 53  # 50 chars + "..."
        assert doc_contexts[0]["content"].endswith("...")

    def test_build_context_summary_no_content(self, context_utils: ContextUtils) -> None:
        """Test context summary without content."""
        # Arrange
        conversation_id = uuid4()
        context_id = uuid4()
        created_at = datetime.utcnow().isoformat()
        
        # Mock the get_conversation_contexts method
        context_utils.get_conversation_contexts = Mock(return_value=[
            {
                "context_id": context_id,
                "type": ContextType.DOCUMENT,
                "content": "Document content",
                "created_at": created_at,
                "relevance": 0.9
            }
        ])
        
        # Act
        summary = context_utils.build_context_summary(
            conversation_id=conversation_id,
            include_content=False
        )
        
        # Assert
        doc_contexts = summary["contexts_by_type"][ContextType.DOCUMENT]
        assert "content" not in doc_contexts[0]
