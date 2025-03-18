"""
Unit tests for the ConversationRepository.

This module contains tests for the ConversationRepository class,
which manages conversations and their related entities in Neo4j.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID, uuid4

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.repositories.conversation import ConversationRepository


@pytest.fixture
def mock_neo4j_service() -> Mock:
    """Create a mock Neo4j service for testing."""
    service = Mock(spec=Neo4jServiceProtocol)
    return service


@pytest.fixture
def conversation_repo(mock_neo4j_service: Mock) -> ConversationRepository:
    """Create a ConversationRepository with a mock Neo4j service."""
    return ConversationRepository(mock_neo4j_service)


class TestConversationRepository:
    """Tests for the ConversationRepository class."""

    def test_create_conversation(self, conversation_repo: ConversationRepository, 
                                mock_neo4j_service: Mock) -> None:
        """Test creating a conversation."""
        # Arrange
        mock_uuid = uuid4()
        mock_neo4j_service.create_node.return_value = mock_uuid
        title = "Test Conversation"
        description = "A test conversation"
        metadata = {"key": "value"}
        
        # Act
        result = conversation_repo.create_conversation(title, description, metadata)
        
        # Assert
        assert result == mock_uuid
        mock_neo4j_service.create_node.assert_called_once()
        
        # Verify properties passed to create_node
        args, kwargs = mock_neo4j_service.create_node.call_args
        assert args[0] == "Conversation"
        properties = args[1]
        assert properties["title"] == title
        assert properties["description"] == description
        assert json.loads(properties["metadata"]) == metadata
        assert properties["status"] == "ACTIVE"
        assert "created_at" in properties
        assert "updated_at" in properties

    def test_get_conversation(self, conversation_repo: ConversationRepository, 
                             mock_neo4j_service: Mock) -> None:
        """Test retrieving a conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_conversation = {
            "conversation_id": str(conversation_id),
            "title": "Test Conversation",
            "description": "A test conversation",
            "status": "ACTIVE",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": json.dumps({"key": "value"})
        }
        
        mock_neo4j_service.run_query.return_value = [
            {"conversation": mock_conversation}
        ]
        
        # Act
        result = conversation_repo.get_conversation(conversation_id)
        
        # Assert
        assert result is not None
        assert result["title"] == mock_conversation["title"]
        assert result["description"] == mock_conversation["description"]
        assert result["metadata"] == {"key": "value"}  # Should be parsed from JSON
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation" in query
        assert params["conversation_id"] == str(conversation_id)

    def test_get_conversation_not_found(self, conversation_repo: ConversationRepository, 
                                       mock_neo4j_service: Mock) -> None:
        """Test retrieving a non-existent conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_neo4j_service.run_query.return_value = []
        
        # Act
        result = conversation_repo.get_conversation(conversation_id)
        
        # Assert
        assert result is None

    def test_update_conversation(self, conversation_repo: ConversationRepository, 
                                mock_neo4j_service: Mock) -> None:
        """Test updating a conversation."""
        # Arrange
        conversation_id = uuid4()
        properties = {
            "title": "Updated Title",
            "metadata": {"new": "value"}
        }
        
        mock_neo4j_service.run_query.return_value = [{"updated": True}]
        
        # Act
        result = conversation_repo.update_conversation(conversation_id, properties)
        
        # Assert
        assert result is True
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation" in query
        assert params["conversation_id"] == str(conversation_id)
        assert params["properties"]["title"] == "Updated Title"
        assert isinstance(params["properties"]["metadata"], str)
        assert json.loads(params["properties"]["metadata"]) == {"new": "value"}
        assert "updated_at" in params["properties"]

    def test_add_message_to_conversation(self, conversation_repo: ConversationRepository, 
                                        mock_neo4j_service: Mock) -> None:
        """Test adding a message to a conversation."""
        # Arrange
        conversation_id = uuid4()
        message_id = uuid4()
        content = "Hello, world!"
        role = "user"
        metadata = {"client": "web"}
        
        # Mock the count query to return message count
        mock_neo4j_service.run_query.side_effect = [
            [{"message_count": 5}],  # First call returns message count
            [{"message_id": str(message_id)}]  # Second call returns message ID
        ]
        
        # Act
        result = conversation_repo.add_message_to_conversation(
            conversation_id, content, role, metadata
        )
        
        # Assert
        assert result is not None
        
        # Verify query executions (should be called twice)
        assert mock_neo4j_service.run_query.call_count == 2
        
        # Check the second call (create message query)
        create_query, create_params = mock_neo4j_service.run_query.call_args[0]
        assert "CREATE (m:Message" in create_query
        assert create_params["conversation_id"] == str(conversation_id)
        assert create_params["message_properties"]["content"] == content
        assert create_params["message_properties"]["role"] == role
        assert create_params["order"] == 5  # Should match the mocked message count

    def test_add_message_conversation_not_found(self, conversation_repo: ConversationRepository, 
                                              mock_neo4j_service: Mock) -> None:
        """Test adding a message to a non-existent conversation."""
        # Arrange
        conversation_id = uuid4()
        content = "Hello, world!"
        role = "user"
        
        # Mock the count query to return empty result
        mock_neo4j_service.run_query.return_value = []
        
        # Act/Assert
        with pytest.raises(ValueError, match=f"Conversation {conversation_id} not found"):
            conversation_repo.add_message_to_conversation(conversation_id, content, role)

    def test_get_conversation_messages(self, conversation_repo: ConversationRepository, 
                                      mock_neo4j_service: Mock) -> None:
        """Test retrieving messages for a conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_messages = [
            {
                "message": {
                    "message_id": str(uuid4()),
                    "content": "Message 1",
                    "role": "user",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": json.dumps({"client": "web"})
                }
            },
            {
                "message": {
                    "message_id": str(uuid4()),
                    "content": "Message 2",
                    "role": "assistant",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": json.dumps({"model": "gpt-4"})
                }
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_messages
        
        # Act
        result = conversation_repo.get_conversation_messages(conversation_id, limit=10)
        
        # Assert
        assert len(result) == 2
        assert result[0]["content"] == "Message 1"
        assert result[0]["role"] == "user"
        assert result[0]["metadata"] == {"client": "web"}
        assert result[1]["content"] == "Message 2"
        assert result[1]["role"] == "assistant"
        assert result[1]["metadata"] == {"model": "gpt-4"}
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation" in query
        assert "ORDER BY r.order ASC" in query
        assert params["conversation_id"] == str(conversation_id)
        assert params["limit"] == 10
        assert params["offset"] == 0

    def test_add_context_to_conversation(self, conversation_repo: ConversationRepository, 
                                        mock_neo4j_service: Mock) -> None:
        """Test adding a context to a conversation."""
        # Arrange
        conversation_id = uuid4()
        context_id = uuid4()
        context_type = "document"
        content = "Document content"
        metadata = {"source": "file.txt"}
        relevance = 0.85
        
        mock_neo4j_service.run_query.return_value = [{"context_id": str(context_id)}]
        
        # Act
        result = conversation_repo.add_context_to_conversation(
            conversation_id, context_type, content, metadata, relevance
        )
        
        # Assert
        assert result is not None
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "CREATE (ctx:Context" in query
        assert params["conversation_id"] == str(conversation_id)
        assert params["context_properties"]["type"] == context_type
        assert params["context_properties"]["content"] == content
        assert json.loads(params["context_properties"]["metadata"]) == metadata
        assert params["relevance"] == 0.85

    def test_add_context_invalid_relevance(self, conversation_repo: ConversationRepository) -> None:
        """Test adding a context with invalid relevance score."""
        # Arrange
        conversation_id = uuid4()
        context_type = "document"
        content = "Document content"
        
        # Act/Assert
        with pytest.raises(ValueError, match="Relevance must be between 0.0 and 1.0"):
            conversation_repo.add_context_to_conversation(
                conversation_id, context_type, content, relevance=1.5
            )

    def test_get_conversation_contexts(self, conversation_repo: ConversationRepository, 
                                      mock_neo4j_service: Mock) -> None:
        """Test retrieving contexts for a conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_contexts = [
            {
                "context": {
                    "context_id": str(uuid4()),
                    "type": "document",
                    "content": "Document content",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": json.dumps({"source": "file.txt"})
                },
                "relevance": 0.9
            },
            {
                "context": {
                    "context_id": str(uuid4()),
                    "type": "memory",
                    "content": "Memory content",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": json.dumps({"source": "memory_store"})
                },
                "relevance": 0.7
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_contexts
        
        # Act
        result = conversation_repo.get_conversation_contexts(
            conversation_id, 
            context_types=["document", "memory"],
            min_relevance=0.5
        )
        
        # Assert
        assert len(result) == 2
        assert result[0]["type"] == "document"
        assert result[0]["content"] == "Document content"
        assert result[0]["metadata"] == {"source": "file.txt"}
        assert result[0]["relevance"] == 0.9
        assert result[1]["type"] == "memory"
        assert result[1]["relevance"] == 0.7
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation" in query
        assert "WHERE r.relevance >=" in query
        assert "AND ctx.type IN" in query
        assert params["conversation_id"] == str(conversation_id)
        assert params["min_relevance"] == 0.5
        assert params["context_types"] == ["document", "memory"]

    def test_delete_conversation(self, conversation_repo: ConversationRepository, 
                               mock_neo4j_service: Mock) -> None:
        """Test deleting a conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_neo4j_service.run_query.return_value = [{"deleted": True}]
        
        # Act
        result = conversation_repo.delete_conversation(conversation_id)
        
        # Assert
        assert result is True
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation" in query
        assert "DETACH DELETE" in query
        assert params["conversation_id"] == str(conversation_id)

    def test_delete_conversation_not_found(self, conversation_repo: ConversationRepository, 
                                         mock_neo4j_service: Mock) -> None:
        """Test deleting a non-existent conversation."""
        # Arrange
        conversation_id = uuid4()
        mock_neo4j_service.run_query.return_value = [{"deleted": False}]
        
        # Act
        result = conversation_repo.delete_conversation(conversation_id)
        
        # Assert
        assert result is False

    def test_get_recent_conversations(self, conversation_repo: ConversationRepository, 
                                    mock_neo4j_service: Mock) -> None:
        """Test retrieving recent conversations."""
        # Arrange
        mock_conversations = [
            {
                "conversation": {
                    "conversation_id": str(uuid4()),
                    "title": "First Conversation",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": json.dumps({"key": "value1"})
                }
            },
            {
                "conversation": {
                    "conversation_id": str(uuid4()),
                    "title": "Second Conversation",
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": json.dumps({"key": "value2"})
                }
            }
        ]
        
        mock_neo4j_service.run_query.return_value = mock_conversations
        
        # Act
        result = conversation_repo.get_recent_conversations(limit=5, offset=0)
        
        # Assert
        assert len(result) == 2
        assert result[0]["title"] == "First Conversation"
        assert result[0]["metadata"] == {"key": "value1"}
        assert result[1]["title"] == "Second Conversation"
        assert result[1]["metadata"] == {"key": "value2"}
        
        # Verify query execution
        mock_neo4j_service.run_query.assert_called_once()
        query, params = mock_neo4j_service.run_query.call_args[0]
        assert "MATCH (c:Conversation)" in query
        assert "ORDER BY c.created_at DESC" in query
        assert params["limit"] == 5
        assert params["offset"] == 0
