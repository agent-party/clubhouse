"""
Unit tests for Neo4j entity relationship repositories.

This module contains tests for the relationship repositories that handle
core entity relationships including Agent-Conversation, Message-Context,
and Agent-Knowledge relationships in the Neo4j graph database.
"""

import pytest
import uuid
from unittest.mock import MagicMock, patch, call
from datetime import datetime
from typing import Dict, Any, List

from neo4j.exceptions import Neo4jError, ClientError

from clubhouse.services.neo4j.repositories.entity_relationships import EntityRelationshipRepository


@pytest.fixture
def mock_neo4j_service():
    """Create a mock Neo4j service."""
    mock_service = MagicMock()
    
    # Mock session and transaction
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_service.session.return_value = mock_session
    mock_session.begin_transaction.return_value = mock_tx
    mock_session.__enter__.return_value = mock_session
    mock_tx.__enter__.return_value = mock_tx
    
    # Set up run method to return useful results
    mock_result = MagicMock()
    mock_record = MagicMock()
    mock_result.single.return_value = mock_record
    mock_tx.run.return_value = mock_result
    
    return mock_service


@pytest.fixture
def entity_repo(mock_neo4j_service):
    """Create an EntityRelationshipRepository with a mock Neo4j service."""
    return EntityRelationshipRepository(mock_neo4j_service)


class TestEntityRelationshipRepository:
    """Test suite for the EntityRelationshipRepository."""
    
    def test_link_agent_to_conversation(self, entity_repo, mock_neo4j_service):
        """Test linking an agent to a conversation."""
        # Test data
        agent_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        relationship_type = "PARTICIPATES_IN"
        properties = {"role": "primary", "joined_at": datetime.now().isoformat()}
        
        # Call method
        result = entity_repo.link_agent_to_conversation(
            agent_id=agent_id,
            conversation_id=conversation_id,
            relationship_type=relationship_type,
            properties=properties
        )
        
        # Assert transaction was used
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_tx.run.assert_called_once()
        
        # Check query contains correct relationship information
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (a:Agent), (c:Conversation)" in query
        assert "WHERE a.id = $agent_id AND c.id = $conversation_id" in query
        assert f"CREATE (a)-[r:{relationship_type} $properties]->(c)" in query
        assert params["agent_id"] == agent_id
        assert params["conversation_id"] == conversation_id
        assert params["properties"] == properties
        
        # Ensure result is as expected
        assert result is True
    
    def test_link_message_to_context(self, entity_repo, mock_neo4j_service):
        """Test linking a message to context."""
        # Test data
        message_id = str(uuid.uuid4())
        context_id = str(uuid.uuid4())
        relationship_type = "HAS_CONTEXT"
        properties = {"relevance_score": 0.85, "created_at": datetime.now().isoformat()}
        
        # Call method
        result = entity_repo.link_message_to_context(
            message_id=message_id,
            context_id=context_id,
            relationship_type=relationship_type,
            properties=properties
        )
        
        # Assert transaction was used
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_tx.run.assert_called_once()
        
        # Check query contains correct relationship information
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (m:Message), (ctx:Context)" in query
        assert "WHERE m.id = $message_id AND ctx.id = $context_id" in query
        assert f"CREATE (m)-[r:{relationship_type} $properties]->(ctx)" in query
        assert params["message_id"] == message_id
        assert params["context_id"] == context_id
        assert params["properties"] == properties
        
        # Ensure result is as expected
        assert result is True
    
    def test_link_agent_to_knowledge(self, entity_repo, mock_neo4j_service):
        """Test linking an agent to knowledge."""
        # Test data
        agent_id = str(uuid.uuid4())
        knowledge_id = str(uuid.uuid4()) 
        relationship_type = "KNOWS_ABOUT"
        properties = {"confidence": 0.92, "last_updated": datetime.now().isoformat()}
        
        # Call method
        result = entity_repo.link_agent_to_knowledge(
            agent_id=agent_id,
            knowledge_id=knowledge_id,
            relationship_type=relationship_type,
            properties=properties
        )
        
        # Assert transaction was used
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_tx.run.assert_called_once()
        
        # Check query contains correct relationship information
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (a:Agent), (k:Knowledge)" in query
        assert "WHERE a.id = $agent_id AND k.id = $knowledge_id" in query
        assert f"CREATE (a)-[r:{relationship_type} $properties]->(k)" in query
        assert params["agent_id"] == agent_id
        assert params["knowledge_id"] == knowledge_id
        assert params["properties"] == properties
        
        # Ensure result is as expected
        assert result is True
    
    def test_get_conversation_agents(self, entity_repo, mock_neo4j_service):
        """Test getting all agents linked to a conversation."""
        # Test data
        conversation_id = str(uuid.uuid4())
        
        # Mock result
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_result = mock_tx.run.return_value
        
        # Define expected agent records
        expected_agents = [
            {"id": str(uuid.uuid4()), "name": "Agent 1", "role": "primary"},
            {"id": str(uuid.uuid4()), "name": "Agent 2", "role": "assistant"}
        ]
        
        # Configure mock to return agents
        mock_record1 = MagicMock()
        mock_record2 = MagicMock()
        mock_record1["agent"] = expected_agents[0]
        mock_record2["agent"] = expected_agents[1]
        mock_result.data.return_value = [
            {"agent": expected_agents[0], "relationship": {"role": "primary"}}, 
            {"agent": expected_agents[1], "relationship": {"role": "assistant"}}
        ]
        
        # Call method
        result = entity_repo.get_conversation_agents(conversation_id)
        
        # Assert transaction was used
        mock_tx.run.assert_called_once()
        
        # Check query is correct
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (a:Agent)-[r]->(c:Conversation)" in query
        assert "WHERE c.id = $conversation_id" in query
        assert "RETURN a as agent, r as relationship" in query
        assert params["conversation_id"] == conversation_id
        
        # Ensure result is as expected
        assert len(result) == 2
        assert result[0]["agent"]["id"] == expected_agents[0]["id"]
        assert result[1]["agent"]["id"] == expected_agents[1]["id"]
        
    def test_get_message_context(self, entity_repo, mock_neo4j_service):
        """Test getting all context items linked to a message."""
        # Test data
        message_id = str(uuid.uuid4())
        
        # Mock result
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_result = mock_tx.run.return_value
        
        # Define expected context records
        expected_contexts = [
            {"id": str(uuid.uuid4()), "type": "document", "content": "Document content"},
            {"id": str(uuid.uuid4()), "type": "web_search", "content": "Search results"}
        ]
        
        # Configure mock to return contexts
        mock_result.data.return_value = [
            {"context": expected_contexts[0], "relationship": {"relevance_score": 0.95}}, 
            {"context": expected_contexts[1], "relationship": {"relevance_score": 0.82}}
        ]
        
        # Call method
        result = entity_repo.get_message_context(message_id)
        
        # Assert transaction was used
        mock_tx.run.assert_called_once()
        
        # Check query is correct
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (m:Message)-[r]->(ctx:Context)" in query
        assert "WHERE m.id = $message_id" in query
        assert "RETURN ctx as context, r as relationship" in query
        assert params["message_id"] == message_id
        
        # Ensure result is as expected
        assert len(result) == 2
        assert result[0]["context"]["id"] == expected_contexts[0]["id"]
        assert result[1]["context"]["id"] == expected_contexts[1]["id"]
    
    def test_get_agent_knowledge(self, entity_repo, mock_neo4j_service):
        """Test getting all knowledge linked to an agent."""
        # Test data
        agent_id = str(uuid.uuid4())
        
        # Mock result
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_result = mock_tx.run.return_value
        
        # Define expected knowledge records
        expected_knowledge = [
            {"id": str(uuid.uuid4()), "type": "concept", "content": "AI capabilities"},
            {"id": str(uuid.uuid4()), "type": "fact", "content": "Python is a programming language"}
        ]
        
        # Configure mock to return knowledge
        mock_result.data.return_value = [
            {"knowledge": expected_knowledge[0], "relationship": {"confidence": 0.92}}, 
            {"knowledge": expected_knowledge[1], "relationship": {"confidence": 0.98}}
        ]
        
        # Call method
        result = entity_repo.get_agent_knowledge(agent_id)
        
        # Assert transaction was used
        mock_tx.run.assert_called_once()
        
        # Check query is correct
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (a:Agent)-[r]->(k:Knowledge)" in query
        assert "WHERE a.id = $agent_id" in query
        assert "RETURN k as knowledge, r as relationship" in query
        assert params["agent_id"] == agent_id
        
        # Ensure result is as expected
        assert len(result) == 2
        assert result[0]["knowledge"]["id"] == expected_knowledge[0]["id"]
        assert result[1]["knowledge"]["id"] == expected_knowledge[1]["id"]
        
    def test_delete_relationship(self, entity_repo, mock_neo4j_service):
        """Test deleting a relationship between two entities."""
        # Test data
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        relationship_type = "KNOWS_ABOUT"
        
        # Call method
        result = entity_repo.delete_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type
        )
        
        # Assert transaction was used
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_tx.run.assert_called_once()
        
        # Check query contains correct delete information
        call_args = mock_tx.run.call_args
        query = call_args[0][0]
        params = call_args[1]  # Use kwargs instead of positional args
        
        assert "MATCH (source)-[r:KNOWS_ABOUT]->(target)" in query
        assert "WHERE source.id = $source_id AND target.id = $target_id" in query
        assert "DELETE r" in query
        assert params["source_id"] == source_id
        assert params["target_id"] == target_id
        
        # Ensure result is as expected
        assert result is True
    
    def test_exception_handling(self, entity_repo, mock_neo4j_service):
        """Test exception handling for relationship operations."""
        # Configure mock to raise exception
        mock_session = mock_neo4j_service.session.return_value
        mock_tx = mock_session.begin_transaction.return_value
        mock_tx.run.side_effect = Neo4jError("Test error", "Neo.ClientError.Statement.SyntaxError")
        
        # Test data
        agent_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        
        # Call method - should handle exception and return False
        result = entity_repo.link_agent_to_conversation(
            agent_id=agent_id,
            conversation_id=conversation_id,
            relationship_type="PARTICIPATES_IN"
        )
        
        # Ensure result indicates failure
        assert result is False
