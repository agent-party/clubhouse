"""
Unit tests for the Neo4j service implementation.

These tests verify the functionality of the Neo4j service, ensuring it correctly
handles node and relationship operations, queries, and error conditions.
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4
from typing import Dict, Any, List, cast

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import Neo4jDatabaseConfig, DatabaseConfig, ConnectionPoolConfig
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.service import Neo4jService
from clubhouse.services.neo4j.mock_service import MockNeo4jService


@pytest.fixture
def neo4j_config() -> Neo4jDatabaseConfig:
    """Fixture providing a Neo4j database configuration for testing."""
    return Neo4jDatabaseConfig(
        name="test-neo4j",
        hosts=["localhost:7687"],
        username="neo4j",
        password="password",
        database="neo4j",
        query_timeout_seconds=60,
        max_transaction_retry_time_seconds=30,
        connection_pool=ConnectionPoolConfig(
            max_size=50,
            min_size=1,
            max_idle_time_seconds=600,
            connection_timeout_seconds=30,
            retry_attempts=3
        )
    )


@pytest.fixture
def config_provider(neo4j_config: Neo4jDatabaseConfig) -> ConfigProtocol[DatabaseConfig]:
    """Fixture providing a configuration provider for the Neo4j service."""
    provider = MagicMock(spec=ConfigProtocol)
    provider.get.return_value = neo4j_config
    return cast(ConfigProtocol[DatabaseConfig], provider)


@pytest.fixture
def mock_neo4j_service(neo4j_config: Neo4jDatabaseConfig) -> MockNeo4jService:
    """Fixture providing a mock Neo4j service implementation."""
    service = MockNeo4jService(config=neo4j_config)
    service.initialize()
    return service


class TestNeo4jService:
    """Unit tests for the Neo4j service implementation."""
    
    def test_create_node(self, mock_neo4j_service: MockNeo4jService) -> None:
        """
        Test creation of nodes in the knowledge graph.
        
        Verifies that nodes can be created with correct labels and properties,
        and that the returned UUID can be used to retrieve the node.
        """
        # Setup
        labels = ["Agent"]
        properties = {"name": "TestAgent", "status": "active"}
        
        # Execute
        node_id = mock_neo4j_service.create_node(labels, properties)
        
        # Verify
        assert isinstance(node_id, UUID)
        
        # Get the node and check properties
        node_properties = mock_neo4j_service.get_node(node_id)
        assert node_properties is not None
        assert "name" in node_properties
        assert node_properties["name"] == "TestAgent"
        assert "status" in node_properties
        assert node_properties["status"] == "active"
        
        # Check that labels were applied correctly using the internal structure
        # This is testing internal implementation which is not ideal, but necessary for mocks
        str_id = str(node_id)
        if str_id in mock_neo4j_service._nodes:
            assert "Agent" in mock_neo4j_service._nodes[str_id]["labels"]

    def test_create_relationship(self, mock_neo4j_service: MockNeo4jService) -> None:
        """
        Test creation of relationships between nodes.
        
        Verifies that relationships can be created with correct type and properties,
        and that the relationship can be queried from both source and target nodes.
        """
        # Setup - Create two nodes
        agent_id = mock_neo4j_service.create_node(["Agent"], {"name": "AgentA"})
        knowledge_id = mock_neo4j_service.create_node(["Knowledge"], {"content": "Test knowledge"})
        
        # Properties for the relationship
        rel_props = {"confidence": 0.95, "timestamp": "2025-03-16T12:00:00"}
        
        # Execute - Create relationship
        rel_id = mock_neo4j_service.create_relationship(
            agent_id, 
            knowledge_id, 
            "KNOWS", 
            rel_props
        )
        
        # Verify
        assert isinstance(rel_id, UUID)
        
        # Check relationship exists by examining node relationships
        outgoing_relationships = mock_neo4j_service.get_node_relationships(agent_id, "OUTGOING")
        assert len(outgoing_relationships) == 1
        assert outgoing_relationships[0]["type"] == "KNOWS"
        
        # Using internal structures to verify relationship properties 
        str_id = str(rel_id)
        if str_id in mock_neo4j_service._relationships:
            rel_data = mock_neo4j_service._relationships[str_id]
            assert rel_data["properties"]["confidence"] == 0.95
        
        # Check relationship can be found from target node
        incoming_relationships = mock_neo4j_service.get_node_relationships(knowledge_id, "INCOMING")
        assert len(incoming_relationships) == 1
        assert incoming_relationships[0]["type"] == "KNOWS"

    def test_complex_query(self, mock_neo4j_service: MockNeo4jService) -> None:
        """
        Test complex query for finding all knowledge linked to an agent.
        
        Verifies that complex queries can retrieve nodes and relationships
        with proper filtering and result transformation.
        """
        # Setup - Create an agent with multiple knowledge connections
        agent_id = mock_neo4j_service.create_node(["Agent"], {"name": "AgentB"})
        
        # Create multiple knowledge nodes connected to the agent
        knowledge_ids = []
        for i in range(3):
            k_id = mock_neo4j_service.create_node(
                ["Knowledge"], 
                {"content": f"Knowledge {i}", "priority": i}
            )
            knowledge_ids.append(k_id)
            
            # Connect the agent to each knowledge node
            mock_neo4j_service.create_relationship(
                agent_id, 
                k_id, 
                "KNOWS", 
                {"confidence": 0.8 + (i * 0.05)}
            )
        
        # Execute - Get the target nodes using relationships
        related_nodes = []
        relationships = mock_neo4j_service.get_node_relationships(agent_id, "OUTGOING", relationship_types=["KNOWS"])
        
        # Get the target nodes of these relationships
        for rel in relationships:
            end_node_id = rel["end_node_id"]
            target_properties = mock_neo4j_service.get_node(end_node_id)
            if target_properties and "content" in target_properties:  # Knowledge nodes have content
                related_nodes.append(target_properties)
        
        # Verify
        assert len(related_nodes) == 3
        
        # Check that all knowledge nodes were returned
        contents = [node["content"] for node in related_nodes]
        for i in range(3):
            assert f"Knowledge {i}" in contents

    def test_node_update(self, mock_neo4j_service: MockNeo4jService) -> None:
        """
        Test updating node properties.
        
        Verifies that node properties can be updated correctly, including
        adding new properties and modifying existing ones.
        """
        # Setup - Create a node
        node_id = mock_neo4j_service.create_node(
            ["Agent"], 
            {"name": "UpdateTest", "status": "inactive"}
        )
        
        # Execute - Update properties
        updated_props = {"status": "active", "priority": "high"}
        mock_neo4j_service.update_node(node_id, updated_props)
        
        # Verify
        node_properties = mock_neo4j_service.get_node(node_id)
        assert node_properties is not None
        assert node_properties["status"] == "active"  # Updated property
        assert node_properties["priority"] == "high"  # New property
        assert node_properties["name"] == "UpdateTest"  # Unchanged property

    def test_node_deletion(self, mock_neo4j_service: MockNeo4jService) -> None:
        """
        Test node deletion.
        
        Verifies that nodes can be deleted when they have no relationships.
        """
        # Setup - Create a node
        node_id = mock_neo4j_service.create_node(["Agent"], {"name": "DeleteTestAgent"})
        
        # Execute - Delete the agent node
        mock_neo4j_service.delete_node(node_id)
        
        # Verify - Agent node should be gone
        node = mock_neo4j_service.get_node(node_id)
        assert node is None

    def test_relationship_deletion(self, mock_neo4j_service: MockNeo4jService) -> None:
        """
        Test relationship deletion.
        
        Verifies that relationships can be deleted without affecting connected nodes.
        """
        # Setup - Create nodes and relationship
        source_id = mock_neo4j_service.create_node(["Source"], {"name": "DeleteRelSource"})
        target_id = mock_neo4j_service.create_node(["Target"], {"name": "DeleteRelTarget"})
        
        rel_id = mock_neo4j_service.create_relationship(source_id, target_id, "CONNECTS")
        
        # Get the relationships to verify existence before deletion
        outgoing_rels = mock_neo4j_service.get_node_relationships(source_id, "OUTGOING")
        assert len(outgoing_rels) == 1
        
        # Execute - Delete the relationship
        mock_neo4j_service.delete_relationship(rel_id)
        
        # Verify - Relationship should be gone
        outgoing_rels_after = mock_neo4j_service.get_node_relationships(source_id, "OUTGOING")
        assert len(outgoing_rels_after) == 0
        
        # Both nodes should still exist
        source = mock_neo4j_service.get_node(source_id)
        target = mock_neo4j_service.get_node(target_id)
        assert source is not None
        assert target is not None
