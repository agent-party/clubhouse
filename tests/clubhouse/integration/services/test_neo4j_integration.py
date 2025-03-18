"""
Integration tests for Neo4j service using a real Neo4j database running in Docker.

This test module connects to a real Neo4j instance to verify the functionality
of the Neo4j service implementation beyond unit tests and mocks.

To run these tests:
1. Start the Neo4j Docker container: `docker-compose -f docker-compose-neo4j.yml up -d`
2. Run the tests: `pytest tests/integration/services/test_neo4j_integration.py -v`
3. Stop the container when done: `docker-compose -f docker-compose-neo4j.yml down`
"""

import os
import time
import logging
from datetime import datetime
from uuid import UUID, uuid4

import pytest
from neo4j.exceptions import ServiceUnavailable
from typing import Dict, Any, Generator

from clubhouse.services.neo4j.service import Neo4jService
from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import (
    Neo4jDatabaseConfig, 
    DatabaseConfig, 
    ConnectionPoolConfig
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def neo4j_config() -> Neo4jDatabaseConfig:
    """Fixture providing a Neo4j database configuration for the Docker container."""
    return Neo4jDatabaseConfig(
        name="test-neo4j",
        hosts=["localhost:7687"],
        username="neo4j",
        password="testpassword",
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


class MockConfigProvider:
    """A simple config provider for testing."""
    
    def __init__(self, config: Neo4jDatabaseConfig) -> None:
        self.config = config
    
    def get(self) -> Neo4jDatabaseConfig:
        """Return the configuration."""
        return self.config


@pytest.fixture(scope="module")
def config_provider(neo4j_config: Neo4jDatabaseConfig) -> ConfigProtocol[DatabaseConfig]:
    """Fixture providing a configuration provider with the Neo4j config."""
    return MockConfigProvider(neo4j_config)


@pytest.fixture(scope="module")
def neo4j_service(config_provider: ConfigProtocol[DatabaseConfig]) -> Generator[Neo4jService, None, None]:
    """
    Fixture providing an initialized Neo4j service connected to the Docker container.
    
    This fixture has module scope to avoid repeatedly initializing the service.
    """
    service = Neo4jService(config_provider)
    
    # Give the Docker container time to start up if just launched
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            service.initialize()
            if service.is_healthy():
                logger.info("Neo4j service successfully connected to Docker container")
                break
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j (attempt {retry_count+1}/{max_retries}): {e}")
            retry_count += 1
            time.sleep(2)  # Wait before retrying
    
    if retry_count == max_retries:
        pytest.skip("Could not connect to Neo4j Docker container - tests skipped")
    
    # Run setup queries to clean the database for testing
    service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
    
    # Yield the service for test use
    yield service
    
    # Proper teardown - ensure all queries are finished and resources released
    try:
        # Clean up any remaining test data
        service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
        # Explicitly shutdown the service to ensure all connections are properly closed
        service.shutdown()
    except Exception as e:
        logger.warning(f"Error during Neo4j service teardown: {e}")


class TestNeo4jIntegration:
    """Integration tests for Neo4j service with a real database."""
    
    def test_service_health(self, neo4j_service: Neo4jService) -> None:
        """Test that the Neo4j service reports as healthy."""
        assert neo4j_service.is_healthy(), "Neo4j service should be healthy"
    
    def test_crud_operations(self, neo4j_service: Neo4jService) -> None:
        """Test basic CRUD operations on nodes and relationships."""
        # Create nodes
        agent_uuid = uuid4()
        agent_labels = ["Agent", "TestEntity"]
        agent_props = {
            "name": "Test Agent", 
            "created_at": datetime.now().isoformat(),
            "uuid": str(agent_uuid)  # Explicitly add UUID property
        }
        agent_id = neo4j_service.create_node(agent_labels, agent_props)
        
        knowledge_uuid = uuid4()
        knowledge_labels = ["Knowledge", "TestEntity"]
        knowledge_props = {
            "content": "Test knowledge", 
            "created_at": datetime.now().isoformat(),
            "uuid": str(knowledge_uuid)  # Explicitly add UUID property
        }
        knowledge_id = neo4j_service.create_node(knowledge_labels, knowledge_props)
        
        # Verify nodes were created
        agent_node = neo4j_service.get_node(agent_uuid)
        knowledge_node = neo4j_service.get_node(knowledge_uuid)
        
        assert agent_node is not None
        assert knowledge_node is not None
        assert agent_node["name"] == "Test Agent"
        assert knowledge_node["content"] == "Test knowledge"
        
        # Create relationship
        rel_uuid = uuid4()
        rel_props = {
            "created_at": datetime.now().isoformat(),
            "uuid": str(rel_uuid)  # Explicitly add UUID property to relationship
        }
        rel_id = neo4j_service.create_relationship(
            agent_uuid, knowledge_uuid, "KNOWS_ABOUT", rel_props
        )
        
        # Verify relationship was created
        rel = neo4j_service.get_relationship(rel_uuid)
        assert rel is not None
        assert "uuid" in rel
        assert str(rel_uuid) == rel["uuid"]
        
        # The relationship type should be accessible via a key that contains the relationship type name
        # or stored as a property
        if "properties" in rel and "type" in rel["properties"]:
            assert rel["properties"]["type"] == "KNOWS_ABOUT"
        elif "type" in rel:
            assert rel["type"] == "KNOWS_ABOUT"
        else:
            # Handle alternative formats where the relationship type may be directly accessible
            relationship_type_found = False
            for key in rel:
                if "type" in key.lower() or "label" in key.lower():
                    if "KNOWS_ABOUT" in str(rel[key]):
                        relationship_type_found = True
                        break
            assert relationship_type_found, f"Relationship type 'KNOWS_ABOUT' not found in relationship keys: {rel.keys()}"
        
        # Get relationships for node
        agent_rels = neo4j_service.get_node_relationships(str(agent_uuid))
        assert len(agent_rels) == 1
        assert agent_rels[0]["uuid"] == str(rel_uuid)
        
        # Check directional relationships
        outgoing_rels = neo4j_service.get_node_relationships(str(agent_uuid), direction="outgoing")
        assert len(outgoing_rels) == 1
        
        incoming_rels = neo4j_service.get_node_relationships(str(agent_uuid), direction="incoming")
        assert len(incoming_rels) == 0
        
        # Update node
        updated_props = {"name": "Updated Agent", "updated_at": datetime.now().isoformat()}
        neo4j_service.update_node(str(agent_uuid), updated_props)
        
        # Verify update
        updated_agent = neo4j_service.get_node(str(agent_uuid))
        assert updated_agent["name"] == "Updated Agent"
        assert "updated_at" in updated_agent
        
        # Update relationship
        rel_update = {"updated_at": datetime.now().isoformat()}
        neo4j_service.update_relationship(rel_uuid, rel_update)
        
        # Delete relationship
        neo4j_service.delete_relationship(rel_uuid)
        
        # Verify relationship was deleted
        assert neo4j_service.get_relationship(rel_uuid) is None
        
        agent_rels = neo4j_service.get_node_relationships(str(agent_uuid))
        assert len(agent_rels) == 0
        
        # Delete nodes
        neo4j_service.delete_node(str(agent_uuid))
        neo4j_service.delete_node(str(knowledge_uuid))
        
        # Verify nodes were deleted
        assert neo4j_service.get_node(str(agent_uuid)) is None
        assert neo4j_service.get_node(str(knowledge_uuid)) is None
    
    def test_query_execution(self, neo4j_service: Neo4jService) -> None:
        """Test executing Cypher queries directly."""
        # Create some test data
        result = neo4j_service.execute_query(
            "CREATE (a:TestNode {name: $name, value: $value}) RETURN a",
            {"name": "Query Test", "value": 42},
            readonly=False
        )
        
        assert len(result) == 1
        assert "a" in result[0]
        
        # Query the data
        result = neo4j_service.execute_query(
            "MATCH (n:TestNode {name: $name}) RETURN n.value AS value",
            {"name": "Query Test"}
        )
        
        assert len(result) == 1
        assert result[0]["value"] == 42
        
        # Test with multiple results
        neo4j_service.execute_query(
            """
            CREATE (a:BatchNode {id: 1})
            CREATE (b:BatchNode {id: 2})
            CREATE (c:BatchNode {id: 3})
            """,
            readonly=False
        )
        
        result = neo4j_service.execute_query("MATCH (n:BatchNode) RETURN n.id AS id ORDER BY n.id")
        assert len(result) == 3
        assert [r["id"] for r in result] == [1, 2, 3]
        
        # Clean up
        neo4j_service.execute_query("MATCH (n:TestNode) DETACH DELETE n", readonly=False)
        neo4j_service.execute_query("MATCH (n:BatchNode) DETACH DELETE n", readonly=False)

    def test_batch_operation(self, neo4j_service: Neo4jService) -> None:
        """Test batch operations."""
        # Create multiple nodes in a batch
        batch_queries = [
            {
                "query": "CREATE (n:BatchTest {id: $id, name: $name, uuid: $uuid}) RETURN n",
                "parameters": {"id": i, "name": f"Node {i}", "uuid": str(uuid4())},
                "return_results": True
            }
            for i in range(1, 6)
        ]
        
        # Corrected method name from execute_batch_operation to execute_batch_operations
        results = neo4j_service.execute_batch_operations(batch_queries)
        assert len(results) == 5
        
        # Query to verify
        result = neo4j_service.execute_query(
            "MATCH (n:BatchTest) RETURN n.id AS id, n.name AS name ORDER BY n.id"
        )
        
        assert len(result) == 5
        assert [r["id"] for r in result] == [1, 2, 3, 4, 5]
        assert [r["name"] for r in result] == ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
        
        # Clean up
        neo4j_service.execute_query("MATCH (n:BatchTest) DETACH DELETE n", readonly=False)
