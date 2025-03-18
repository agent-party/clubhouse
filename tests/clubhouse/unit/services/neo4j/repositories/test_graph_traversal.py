"""
Tests for the GraphTraversalRepository using real Neo4j infrastructure.

These tests verify the functionality of graph traversal utilities,
including path finding, subgraph extraction, and result transformation.
Following our core principle of using real infrastructure over mocks.
"""

import pytest
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Generator

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from clubhouse.core.config.models.database import (
    Neo4jDatabaseConfig,
    ConnectionPoolConfig
)
from clubhouse.services.neo4j.service import Neo4jService
from clubhouse.services.neo4j.repositories.graph_traversal import (
    GraphTraversalRepository,
    PathResult
)
from pydantic import BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def neo4j_config() -> Neo4jDatabaseConfig:
    """Fixture providing Neo4j configuration for testing."""
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


class ConfigProvider:
    """Config provider for Neo4j service."""
    
    def __init__(self, config: Neo4jDatabaseConfig):
        self.config = config
    
    def get(self) -> Neo4jDatabaseConfig:
        return self.config


def wait_for_neo4j(config: Neo4jDatabaseConfig, timeout: int = 30) -> bool:
    """Wait for Neo4j to be ready."""
    start_time = time.time()
    uri = f"bolt://{config.hosts[0]}"
    
    while time.time() - start_time < timeout:
        try:
            driver = GraphDatabase.driver(
                uri,
                auth=(config.username, config.password)
            )
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    driver.close()
                    return True
        except Exception as e:
            logger.info(f"Waiting for Neo4j: {str(e)}")
            time.sleep(1)
        finally:
            if 'driver' in locals():
                driver.close()
    
    return False


@pytest.fixture(scope="session")
def neo4j_service(neo4j_config: Neo4jDatabaseConfig) -> Generator[Neo4jService, None, None]:
    """Fixture providing an initialized Neo4j service."""
    # Wait for Neo4j to be ready
    assert wait_for_neo4j(neo4j_config), "Neo4j not ready"
    
    config_provider = ConfigProvider(neo4j_config)
    service = Neo4jService(config_provider)
    
    try:
        service.initialize()
        yield service
    finally:
        service.shutdown()


@pytest.fixture(scope="function")
def clean_neo4j(neo4j_service: Neo4jService) -> None:
    """Clean the Neo4j database before and after each test."""
    neo4j_service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
    yield
    neo4j_service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)


@pytest.fixture(scope="function")
def graph_repo(neo4j_service: Neo4jService) -> GraphTraversalRepository:
    """Create a GraphTraversalRepository with real Neo4j service."""
    return GraphTraversalRepository(neo4j_service=neo4j_service)


@pytest.fixture(scope="function")
def test_nodes(neo4j_service: Neo4jService, clean_neo4j) -> Dict[str, Dict[str, Any]]:
    """Create test nodes and relationships."""
    nodes = {}
    
    try:
        # Create test nodes
        for name in ['A', 'B', 'C']:
            node_uuid = str(uuid.uuid4())
            result = neo4j_service.execute_query(
                """
                CREATE (n:TestNode {uuid: $uuid, name: $name, created_at: $created_at})
                RETURN n
                """,
                parameters={
                    "uuid": node_uuid,
                    "name": f"Node {name}",
                    "created_at": datetime.now().isoformat()
                },
                readonly=False
            )
            nodes[name] = {
                "uuid": node_uuid,
                "name": f"Node {name}"
            }
        
        # Create relationships
        neo4j_service.execute_query(
            """
            MATCH (a:TestNode {uuid: $a_uuid}), (b:TestNode {uuid: $b_uuid})
            CREATE (a)-[:KNOWS {uuid: $rel_uuid}]->(b)
            """,
            parameters={
                "a_uuid": nodes['A']['uuid'],
                "b_uuid": nodes['B']['uuid'],
                "rel_uuid": str(uuid.uuid4())
            },
            readonly=False
        )
        
        neo4j_service.execute_query(
            """
            MATCH (b:TestNode {uuid: $b_uuid}), (c:TestNode {uuid: $c_uuid})
            CREATE (b)-[:WORKS_WITH {uuid: $rel_uuid}]->(c)
            """,
            parameters={
                "b_uuid": nodes['B']['uuid'],
                "c_uuid": nodes['C']['uuid'],
                "rel_uuid": str(uuid.uuid4())
            },
            readonly=False
        )
        
        return nodes
        
    except Exception as e:
        logger.error(f"Error creating test nodes: {str(e)}")
        raise


class TestPathResult:
    """Tests for the PathResult data class."""
    
    def test_from_neo4j_path(self, graph_repo: GraphTraversalRepository, test_nodes: Dict[str, Dict[str, Any]]) -> None:
        """Test converting a Neo4j Path to a PathResult using real nodes."""
        # Find a path between nodes A and C
        path_result = graph_repo.find_shortest_path(
            source_id=test_nodes['A']['uuid'],
            target_id=test_nodes['C']['uuid']
        )
        
        assert path_result is not None
        assert len(path_result.nodes) == 3
        assert path_result.length == 2
        assert path_result.start_node['uuid'] == test_nodes['A']['uuid']
        assert path_result.end_node['uuid'] == test_nodes['C']['uuid']


class TestGraphTraversalRepository:
    """Tests for the GraphTraversalRepository class using real Neo4j."""
    
    def test_find_shortest_path(self, graph_repo: GraphTraversalRepository, test_nodes: Dict[str, Dict[str, Any]]) -> None:
        """Test finding the shortest path between two nodes."""
        path = graph_repo.find_shortest_path(
            source_id=test_nodes['A']['uuid'],
            target_id=test_nodes['C']['uuid']
        )
        
        assert path is not None
        assert len(path.nodes) == 3
        assert path.length == 2
        assert path.start_node['uuid'] == test_nodes['A']['uuid']
        assert path.end_node['uuid'] == test_nodes['C']['uuid']
    
    def test_find_shortest_path_not_found(self, graph_repo: GraphTraversalRepository, neo4j_service: Neo4jService, test_nodes: Dict[str, Dict[str, Any]]) -> None:
        """Test when no path is found between nodes."""
        # Create an isolated node
        isolated_uuid = str(uuid.uuid4())
        neo4j_service.execute_query(
            """
            CREATE (n:TestNode {uuid: $uuid, name: 'Isolated'})
            """,
            parameters={"uuid": isolated_uuid},
            readonly=False
        )
        
        # Try to find path to isolated node
        path = graph_repo.find_shortest_path(
            source_id=test_nodes['A']['uuid'],
            target_id=isolated_uuid
        )
        
        assert path is None
    
    def test_find_all_paths(self, graph_repo: GraphTraversalRepository, test_nodes: Dict[str, Dict[str, Any]]) -> None:
        """Test finding all paths between two nodes."""
        paths = graph_repo.find_all_paths(
            source_id=test_nodes['A']['uuid'],
            target_id=test_nodes['C']['uuid']
        )
        
        assert len(paths) == 1  # Only one path exists in our test graph
        assert paths[0].length == 2
        assert paths[0].start_node['uuid'] == test_nodes['A']['uuid']
        assert paths[0].end_node['uuid'] == test_nodes['C']['uuid']
    
    def test_get_subgraph(self, graph_repo: GraphTraversalRepository, test_nodes: Dict[str, Dict[str, Any]]) -> None:
        """Test extracting a subgraph from a root node."""
        # Call the repository method instead of implementing our own query
        subgraph = graph_repo.get_subgraph(
            root_id=test_nodes['A']['uuid'],
            max_depth=2,
            direction="BOTH"  # Use BOTH to ensure we catch all relationships
        )
        
        assert subgraph is not None
        assert 'nodes' in subgraph
        assert 'relationships' in subgraph
        
        nodes = subgraph['nodes']
        relationships = subgraph['relationships']
        
        # Verify we have all 3 nodes in our test graph
        assert len(nodes) == 3
        
        # Verify node UUIDs
        node_uuids = [node['uuid'] for node in nodes]
        for node_id in [test_nodes['A']['uuid'], test_nodes['B']['uuid'], test_nodes['C']['uuid']]:
            assert node_id in node_uuids
        
        # Verify we have all relationships
        assert len(relationships) == 2


class SampleModel(BaseModel):
    """Sample model for testing result transformation."""
    uuid: str
    name: str
    created_at: str = ""


def test_transform_results(graph_repo: GraphTraversalRepository, test_nodes: Dict[str, Dict[str, Any]], clean_neo4j) -> None:
    """Test transforming Neo4j results to domain models."""
    # Query all test nodes
    results = graph_repo.neo4j_service.execute_query(
        """
        MATCH (n:TestNode)
        RETURN n
        """,
        readonly=True
    )
    
    # Transform to domain models
    models = graph_repo.transform_results(
        [record['n'] for record in results],
        SampleModel
    )
    
    assert len(models) == 3
    for model in models:
        assert isinstance(model, SampleModel)
        assert model.uuid in [node['uuid'] for node in test_nodes.values()]
