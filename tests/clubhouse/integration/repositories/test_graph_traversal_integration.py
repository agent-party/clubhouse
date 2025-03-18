"""
Integration tests for GraphTraversalRepository using a real Neo4j database.

This test module validates the graph traversal operations against a real Neo4j instance
to ensure that repository behavior matches expected outcomes with production-like data.
It creates complex graph topologies to test traversal algorithms under realistic conditions.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Generator

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import (
    Neo4jDatabaseConfig, 
    DatabaseConfig,
    ConnectionPoolConfig
)
from clubhouse.services.neo4j.service import Neo4jService
from clubhouse.services.neo4j.repositories.graph_traversal import GraphTraversalRepository, PathResult


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def neo4j_config() -> Neo4jDatabaseConfig:
    """Fixture providing a Neo4j database configuration for the Docker container."""
    # Make sure this matches the port used in docker-compose
    return Neo4jDatabaseConfig(
        name="test-neo4j",
        hosts=["localhost:7687"],  # Standard Neo4j port
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


def test_direct_connection(neo4j_config: Neo4jDatabaseConfig) -> None:
    """Test direct connection to Neo4j using the driver."""
    uri = f"bolt://{neo4j_config.hosts[0]}"
    logger.info(f"Testing direct connection to {uri}")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(neo4j_config.username, neo4j_config.password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info("Direct connection test successful!")
                assert True
            else:
                logger.error("Direct connection test failed: unexpected result")
                assert False
    except Exception as e:
        logger.error(f"Direct connection failed: {str(e)}")
        assert False
    finally:
        if 'driver' in locals():
            driver.close()


@pytest.fixture(scope="module")
def neo4j_service(neo4j_config: Neo4jDatabaseConfig) -> Generator[Neo4jService, None, None]:
    """
    Fixture providing an initialized Neo4j service connected to the Docker container.
    
    This fixture has module scope to avoid repeatedly initializing the service.
    """
    config_provider = MockConfigProvider(neo4j_config)
    service = Neo4jService(config_provider)
    
    # First, test a direct connection to ensure Neo4j is available
    max_direct_attempts = 5
    direct_connected = False
    
    for attempt in range(1, max_direct_attempts + 1):
        logger.info(f"Direct connection attempt {attempt}/{max_direct_attempts}")
        
        # Test direct connection
        uri = f"bolt://{neo4j_config.hosts[0]}"
        logger.info(f"Testing direct connection to {uri}")
        
        try:
            driver = GraphDatabase.driver(uri, auth=(neo4j_config.username, neo4j_config.password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("Direct connection test successful!")
                    direct_connected = True
                    break
                else:
                    logger.error("Direct connection test failed: unexpected result")
        except Exception as e:
            logger.error(f"Direct connection failed: {str(e)}")
        finally:
            if 'driver' in locals():
                driver.close()
                
        time.sleep(2)
    
    if not direct_connected:
        pytest.skip("Could not establish direct connection to Neo4j - skipping tests")
    
    # Now initialize the Neo4j service
    logger.info("Initializing Neo4jService...")
    
    try:
        service.initialize()
        logger.info("Neo4jService initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4jService: {str(e)}")
        pytest.skip(f"Failed to initialize Neo4jService: {str(e)}")
    
    # Wait for Neo4j to be available (max 30 seconds)
    start_time = time.time()
    connected = False
    connection_errors = []
    
    logger.info("Testing Neo4jService connection...")
    
    while time.time() - start_time < 30:
        try:
            # First try a simple query to test connection
            result = service.execute_query("RETURN 1 as test", readonly=True)
            if result and result[0]['test'] == 1:
                logger.info("Neo4j service connection test successful!")
                connected = True
                break
        except Exception as e:
            error_message = f"Connection attempt failed: {str(e)}"
            logger.info(error_message)
            connection_errors.append(error_message)
            time.sleep(1)
    
    if not connected:
        error_message = f"Neo4j service is not available after 30 seconds. Errors: {connection_errors}"
        logger.error(error_message)
        pytest.skip(error_message)
    
    # Clean database before tests
    try:
        service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
        logger.info("Database cleaned successfully")
    except Exception as e:
        logger.warning(f"Error cleaning database: {e}")
    
    yield service
    
    # Clean up after tests
    try:
        service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
        service.shutdown()
        logger.info("Neo4j service shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during Neo4j service teardown: {e}")


@pytest.fixture(scope="module")
def graph_traversal_repo(neo4j_service: Neo4jService) -> GraphTraversalRepository:
    """Fixture providing a GraphTraversalRepository instance."""
    return GraphTraversalRepository(neo4j_service)


@pytest.fixture(scope="module")
def complex_graph(neo4j_service: Neo4jService) -> Dict[str, Dict[str, Any]]:
    """
    Fixture creating a complex graph for testing traversal operations.
    
    Creates a graph with the following topology:
    
    (A) -- KNOWS --> (B) -- KNOWS --> (C) -- KNOWS --> (D)
     |                |                 |                |
     |                |                 |                |
     v                v                 v                v
    (E) -- KNOWS --> (F) -- WORKS_WITH (G) -- WORKS_WITH (H)
     |                                  |
     |                                  |
     v                                  v
    (I) -- COLLABORATES --> (J) -- COLLABORATES --> (K)
    
    Returns a dictionary mapping node names to node IDs for test reference.
    """
    node_map = {}
    relationship_types = ["KNOWS", "WORKS_WITH", "COLLABORATES"]
    
    try:
        # Clean the database before creating the test graph
        neo4j_service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
        logger.info("Database cleaned before creating complex graph")
        
        # Create nodes A through K with UUID field
        for node_name in "ABCDEFGHIJK":
            node_id = str(uuid.uuid4())
            create_result = neo4j_service.execute_query(
                """
                CREATE (n:TestNode {uuid: $uuid, name: $name, created_at: $created_at})
                RETURN n
                """,
                parameters={
                    "uuid": node_id,
                    "name": f"Node {node_name}",
                    "created_at": datetime.now().isoformat()
                },
                readonly=False
            )
            
            if create_result and len(create_result) > 0:
                # Store id as "uuid" to match the repository's expectations
                node_map[node_name] = {"uuid": node_id, "name": f"Node {node_name}"}
                logger.info(f"Created node {node_name} with UUID {node_id}")
            else:
                raise Exception(f"Failed to create node {node_name}")
        
        # Create relationships according to the topology
        relationships = [
            ("A", "B", "KNOWS"),
            ("B", "C", "KNOWS"),
            ("C", "D", "KNOWS"),
            ("A", "E", "KNOWS"),
            ("B", "F", "KNOWS"),
            ("C", "G", "KNOWS"),
            ("D", "H", "KNOWS"),
            ("E", "F", "KNOWS"),
            ("F", "G", "WORKS_WITH"),
            ("G", "H", "WORKS_WITH"),
            ("E", "I", "KNOWS"),
            ("G", "J", "KNOWS"),
            ("I", "J", "COLLABORATES"),
            ("J", "K", "COLLABORATES")
        ]
        
        for source, target, rel_type in relationships:
            rel_id = str(uuid.uuid4())
            rel_result = neo4j_service.execute_query(
                f"""
                MATCH (a:TestNode), (b:TestNode)
                WHERE a.uuid = $start_uuid AND b.uuid = $end_uuid
                CREATE (a)-[r:{rel_type} {{uuid: $rel_uuid, created_at: $created_at}}]->(b)
                RETURN r
                """,
                parameters={
                    "start_uuid": node_map[source]["uuid"],
                    "end_uuid": node_map[target]["uuid"],
                    "rel_uuid": rel_id,
                    "created_at": datetime.now().isoformat()
                },
                readonly=False
            )
            
            if rel_result and len(rel_result) > 0:
                logger.info(f"Created relationship {source}-[{rel_type}]->{target}")
            else:
                raise Exception(f"Failed to create relationship {source}-[{rel_type}]->{target}")
        
        # Verify nodes exist
        for node_name, node_info in node_map.items():
            verify_result = neo4j_service.execute_query(
                """
                MATCH (n:TestNode)
                WHERE n.uuid = $uuid
                RETURN count(n) as node_exists
                """,
                parameters={"uuid": node_info["uuid"]},
                readonly=True
            )
            
            if not verify_result or verify_result[0]["node_exists"] == 0:
                raise Exception(f"Node {node_name} with UUID {node_info['uuid']} was not created properly")
        
        # Verify relationships between specific nodes
        # For test_find_shortest_path_with_relationship_filter
        logger.info("Verifying relationship from E to F")
        verify_e_f = neo4j_service.execute_query(
            """
            MATCH (a:TestNode)-[r:KNOWS]->(b:TestNode)
            WHERE a.uuid = $a_uuid AND b.uuid = $b_uuid
            RETURN count(r) as rel_exists
            """,
            parameters={
                "a_uuid": node_map["E"]["uuid"],
                "b_uuid": node_map["F"]["uuid"]
            },
            readonly=True
        )
        if verify_e_f and verify_e_f[0]["rel_exists"] > 0:
            logger.info("Verified E-[KNOWS]->F relationship exists")
        else:
            logger.error("E-[KNOWS]->F relationship is missing")
            
        logger.info("Verifying relationship from F to G")
        verify_f_g = neo4j_service.execute_query(
            """
            MATCH (a:TestNode)-[r:WORKS_WITH]->(b:TestNode)
            WHERE a.uuid = $a_uuid AND b.uuid = $b_uuid
            RETURN count(r) as rel_exists
            """,
            parameters={
                "a_uuid": node_map["F"]["uuid"],
                "b_uuid": node_map["G"]["uuid"]
            },
            readonly=True
        )
        if verify_f_g and verify_f_g[0]["rel_exists"] > 0:
            logger.info("Verified F-[WORKS_WITH]->G relationship exists")
        else:
            logger.error("F-[WORKS_WITH]->G relationship is missing")
        
        logger.info(f"Created complex graph with {len(node_map)} nodes and {len(relationships)} relationships")
        return node_map
        
    except Exception as e:
        logger.error(f"Error creating complex graph: {str(e)}")
        raise


class TestGraphTraversalIntegration:
    """Integration tests for the GraphTraversalRepository with a real Neo4j database."""
    
    def test_repository_initialization(self, graph_traversal_repo: GraphTraversalRepository):
        """Test that the repository can be properly initialized."""
        assert graph_traversal_repo is not None
        assert graph_traversal_repo.neo4j_service is not None
    
    def test_find_shortest_path_direct(self, graph_traversal_repo: GraphTraversalRepository, complex_graph: Dict):
        """Test finding a shortest path between directly connected nodes."""
        # Find path from A to B (direct connection)
        path = graph_traversal_repo.find_shortest_path(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["B"]["uuid"]
        )
        
        # Verify path exists and has expected properties
        assert path is not None
        assert isinstance(path, PathResult)
        assert path.length == 1
        assert path.start_node["name"] == "Node A"
        assert path.end_node["name"] == "Node B"
        assert path.relationships[0]["type"] == "KNOWS"
    
    def test_find_shortest_path_indirect(self, graph_traversal_repo: GraphTraversalRepository, complex_graph: Dict):
        """Test finding a shortest path between indirectly connected nodes."""
        # Find path from A to D (A->B->C->D)
        path = graph_traversal_repo.find_shortest_path(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["D"]["uuid"]
        )
        
        # Verify path exists and has expected properties
        assert path is not None
        assert isinstance(path, PathResult)
        assert path.length == 3
        assert path.start_node["name"] == "Node A"
        assert path.end_node["name"] == "Node D"
        
        # Verify path goes through expected nodes
        node_names = [node["name"] for node in path.nodes]
        assert node_names == ["Node A", "Node B", "Node C", "Node D"]
    
    def test_find_shortest_path_with_relationship_filter(self, graph_traversal_repo, complex_graph, neo4j_service):
        """Test path finding with relationship type filtering."""
        # Use nodes E, F, G for this test (follows E-[KNOWS]->F-[WORKS_WITH]->G)
        start_node = complex_graph["E"]
        end_node = complex_graph["G"]
        
        logger.info(f"Testing path finding with relationship filters from {start_node['name']} to {end_node['name']}")
        logger.info(f"Start node UUID: {start_node['uuid']}, End node UUID: {end_node['uuid']}")
        
        # Verify nodes exist with a direct query
        neo4j_service.execute_query(
            """
            MATCH (n:TestNode)
            WHERE n.uuid IN [$start_id, $end_id]
            RETURN n.name, n.uuid
            """,
            parameters={
                "start_id": start_node["uuid"],
                "end_id": end_node["uuid"]
            },
            readonly=True
        )
        
        # First, verify there's a path when including both relationship types
        path_all_types = graph_traversal_repo.find_shortest_path(
            source_id=start_node["uuid"],
            target_id=end_node["uuid"],
            max_depth=3
        )
        
        assert path_all_types is not None, "Should find a path with all relationship types"
        logger.info(f"Found path with all relationship types, length: {path_all_types.length}")
        
        # Log the relationships in the path
        for i, rel in enumerate(path_all_types.relationships):
            logger.info(f"Relationship {i+1} in path: {rel.get('type', 'unknown')}")
        
        # Then test with specific relationship filter
        # Should follow E-[KNOWS]->F-[WORKS_WITH]->G
        # Verify these relationships exist with a direct query
        logger.info("Verifying path via direct query")
        verification_query = """
        MATCH path = (start:TestNode)-[:KNOWS]->(:TestNode)-[:WORKS_WITH]->(end:TestNode)
        WHERE start.uuid = $start_id AND end.uuid = $end_id
        RETURN path
        """
        
        direct_path = neo4j_service.execute_query(
            verification_query,
            parameters={
                "start_id": start_node["uuid"],
                "end_id": end_node["uuid"]
            },
            readonly=True
        )
        
        logger.info(f"Direct path query results: {direct_path}")
        
        # Now use the repository with relationship filters
        path_with_filter = graph_traversal_repo.find_shortest_path(
            source_id=start_node["uuid"],
            target_id=end_node["uuid"],
            relationship_types=["KNOWS", "WORKS_WITH"],
            max_depth=3
        )
        
        assert path_with_filter is not None, "Should find a path with KNOWS and WORKS_WITH relationship types"
        
        # Verify the path length and relationship types
        assert path_with_filter.length == 2, f"Path should have 2 relationships, but has {path_with_filter.length}"
        
        # Verify the relationships are of expected types
        rel_types = [rel.get("type") for rel in path_with_filter.relationships]
        logger.info(f"Path relationship types: {rel_types}")
        
        assert "KNOWS" in rel_types, "Path should include KNOWS relationship"
        assert "WORKS_WITH" in rel_types, "Path should include WORKS_WITH relationship"
    
    def test_find_shortest_path_max_depth(
        self, 
        graph_traversal_repo: GraphTraversalRepository, 
        complex_graph: Dict
    ):
        """Test that max_depth parameter properly limits path finding."""
        # Find path from A to D with max_depth=2 (should fail as path is length 3)
        path = graph_traversal_repo.find_shortest_path(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["D"]["uuid"],
            max_depth=2
        )
        
        # Path should not be found due to depth restriction
        assert path is None
        
        # Now try with sufficient depth
        path = graph_traversal_repo.find_shortest_path(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["D"]["uuid"],
            max_depth=3
        )
        
        # Now path should be found
        assert path is not None
        assert path.length == 3
    
    def test_find_all_paths(
        self, 
        graph_traversal_repo: GraphTraversalRepository, 
        complex_graph: Dict
    ):
        """Test finding all paths between two nodes."""
        # Find all paths from A to G (multiple possible paths)
        paths = graph_traversal_repo.find_all_paths(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["G"]["uuid"],
            max_depth=4
        )
        
        # Verify multiple paths are found
        assert paths is not None
        assert len(paths) >= 2  # There should be at least 2 paths (A->B->C->G and A->E->F->G)
        
        # Verify paths have expected properties
        for path in paths:
            assert isinstance(path, PathResult)
            assert path.start_node["name"] == "Node A"
            assert path.end_node["name"] == "Node G"
            
        # Verify paths have different routes
        path_strings = []
        for path in paths:
            path_str = "->".join([node["name"] for node in path.nodes])
            path_strings.append(path_str)
        
        # Check that we have distinct paths
        assert len(set(path_strings)) == len(paths)
    
    def test_find_all_paths_with_limit(
        self, 
        graph_traversal_repo: GraphTraversalRepository, 
        complex_graph: Dict
    ):
        """Test that the limit parameter properly restricts the number of returned paths."""
        # Find all paths from A to K with a small limit
        paths = graph_traversal_repo.find_all_paths(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["K"]["uuid"],
            max_depth=6,
            limit=2
        )
        
        # Verify limit is respected
        assert paths is not None
        assert len(paths) <= 2
        
        # Now try with a larger limit
        more_paths = graph_traversal_repo.find_all_paths(
            source_id=complex_graph["A"]["uuid"],
            target_id=complex_graph["K"]["uuid"],
            max_depth=6,
            limit=10
        )
        
        # Verify we can get more paths with a larger limit
        assert more_paths is not None
        # Might have more paths (depending on graph structure)
        assert len(more_paths) >= len(paths)
    
    def test_path_not_found(
        self, 
        graph_traversal_repo: GraphTraversalRepository, 
        neo4j_service: Neo4jService
    ):
        """Test behavior when no path exists between nodes."""
        # Create two completely disconnected nodes
        isolated_node_1_id = str(uuid.uuid4())
        isolated_node_2_id = str(uuid.uuid4())
        
        # Create first isolated node
        neo4j_service.execute_query(
            """
            CREATE (n:IsolatedNode {uuid: $uuid, name: $name})
            RETURN n
            """,
            parameters={
                "uuid": isolated_node_1_id,
                "name": "Isolated Node 1"
            },
            readonly=False
        )
        
        # Create second isolated node
        neo4j_service.execute_query(
            """
            CREATE (n:IsolatedNode {uuid: $uuid, name: $name})
            RETURN n
            """,
            parameters={
                "uuid": isolated_node_2_id,
                "name": "Isolated Node 2"
            },
            readonly=False
        )
        
        # Try to find path between isolated nodes
        path = graph_traversal_repo.find_shortest_path(
            source_id=isolated_node_1_id,
            target_id=isolated_node_2_id
        )
        
        # Path should not exist
        assert path is None
        
        # Also try find_all_paths
        paths = graph_traversal_repo.find_all_paths(
            source_id=isolated_node_1_id,
            target_id=isolated_node_2_id
        )
        
        # No paths should be found
        assert paths == []
    
    def test_performance_with_larger_graph(
        self, 
        graph_traversal_repo: GraphTraversalRepository, 
        neo4j_service: Neo4jService
    ):
        """Test performance with a larger graph."""
        num_nodes = 50
        node_ids = []
        
        try:
            # Clean database
            neo4j_service.execute_query("MATCH (n) DETACH DELETE n", readonly=False)
            logger.info(f"Database cleaned before creating performance test graph with {num_nodes} nodes")
            
            # Create a larger graph with a chain of nodes
            # This creates a linear chain A->B->C->...->Z where we want to find the path from A to Z
            
            # Create nodes first
            for i in range(num_nodes):
                node_id = str(uuid.uuid4())
                node_ids.append(node_id)
                
                result = neo4j_service.execute_query(
                    """
                    CREATE (n:TestNode {uuid: $uuid, name: $name, index: $index})
                    RETURN n
                    """,
                    parameters={
                        "uuid": node_id,
                        "name": f"Node {i}",
                        "index": i
                    },
                    readonly=False
                )
                
                if not result:
                    raise Exception(f"Failed to create node {i}")
            
            # Verify nodes were created
            count_result = neo4j_service.execute_query(
                """
                MATCH (n:TestNode) RETURN count(n) as node_count
                """,
                readonly=True
            )
            
            if count_result and count_result[0]["node_count"] == num_nodes:
                logger.info(f"Created {num_nodes} nodes for performance test")
            else:
                logger.error(f"Expected {num_nodes} nodes but found {count_result[0]['node_count'] if count_result else 'unknown'}")
                raise Exception("Node creation failed")
            
            # Create relationships (linear chain A->B->C->...->Z)
            relationship_count = 0
            for i in range(num_nodes - 1):
                rel_result = neo4j_service.execute_query(
                    """
                    MATCH (a:TestNode), (b:TestNode)
                    WHERE a.uuid = $src_id AND b.uuid = $dst_id
                    CREATE (a)-[r:CONNECTED_TO {weight: $weight}]->(b)
                    RETURN r
                    """,
                    parameters={
                        "src_id": node_ids[i],
                        "dst_id": node_ids[i + 1],
                        "weight": 1
                    },
                    readonly=False
                )
                
                if rel_result:
                    relationship_count += 1
            
            # Verify relationships were created
            rel_count_result = neo4j_service.execute_query(
                """
                MATCH ()-[r:CONNECTED_TO]->() RETURN count(r) as rel_count
                """,
                readonly=True
            )
            
            expected_relationships = num_nodes - 1
            if rel_count_result and rel_count_result[0]["rel_count"] == expected_relationships:
                logger.info(f"Created {expected_relationships} relationships for performance test")
            else:
                logger.error(f"Expected {expected_relationships} relationships but found {rel_count_result[0]['rel_count'] if rel_count_result else 'unknown'}")
                raise Exception("Relationship creation failed")
                
            # Verify a direct path exists from start to end
            verification_query = """
            MATCH path = (start:TestNode)-[:CONNECTED_TO*]->(end:TestNode)
            WHERE start.uuid = $start_id AND end.uuid = $end_id
            RETURN path
            LIMIT 1
            """
            
            direct_path = neo4j_service.execute_query(
                verification_query,
                parameters={
                    "start_id": node_ids[0],
                    "end_id": node_ids[-1]
                },
                readonly=True
            )
            
            logger.info(f"Direct path verification result: {direct_path is not None and len(direct_path) > 0}")
            
            # Find path from start to end node
            # This test should pass if the graph is correctly created
            start_time = time.time()
            
            logger.info(f"Finding path from Node 0 (UUID: {node_ids[0]}) to Node {num_nodes-1} (UUID: {node_ids[-1]})")
            
            path = graph_traversal_repo.find_shortest_path(
                source_id=node_ids[0],
                target_id=node_ids[-1],
                max_depth=num_nodes  # Need enough depth to reach the end
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Path should exist and be linear from start to end
            assert path is not None, "Path should not be None - expected to find a valid path"
            logger.info(f"Found path with {path.length} relationships in {elapsed:.6f} seconds")
            
            # Verify it's the correct path by checking node count and relationship count
            assert len(path.nodes) == num_nodes, f"Expected {num_nodes} nodes in path, found {len(path.nodes)}"
            assert len(path.relationships) == num_nodes - 1, f"Expected {num_nodes-1} relationships in path, found {len(path.relationships)}"
            
            # Verify start and end nodes
            assert path.start_node["uuid"] == node_ids[0], "Start node UUID doesn't match"
            assert path.end_node["uuid"] == node_ids[-1], "End node UUID doesn't match"
            
            logger.info(f"Performance test passed: found path with expected topology in {elapsed:.6f} seconds")
            
        except Exception as e:
            logger.error(f"Error in performance test: {str(e)}")
            raise
