"""
Unit tests for Neo4j service core operations.

These tests focus on the core CRUD operations of the Neo4j service implementation,
ensuring that node and relationship operations function correctly. These tests
use proper mocking to avoid actual database connections while providing comprehensive
test coverage.
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, cast
from unittest.mock import MagicMock, patch, call

from neo4j import Record
from neo4j.graph import Node, Relationship
from neo4j.exceptions import Neo4jError, ServiceUnavailable, DatabaseError, TransientError
from clubhouse.services.neo4j.utils import format_direction

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import Neo4jDatabaseConfig, DatabaseConfig, ConnectionPoolConfig
from clubhouse.services.neo4j.service import Neo4jService
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol


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
def initialized_service(config_provider: ConfigProtocol[DatabaseConfig]) -> Neo4jService:
    """
    Fixture providing an initialized Neo4j service with mocked connections.
    
    This fixture creates a Neo4jService instance and mocks all the necessary
    dependencies to simulate a successfully initialized service without actually
    connecting to a database.
    """
    with patch('clubhouse.services.neo4j.service.GraphDatabase') as mock_graph_db, \
         patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
        
        # Set up mock drivers and session
        mock_driver = MagicMock()
        mock_async_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Configure successful connection test
        mock_graph_db.driver.return_value = mock_driver
        mock_async_graph_db.driver.return_value = mock_async_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value.single.return_value = 1
        
        # Create and initialize service
        service = Neo4jService(config_provider)
        service.initialize()
        
        # Return initialized service for tests to use
        return service


class TestNeo4jCoreOperations:
    """Tests for Neo4j service core CRUD operations."""
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_create_node(self, mock_transaction, initialized_service):
        """
        Test node creation with various label formats and property types.
        
        Verifies that:
        1. Single string labels are handled correctly
        2. List of labels are handled correctly
        3. Properties are properly converted to Neo4j format
        4. UUID properties are correctly handled
        """
        # Setup mock node result
        node_id = uuid.uuid4()
        mock_node = {
            "uuid": str(node_id),
            "name": "Test Node",
            "created_at": datetime.now().isoformat(),
            "numeric_prop": 42,
            "boolean_prop": True
        }
        
        # Configure transaction mock to return a successful result
        mock_transaction.return_value = mock_node
        
        # Test with single string label
        result1 = initialized_service.create_node(
            labels="TestNode", 
            properties={
                "uuid": node_id,
                "name": "Test Node",
                "created_at": datetime.now(),
                "numeric_prop": 42,
                "boolean_prop": True
            }
        )
        
        # Verify the result
        assert result1 == mock_node
        assert mock_transaction.call_count == 1
        
        # Reset mock for next test
        mock_transaction.reset_mock()
        
        # Test with list of labels
        result2 = initialized_service.create_node(
            labels=["TestNode", "Entity"], 
            properties={
                "uuid": node_id,
                "name": "Test Node",
                "numeric_prop": 42
            }
        )
        
        # Verify the result
        assert result2 == mock_node
        assert mock_transaction.call_count == 1
        
        # Verify that the transaction function was called with the correct parameters
        # The first argument should be a driver, second a database name, and third a function
        assert mock_transaction.call_args[1]["readonly"] == False
        
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_get_node(self, mock_transaction, initialized_service):
        """
        Test node retrieval by UUID.
        
        Verifies that:
        1. Existing nodes can be retrieved correctly
        2. Non-existent nodes return None
        3. UUID conversion is handled properly
        """
        # Setup mock node result for existing node
        node_id = uuid.uuid4()
        mock_node = {
            "uuid": str(node_id),
            "name": "Test Node",
            "created_at": datetime.now().isoformat(),
            "numeric_prop": 42
        }
        
        # Configure transaction mock to return a successful result
        mock_transaction.return_value = mock_node
        
        # Test retrieving an existing node
        result = initialized_service.get_node(node_id)
        
        # Verify the result
        assert result == mock_node
        assert mock_transaction.call_count == 1
        
        # Verify that the transaction was read-only
        assert mock_transaction.call_args[1]["readonly"] == True
        
        # Reset mock and configure for non-existent node
        mock_transaction.reset_mock()
        mock_transaction.return_value = None
        
        # Test retrieving a non-existent node
        non_existent_result = initialized_service.get_node(uuid.uuid4())
        
        # Verify the result
        assert non_existent_result is None
        assert mock_transaction.call_count == 1
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_update_node(self, mock_transaction, initialized_service):
        """
        Test node property updates.
        
        Verifies that:
        1. Node properties can be updated when the node exists
        2. Updates to non-existent nodes return None
        3. Merge mode correctly preserves existing properties
        4. Replace mode correctly overwrites all properties
        """
        # Setup mock node result for existing node
        node_id = uuid.uuid4()
        original_props = {
            "uuid": str(node_id),
            "name": "Original Name",
            "created_at": datetime.now().isoformat(),
            "count": 5
        }
        
        updated_props = {
            "uuid": str(node_id),
            "name": "Updated Name",
            "created_at": original_props["created_at"],
            "count": 10,
            "new_prop": "new value"
        }
        
        # Configure transaction mock to return a successful result with updated properties
        mock_transaction.return_value = updated_props
        
        # Test updating node with merge=True (default)
        result = initialized_service.update_node(
            node_id=node_id,
            properties={"name": "Updated Name", "count": 10, "new_prop": "new value"},
            merge=True
        )
        
        # Verify the result
        assert result == updated_props
        assert mock_transaction.call_count == 1
        
        # Reset mock for next test
        mock_transaction.reset_mock()
        
        # Setup result for replace mode (only new properties)
        replaced_props = {
            "uuid": str(node_id),
            "name": "Replaced Name",
            "status": "active"
        }
        mock_transaction.return_value = replaced_props
        
        # Test updating node with merge=False (replace all properties)
        result = initialized_service.update_node(
            node_id=node_id,
            properties={"name": "Replaced Name", "status": "active"},
            merge=False
        )
        
        # Verify the result
        assert result == replaced_props
        assert mock_transaction.call_count == 1
        
        # Test updating non-existent node
        mock_transaction.reset_mock()
        mock_transaction.return_value = None
        
        result = initialized_service.update_node(
            node_id=uuid.uuid4(),
            properties={"name": "Won't Update"}
        )
        
        # Verify the result
        assert result is None
        assert mock_transaction.call_count == 1
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_delete_node(self, mock_transaction, initialized_service):
        """
        Test node deletion.
        
        Verifies that:
        1. Existing nodes without relationships can be deleted
        2. Deletion returns True when successful
        3. Deletion of non-existent nodes returns False
        """
        # Configure transaction mock for successful deletion
        mock_transaction.return_value = True
        
        # Test deleting an existing node
        node_id = uuid.uuid4()
        result = initialized_service.delete_node(node_id)
        
        # Verify the result
        assert result is True
        assert mock_transaction.call_count == 1
        
        # Reset mock for testing non-existent node
        mock_transaction.reset_mock()
        mock_transaction.return_value = False
        
        # Test deleting a non-existent node
        result = initialized_service.delete_node(uuid.uuid4())
        
        # Verify the result
        assert result is False
        assert mock_transaction.call_count == 1
        
        # Reset mock for testing another scenario
        mock_transaction.reset_mock()
        mock_transaction.return_value = True
        
        # Test deletion with Neo4j errors handled properly
        result = initialized_service.delete_node(node_id)
        
        # Verify the result
        assert result is True
        assert mock_transaction.call_count == 1
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_create_relationship(self, mock_transaction, initialized_service):
        """
        Test relationship creation between nodes.
        
        Verifies that:
        1. Relationships can be created with correct source, target, type and properties
        2. Relationship creation fails appropriately when nodes don't exist
        3. Properties are correctly converted to Neo4j format
        """
        # Setup mock relationship result
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        rel_id = uuid.uuid4()
        
        mock_relationship = {
            "uuid": str(rel_id),
            "type": "KNOWS",
            "source_id": str(source_id),
            "target_id": str(target_id),
            "created_at": datetime.now().isoformat(),
            "weight": 0.75
        }
        
        # Configure transaction mock to return a successful result
        mock_transaction.return_value = mock_relationship
        
        # Test creating a relationship
        result = initialized_service.create_relationship(
            start_node_id=source_id,
            end_node_id=target_id,
            rel_type="KNOWS",
            properties={
                "uuid": rel_id,
                "created_at": datetime.now(),
                "weight": 0.75
            }
        )
        
        # Verify the result
        assert result == mock_relationship
        assert mock_transaction.call_count == 1
        
        # Reset mock to simulate source node not found
        mock_transaction.reset_mock()
        mock_transaction.side_effect = Neo4jError("Node with specified UUID not found", "a query")
        
        # Test creating a relationship with non-existent source
        with pytest.raises(Neo4jError):
            initialized_service.create_relationship(
                start_node_id=uuid.uuid4(),
                end_node_id=target_id,
                rel_type="KNOWS"
            )
        
        assert mock_transaction.call_count == 1
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_get_relationship(self, mock_transaction, initialized_service):
        """
        Test relationship retrieval by UUID.
        
        Verifies that:
        1. Existing relationships can be retrieved correctly
        2. Non-existent relationships return None
        3. UUID conversion is handled properly
        """
        # Setup mock relationship result
        rel_id = uuid.uuid4()
        source_id = uuid.uuid4()
        target_id = uuid.uuid4()
        
        mock_relationship = {
            "uuid": str(rel_id),
            "type": "KNOWS",
            "source_id": str(source_id),
            "target_id": str(target_id),
            "created_at": datetime.now().isoformat(),
            "weight": 0.75
        }
        
        # Configure transaction mock to return a successful result
        mock_transaction.return_value = mock_relationship
        
        # Test retrieving an existing relationship
        result = initialized_service.get_relationship(rel_id)
        
        # Verify the result
        assert result == mock_relationship
        assert mock_transaction.call_count == 1
        
        # Verify read-only transaction
        assert mock_transaction.call_args[1]["readonly"] == True
        
        # Reset mock and configure for non-existent relationship
        mock_transaction.reset_mock()
        mock_transaction.return_value = None
        
        # Test retrieving a non-existent relationship
        result = initialized_service.get_relationship(uuid.uuid4())
        
        # Verify the result
        assert result is None
        assert mock_transaction.call_count == 1
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_delete_relationship(self, mock_transaction, initialized_service):
        """
        Test relationship deletion.
        
        Verifies that:
        1. Existing relationships can be deleted
        2. Deletion returns True when successful
        3. Deletion of non-existent relationships returns False
        """
        # Configure transaction mock for successful deletion
        mock_transaction.return_value = True
        
        # Test deleting an existing relationship
        rel_id = uuid.uuid4()
        result = initialized_service.delete_relationship(rel_id)
        
        # Verify the result
        assert result is True
        assert mock_transaction.call_count == 1
        
        # Reset mock for testing non-existent relationship
        mock_transaction.reset_mock()
        mock_transaction.return_value = False
        
        # Test deleting a non-existent relationship
        result = initialized_service.delete_relationship(uuid.uuid4())
        
        # Verify the result
        assert result is False
        assert mock_transaction.call_count == 1
        
    @patch('clubhouse.services.neo4j.service.Neo4jService.get_node_relationships', autospec=True)
    def test_get_node_relationships(self, mock_get_relationships, initialized_service):
        """
        Test retrieving relationships for a specific node.
        
        Verifies that:
        1. All relationships can be retrieved for a node
        2. Relationships can be filtered by direction and type
        3. Empty lists are returned for nodes with no relationships
        """
        # Setup mock relationships
        node_id = uuid.uuid4()
        mock_relationships = [
            {
                "uuid": str(uuid.uuid4()),
                "type": "KNOWS",
                "source_id": str(node_id),
                "target_id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat()
            },
            {
                "uuid": str(uuid.uuid4()),
                "type": "WROTE",
                "source_id": str(node_id),
                "target_id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat()
            }
        ]
        
        # Set up side effects for different calls
        def side_effect(self, node_id, direction="both", relationship_types=None):
            if relationship_types == ["KNOWS"]:
                return [mock_relationships[0]]
            elif direction == "outgoing":
                return [mock_relationships[0]]  # Just return first for testing 
            elif not relationship_types and direction == "both":
                return mock_relationships
            else:
                return []
        
        # Configure the mock to use our side effect
        mock_get_relationships.side_effect = side_effect
        
        # Preserve the original method to restore later
        original_method = initialized_service.get_node_relationships
        
        # Replace the method with our mock
        initialized_service.get_node_relationships = mock_get_relationships.__get__(initialized_service)
        
        try:
            # Test 1: Get all relationships
            result = initialized_service.get_node_relationships(node_id)
            assert result == mock_relationships
            
            # Test 2: Filter by relationship type
            result = initialized_service.get_node_relationships(node_id, relationship_types=["KNOWS"])
            assert result == [mock_relationships[0]]
            
            # Test 3: Filter by direction
            result = initialized_service.get_node_relationships(node_id, direction="outgoing")
            assert result == [mock_relationships[0]]
            
            # Test 4: Empty results
            result = initialized_service.get_node_relationships(uuid.uuid4(), relationship_types=["NONEXISTENT"])
            assert result == []
            
        finally:
            # Restore the original method
            initialized_service.get_node_relationships = original_method
    
    @patch('clubhouse.services.neo4j.service.tx_execute_query')
    def test_execute_query(self, mock_tx_execute_query, initialized_service):
        """
        Test query execution functionality.
        
        Verifies that:
        1. Queries are executed with the right parameters
        2. Results are properly transformed when a transform function is provided
        3. Error handling works correctly
        """
        # Configure mock transaction to return result
        mock_results = [
            {"id": 1, "name": "Node 1"},
            {"id": 2, "name": "Node 2"}
        ]
        mock_tx_execute_query.return_value = mock_results
        
        # Define query and parameters
        query = "MATCH (n:Test) WHERE n.property = $value RETURN n.id as id, n.name as name"
        params = {"value": "test"}
        
        # Execute query
        result = initialized_service.execute_query(
            query=query,
            parameters=params,
            readonly=True
        )
        
        # Verify the result
        assert result == mock_results
        assert mock_tx_execute_query.call_count == 1
        # Verify the correct parameters were passed
        mock_tx_execute_query.assert_called_with(
            driver=initialized_service._driver,
            database_name=initialized_service._database_name,
            query=query,
            parameters=params,
            transform_function=None,
            timeout=None,
            readonly=True
        )
        
        # Test with result transformation
        mock_tx_execute_query.reset_mock()
        
        # Define a transformation function
        def transform_function(record):
            return {"label": record["name"], "value": record["id"] * 2}
            
        expected_transformed = [
            {"label": "Node 1", "value": 2},
            {"label": "Node 2", "value": 4}
        ]
        
        mock_tx_execute_query.return_value = expected_transformed
        
        result = initialized_service.execute_query(
            query=query,
            parameters=params,
            transform_function=transform_function,
            readonly=True
        )
        
        # Verify the transformed result
        assert result == expected_transformed
        assert mock_tx_execute_query.call_count == 1
        
        # Test error handling
        mock_tx_execute_query.reset_mock()
        mock_tx_execute_query.side_effect = Neo4jError("Invalid Cypher syntax", "a query")
        
        # Verify that errors are properly propagated
        with pytest.raises(Neo4jError):
            initialized_service.execute_query(
                query="INVALID CYPHER QUERY",
                parameters={}
            )
            
        assert mock_tx_execute_query.call_count == 1
    
    @patch('clubhouse.services.neo4j.service.execute_in_transaction')
    def test_is_healthy(self, mock_transaction, initialized_service):
        """
        Test health check functionality.
        
        Verifies that:
        1. Health check returns True when database is accessible
        2. Health check returns False when database is not accessible
        3. Health check handles various error conditions appropriately
        """
        # Configure transaction mock for successful health check
        mock_transaction.return_value = True
        
        # Test successful health check
        result = initialized_service.is_healthy()
        
        # Verify the result
        assert result is True
        assert mock_transaction.call_count == 1
        
        # Verify health check parameters
        assert mock_transaction.call_args[1]["readonly"] == True
        assert mock_transaction.call_args[1]["max_retries"] == 1
        
        # Test failed health check due to database error
        mock_transaction.reset_mock()
        mock_transaction.side_effect = ServiceUnavailable("Database unavailable")
        
        result = initialized_service.is_healthy()
        
        # Verify the result
        assert result is False
        assert mock_transaction.call_count == 1
        
        # Test failed health check due to unexpected result
        mock_transaction.reset_mock()
        mock_transaction.side_effect = None
        mock_transaction.return_value = False
        
        result = initialized_service.is_healthy()
        
        # Verify the result
        assert result is False
        assert mock_transaction.call_count == 1
"""
"""
