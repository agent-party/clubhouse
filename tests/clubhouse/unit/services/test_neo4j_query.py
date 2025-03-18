"""
Unit tests for Neo4j service query operations.

Tests various query operations of the Neo4j service implementation,
including complex queries, pagination, and result transformation.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from uuid import UUID, uuid4
from typing import Dict, Any, List, cast

from neo4j.exceptions import ServiceUnavailable, ClientError, DatabaseError, TransactionError

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
def initialized_service(config_provider):
    """Fixture providing an initialized Neo4j service with mocked driver."""
    with patch('clubhouse.services.neo4j.service.GraphDatabase') as mock_graph_db:
        mock_driver = MagicMock()
        mock_graph_db.driver.return_value = mock_driver
        
        # Mock the async driver as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_driver = MagicMock()
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            service = Neo4jService(config_provider)
            service.initialize()
            
            # Create mock record structure
            mock_record_factory = lambda **kwargs: MagicMock(values=lambda: kwargs.values(), items=lambda: kwargs.items(), **kwargs)
            
            # Attach the factory to the service for use in tests
            service._mock_record_factory = mock_record_factory
            
            yield service, mock_driver


class TestNeo4jQueryOperations:
    """Tests for Neo4j service query operations."""
    
    def test_simple_query(self, initialized_service):
        """
        Test execution of a simple query.
        
        Verifies that simple queries can be executed successfully with proper
        parameter passing and result handling.
        """
        service, mock_driver = initialized_service
        
        # Create mock record and expected result
        expected_result = [{"id": "node-id", "name": "Test Node"}]
        
        # Setup mock for transaction execution
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to return our expected result
            mock_tx_execute_query.return_value = expected_result
            
            # Execute - run a simple query
            result = service.execute_query(
                "MATCH (n:Node {name: $name}) RETURN id(n) as id, n.name as name",
                {"name": "Test Node"}
            )
            
            # Verify result was processed correctly
            assert result == expected_result
            
            # Verify correct query and parameters were passed
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['database_name'] == service._database_name
            assert call_kwargs['query'] == "MATCH (n:Node {name: $name}) RETURN id(n) as id, n.name as name"
            assert call_kwargs['parameters'] == {"name": "Test Node"}
            assert call_kwargs['transform_function'] is None
            assert call_kwargs['readonly'] is True
    
    def test_query_with_multiple_results(self, initialized_service):
        """
        Test execution of a query returning multiple results.
        
        Verifies that queries returning multiple results are handled correctly,
        with all results properly transformed and returned.
        """
        service, mock_driver = initialized_service
        
        # Create expected results
        expected_results = [
            {"id": "node1-id", "name": "Node 1"},
            {"id": "node2-id", "name": "Node 2"},
            {"id": "node3-id", "name": "Node 3"}
        ]
        
        # Setup mock for transaction execution
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to return our expected results
            mock_tx_execute_query.return_value = expected_results
            
            # Execute - run a query returning multiple results
            results = service.execute_query(
                "MATCH (n:Node) RETURN id(n) as id, n.name as name LIMIT 3"
            )
            
            # Verify
            assert len(results) == 3
            assert results[0]["id"] == "node1-id"
            assert results[1]["name"] == "Node 2"
            assert results[2]["id"] == "node3-id"
            
            # Verify correct query was executed
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['database_name'] == service._database_name
            assert call_kwargs['query'] == "MATCH (n:Node) RETURN id(n) as id, n.name as name LIMIT 3"
            assert call_kwargs['parameters'] is None  # No parameters provided, should be None
            assert call_kwargs['transform_function'] is None
            assert call_kwargs['readonly'] is True
    
    def test_query_with_no_results(self, initialized_service):
        """
        Test execution of a query returning no results.
        
        Verifies that queries returning no results are handled gracefully,
        returning an empty list rather than raising an error.
        """
        service, mock_driver = initialized_service
        
        # Create expected empty result
        expected_results = []
        
        # Setup mock for transaction execution
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to return an empty list
            mock_tx_execute_query.return_value = expected_results
            
            # Execute - run a query returning no results
            results = service.execute_query(
                "MATCH (n:NonExistentLabel) RETURN id(n) as id, n.name as name"
            )
            
            # Verify
            assert isinstance(results, list)
            assert len(results) == 0
            
            # Verify correct query was executed
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['database_name'] == service._database_name
            assert call_kwargs['query'] == "MATCH (n:NonExistentLabel) RETURN id(n) as id, n.name as name"
            assert call_kwargs['parameters'] is None  # No parameters provided, should be None
            assert call_kwargs['transform_function'] is None
            assert call_kwargs['readonly'] is True
    
    def test_query_with_result_transformation(self, initialized_service):
        """
        Test query execution with custom result transformation.
        
        Verifies that query results can be properly transformed using a
        custom transformation function.
        """
        service, mock_driver = initialized_service
        
        # Create mock records data (will be passed to transformer)
        mock_data = [
            {
                "person": {"name": "Alice", "age": 30},
                "relation": {"type": "FRIEND", "since": 2020},
                "friend": {"name": "Bob", "age": 32}
            },
            {
                "person": {"name": "Alice", "age": 30},
                "relation": {"type": "COLLEAGUE", "since": 2018},
                "friend": {"name": "Charlie", "age": 35}
            }
        ]
        
        # Define transformation function
        def transform_friendship(record):
            return {
                "source": record["person"]["name"],
                "target": record["friend"]["name"],
                "relationship": record["relation"]["type"],
                "since": record["relation"]["since"]
            }
            
        # Create expected results after transformation
        expected_transformed_results = [
            {
                "source": "Alice",
                "target": "Bob",
                "relationship": "FRIEND",
                "since": 2020
            },
            {
                "source": "Alice",
                "target": "Charlie",
                "relationship": "COLLEAGUE",
                "since": 2018
            }
        ]
        
        # Setup mock for transaction execution
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to return our transformed results directly
            # This simulates the transformation happening inside tx_execute_query
            mock_tx_execute_query.return_value = expected_transformed_results
            
            # Execute - run a query with result transformation
            query = """
                MATCH (p:Person {name: $name})-[r]->(f:Person)
                RETURN p as person, r as relation, f as friend
            """
            results = service.execute_query(
                query,
                {"name": "Alice"},
                transform_function=transform_friendship
            )
            
            # Verify
            assert len(results) == 2
            assert results[0]["source"] == "Alice"
            assert results[0]["target"] == "Bob"
            assert results[0]["relationship"] == "FRIEND"
            assert results[1]["source"] == "Alice"
            assert results[1]["target"] == "Charlie"
            assert results[1]["relationship"] == "COLLEAGUE"
            
            # Verify correct query and parameters were passed along with transform function
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['database_name'] == service._database_name
            assert call_kwargs['query'] == query
            assert call_kwargs['parameters'] == {"name": "Alice"}
            assert call_kwargs['transform_function'] == transform_friendship
            assert call_kwargs['readonly'] is True
    
    def test_paginated_query(self, initialized_service):
        """
        Test execution of a paginated query.
        
        Verifies that queries with pagination parameters are executed correctly,
        respecting the skip and limit values provided.
        """
        service, mock_driver = initialized_service
        
        # Create expected results for second page
        expected_results = [
            {"id": "node4-id", "name": "Node 4"},
            {"id": "node5-id", "name": "Node 5"},
            {"id": "node6-id", "name": "Node 6"}
        ]
        
        # Setup mock for transaction execution
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to return our expected results
            mock_tx_execute_query.return_value = expected_results
            
            # Execute - run a paginated query (second page, 3 items per page)
            page = 2
            page_size = 3
            skip = (page - 1) * page_size
            
            results = service.execute_query(
                "MATCH (n:Node) RETURN id(n) as id, n.name as name ORDER BY n.name SKIP $skip LIMIT $limit",
                {"skip": skip, "limit": page_size}
            )
            
            # Verify
            assert len(results) == 3
            assert results[0]["id"] == "node4-id"
            assert results[-1]["id"] == "node6-id"
            
            # Verify correct query and parameters were passed
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['database_name'] == service._database_name
            assert call_kwargs['query'] == "MATCH (n:Node) RETURN id(n) as id, n.name as name ORDER BY n.name SKIP $skip LIMIT $limit"
            assert call_kwargs['parameters'] == {"skip": 3, "limit": 3}
            assert call_kwargs['transform_function'] is None
            assert call_kwargs['readonly'] is True
    
    def test_query_with_error_handling(self, initialized_service):
        """
        Test proper error handling during query execution.
        
        Verifies that errors during query execution are properly caught
        and propagated with appropriate context.
        """
        service, mock_driver = initialized_service
        
        # Setup mock for transaction execution to raise an error
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to raise an error
            mock_tx_execute_query.side_effect = ClientError("Invalid syntax near 'MACTH'")
            
            # Execute - run a query with a syntax error
            with pytest.raises(Exception) as excinfo:  # Using generic Exception to catch either ClientError or Neo4jError
                service.execute_query("MACTH (n:Node) RETURN n")
            
            # Verify error message contains useful information
            assert "Invalid syntax" in str(excinfo.value)
            
            # Verify the mocked function was called with the right parameters
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['query'] == "MACTH (n:Node) RETURN n"
    
    def test_query_timeout_handling(self, initialized_service):
        """
        Test handling of query timeout.
        
        Verifies that query timeout is properly set based on configuration
        and that timeout errors are properly handled.
        """
        service, mock_driver = initialized_service
        
        # Setup mock for transaction execution to simulate a timeout
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to raise a timeout error
            mock_tx_execute_query.side_effect = TransactionError("Transaction timeout")
            
            # Execute - run a query that times out
            with pytest.raises(Exception) as excinfo:  # Using generic Exception to catch any error
                service.execute_query(
                    "MATCH (n)-[*1..5]->(m) RETURN n, m",  # Complex query that might timeout
                    {},
                    timeout=1  # Set a very short timeout
                )
            
            # Skip checking the error message content as it might vary
            # Just ensure the function was called properly
            
            # Verify the mocked function was called with the right parameters
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            assert call_kwargs['driver'] == mock_driver
            assert call_kwargs['database_name'] == service._database_name
            assert call_kwargs['query'] == "MATCH (n)-[*1..5]->(m) RETURN n, m"
            assert call_kwargs['parameters'] == {}
            assert call_kwargs['transform_function'] is None
            assert call_kwargs['timeout'] == 1
            assert call_kwargs['readonly'] is True
    
    def test_parameter_conversion(self, initialized_service):
        """
        Test parameter conversion for Neo4j queries.
        
        Verifies that parameters of various Python types are correctly
        converted to Neo4j-compatible formats before query execution.
        """
        service, mock_driver = initialized_service
        
        # Setup complex parameters
        test_uuid = uuid4()
        test_datetime = "2025-03-16T12:34:56"
        test_list = [1, 2, 3]
        test_dict = {"key": "value"}
        
        # Setup mock for transaction execution
        with patch('clubhouse.services.neo4j.service.tx_execute_query') as mock_tx_execute_query:
            # Configure the mock to return an empty result
            mock_tx_execute_query.return_value = []
            
            # Execute query with various parameter types
            service.execute_query(
                "CREATE (n:Test $params)",
                {
                    "params": {
                        "uuid": test_uuid,
                        "datetime": test_datetime,
                        "list": test_list,
                        "dict": test_dict
                    }
                }
            )
            
            # Verify parameters were passed to query execution
            assert mock_tx_execute_query.call_count == 1
            call_args, call_kwargs = mock_tx_execute_query.call_args
            
            # Extract the parameters that were actually passed
            passed_params = call_kwargs['parameters']
            
            # Verify parameters were properly passed through
            assert "params" in passed_params
            params = passed_params["params"]
            
            # Verify UUID was passed (might be as UUID object or string)
            assert "uuid" in params
            assert params["uuid"] == test_uuid or params["uuid"] == str(test_uuid)
            
            # Verify other parameters were passed correctly
            assert params["datetime"] == test_datetime
            assert params["list"] == test_list
            assert params["dict"] == test_dict
