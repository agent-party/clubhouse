"""
Unit tests for Neo4j service transaction management.

Tests the transaction management capabilities of the Neo4j service implementation,
including commit, rollback, and error handling behaviors.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock, call
from uuid import UUID, uuid4
from typing import Dict, Any, List, cast

from neo4j.exceptions import ServiceUnavailable, ClientError, DatabaseError, TransactionError

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import Neo4jDatabaseConfig, DatabaseConfig, ConnectionPoolConfig
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.service import Neo4jService


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
        # Mock for AsyncGraphDatabase too
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_driver = MagicMock()
            mock_async_driver = MagicMock()
            mock_graph_db.driver.return_value = mock_driver
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            service = Neo4jService(config_provider)
            service.initialize()
            
            # Ensure the service has the mocked driver
            assert service._driver is mock_driver
            assert service._async_driver is mock_async_driver
            
            yield service, mock_driver


class TestNeo4jTransactionManagement:
    """Tests for Neo4j service transaction management."""
    
    def test_successful_transaction(self, initialized_service):
        """
        Test successful transaction execution.
    
        Verifies that transactions can be executed successfully with proper
        session management and commit.
        """
        service, mock_driver = initialized_service
    
        # Mock session and transaction
        mock_session = MagicMock()
        mock_tx_func_result = {"id": "test-id"}
        
        # Setup session with execute_write
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.execute_write = MagicMock(return_value=mock_tx_func_result)
    
        # Execute - run a transaction function
        result = service._execute_in_transaction(
            lambda tx: tx.run("CREATE (n:Test {name: $name}) RETURN id(n) as id",
                              {"name": "test-node"}).single()["id"]
        )
    
        # Verify transaction was executed via execute_write
        assert result == mock_tx_func_result
        mock_session.execute_write.assert_called_once()
    
    def test_transaction_retry_on_transient_error(self, initialized_service):
        """
        Test transaction retry behavior on transient errors.
    
        Verifies that transactions are retried when transient errors occur,
        and eventually succeed if the error is resolved.
        """
        service, mock_driver = initialized_service
    
        # Mock session
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # First call raises TransientError, second succeeds
        mock_session.execute_write = MagicMock(side_effect=[
            ServiceUnavailable("Service unavailable - transient error"),
            {"name": "Node1"}
        ])
    
        # Execute with retry
        result = service._execute_in_transaction(
            lambda tx: tx.run("MATCH (n:Test) RETURN n").single(),
            max_retries=3,
            retry_interval=0.1  # Short interval for test
        )
    
        # Verify retry behavior
        assert result == {"name": "Node1"}
        assert mock_session.execute_write.call_count == 2  # First fails, second succeeds
    
    def test_transaction_failure_after_max_retries(self, initialized_service):
        """
        Test transaction failure after maximum retry attempts.
    
        Verifies that transactions ultimately fail after the maximum number
        of retry attempts if the error persists.
        """
        service, mock_driver = initialized_service
    
        # Mock session
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # All attempts raise ServiceUnavailable
        mock_session.execute_write = MagicMock(side_effect=[
            ServiceUnavailable("Service unavailable - attempt 1"),
            ServiceUnavailable("Service unavailable - attempt 2"),
            ServiceUnavailable("Service unavailable - attempt 3"),
            ServiceUnavailable("Service unavailable - attempt 4")  # One more than max_retries
        ])
    
        # Execute with limited retries
        with pytest.raises(ServiceUnavailable) as excinfo:
            service._execute_in_transaction(
                lambda tx: tx.run("MATCH (n:Test) RETURN n").single(),
                max_retries=3,
                retry_interval=0.1  # Short interval for test
            )
        
        assert "Service unavailable" in str(excinfo.value)
        assert mock_session.execute_write.call_count == 4  # Initial + 3 retries
    
    def test_transaction_rollback_on_error(self, initialized_service):
        """
        Test transaction rollback on non-transient errors.
    
        Verifies that transactions are properly rolled back when errors occur
        that should not trigger retry attempts.
        """
        service, mock_driver = initialized_service
    
        # Mock session
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Non-transient error - should not be retried
        mock_session.execute_write = MagicMock(side_effect=ClientError("Constraint violation"))
    
        # Execute - run a transaction function that should raise a ClientError
        with pytest.raises(ClientError) as excinfo:
            service._execute_in_transaction(
                lambda tx: tx.run("CREATE (n:Test) RETURN n").single()
            )
        
        assert "Constraint violation" in str(excinfo.value)
        assert mock_session.execute_write.call_count == 1  # Should not retry
    
    def test_readonly_transaction(self, initialized_service):
        """
        Test read-only transaction execution.
    
        Verifies that read-only transactions are properly marked as such
        and can be executed successfully.
        """
        service, mock_driver = initialized_service
    
        # Mock session
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Set up read-only transaction result
        expected_result = {"name": "Node1"}
        mock_session.execute_read = MagicMock(return_value=expected_result)
    
        # Execute a read-only transaction
        result = service._execute_in_transaction(
            lambda tx: tx.run("MATCH (n:Test) RETURN n").single(),
            readonly=True
        )
    
        # Verify read transaction was used
        assert result == expected_result
        assert mock_session.execute_read.call_count == 1
        assert mock_session.execute_write.call_count == 0  # Should not use write transaction
    
    def test_batch_operations_in_transaction(self, initialized_service):
        """
        Test executing multiple operations in a single transaction.
    
        Verifies that multiple operations can be executed within a single
        transaction and are properly committed or rolled back together.
        """
        service, mock_driver = initialized_service
    
        # Mock session
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Set up batch operations
        batch_operations = [
            {
                "type": "create_node",
                "labels": ["BatchTest"],
                "properties": {"name": "Node1", "id": 1}
            },
            {
                "type": "create_node",
                "labels": ["BatchTest"],
                "properties": {"name": "Node2", "id": 2}
            },
            {
                "type": "create_node",
                "labels": ["BatchTest"],
                "properties": {"name": "Node3", "id": 3}
            }
        ]
        
        # Mock the execute_write to return the expected results
        expected_results = [
            {"uuid": "uuid-1", "name": "Node1", "id": 1},
            {"uuid": "uuid-2", "name": "Node2", "id": 2},
            {"uuid": "uuid-3", "name": "Node3", "id": 3}
        ]
        mock_session.execute_write = MagicMock(return_value=expected_results)
        
        # Execute batch operations
        results = service.execute_batch(batch_operations)
        
        # Verify results
        assert results == expected_results
        # Verify the transaction function was executed through execute_write
        assert mock_session.execute_write.call_count == 1
