"""
Unit tests for Neo4j service connection management.

Tests the connection handling, initialization, and error recovery capabilities
of the Neo4j service implementation.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import UUID, uuid4
from typing import Dict, Any, List, cast

from neo4j.exceptions import ServiceUnavailable, ClientError, DatabaseError

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


class TestNeo4jServiceConnection:
    """Tests for Neo4j service connection management."""
    
    @patch('clubhouse.services.neo4j.service.GraphDatabase')
    def test_initialize_connection(self, mock_graph_db, config_provider):
        """
        Test successful initialization of Neo4j connection.
        
        Verifies that the service initializes correctly with the provided
        configuration and establishes a connection to the database.
        """
        # Setup mock driver
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        # Setup mock for AsyncGraphDatabase as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_driver = MagicMock()
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            # Configure driver and session mocks
            mock_graph_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = mock_result
            mock_result.single.return_value = mock_record
            mock_record.get.return_value = 1
            
            # Instantiate service
            service = Neo4jService(config_provider)
            
            # Execute - initialize connection
            service.initialize()
            
            # Verify
            assert service._initialized is True
            assert service._driver is mock_driver
            assert service._async_driver is mock_async_driver
            
            # Verify connection parameters were used correctly
            config = config_provider.get()
            mock_graph_db.driver.assert_called_once()
            # Verify URI has the correct format for neo4j connection
            uri_arg = mock_graph_db.driver.call_args[0][0]
            assert "bolt://" in uri_arg
            assert "localhost:7687" in uri_arg
            
            # Verify authentication info was used (using keywords which is more reliable)
            auth_kwarg = mock_graph_db.driver.call_args[1].get('auth')
            assert auth_kwarg == ('neo4j', 'password')
    
    @patch('clubhouse.services.neo4j.service.GraphDatabase')
    def test_connection_failure_handling(self, mock_graph_db, config_provider):
        """
        Test handling of connection failures during initialization.
        
        Verifies that the service properly handles and reports connection
        failures during initialization.
        """
        # Setup mock to raise connection error
        mock_graph_db.driver.side_effect = ServiceUnavailable("Connection refused")
        
        # Setup mock for AsyncGraphDatabase as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_graph_db.driver.side_effect = ServiceUnavailable("Connection refused")
            
            # Instantiate service
            service = Neo4jService(config_provider)
            
            # Execute & Verify - should handle exception and set initialized to False
            with pytest.raises(ServiceUnavailable):
                service.initialize()
            
            assert service._initialized is False
            assert service._last_error is not None
            assert service._driver is None
            assert service._async_driver is None
    
    @patch('clubhouse.services.neo4j.service.GraphDatabase')
    def test_shutdown(self, mock_graph_db, config_provider):
        """
        Test successful shutdown of Neo4j connection.
        
        Verifies that the service properly closes the connection to the database
        during shutdown.
        """
        # Setup mock driver
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"test": 1}
        
        # Setup mock for AsyncGraphDatabase as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_driver = MagicMock()
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            # Configure driver and session mocks
            mock_graph_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = mock_result
            
            # Instantiate service and initialize
            service = Neo4jService(config_provider)
            service.initialize()
            
            # Execute - shutdown service
            service.shutdown()
            
            # Verify connections were closed
            mock_driver.close.assert_called_once()
            mock_async_driver.close.assert_called_once()
            assert service._driver is None
            assert service._async_driver is None
            assert service._initialized is False
    
    @patch('clubhouse.services.neo4j.service.GraphDatabase')
    def test_reconnection_after_failure(self, mock_graph_db, config_provider):
        """
        Test reconnection after a connection failure.
        
        Verifies that the service properly attempts to reconnect after
        encountering a connection failure.
        """
        # Setup mocks for first connection attempt (should fail)
        mock_graph_db.driver.side_effect = [
            ServiceUnavailable("Connection refused"),  # First attempt fails
            MagicMock()  # Second attempt succeeds
        ]
        
        # Setup mock for AsyncGraphDatabase as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_graph_db.driver.side_effect = [
                ServiceUnavailable("Connection refused"),  # First attempt fails
                MagicMock()  # Second attempt succeeds
            ]
            
            # Instantiate service
            service = Neo4jService(config_provider)
            
            # First connection attempt should fail
            with pytest.raises(ServiceUnavailable):
                service.initialize()
                
            assert service._initialized is False
            assert service._driver is None
            assert service._async_driver is None
            
            # Setup mock session and result for second attempt
            mock_driver = MagicMock()
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_graph_db.driver.side_effect = None
            mock_graph_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value = mock_result
            mock_result.single.return_value = {"test": 1}
            
            # Set up async mock for second attempt
            mock_async_driver = MagicMock()
            mock_async_graph_db.driver.side_effect = None
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            # Second attempt should succeed
            service.initialize()
            
            assert service._initialized is True
            assert service._driver is mock_driver
            assert service._async_driver is mock_async_driver
    
    @patch('clubhouse.services.neo4j.service.GraphDatabase')
    def test_connection_health_check(self, mock_graph_db, config_provider):
        """
        Test health check functionality for Neo4j connection.
        
        Verifies that the health check properly reports a healthy connection
        when the database is responding correctly.
        """
        # Setup mock driver and session for health check
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        # Setup transaction mocks
        mock_tx = MagicMock()
        
        # Configure the mocks
        mock_graph_db.driver.return_value = mock_driver
        
        # Setup mock for transaction execution based on how _execute_in_transaction works
        def mock_execute_tx_side_effect(transaction_function, **kwargs):
            return transaction_function(mock_tx)
            
        # Setup mock for AsyncGraphDatabase as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_driver = MagicMock()
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            # Configure the transaction result
            mock_tx.run.return_value = mock_result
            mock_result.single.return_value = mock_record
            mock_record.get.return_value = 1
            
            # Instantiate and initialize service
            service = Neo4jService(config_provider)
            service.initialize()
            
            # Mock the _execute_in_transaction method to control transaction execution
            with patch.object(service, '_execute_in_transaction', side_effect=mock_execute_tx_side_effect):
                # Execute health check
                is_healthy = service.is_healthy()
                
                # Verify health is reported correctly
                assert is_healthy is True
                
                # Verify correct query was executed in the transaction
                mock_tx.run.assert_called_with("RETURN 1 AS test")
    
    @patch('clubhouse.services.neo4j.service.GraphDatabase')
    def test_connection_health_check_unhealthy(self, mock_graph_db, config_provider):
        """
        Test health check failure detection.
        
        Verifies that the health check correctly reports an unhealthy connection
        when the database is not responding as expected.
        """
        # Setup mock driver and session to simulate failure
        mock_driver = MagicMock()
        
        # Setup mock for transaction execution that simulates a failure
        def mock_execute_tx_side_effect(*args, **kwargs):
            raise ServiceUnavailable("Database unavailable")
            
        # Setup mock for AsyncGraphDatabase as well
        with patch('clubhouse.services.neo4j.service.AsyncGraphDatabase') as mock_async_graph_db:
            mock_async_driver = MagicMock()
            mock_async_graph_db.driver.return_value = mock_async_driver
            
            # Configure the driver mock
            mock_graph_db.driver.return_value = mock_driver
            
            # Instantiate and initialize service
            service = Neo4jService(config_provider)
            service.initialize()
            
            # Mock the _execute_in_transaction method to simulate failure
            with patch.object(service, '_execute_in_transaction', side_effect=mock_execute_tx_side_effect):
                # Execute health check which should detect unhealthy state
                is_healthy = service.is_healthy()
                
                # Verify health is reported correctly as unhealthy
                assert is_healthy is False
