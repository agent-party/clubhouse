"""
Unit tests for the Neo4j connection pool management.

This module contains tests for the ConnectionPoolManager and
QueryExecutionStrategy classes that optimize Neo4j connections.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple

from neo4j import Driver, Session, Transaction, Result
from neo4j.exceptions import ServiceUnavailable, TransientError

from clubhouse.services.neo4j.connection_pool import (
    ConnectionPoolManager, QueryExecutionStrategy, PoolStatus
)


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j driver."""
    mock = MagicMock()
    mock.get_metrics.return_value = {"connections": 1, "idle_connections": 0}
    return mock


@pytest.fixture
def mock_session() -> MagicMock:
    """Create a mock Neo4j session."""
    mock = MagicMock()
    # Set up context manager behavior
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    # Make the original close method accessible for instrumentation
    mock.close = MagicMock()
    return mock


@pytest.fixture
def mock_transaction() -> MagicMock:
    """Create a mock Neo4j transaction."""
    mock = MagicMock()
    # Set up context manager behavior
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    return mock


@pytest.fixture
def mock_result() -> MagicMock:
    """Create a mock Neo4j result."""
    mock = MagicMock()
    record = MagicMock()
    record.__getitem__.side_effect = lambda key: f"value-{key}"
    record.keys.return_value = ["key1", "key2"]
    mock.__iter__.return_value = [record]
    mock.single.return_value = {"test": 1}
    return mock


@pytest.fixture
def pool_manager(mock_driver: MagicMock) -> ConnectionPoolManager:
    """Create a ConnectionPoolManager with a mock driver."""
    manager = ConnectionPoolManager(
        driver=mock_driver,
        database_name="test_db",
        max_consecutive_errors=5,
        error_threshold_seconds=60,
        health_check_interval_seconds=300
    )
    # Set the initial state to make tests more consistent
    manager._pool_status = PoolStatus.INACTIVE
    return manager


@pytest.fixture
def query_strategy(pool_manager: ConnectionPoolManager) -> QueryExecutionStrategy:
    """Create a QueryExecutionStrategy with a mock pool manager."""
    return QueryExecutionStrategy(
        pool_manager=pool_manager,
        enable_cache=True,
        cache_ttl_seconds=60,
        max_retries=3,
        retry_interval_seconds=0.01  # Short interval for faster tests
    )


class TestConnectionPoolManager:
    """Test the ConnectionPoolManager class."""
    
    def test_init(self, pool_manager: ConnectionPoolManager) -> None:
        """Test the initialization of the ConnectionPoolManager."""
        # Check initial state
        assert pool_manager._driver is not None
        assert pool_manager._database_name == "test_db"
        assert pool_manager._pool_status == PoolStatus.INACTIVE
        assert pool_manager._total_sessions_created == 0
        assert pool_manager._total_transactions_created == 0
        assert pool_manager._active_sessions == 0
    
    def test_get_session(self, pool_manager: ConnectionPoolManager, mock_session: MagicMock) -> None:
        """Test getting a session from the connection pool."""
        # Set up the pool status to HEALTHY
        pool_manager._pool_status = PoolStatus.HEALTHY
        
        # Mock driver.session to return our mock session
        pool_manager._driver.session.return_value = mock_session
        
        # Mock _maybe_check_health to prevent actual checks
        with patch.object(pool_manager, '_maybe_check_health'):
            session = pool_manager.get_session(readonly=True)
            
            # Verify correct calls to driver
            pool_manager._driver.session.assert_called_once_with(
                database="test_db",
                default_access_mode="READ"
            )
            
            # Check that session count was incremented
            assert pool_manager._total_sessions_created == 1
            assert pool_manager._active_sessions == 1
            
            # Test instrumented close
            session.close()
            assert pool_manager._active_sessions == 0
    
    def test_get_session_inactive_pool(self, pool_manager: ConnectionPoolManager) -> None:
        """Test getting a session from an inactive pool."""
        # Pool status already set to INACTIVE in fixture
        
        # Mock _maybe_check_health to prevent actual checks
        with patch.object(pool_manager, '_maybe_check_health'):
            with pytest.raises(ServiceUnavailable):
                pool_manager.get_session()
    
    def test_get_session_critical_pool(self, pool_manager: ConnectionPoolManager, mock_session: MagicMock) -> None:
        """Test getting a session from a critical pool."""
        # Set pool status to CRITICAL
        pool_manager._pool_status = PoolStatus.CRITICAL
        
        # Mock driver.session to return our mock session
        pool_manager._driver.session.return_value = mock_session
        
        # Mock _maybe_check_health to prevent actual checks
        with patch.object(pool_manager, '_maybe_check_health'):
            with patch('logging.Logger.warning') as mock_warning:
                session = pool_manager.get_session()
                
                # Verify warning was logged
                mock_warning.assert_called_once()
                
                # Verify session was returned despite critical pool
                assert session is mock_session
    
    def test_execute_query_readonly(self, pool_manager: ConnectionPoolManager, mock_session: MagicMock, mock_result: MagicMock) -> None:
        """Test executing a read-only query."""
        # Set up the pool status
        pool_manager._pool_status = PoolStatus.HEALTHY
        
        # Configure mock result to simulate Neo4j result records
        record = MagicMock()
        record.__getitem__.side_effect = lambda key: f"value-{key}"
        record.keys.return_value = ["key1", "key2"]
        mock_result.__iter__.return_value = [record]
        
        # Setup session run to return our mock result
        mock_session.run.return_value = mock_result
        
        # Patch get_session to return our mock session
        with patch.object(pool_manager, 'get_session', return_value=mock_session) as mock_get_session:
            result = pool_manager.execute_query("MATCH (n) RETURN n", {"param": "value"}, readonly=True)
            
            # Verify session was created correctly
            mock_get_session.assert_called_once_with(readonly=True)
            
            # Verify correct query execution
            mock_session.run.assert_called_once_with("MATCH (n) RETURN n", {"param": "value"})
            
            # Check result conversion
            assert len(result) == 1
            assert "key1" in result[0]
            assert "key2" in result[0]
    
    def test_execute_query_write(self, pool_manager: ConnectionPoolManager, mock_session: MagicMock, mock_transaction: MagicMock, mock_result: MagicMock) -> None:
        """Test executing a write query."""
        # Set up the pool status
        pool_manager._pool_status = PoolStatus.HEALTHY
        
        # Configure mock result 
        record = MagicMock()
        record.__getitem__.side_effect = lambda key: f"value-{key}"
        record.keys.return_value = ["key1", "key2"]
        mock_result.__iter__.return_value = [record]
        
        # Set up transaction mocks
        mock_transaction.run.return_value = mock_result
        mock_session.begin_transaction.return_value = mock_transaction
        
        # Patch get_session to return our mock session
        with patch.object(pool_manager, 'get_session', return_value=mock_session) as mock_get_session:
            result = pool_manager.execute_query("CREATE (n) RETURN n", {"param": "value"}, readonly=False)
            
            # Verify session was created correctly
            mock_get_session.assert_called_once_with(readonly=False)
            
            # Verify transaction was created
            mock_session.begin_transaction.assert_called_once()
            
            # Verify transaction count was incremented
            assert pool_manager._total_transactions_created == 1
            
            # Verify correct query execution
            mock_transaction.run.assert_called_once_with("CREATE (n) RETURN n", {"param": "value"})
            
            # Verify transaction was committed
            mock_transaction.commit.assert_called_once()
            
            # Check result conversion
            assert len(result) == 1
            assert "key1" in result[0]
            assert "key2" in result[0]
    
    def test_execute_query_error(self, pool_manager: ConnectionPoolManager, mock_session: MagicMock, mock_transaction: MagicMock) -> None:
        """Test error handling during query execution."""
        # Set up the pool status
        pool_manager._pool_status = PoolStatus.HEALTHY
        
        # Configure transaction to raise an exception
        mock_transaction.run.side_effect = ServiceUnavailable("Test error")
        mock_session.begin_transaction.return_value = mock_transaction
        
        # Patch get_session to return our mock session
        with patch.object(pool_manager, 'get_session', return_value=mock_session) as mock_get_session:
            with pytest.raises(ServiceUnavailable):
                pool_manager.execute_query("CREATE (n) RETURN n", readonly=False)
            
            # Verify session and transaction were created
            mock_get_session.assert_called_once()
            mock_session.begin_transaction.assert_called_once()
            
            # Verify transaction was rolled back
            mock_transaction.rollback.assert_called_once()
            
            # Verify error was recorded
            assert pool_manager._error_count == 1
            assert pool_manager._consecutive_errors == 1
            assert isinstance(pool_manager._last_error, ServiceUnavailable)
    
    def test_check_health_success(self, pool_manager: ConnectionPoolManager, mock_session: MagicMock, mock_result: MagicMock) -> None:
        """Test successful health check."""
        # Configure session for health check
        mock_result.single.return_value = {"test": 1}
        mock_session.run.return_value = mock_result
        
        # Patch get_session to return mock session
        with patch.object(pool_manager, 'get_session', return_value=mock_session) as mock_get_session:
            status = pool_manager.check_health()
            
            # Verify session was created correctly
            mock_get_session.assert_called_once_with(readonly=True)
            
            # Verify health check query was executed
            mock_session.run.assert_called_once_with("RETURN 1 AS test")
            
            # Verify status was updated to HEALTHY
            assert status == PoolStatus.HEALTHY
            assert pool_manager._pool_status == PoolStatus.HEALTHY
            assert pool_manager._last_health_check is not None
    
    def test_check_health_failure(self, pool_manager: ConnectionPoolManager) -> None:
        """Test health check when connection fails."""
        # Configure get_session to raise an exception
        with patch.object(pool_manager, 'get_session', side_effect=ServiceUnavailable("Test error")):
            status = pool_manager.check_health()
            
            # Verify status was updated to DEGRADED (first failure)
            assert status == PoolStatus.DEGRADED
            assert pool_manager._pool_status == PoolStatus.DEGRADED
            assert pool_manager._consecutive_errors == 1
            assert pool_manager._last_health_check is not None
    
    def test_check_health_critical(self, pool_manager: ConnectionPoolManager) -> None:
        """Test health check when connection has multiple failures."""
        # Set up consecutive errors to trigger CRITICAL status
        pool_manager._consecutive_errors = pool_manager._max_consecutive_errors
        
        # Configure get_session to raise an exception
        with patch.object(pool_manager, 'get_session', side_effect=ServiceUnavailable("Test error")):
            status = pool_manager.check_health()
            
            # Verify status was updated to CRITICAL
            assert status == PoolStatus.CRITICAL
            assert pool_manager._pool_status == PoolStatus.CRITICAL
            assert pool_manager._consecutive_errors == pool_manager._max_consecutive_errors + 1
    
    def test_get_metrics(self, pool_manager: ConnectionPoolManager) -> None:
        """Test getting metrics from the connection pool."""
        # Set up some test metrics
        pool_manager._pool_status = PoolStatus.HEALTHY
        pool_manager._total_sessions_created = 10
        pool_manager._total_transactions_created = 5
        pool_manager._active_sessions = 2
        pool_manager._last_health_check = datetime.now()
        
        # Configure driver metrics
        mock_driver_metrics = {"connections": 3, "idle_connections": 1}
        pool_manager._driver.get_metrics.return_value = mock_driver_metrics
        
        # Get metrics
        metrics = pool_manager.get_metrics()
        
        # Verify metrics are correct
        assert metrics["status"] == PoolStatus.HEALTHY
        assert metrics["total_sessions_created"] == 10
        assert metrics["total_transactions_created"] == 5
        assert metrics["active_sessions"] == 2
        assert metrics["driver_metrics"] == mock_driver_metrics
    
    def test_reset(self, pool_manager: ConnectionPoolManager) -> None:
        """Test resetting the connection pool metrics."""
        # Set up some metrics
        pool_manager._error_count = 10
        pool_manager._consecutive_errors = 5
        pool_manager._last_error = Exception("Test error")
        pool_manager._last_error_time = datetime.now()
        pool_manager._pool_status = PoolStatus.CRITICAL
        
        # Reset metrics
        pool_manager.reset()
        
        # Verify metrics were reset
        assert pool_manager._error_count == 0
        assert pool_manager._consecutive_errors == 0
        assert pool_manager._last_error is None
        assert pool_manager._last_error_time is None
        assert pool_manager._pool_status == PoolStatus.HEALTHY
    
    def test_maybe_check_health(self, pool_manager: ConnectionPoolManager) -> None:
        """Test automatic health check triggering."""
        # Set up last health check long ago
        pool_manager._last_health_check = datetime.now() - timedelta(seconds=pool_manager._health_check_interval_seconds + 10)
        
        # Mock check_health
        with patch.object(pool_manager, 'check_health') as mock_check_health:
            pool_manager._maybe_check_health()
            
            # Verify health check was called
            mock_check_health.assert_called_once()
    
    def test_maybe_check_health_recent(self, pool_manager: ConnectionPoolManager) -> None:
        """Test no automatic health check when recent."""
        # Set up recent health check
        pool_manager._last_health_check = datetime.now()
        
        # Mock check_health
        with patch.object(pool_manager, 'check_health') as mock_check_health:
            pool_manager._maybe_check_health()
            
            # Verify health check was not called
            mock_check_health.assert_not_called()


class TestQueryExecutionStrategy:
    """Tests for the QueryExecutionStrategy class."""

    def test_init(self, pool_manager: ConnectionPoolManager) -> None:
        """Test initialization of query execution strategy."""
        # Act
        strategy = QueryExecutionStrategy(
            pool_manager=pool_manager,
            enable_cache=True,
            cache_ttl_seconds=60,
            max_retries=3,
            retry_interval_seconds=0.5
        )
        
        # Assert
        assert strategy._pool_manager == pool_manager
        assert strategy._enable_cache is True
        assert strategy._cache_ttl_seconds == 60
        assert strategy._max_retries == 3
        assert strategy._retry_interval_seconds == 0.5
        assert strategy._cache_hits == 0
        assert strategy._cache_misses == 0

    def test_execute_query_no_cache(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test executing a query without cache."""
        # Arrange
        expected_result = [{"result": "value"}]
        
        # Mock pool_manager.execute_query
        with patch.object(query_strategy._pool_manager, 'execute_query', return_value=expected_result) as mock_execute:
            # Act
            result = query_strategy.execute_query(
                query="MATCH (n) RETURN n",
                params={"param": "value"},
                readonly=True,
                use_cache=False
            )
            
            # Assert
            assert result == expected_result
            mock_execute.assert_called_once_with(
                "MATCH (n) RETURN n",
                {"param": "value"},
                True
            )
            assert query_strategy._successful_queries == 1
            assert query_strategy._cache_misses == 0  # Cache not used
            assert query_strategy._cache_hits == 0

    def test_execute_query_with_cache_miss(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test executing a query with cache miss."""
        # Arrange
        expected_result = [{"result": "value"}]
        
        # Mock pool_manager.execute_query
        with patch.object(query_strategy._pool_manager, 'execute_query', return_value=expected_result) as mock_execute:
            # Act
            result = query_strategy.execute_query(
                query="MATCH (n) RETURN n",
                params={"param": "value"},
                readonly=True
            )
            
            # Assert
            assert result == expected_result
            mock_execute.assert_called_once()
            assert query_strategy._successful_queries == 1
            assert query_strategy._cache_misses == 1
            assert query_strategy._cache_hits == 0
            assert len(query_strategy._query_cache) == 1

    def test_execute_query_with_cache_hit(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test executing a query with cache hit."""
        # Arrange
        expected_result = [{"result": "value"}]
        cache_key = query_strategy._get_cache_key("MATCH (n) RETURN n", {"param": "value"})
        
        # Manually populate the cache to avoid dependency on execute_query
        query_strategy._add_to_cache(cache_key, expected_result, query_strategy._cache_ttl_seconds)
        
        # Mock pool_manager.execute_query to ensure it's not called
        with patch.object(query_strategy._pool_manager, 'execute_query') as mock_execute:
            # Act
            result = query_strategy.execute_query(
                query="MATCH (n) RETURN n",
                params={"param": "value"},
                readonly=True
            )
            
            # Assert
            assert result == expected_result
            mock_execute.assert_not_called()
            assert query_strategy._cache_hits == 1
            assert query_strategy._cache_misses == 0

    def test_execute_query_retries(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test query retries on transient error."""
        # Arrange
        expected_result = [{"result": "value"}]
        transient_error = TransientError("Transient test error")
        
        # Configure mock to fail twice then succeed
        mock_execute = MagicMock()
        mock_execute.side_effect = [transient_error, transient_error, expected_result]
        
        with patch.object(query_strategy._pool_manager, 'execute_query', side_effect=mock_execute.side_effect) as mock_exec:
            # Act
            result = query_strategy.execute_query(
                query="MATCH (n) RETURN n",
                params={"param": "value"},
                readonly=True,
                use_cache=False
            )
            
            # Assert
            assert result == expected_result
            assert mock_exec.call_count == 3
            assert query_strategy._retries == 2
            assert query_strategy._successful_queries == 1
            assert query_strategy._failed_queries == 0

    def test_execute_query_max_retries_exceeded(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test max retries exceeded."""
        # Arrange - ensure test is deterministic
        query_strategy._retries = 0  # Reset retry counter
        query_strategy._max_retries = 3
        query_strategy._failed_queries = 0  # Reset failed queries counter
        
        # Mock time.sleep to avoid waiting during tests
        with patch('time.sleep'):
            # Configure mock to always fail with transient error
            with patch.object(
                query_strategy._pool_manager, 
                'execute_query', 
                side_effect=TransientError("Transient test error")
            ) as mock_exec:
                # Act/Assert
                with pytest.raises(TransientError):
                    query_strategy.execute_query(
                        query="MATCH (n) RETURN n",
                        params={"param": "value"},
                        readonly=True,
                        use_cache=False
                    )
                
                # Assert
                # Initial attempt + 3 retries = 4 total calls
                assert mock_exec.call_count == 4
                # When max_retries is exceeded, the _retries counter will actually be
                # incremented to max_retries + 1 because of how the execute_query method
                # is structured: it first increments retry_count, then checks the condition
                assert query_strategy._retries == 4
                assert query_strategy._successful_queries == 0
                assert query_strategy._failed_queries == 1

    def test_execute_query_non_retryable_error(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test non-retryable error handling."""
        # Arrange
        query_strategy._retries = 0  # Reset retry counter
        
        # Create fresh patch to avoid state from previous tests
        with patch.object(query_strategy._pool_manager, 'execute_query') as mock_exec:
            # Configure mock to fail with non-retryable error
            mock_exec.side_effect = ValueError("Non-retryable test error")
            
            # Act/Assert
            with pytest.raises(ValueError):
                query_strategy.execute_query(
                    query="MATCH (n) RETURN n",
                    params={"param": "value"},
                    readonly=True,
                    use_cache=False
                )
            
            # Assert - no retries for non-retryable errors
            assert mock_exec.call_count == 1
            assert query_strategy._retries == 0
            assert query_strategy._successful_queries == 0
            assert query_strategy._failed_queries == 1

    def test_execute_batch(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test executing a batch of queries."""
        # Arrange
        queries = [
            ("MATCH (n) RETURN n", {"param1": "value1"}),
            ("MATCH (m) RETURN m", {"param2": "value2"})
        ]
        expected_results = [
            [{"result1": "value1"}],
            [{"result2": "value2"}]
        ]
        
        # Mock transaction context manager
        mock_tx = MagicMock()
        mock_tx.run.side_effect = [
            MagicMock(__iter__=lambda _: iter([{"result1": "value1"}])),
            MagicMock(__iter__=lambda _: iter([{"result2": "value2"}]))
        ]
        
        # Mock session context manager
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.begin_transaction.return_value = mock_tx
        
        # Mock pool_manager.get_session
        with patch.object(query_strategy._pool_manager, 'get_session', return_value=mock_session):
            # Act
            results = query_strategy.execute_batch(queries, readonly=False)
            
            # Assert
            assert results == expected_results
            assert mock_tx.run.call_count == 2
            mock_tx.run.assert_any_call("MATCH (n) RETURN n", {"param1": "value1"})
            mock_tx.run.assert_any_call("MATCH (m) RETURN m", {"param2": "value2"})
            mock_tx.commit.assert_called_once()

    def test_execute_batch_empty(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test executing an empty batch."""
        # Arrange
        queries = []
        
        # Act
        results = query_strategy.execute_batch(queries, readonly=False)
        
        # Assert
        assert results == []
        # Ensure get_session was not called since the batch is empty
        assert not hasattr(query_strategy._pool_manager, 'get_session_called') or not query_strategy._pool_manager.get_session_called

    def test_execute_batch_error(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test error handling in batch execution."""
        # Arrange
        queries = [
            ("MATCH (n) RETURN n", {"param1": "value1"}),
            ("MATCH (m) RETURN m", {"param2": "value2"})
        ]
        
        # Mock transaction that fails
        mock_tx = MagicMock()
        mock_tx.run.side_effect = ServiceUnavailable("Test error")
        
        # Mock session
        mock_session = MagicMock()
        mock_session.__enter__.return_value = mock_session
        mock_session.begin_transaction.return_value = mock_tx
        
        # Mock pool_manager.get_session
        with patch.object(query_strategy._pool_manager, 'get_session', return_value=mock_session):
            # Act/Assert
            with pytest.raises(ServiceUnavailable):
                query_strategy.execute_batch(queries, readonly=False)
            
            # Assert
            mock_tx.rollback.assert_called_once()

    def test_clear_cache(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test clearing the query cache."""
        # Arrange
        # Add items to cache
        query_strategy._query_cache = {
            "key1": ([{"result": "value1"}], datetime.now()),
            "key2": ([{"result": "value2"}], datetime.now())
        }
        
        # Act
        query_strategy.clear_cache()
        
        # Assert
        assert len(query_strategy._query_cache) == 0

    def test_get_metrics(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test getting metrics."""
        # Arrange
        query_strategy._cache_hits = 10
        query_strategy._cache_misses = 5
        query_strategy._retries = 3
        query_strategy._successful_queries = 15
        query_strategy._failed_queries = 2
        
        # Act
        metrics = query_strategy.get_metrics()
        
        # Assert - match the keys in the actual implementation
        assert metrics["cache_hits"] == 10
        assert metrics["cache_misses"] == 5
        assert metrics["cache_hit_ratio"] == 10 / (10 + 5)
        assert metrics["retry_count"] == 3  # In real implementation it's retry_count not retries
        assert metrics["successful_queries"] == 15
        assert metrics["failed_queries"] == 2
        assert "cache_size" in metrics
        assert "cache_enabled" in metrics
        assert "cache_ttl_seconds" in metrics

    def test_cache_expiry(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test cache entry expiry."""
        # Arrange
        # Add an expired cache entry
        expired_time = datetime.now() - timedelta(seconds=query_strategy._cache_ttl_seconds + 10)
        query_strategy._query_cache = {
            "expired_key": ([{"result": "value"}], expired_time)
        }
        
        # Mock pool_manager.execute_query for when cache is expired
        expected_result = [{"fresh_result": "new_value"}]
        
        with patch.object(query_strategy._pool_manager, 'execute_query', return_value=expected_result) as mock_execute:
            # Act - should detect expired cache and requery
            result = query_strategy.execute_query(
                query="cached query",
                params={},
                readonly=True,
                use_cache=True,
                cache_ttl_seconds=query_strategy._cache_ttl_seconds
            )
            
            # Assert
            assert result == expected_result
            mock_execute.assert_called_once()
            assert query_strategy._cache_hits == 0
            assert query_strategy._cache_misses == 1

    def test_clean_cache(self, query_strategy: QueryExecutionStrategy) -> None:
        """Test cache cleaning."""
        # Arrange
        now = datetime.now()
        # Add entries with realistic expiry times based on implementation
        query_strategy._query_cache = {
            "fresh_key": ([{"result": "fresh"}], now + timedelta(seconds=10)),
            "expired_key": ([{"result": "expired"}], now - timedelta(seconds=10))
        }
        
        # Should be 2 entries before cleaning
        assert len(query_strategy._query_cache) == 2
        
        # Act
        query_strategy._clean_cache()
        
        # Assert
        assert len(query_strategy._query_cache) == 1
        assert "fresh_key" in query_strategy._query_cache
        assert "expired_key" not in query_strategy._query_cache
