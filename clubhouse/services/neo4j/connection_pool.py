"""
Neo4j connection pooling and management for the Clubhouse platform.

This module provides enhanced connection pool management for Neo4j,
ensuring efficient connection handling, monitoring, and optimization.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from neo4j import Driver, Session, Transaction, Result
from neo4j.exceptions import (
    Neo4jError, ServiceUnavailable, ClientError, DatabaseError, 
    TransientError
)

logger = logging.getLogger(__name__)


class PoolStatus(str, Enum):
    """Status of the connection pool."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    INACTIVE = "INACTIVE"


class ConnectionPoolManager:
    """
    Manager for Neo4j connection pool.
    
    This class provides enhanced management of Neo4j connection pools,
    including health monitoring, connection recycling, and optimization
    of connection utilization for improved performance and reliability.
    """
    
    def __init__(
        self, 
        driver: Driver, 
        database_name: str = "neo4j",
        max_consecutive_errors: int = 5,
        error_threshold_seconds: int = 60,
        health_check_interval_seconds: int = 300
    ) -> None:
        """
        Initialize the connection pool manager.
        
        Args:
            driver: Neo4j driver instance
            database_name: Name of the database to connect to
            max_consecutive_errors: Maximum consecutive errors before marking pool as degraded
            error_threshold_seconds: Time window for counting errors
            health_check_interval_seconds: Interval between automatic health checks
        """
        self._driver = driver
        self._database_name = database_name
        self._max_consecutive_errors = max_consecutive_errors
        self._error_threshold_seconds = error_threshold_seconds
        self._health_check_interval_seconds = health_check_interval_seconds
        
        # Connection pool metrics
        self._last_health_check: Optional[datetime] = None
        self._last_error: Optional[Exception] = None
        self._error_count = 0
        self._last_error_time: Optional[datetime] = None
        self._consecutive_errors = 0
        self._total_sessions_created = 0
        self._total_transactions_created = 0
        self._active_sessions = 0
        self._pool_status = PoolStatus.INACTIVE
        
        # Thread safety
        self._lock = Lock()
    
    def get_session(self, readonly: bool = False) -> Session:
        """
        Get a session from the connection pool.
        
        Args:
            readonly: Whether the session should be read-only
            
        Returns:
            Neo4j session object
            
        Raises:
            ServiceUnavailable: If the connection pool is not healthy
        """
        # Check if health check is needed
        self._maybe_check_health()
        
        # Check if pool is usable
        if self._pool_status == PoolStatus.INACTIVE:
            raise ServiceUnavailable("Connection pool is not initialized")
        
        if self._pool_status == PoolStatus.CRITICAL:
            logger.warning("Using connection pool in CRITICAL state")
        
        try:
            session = self._driver.session(
                database=self._database_name,
                default_access_mode="READ" if readonly else "WRITE"
            )
            
            with self._lock:
                self._total_sessions_created += 1
                self._active_sessions += 1
            
            # Wrap session to track when it's closed
            original_close = session.close
            
            def instrumented_close():
                with self._lock:
                    self._active_sessions = max(0, self._active_sessions - 1)
                return original_close()
            
            session.close = instrumented_close  # type: ignore
            
            return session
        
        except Exception as e:
            self._record_error(e)
            raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        readonly: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and return the results.
        
        This is a convenience method that manages the session and transaction
        lifecycle, while recording metrics.
        
        Args:
            query: Cypher query to execute
            params: Optional parameters for the query
            readonly: Whether the query is read-only
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            Neo4jError: If there is an error executing the query
        """
        with self.get_session(readonly=readonly) as session:
            result: Result
            
            if readonly:
                result = session.run(query, params or {})
            else:
                tx = session.begin_transaction()
                with self._lock:
                    self._total_transactions_created += 1
                
                try:
                    result = tx.run(query, params or {})
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    self._record_error(e)
                    raise
            
            # Convert result to a list of dictionaries
            records = []
            for record in result:
                records.append(dict(record))
            
            return records
    
    def check_health(self) -> PoolStatus:
        """
        Check the health of the connection pool.
        
        This method executes a simple query to verify that the connection
        pool is working correctly.
        
        Returns:
            Current status of the connection pool
        """
        try:
            # Execute a simple query to check connection
            with self.get_session(readonly=True) as session:
                session.run("RETURN 1 AS test").single()
            
            # Reset error count on successful health check
            with self._lock:
                self._last_health_check = datetime.now()
                
                # If we had consecutive errors but now succeeded, reset counter
                if self._consecutive_errors > 0:
                    logger.info(f"Connection pool recovered after {self._consecutive_errors} consecutive errors")
                    self._consecutive_errors = 0
                
                # Update pool status based on recent error history
                if self._error_count > 0 and self._last_error_time:
                    error_window = datetime.now() - timedelta(seconds=self._error_threshold_seconds)
                    
                    if self._last_error_time > error_window:
                        if self._error_count >= self._max_consecutive_errors:
                            self._pool_status = PoolStatus.DEGRADED
                        else:
                            self._pool_status = PoolStatus.HEALTHY
                    else:
                        # Errors were outside our window, reset count
                        self._error_count = 0
                        self._pool_status = PoolStatus.HEALTHY
                else:
                    self._pool_status = PoolStatus.HEALTHY
            
            return self._pool_status
            
        except Exception as e:
            self._record_error(e)
            
            with self._lock:
                self._last_health_check = datetime.now()
                
                if self._consecutive_errors >= self._max_consecutive_errors:
                    self._pool_status = PoolStatus.CRITICAL
                else:
                    self._pool_status = PoolStatus.DEGRADED
            
            logger.warning(f"Connection pool health check failed: {str(e)}")
            return self._pool_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the connection pool.
        
        Returns:
            Dictionary containing pool metrics and status
        """
        with self._lock:
            driver_metrics = {}
            
            # Try to get metrics from the driver if available
            try:
                driver_metrics = self._driver.get_metrics()
            except (AttributeError, Exception) as e:
                logger.debug(f"Unable to get driver metrics: {str(e)}")
            
            metrics = {
                "status": self._pool_status,
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
                "consecutive_errors": self._consecutive_errors,
                "error_count": self._error_count,
                "last_error": str(self._last_error) if self._last_error else None,
                "last_error_time": self._last_error_time.isoformat() if self._last_error_time else None,
                "total_sessions_created": self._total_sessions_created,
                "total_transactions_created": self._total_transactions_created,
                "active_sessions": self._active_sessions,
                "driver_metrics": driver_metrics
            }
            
            return metrics
    
    def reset(self) -> None:
        """
        Reset pool metrics and status.
        
        This can be used after resolving connection issues to reset
        error counters and pool status.
        """
        with self._lock:
            self._error_count = 0
            self._consecutive_errors = 0
            self._last_error = None
            self._last_error_time = None
            self._pool_status = PoolStatus.HEALTHY
            logger.info("Connection pool metrics reset")
    
    def _record_error(self, error: Exception) -> None:
        """
        Record a connection error.
        
        Args:
            error: The exception that occurred
        """
        with self._lock:
            self._last_error = error
            self._last_error_time = datetime.now()
            self._error_count += 1
            self._consecutive_errors += 1
            
            # Update pool status based on consecutive errors
            if self._consecutive_errors >= self._max_consecutive_errors:
                old_status = self._pool_status
                self._pool_status = PoolStatus.CRITICAL
                
                if old_status != PoolStatus.CRITICAL:
                    logger.error(f"Connection pool status changed to CRITICAL after {self._consecutive_errors} consecutive errors")
            elif self._consecutive_errors > 0:
                old_status = self._pool_status
                self._pool_status = PoolStatus.DEGRADED
                
                if old_status == PoolStatus.HEALTHY:
                    logger.warning(f"Connection pool status changed to DEGRADED after {self._consecutive_errors} errors")
    
    def _maybe_check_health(self) -> None:
        """
        Check health if the last check was too long ago.
        """
        if not self._last_health_check or (
            datetime.now() - self._last_health_check > timedelta(seconds=self._health_check_interval_seconds)
        ):
            # Avoid concurrent health checks with lock
            with self._lock:
                # Double-check after acquiring lock
                if not self._last_health_check or (
                    datetime.now() - self._last_health_check > timedelta(seconds=self._health_check_interval_seconds)
                ):
                    logger.debug("Performing automatic health check")
                    # Release lock during potentially slow operation
                    self._lock.release()
                    try:
                        self.check_health()
                    finally:
                        self._lock.acquire()


class QueryExecutionStrategy:
    """
    Strategy for executing Neo4j queries with optimizations.
    
    This class provides various strategies for executing Neo4j queries,
    including caching, batching, and automatic retry, to optimize
    performance and reliability.
    """
    
    def __init__(
        self, 
        pool_manager: ConnectionPoolManager,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 60,
        max_retries: int = 3,
        retry_interval_seconds: float = 1.0
    ) -> None:
        """
        Initialize the query execution strategy.
        
        Args:
            pool_manager: Connection pool manager
            enable_cache: Whether to enable result caching
            cache_ttl_seconds: Time-to-live for cached results in seconds
            max_retries: Maximum number of retries for transient errors
            retry_interval_seconds: Interval between retries in seconds
        """
        self._pool_manager = pool_manager
        self._enable_cache = enable_cache
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_retries = max_retries
        self._retry_interval_seconds = retry_interval_seconds
        
        # Simple in-memory cache
        self._query_cache: Dict[str, Tuple[List[Dict[str, Any]], datetime]] = {}
        
        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._retries = 0
        self._successful_queries = 0
        self._failed_queries = 0
        
        # Thread safety
        self._lock = Lock()
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        readonly: bool = True,
        use_cache: Optional[bool] = None,
        cache_ttl_seconds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query with caching and retry support.
        
        Args:
            query: Cypher query to execute
            params: Optional parameters for the query
            readonly: Whether the query is read-only
            use_cache: Override default cache setting
            cache_ttl_seconds: Override default cache TTL
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            Neo4jError: If there is an error executing the query after retries
        """
        # Determine if we should use cache
        should_use_cache = (
            use_cache if use_cache is not None else 
            (self._enable_cache and readonly)
        )
        
        # Check cache if enabled and query is readonly
        if should_use_cache:
            cache_key = self._get_cache_key(query, params)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result is not None:
                with self._lock:
                    self._cache_hits += 1
                return cached_result
            
            with self._lock:
                self._cache_misses += 1
        
        # Execute query with retry
        retry_count = 0
        last_error = None
        
        while retry_count <= self._max_retries:
            try:
                result = self._pool_manager.execute_query(query, params, readonly)
                
                with self._lock:
                    self._successful_queries += 1
                
                # Cache the result if needed
                if should_use_cache:
                    ttl = cache_ttl_seconds or self._cache_ttl_seconds
                    self._add_to_cache(self._get_cache_key(query, params), result, ttl)
                
                return result
                
            except (ServiceUnavailable, TransientError) as e:
                # These are retryable errors
                retry_count += 1
                last_error = e
                
                with self._lock:
                    self._retries += 1
                
                if retry_count <= self._max_retries:
                    logger.warning(
                        f"Retryable error on attempt {retry_count}/{self._max_retries}: {str(e)}"
                    )
                    time.sleep(self._retry_interval_seconds * retry_count)  # Exponential backoff
                else:
                    with self._lock:
                        self._failed_queries += 1
                    raise
                    
            except Exception as e:
                # Non-retryable errors
                with self._lock:
                    self._failed_queries += 1
                raise
        
        # If we get here, we've exhausted retries
        with self._lock:
            self._failed_queries += 1
        
        if last_error:
            raise last_error
        else:
            raise ServiceUnavailable("Query failed after retries, no specific error recorded")
    
    def execute_batch(
        self, 
        queries: List[Tuple[str, Optional[Dict[str, Any]]]], 
        readonly: bool = False
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute a batch of queries in a single transaction.
        
        Args:
            queries: List of (query, params) tuples
            readonly: Whether the queries are read-only
            
        Returns:
            List of result lists, one for each query
            
        Raises:
            Neo4jError: If there is an error executing the batch
        """
        if not queries:
            return []
        
        with self._pool_manager.get_session(readonly=readonly) as session:
            tx = session.begin_transaction()
            
            try:
                results = []
                
                for query, params in queries:
                    result = tx.run(query, params or {})
                    records = [dict(record) for record in result]
                    results.append(records)
                
                tx.commit()
                
                with self._lock:
                    self._successful_queries += len(queries)
                
                return results
                
            except Exception as e:
                tx.rollback()
                
                with self._lock:
                    self._failed_queries += len(queries)
                
                raise
    
    def clear_cache(self) -> None:
        """Clear the query result cache."""
        with self._lock:
            self._query_cache.clear()
            logger.debug("Query cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about query execution.
        
        Returns:
            Dictionary containing query execution metrics
        """
        with self._lock:
            cache_size = len(self._query_cache)
            
            # Calculate cache memory usage (approximate)
            cache_memory = 0
            for cache_key, (result, _) in self._query_cache.items():
                # Rough estimation based on key length and result size
                cache_memory += len(cache_key) * 2  # 2 bytes per character
                cache_memory += sum(len(str(r)) * 2 for r in result)  # Rough approximation
            
            # Clean expired cache entries while we're here
            self._clean_cache()
            
            metrics = {
                "cache_enabled": self._enable_cache,
                "cache_ttl_seconds": self._cache_ttl_seconds,
                "cache_size": cache_size,
                "cache_memory_bytes": cache_memory,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "retry_count": self._retries,
                "successful_queries": self._successful_queries,
                "failed_queries": self._failed_queries,
                "cache_hit_ratio": (
                    self._cache_hits / (self._cache_hits + self._cache_misses)
                    if (self._cache_hits + self._cache_misses) > 0 else 0
                )
            }
            
            return metrics
    
    def _get_cache_key(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """
        Generate a cache key for a query and params.
        
        Args:
            query: Cypher query
            params: Query parameters
            
        Returns:
            Cache key string
        """
        # Simple implementation - in production, consider a more efficient approach
        params_str = str(sorted(params.items())) if params else ""
        return f"{query}:{params_str}"
    
    def _add_to_cache(
        self, 
        key: str, 
        result: List[Dict[str, Any]], 
        ttl_seconds: int
    ) -> None:
        """
        Add a result to the cache.
        
        Args:
            key: Cache key
            result: Query result
            ttl_seconds: Time-to-live in seconds
        """
        with self._lock:
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
            self._query_cache[key] = (result, expiry)
            
            # Clean cache if it's growing too large (simple approach)
            if len(self._query_cache) > 1000:
                self._clean_cache()
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get a result from the cache if it exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found or expired
        """
        with self._lock:
            if key not in self._query_cache:
                return None
            
            result, expiry = self._query_cache[key]
            
            if datetime.now() > expiry:
                # Expired
                del self._query_cache[key]
                return None
            
            return result
    
    def _clean_cache(self) -> None:
        """Remove expired entries from the cache."""
        now = datetime.now()
        keys_to_remove = []
        
        for key, (_, expiry) in self._query_cache.items():
            if now > expiry:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._query_cache[key]
        
        logger.debug(f"Cleaned {len(keys_to_remove)} expired cache entries")
