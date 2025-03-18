"""
Neo4j transaction management utilities.

This module provides utilities for managing Neo4j transactions with proper
error handling, retries, and best practices for database operations.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID

from neo4j import Driver, Transaction
from neo4j.exceptions import (
    Neo4jError, ServiceUnavailable, ClientError, TransientError
)

from clubhouse.services.neo4j.utils import params_to_neo4j

logger = logging.getLogger(__name__)

# Type variables for generic methods
T = TypeVar('T')
R = TypeVar('R')


def execute_in_transaction(
    driver: Driver,
    database_name: str,
    transaction_function: Callable[[Transaction], T],
    readonly: bool = False,
    max_retries: int = 3,
    retry_interval: float = 1.0
) -> T:
    """
    Execute a function within a Neo4j transaction with automatic retry.
    
    This function handles the complexities of Neo4j transactions, including
    automatic retries for transient errors, proper session management,
    and appropriate error handling.
    
    Args:
        driver: Neo4j driver instance
        database_name: Name of the database to use
        transaction_function: Function to execute within the transaction
        readonly: If True, execute as read-only transaction
        max_retries: Maximum number of retry attempts for transient errors
        retry_interval: Base time in seconds to wait between retry attempts
        
    Returns:
        The result of the transaction function
        
    Raises:
        Neo4jError: If a non-transient error occurs
        ServiceUnavailable: If the database is not available after retries
        ValueError: If the driver is None
    """
    if not driver:
        raise ValueError("Neo4j driver not initialized")
    
    attempts = 0
    last_error = None
    
    while attempts <= max_retries:
        try:
            with driver.session(database=database_name) as session:
                if readonly:
                    return session.execute_read(transaction_function)
                else:
                    return session.execute_write(transaction_function)
                
        except (ServiceUnavailable, TransientError) as e:
            # These are retryable errors
            attempts += 1
            last_error = e
            
            if attempts <= max_retries:
                wait_time = retry_interval * attempts  # Exponential backoff
                logger.warning(
                    f"Transient error in Neo4j transaction (attempt {attempts}/{max_retries}): "
                    f"{str(e)}. Retrying in {wait_time:.2f}s"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Max retries ({max_retries}) exceeded for Neo4j transaction: {str(e)}")
                raise
                
        except Neo4jError as e:
            # Non-transient errors should not be retried
            logger.error(f"Non-transient error in Neo4j transaction: {str(e)}")
            raise
    
    # If we get here, we've exceeded retries
    if last_error:
        raise last_error
    
    # This should never happen, but just in case
    raise ServiceUnavailable("Failed to execute Neo4j transaction for unknown reason")


def execute_query(
    driver: Driver,
    database_name: str,
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    transform_function: Optional[Callable[[Dict[str, Any]], R]] = None,
    timeout: Optional[int] = None,
    readonly: bool = True
) -> List[Any]:
    """
    Execute a Cypher query with parameters and return the results.
    
    This function provides a flexible way to execute arbitrary Cypher queries,
    with options for parameter passing, result transformation, and timeout control.
    
    Args:
        driver: Neo4j driver instance
        database_name: Name of the database to use
        query: Cypher query to execute
        parameters: Query parameters
        transform_function: Optional function to transform each result record
        timeout: Query timeout in seconds
        readonly: If True, execute as read-only query
        
    Returns:
        List of result records as dictionaries
        
    Raises:
        Neo4jError: If a database error occurs
        ValueError: If the driver is None
    """
    if not driver:
        raise ValueError("Neo4j driver not initialized")
    
    # Ensure parameters is a dictionary
    params = parameters or {}
    
    # Convert parameters to Neo4j-compatible format
    neo4j_params = params_to_neo4j(params)
    
    # Set up transaction metadata
    tx_metadata = {}
    if timeout:
        tx_metadata["timeout"] = timeout
    
    # Define transaction function
    def run_query(tx):
        result = tx.run(query, neo4j_params, **tx_metadata)
        records = list(result)
        
        # Transform records to dictionaries
        results = []
        for record in records:
            # Convert record to dict
            record_dict = {key: record[key] for key in record.keys()}
            
            # Apply custom transformation if provided
            if transform_function:
                record_dict = transform_function(record_dict)
            
            results.append(record_dict)
            
        return results
    
    # Execute query in transaction
    return execute_in_transaction(
        driver=driver,
        database_name=database_name,
        transaction_function=run_query,
        readonly=readonly
    )


def execute_batch_operation(
    driver: Driver,
    database_name: str,
    batch_statements: List[Dict[str, Any]]
) -> List[Any]:
    """
    Execute a batch of Cypher statements in a single transaction.
    
    This function allows multiple database operations to be executed
    within a single transaction, ensuring atomicity.
    
    Args:
        driver: Neo4j driver instance
        database_name: Name of the database to use
        batch_statements: List of statement dictionaries, each containing:
                          - query: The Cypher query to execute
                          - parameters: Parameters for the query
        
    Returns:
        List of results for each statement
        
    Raises:
        Neo4jError: If a database error occurs
        ValueError: If the driver is None or batch_statements is empty
    """
    if not driver:
        raise ValueError("Neo4j driver not initialized")
    
    if not batch_statements:
        return []
    
    def run_batch(tx):
        results = []
        for statement in batch_statements:
            query = statement.get("query")
            parameters = statement.get("parameters", {})
            
            if not query:
                raise ValueError("Missing query in batch statement")
            
            # Convert parameters to Neo4j-compatible format
            neo4j_params = params_to_neo4j(parameters)
            
            # Execute the statement
            result = tx.run(query, neo4j_params)
            
            # Process the result
            if statement.get("return_results", True):
                # Convert to list of dictionaries
                records = [
                    {key: record[key] for key in record.keys()}
                    for record in result
                ]
                results.append(records)
            else:
                # Just consume the result without returning
                result.consume()
                results.append(None)
        
        return results
    
    # Execute all statements in a single transaction
    return execute_in_transaction(
        driver=driver,
        database_name=database_name,
        transaction_function=run_batch,
        readonly=False  # Batch operations assume write operations by default
    )
