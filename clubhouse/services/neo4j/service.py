"""
Neo4j service implementation for the Clubhouse platform.

This module provides a concrete implementation of the Neo4jServiceProtocol,
offering graph database capabilities for storing and querying agent data.
"""

import logging
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast, Callable
from uuid import UUID, uuid4

from neo4j import GraphDatabase, AsyncGraphDatabase, Driver, AsyncDriver
from neo4j.exceptions import (
    Neo4jError, ServiceUnavailable, ClientError, DatabaseError, 
    TransientError
)
from neo4j.graph import Node, Relationship

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import Neo4jDatabaseConfig, DatabaseConfig
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.transaction import execute_in_transaction, execute_query as tx_execute_query
from clubhouse.services.neo4j.utils import node_to_dict, relationship_to_dict, params_to_neo4j, format_direction

logger = logging.getLogger(__name__)

# Type variables for generic methods
T = TypeVar('T')
R = TypeVar('R')


class Neo4jService(Neo4jServiceProtocol):
    """
    Implementation of the Neo4jServiceProtocol for interacting with Neo4j.
    
    This service provides a comprehensive interface for working with the Neo4j
    graph database, supporting CRUD operations for nodes and relationships,
    as well as more complex graph operations like path finding.
    """
    
    def __init__(self, config: ConfigProtocol[DatabaseConfig]) -> None:
        """
        Initialize the Neo4j service.
        
        Args:
            config: Configuration provider for database settings
        """
        self._config = config
        self._driver: Optional[Driver] = None
        self._async_driver: Optional[AsyncDriver] = None
        self._database_name: str = "neo4j"  # Default database name
        self._initialized = False
        self._last_error: Optional[Exception] = None
        self._connection_attempts = 0
        self._last_connection_attempt: float = 0.0
    
    def initialize(self) -> None:
        """
        Initialize the Neo4j service, establishing connections to the database.
        
        This method is called by the ServiceRegistry during application startup.
        It creates a driver instance using the configuration parameters and
        establishes a connection to the Neo4j database.
        
        Raises:
            ServiceUnavailable: If the Neo4j database is not available
            Neo4jError: If there is an error during initialization
        """
        try:
            logger.info("Initializing Neo4j service")
            self._connection_attempts += 1
            self._last_connection_attempt = time.time()
            
            # Get database configuration
            config = self._config.get()
            if not isinstance(config, Neo4jDatabaseConfig):
                raise ValueError(f"Expected Neo4jDatabaseConfig, got {type(config)}")
            
            # Store database name for later use
            self._database_name = config.database
            
            # Build connection URI from hosts
            hosts_str = ",".join(config.hosts)
            uri = f"bolt://{hosts_str}"
            
            # Configure connection pooling
            connection_kwargs = {}
            if config.connection_pool:
                connection_kwargs.update({
                    "max_connection_pool_size": config.connection_pool.max_size,
                    "connection_acquisition_timeout": config.connection_pool.connection_timeout_seconds
                })
            
            # Configure transaction settings
            if config.max_transaction_retry_time_seconds:
                connection_kwargs["max_transaction_retry_time"] = config.max_transaction_retry_time_seconds
            
            # Initialize synchronous driver
            self._driver = GraphDatabase.driver(
                uri, 
                auth=(config.username, config.password),
                **connection_kwargs
            )
            
            # Initialize asynchronous driver with same configuration
            self._async_driver = AsyncGraphDatabase.driver(
                uri,
                auth=(config.username, config.password),
                **connection_kwargs
            )
            
            # Verify connection by executing a simple query
            with self._driver.session(database=self._database_name) as session:  
                session.run("RETURN 1 AS test").single()
            
            self._initialized = True
            self._last_error = None
            logger.info(f"Successfully connected to Neo4j database at {uri}")
            
        except (ServiceUnavailable, Neo4jError) as e:
            self._initialized = False
            self._last_error = e
            
            # Close drivers if they were initialized
            if self._driver:
                self._driver.close()
                self._driver = None
            
            if self._async_driver:
                # Handle async close in a synchronous context
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._async_driver.close())
                    loop.close()
                except Exception as e:
                    logger.warning(f"Error closing async driver: {e}")
                self._async_driver = None
            
            logger.error(f"Failed to connect to Neo4j database: {str(e)}")
            raise

    def shutdown(self) -> None:
        """
        Shut down the Neo4j service, closing all connections.
        
        This method is called by the ServiceRegistry during application shutdown.
        It ensures that all connections to the database are properly closed.
        """
        logger.info("Shutting down Neo4j service")
        if self._driver:
            self._driver.close()
            self._driver = None
        
        if self._async_driver:
            # Handle async close in a synchronous context
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._async_driver.close())
                loop.close()
            except Exception as e:
                logger.warning(f"Error closing async driver: {e}")
            self._async_driver = None
        
        self._initialized = False
        logger.info("Neo4j service shut down successfully")

    def _ensure_initialized(self) -> None:
        """
        Ensure the service is initialized before performing operations.
        
        Raises:
            RuntimeError: If the service is not initialized
        """
        if not self._initialized or not self._driver:
            raise RuntimeError("Neo4j service not initialized - call initialize() first")

    def _execute_in_transaction(
        self, 
        transaction_function: Callable,
        readonly: bool = False,
        max_retries: int = 3,
        retry_interval: float = 1.0
    ) -> Any:
        """
        Execute a function within a Neo4j transaction with automatic retry.
        
        This is a wrapper around the transaction utility function to maintain
        consistent error handling and logging in the service.
        
        Args:
            transaction_function: Function to execute within the transaction
            readonly: If True, execute as read-only transaction
            max_retries: Maximum number of retry attempts for transient errors
            retry_interval: Time in seconds to wait between retry attempts
            
        Returns:
            The result of the transaction function
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        try:
            return execute_in_transaction(
                driver=self._driver,
                database_name=self._database_name,
                transaction_function=transaction_function,
                readonly=readonly,
                max_retries=max_retries,
                retry_interval=retry_interval
            )
        except (ServiceUnavailable, Neo4jError) as e:
            logger.error(f"Error executing Neo4j transaction: {str(e)}")
            raise

    def is_healthy(self) -> bool:
        """
        Check if the Neo4j service is healthy and can connect to the database.
        
        This method verifies that:
        1. The service is initialized
        2. The driver connection is active
        3. Basic query operations succeed
        
        Returns:
            True if the service is healthy and functioning, False otherwise
        """
        if not self._initialized or not self._driver:
            logger.warning("Health check failed: Neo4j service not initialized")
            return False
            
        try:
            # Define a simple transaction function to verify connectivity
            def health_check(tx):
                result = tx.run("RETURN 1 AS test")
                record = result.single()
                return record and record.get("test") == 1
                
            # Execute with short timeout and fewer retries for health checks
            is_healthy = self._execute_in_transaction(
                transaction_function=health_check,
                readonly=True,
                max_retries=1,
                retry_interval=0.5
            )
            
            if is_healthy:
                logger.debug("Neo4j health check successful")
            else:
                logger.warning("Neo4j health check failed: unexpected query result")
                
            return bool(is_healthy)
            
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {str(e)}")
            return False

    def create_node(
        self, 
        labels: Union[str, List[str]], 
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new node with the given labels and properties.
        
        Args:
            labels: One or more labels for the node
            properties: Properties for the node
            
        Returns:
            Dictionary representation of the created node
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Ensure labels is a list
        label_list = [labels] if isinstance(labels, str) else labels
        
        # Convert properties to Neo4j-compatible format
        neo4j_properties = params_to_neo4j(properties)
        
        # Build Cypher query
        label_clause = ":".join(label_list)
        query = f"CREATE (n:{label_clause} $properties) RETURN n"
        
        def create_node_tx(tx):
            result = tx.run(query, properties=neo4j_properties)
            record = result.single()
            if not record:
                raise Neo4jError("Failed to create node - no record returned", query)
            node = record["n"]
            return node_to_dict(node)
        
        try:
            return self._execute_in_transaction(create_node_tx, readonly=False)
        except Neo4jError as e:
            logger.error(f"Error creating node with labels {label_list}: {str(e)}")
            raise
    
    def get_node(self, node_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its UUID.
        
        Args:
            node_id: UUID of the node to retrieve
            
        Returns:
            Dictionary representation of the node or None if not found
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Convert UUID to string for the query
        uuid_str = str(node_id)
        
        # Build Cypher query
        query = "MATCH (n) WHERE n.uuid = $uuid RETURN n"
        
        def get_node_tx(tx):
            result = tx.run(query, uuid=uuid_str)
            record = result.single()
            if not record:
                return None
            return node_to_dict(record["n"])
        
        try:
            return self._execute_in_transaction(get_node_tx, readonly=True)
        except Neo4jError as e:
            logger.error(f"Error retrieving node with UUID {uuid_str}: {str(e)}")
            raise
    
    def update_node(
        self, 
        node_id: UUID, 
        properties: Dict[str, Any],
        merge: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Update a node's properties.
        
        Args:
            node_id: UUID of the node to update
            properties: New or updated properties for the node
            merge: If True, merge properties with existing ones; if False, replace all properties
            
        Returns:
            Dictionary representation of the updated node or None if not found
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Convert UUID to string for the query
        uuid_str = str(node_id)
        
        # Convert properties to Neo4j-compatible format
        neo4j_properties = params_to_neo4j(properties)
        
        def update_node_tx(tx):
            # First, check if the node exists
            check_result = tx.run(
                "MATCH (n) WHERE n.uuid = $uuid RETURN n",
                uuid=uuid_str
            )
            record = check_result.single()
            if not record:
                return None
                
            # Determine update strategy based on merge flag
            if merge:
                # Merge properties with existing ones
                set_clause = "SET n += $properties"
            else:
                # First reset all properties except uuid, then set new properties
                set_clause = "SET n = {uuid: $uuid} SET n += $properties"
                
            # Execute update
            result = tx.run(
                f"MATCH (n) WHERE n.uuid = $uuid {set_clause} RETURN n",
                uuid=uuid_str,
                properties=neo4j_properties
            )
            
            update_record = result.single()
            if not update_record:
                raise Neo4jError("Failed to update node - no record returned", "update_node")
                
            return node_to_dict(update_record["n"])
            
        try:
            return self._execute_in_transaction(update_node_tx, readonly=False)
        except Neo4jError as e:
            logger.error(f"Error updating node with UUID {uuid_str}: {str(e)}")
            raise
    
    def delete_node(self, node_id: UUID) -> bool:
        """
        Delete a node by its UUID.
        
        Args:
            node_id: UUID of the node to delete
            
        Returns:
            True if the node was deleted, False if it doesn't exist
            
        Raises:
            Neo4jError: If a database error occurs (e.g., constraints violation)
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Convert UUID to string for the query
        uuid_str = str(node_id)
        
        def delete_node_tx(tx):
            # Delete the node and all its relationships
            result = tx.run(
                """
                MATCH (n) 
                WHERE n.uuid = $uuid
                DETACH DELETE n
                RETURN count(n) AS deleted_count
                """,
                uuid=uuid_str
            )
            
            record = result.single()
            # Return True if at least one node was deleted
            return record is not None and record["deleted_count"] > 0
            
        try:
            return self._execute_in_transaction(delete_node_tx, readonly=False)
        except Neo4jError as e:
            logger.error(f"Error deleting node with UUID {uuid_str}: {str(e)}")
            raise
    
    def create_relationship(
        self, 
        start_node_id: UUID, 
        end_node_id: UUID, 
        rel_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            start_node_id: UUID of the start node
            end_node_id: UUID of the end node
            rel_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            Dictionary representation of the created relationship
            
        Raises:
            Neo4jError: If a database error occurs or nodes don't exist
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Ensure properties is a dictionary
        props = properties or {}
        
        # Convert UUIDs to strings for the query
        start_uuid = str(start_node_id)
        end_uuid = str(end_node_id)
        
        # Convert properties to Neo4j-compatible format
        neo4j_properties = params_to_neo4j(props)
        
        def create_relationship_tx(tx):
            # Create the relationship
            query = f"""
                MATCH (a), (b)
                WHERE a.uuid = $start_uuid AND b.uuid = $end_uuid
                CREATE (a)-[r:`{rel_type}` $properties]->(b)
                RETURN r, a, b
                """
            
            result = tx.run(
                query,
                start_uuid=start_uuid,
                end_uuid=end_uuid,
                properties=neo4j_properties
            )
            
            record = result.single()
            if not record:
                # Check if both nodes exist
                check_result = tx.run(
                    """
                    MATCH (a), (b)
                    WHERE a.uuid = $start_uuid AND b.uuid = $end_uuid
                    RETURN count(a) AS start_exists, count(b) AS end_exists
                    """,
                    start_uuid=start_uuid,
                    end_uuid=end_uuid
                )
                
                check_record = check_result.single()
                if check_record:
                    if check_record["start_exists"] == 0:
                        raise Neo4jError(f"Start node with UUID {start_uuid} not found", "create_relationship")
                    if check_record["end_exists"] == 0:
                        raise Neo4jError(f"End node with UUID {end_uuid} not found", "create_relationship")
                
                raise Neo4jError("Failed to create relationship", "create_relationship")
                
            # Convert relationship to dictionary
            rel_dict = relationship_to_dict(record["r"])
            
            # Add node information
            rel_dict["start_node"] = node_to_dict(record["a"])
            rel_dict["end_node"] = node_to_dict(record["b"])
            
            return rel_dict
            
        try:
            return self._execute_in_transaction(create_relationship_tx, readonly=False)
        except Neo4jError as e:
            logger.error(
                f"Error creating relationship of type '{rel_type}' between nodes "
                f"{start_uuid} and {end_uuid}: {str(e)}"
            )
            raise
    
    def get_relationship(self, relationship_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieve a relationship by its UUID.
        
        Args:
            relationship_id: UUID of the relationship to retrieve
            
        Returns:
            Dictionary representation of the relationship or None if not found
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Convert UUID to string for the query
        uuid_str = str(relationship_id)
        
        def get_relationship_tx(tx):
            # Query for the relationship and its connected nodes
            result = tx.run(
                """
                MATCH (a)-[r]->(b)
                WHERE r.uuid = $uuid
                RETURN r, a, b
                """,
                uuid=uuid_str
            )
            
            record = result.single()
            if not record:
                return None
                
            # Convert relationship to dictionary
            rel_dict = relationship_to_dict(record["r"])
            
            # Add node information
            rel_dict["start_node"] = node_to_dict(record["a"])
            rel_dict["end_node"] = node_to_dict(record["b"])
            
            return rel_dict
            
        try:
            return self._execute_in_transaction(get_relationship_tx, readonly=True)
        except Neo4jError as e:
            logger.error(f"Error retrieving relationship with UUID {uuid_str}: {str(e)}")
            raise
    
    def update_relationship(
        self, 
        relationship_id: UUID, 
        properties: Dict[str, Any],
        merge: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Update a relationship's properties.
        
        Args:
            relationship_id: UUID of the relationship to update
            properties: New or updated properties for the relationship
            merge: If True, merge properties with existing ones; if False, replace all properties
            
        Returns:
            Dictionary representation of the updated relationship or None if not found
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        # Convert UUID to string for the query
        uuid_str = str(relationship_id)
        
        # Convert properties to Neo4j-compatible format
        neo4j_properties = params_to_neo4j(properties)
        
        def update_relationship_tx(tx):
            # First, check if the relationship exists
            check_result = tx.run(
                """
                MATCH ()-[r]->()
                WHERE r.uuid = $uuid
                RETURN r
                """,
                uuid=uuid_str
            )
            
            if not check_result.single():
                return None
                
            # Determine update strategy based on merge flag
            if merge:
                # Merge properties with existing ones
                set_clause = "SET r += $properties"
            else:
                # First reset all properties except uuid, then set new properties
                set_clause = "SET r = {uuid: $uuid} SET r += $properties"
                
            # Execute update and return nodes for context
            result = tx.run(
                f"""
                MATCH (a)-[r]->(b)
                WHERE r.uuid = $uuid
                {set_clause}
                RETURN r, a, b
                """,
                uuid=uuid_str,
                properties=neo4j_properties
            )
            
            record = result.single()
            if not record:
                raise Neo4jError("Failed to update relationship - no record returned", "update_relationship")
                
            # Convert relationship to dictionary
            rel_dict = relationship_to_dict(record["r"])
            
            # Add node information
            rel_dict["start_node"] = node_to_dict(record["a"])
            rel_dict["end_node"] = node_to_dict(record["b"])
            
            return rel_dict
            
        try:
            return self._execute_in_transaction(update_relationship_tx, readonly=False)
        except Neo4jError as e:
            logger.error(f"Error updating relationship with UUID {uuid_str}: {str(e)}")
            raise
    
    def delete_relationship(self, relationship_id: UUID) -> bool:
        """
        Delete a relationship from the graph database.
        
        Args:
            relationship_id: UUID of the relationship to delete
            
        Returns:
            True if the relationship was deleted, False if it doesn't exist
            
        Raises:
            RuntimeError: If the service is not initialized
            Neo4jError: If there's an error deleting the relationship
        """
        self._ensure_initialized()
        
        # Convert UUID to string for the query
        uuid_str = str(relationship_id)
        
        def delete_relationship_tx(tx):
            # Delete the relationship
            result = tx.run(
                """
                MATCH ()-[r]->()
                WHERE r.uuid = $uuid
                DELETE r
                RETURN count(r) AS deleted_count
                """,
                uuid=uuid_str
            )
            
            record = result.single()
            # Return True if at least one relationship was deleted
            return record is not None and record["deleted_count"] > 0
            
        try:
            return self._execute_in_transaction(delete_relationship_tx, readonly=False)
        except Neo4jError as e:
            logger.error(f"Error deleting relationship with UUID {uuid_str}: {str(e)}")
            raise
    
    def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        transform_function: Optional[Callable[[Dict[str, Any]], R]] = None,
        timeout: Optional[int] = None,
        readonly: bool = True
    ) -> List[Any]:
        """
        Execute a Cypher query with parameters and return the results.
        
        This method provides a flexible way to execute arbitrary Cypher queries,
        with options for parameter passing, result transformation, and timeout control.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            transform_function: Optional function to transform each result record
            timeout: Query timeout in seconds (overrides config)
            readonly: If True, execute as read-only query
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            Neo4jError: If a database error occurs
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        try:
            return tx_execute_query(
                driver=self._driver,
                database_name=self._database_name,
                query=query,
                parameters=parameters,
                transform_function=transform_function,
                timeout=timeout,
                readonly=readonly
            )
        except (ServiceUnavailable, Neo4jError) as e:
            logger.error(f"Error executing Neo4j query: {str(e)}")
            # Re-raise with additional context
            raise Neo4jError(f"Query execution failed: {str(e)}", query)
    
    def run_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a Cypher query against the Neo4j database.
        
        Args:
            query: Cypher query string
            parameters: Optional parameters for the query
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            RuntimeError: If the service is not initialized
            Neo4jError: If there's an error executing the query
        """
        self._ensure_initialized()
        
        params = params_to_neo4j(parameters or {})
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                result = session.run(query, **params)
                
                # Convert result records to dictionaries
                records = []
                for record in result:
                    record_dict = {}
                    for key, value in record.items():
                        if isinstance(value, Node):
                            record_dict[key] = node_to_dict(value)
                        elif isinstance(value, Relationship):
                            record_dict[key] = relationship_to_dict(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                return records
        except Neo4jError as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_node_relationships(
        self, 
        node_id: UUID, 
        relationship_types: Optional[List[str]] = None, 
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """
        Get relationships connected to a node.
        
        Args:
            node_id: UUID of the node
            relationship_types: Optional list of relationship types to filter by
            direction: Direction of relationships to retrieve ("incoming", "outgoing", or "both")
            
        Returns:
            List of relationship dictionaries
            
        Raises:
            RuntimeError: If the service is not initialized
            ValueError: If direction is invalid
            Neo4jError: If there's an error retrieving relationships
        """
        self._ensure_initialized()
        
        # Build relationship type filter
        rel_type_filter = ""
        if relationship_types:
            rel_type_filter = ":" + "|".join([f"`{rt}`" for rt in relationship_types])
        
        # Get the appropriate direction pattern
        if direction == "outgoing":
            query = f"""
                MATCH (n)-[r{rel_type_filter}]->(m)
                WHERE n.uuid = $uuid
                RETURN r
            """
        elif direction == "incoming":
            query = f"""
                MATCH (n)<-[r{rel_type_filter}]-(m)
                WHERE n.uuid = $uuid
                RETURN r
            """
        elif direction == "both":
            query = f"""
                MATCH (n)-[r{rel_type_filter}]-(m)
                WHERE n.uuid = $uuid
                RETURN r
            """
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'incoming', 'outgoing', or 'both'.")
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                result = session.run(query, uuid=str(node_id))
                
                relationships = []
                for record in result:
                    relationships.append(relationship_to_dict(record["r"]))
                
                return relationships
        except Neo4jError as e:
            logger.error(f"Error retrieving node relationships: {str(e)}")
            raise
    
    def get_connected_nodes(
        self, 
        node_id: UUID, 
        relationship_types: Optional[List[str]] = None, 
        direction: str = "both",
        node_labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get nodes connected to a given node.
        
        Args:
            node_id: UUID of the node
            relationship_types: Optional list of relationship types to filter by
            direction: Direction of relationships to traverse ("incoming", "outgoing", or "both")
            node_labels: Optional list of node labels to filter by
            
        Returns:
            List of connected node dictionaries
            
        Raises:
            RuntimeError: If the service is not initialized
            ValueError: If direction is invalid
            Neo4jError: If there's an error retrieving connected nodes
        """
        self._ensure_initialized()
        
        # Format direction for Cypher
        direction_pattern = format_direction(direction)
        
        # Build relationship type filter
        rel_type_filter = ""
        if relationship_types:
            rel_type_filter = ":" + "|".join(relationship_types)
        
        # Build node label filter
        node_label_filter = ""
        if node_labels:
            node_label_filter = ":" + ":".join(node_labels)
        
        # Build query based on direction
        if direction == "both":
            query = f"""
                MATCH (n){direction_pattern.replace('r', f'r{rel_type_filter}')}(m{node_label_filter})
                WHERE n.uuid = $uuid
                RETURN m
            """
        elif direction == "outgoing":
            query = f"""
                MATCH (n){direction_pattern.replace('r', f'r{rel_type_filter}')}(m{node_label_filter})
                WHERE n.uuid = $uuid
                RETURN m
            """
        elif direction == "incoming":
            query = f"""
                MATCH (n){direction_pattern.replace('r', f'r{rel_type_filter}')}(m{node_label_filter})
                WHERE n.uuid = $uuid
                RETURN m
            """
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'incoming', 'outgoing', or 'both'.")
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                result = session.run(query, uuid=str(node_id))
                
                nodes = []
                for record in result:
                    nodes.append(node_to_dict(record["m"]))
                
                return nodes
        except Neo4jError as e:
            logger.error(f"Error retrieving connected nodes: {str(e)}")
            raise
    
    def find_nodes(
        self, 
        labels: Optional[Union[str, List[str]]] = None, 
        properties: Optional[Dict[str, Any]] = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by labels and/or property values.
        
        Args:
            labels: Optional label or list of labels to filter by
            properties: Optional properties to filter by
            limit: Maximum number of nodes to return
            
        Returns:
            List of matching node dictionaries
            
        Raises:
            RuntimeError: If the service is not initialized
            Neo4jError: If there's an error finding nodes
        """
        self._ensure_initialized()
        
        # Build labels filter
        labels_filter = ""
        if labels:
            labels_filter = build_labels_string(labels)
        
        # Build properties filter
        props_filter = ""
        if properties:
            neo4j_props = params_to_neo4j(properties)
            props_filter = build_where_clause(neo4j_props)
        
        query = f"""
            MATCH (n{labels_filter})
            {props_filter}
            RETURN n
            LIMIT $limit
        """
        
        params = {"limit": limit}
        if properties:
            params.update(neo4j_props)
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                result = session.run(query, **params)
                
                nodes = []
                for record in result:
                    nodes.append(node_to_dict(record["n"]))
                
                return nodes
        except Neo4jError as e:
            logger.error(f"Error finding nodes: {str(e)}")
            raise
    
    def find_paths(
        self, 
        start_node_id: UUID, 
        end_node_id: UUID, 
        relationship_types: Optional[List[str]] = None, 
        max_depth: int = 4
    ) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two nodes.
        
        Args:
            start_node_id: UUID of the start node
            end_node_id: UUID of the end node
            relationship_types: Optional list of relationship types to traverse
            max_depth: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of alternating nodes and relationships
            
        Raises:
            RuntimeError: If the service is not initialized
            Neo4jError: If there's an error finding paths
        """
        self._ensure_initialized()
        
        # Build relationship type filter
        rel_type_filter = ""
        if relationship_types:
            rel_type_filter = ":" + "|".join(relationship_types)
        
        # Use variable length path pattern with max_depth
        path_pattern = f"[*1..{max_depth}]"
        if rel_type_filter:
            path_pattern = f"{rel_type_filter}{path_pattern}"
        
        query = f"""
            MATCH path = (start)-[{path_pattern}]->(end)
            WHERE start.uuid = $start_uuid AND end.uuid = $end_uuid
            RETURN path
            LIMIT 10
        """
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                result = session.run(
                    query,
                    start_uuid=str(start_node_id),
                    end_uuid=str(end_node_id)
                )
                
                paths = []
                for record in result:
                    path = record["path"]
                    path_items = []
                    
                    # Add nodes and relationships to path_items in order
                    for i, node in enumerate(path.nodes):
                        path_items.append(node_to_dict(node))
                        if i < len(path.relationships):
                            path_items.append(relationship_to_dict(path.relationships[i]))
                    
                    paths.append(path_items)
                
                return paths
        except Neo4jError as e:
            logger.error(f"Error finding paths: {str(e)}")
            raise
    
    def create_index(
        self, 
        label: str, 
        properties: List[str], 
        index_name: Optional[str] = None, 
        index_type: str = "btree"
    ) -> bool:
        """
        Create an index on a node label and properties.
        
        Args:
            label: Node label to index
            properties: List of properties to include in the index
            index_name: Optional name for the index
            index_type: Type of index to create ("btree", "text", etc.)
            
        Returns:
            True if the index was created, False otherwise
            
        Raises:
            RuntimeError: If the service is not initialized
            Neo4jError: If there's an error creating the index
        """
        self._ensure_initialized()
        
        # Generate index name if not provided
        name = index_name or f"idx_{label}_{'_'.join(properties)}"
        
        # Format properties for Cypher
        props_str = ", ".join([f"n.{prop}" for prop in properties])
        
        query = f"""
            CREATE {index_type} INDEX {name} IF NOT EXISTS
            FOR (n:{label})
            ON ({props_str})
        """
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                session.run(query)
                return True
        except Neo4jError as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def create_constraint(
        self, 
        label: str, 
        properties: List[str], 
        constraint_type: str = "unique",
        constraint_name: Optional[str] = None
    ) -> bool:
        """
        Create a constraint on a node label and properties.
        
        Args:
            label: Node label to constrain
            properties: List of properties to include in the constraint
            constraint_type: Type of constraint ("unique", "exists", etc.)
            constraint_name: Optional name for the constraint
            
        Returns:
            True if the constraint was created, False otherwise
            
        Raises:
            RuntimeError: If the service is not initialized
            Neo4jError: If there's an error creating the constraint
        """
        self._ensure_initialized()
        
        # Generate constraint name if not provided
        name = constraint_name or f"{constraint_type}_{label}_{'_'.join(properties)}"
        
        # Build constraint query based on type
        if constraint_type == "unique":
            # Format properties for Cypher
            props_str = ", ".join([f"n.{prop}" for prop in properties])
            
            query = f"""
                CREATE CONSTRAINT {name} IF NOT EXISTS
                FOR (n:{label})
                REQUIRE ({props_str}) IS UNIQUE
            """
        elif constraint_type == "exists":
            # Handle existence constraint
            constraints = []
            for prop in properties:
                constraints.append(f"n.{prop} IS NOT NULL")
            
            props_str = " AND ".join(constraints)
            
            query = f"""
                CREATE CONSTRAINT {name} IF NOT EXISTS
                FOR (n:{label})
                REQUIRE {props_str}
            """
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")
        
        try:
            with self._driver.session(database=self._database_name) as session:  
                session.run(query)
                return True
        except Neo4jError as e:
            logger.error(f"Error creating constraint: {str(e)}")
            return False
    
    def execute_batch_operations(
        self, 
        batch_statements: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute a batch of Cypher statements in a single transaction.
        
        This method allows multiple database operations to be executed
        atomically within a single transaction, which improves performance
        and ensures data consistency.
        
        Args:
            batch_statements: List of statement dictionaries, each containing:
                              - query: The Cypher query to execute
                              - parameters: Parameters for the query
                              - return_results: Whether to return results (default: True)
        
        Returns:
            List of results for each statement (None for statements with return_results=False)
            
        Raises:
            Neo4jError: If a database error occurs
            ValueError: If batch_statements is empty or missing required fields
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        if not batch_statements:
            logger.warning("No statements provided for batch operation")
            return []
            
        # Validate statement structure
        for i, statement in enumerate(batch_statements):
            if "query" not in statement:
                raise ValueError(f"Missing 'query' in batch statement at index {i}")
        
        # Define a transaction function to execute all statements
        def run_batch(tx):
            results = []
            for statement in batch_statements:
                query = statement["query"]
                parameters = statement.get("parameters", {})
                
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
        
        try:
            # Execute all statements in a single transaction
            return self._execute_in_transaction(
                transaction_function=run_batch,
                readonly=False,  # Batch operations assume write operations
                max_retries=3,
                retry_interval=1.0
            )
        except Neo4jError as e:
            logger.error(f"Error executing batch operations: {str(e)}")
            raise
    
    def execute_batch(
        self, 
        operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of node and relationship operations in a single transaction.
        
        This method allows for efficient batch operations that may include node and
        relationship creation, updates, and deletions, all within a single atomic
        transaction.
        
        Args:
            operations: List of operation dictionaries. Each operation must contain:
                        - type: Operation type ('create_node', 'update_node', etc.)
                        - Additional parameters specific to the operation type
            
        Returns:
            List of results for each operation
            
        Raises:
            Neo4jError: If a database error occurs
            ValueError: If operations list is empty or contains invalid operations
            RuntimeError: If the service is not initialized
        """
        self._ensure_initialized()
        
        if not operations:
            logger.warning("No operations provided for batch execution")
            return []
            
        # Validate operations
        for i, op in enumerate(operations):
            if "type" not in op:
                raise ValueError(f"Missing 'type' in operation at index {i}")
                
        def execute_batch_tx(tx):
            results = []
            
            for operation in operations:
                op_type = operation["type"]
                
                if op_type == "create_node":
                    # Handle node creation
                    labels = operation.get("labels", [])
                    properties = operation.get("properties", {})
                    
                    # Ensure node has a UUID if not provided
                    if "uuid" not in properties:
                        properties["uuid"] = str(uuid4())
                        
                    # Convert properties for Neo4j
                    neo4j_props = params_to_neo4j(properties)
                    
                    # Build label clause
                    if isinstance(labels, str):
                        labels = [labels]
                    label_clause = ":".join(labels)
                    
                    # Execute query
                    query = f"CREATE (n:{label_clause} $props) RETURN n"
                    result = tx.run(query, props=neo4j_props)
                    record = result.single()
                    if not record:
                        raise Neo4jError(f"Failed to create node with labels {labels}", query)
                    results.append(node_to_dict(record["n"]))
                    
                elif op_type == "update_node":
                    # Handle node update
                    node_id = operation.get("node_id")
                    if not node_id:
                        raise ValueError("Missing node_id in update_node operation")
                        
                    properties = operation.get("properties", {})
                    merge = operation.get("merge", True)
                    
                    # Convert node_id to string if UUID
                    if isinstance(node_id, UUID):
                        node_id = str(node_id)
                        
                    # Convert properties for Neo4j
                    neo4j_props = params_to_neo4j(properties)
                    
                    # Determine update strategy
                    if merge:
                        set_clause = "SET n += $props"
                    else:
                        set_clause = "SET n = {uuid: $uuid} SET n += $props"
                        
                    # Execute query
                    query = f"MATCH (n) WHERE n.uuid = $uuid {set_clause} RETURN n"
                    result = tx.run(query, uuid=node_id, props=neo4j_props)
                    record = result.single()
                    if not record:
                        results.append(None)  # Node not found
                    else:
                        results.append(node_to_dict(record["n"]))
                        
                elif op_type == "delete_node":
                    # Handle node deletion
                    node_id = operation.get("node_id")
                    if not node_id:
                        raise ValueError("Missing node_id in delete_node operation")
                        
                    # Convert node_id to string if UUID
                    if isinstance(node_id, UUID):
                        node_id = str(node_id)
                        
                    # Execute query
                    query = "MATCH (n) WHERE n.uuid = $uuid DETACH DELETE n RETURN count(n) AS deleted_count"
                    result = tx.run(query, uuid=node_id)
                    record = result.single()
                    results.append(record and record["deleted_count"] > 0)
                    
                elif op_type == "create_relationship":
                    # Handle relationship creation
                    start_id = operation.get("start_node_id")
                    end_id = operation.get("end_node_id")
                    rel_type = operation.get("rel_type")
                    properties = operation.get("properties", {})
                    
                    if not start_id or not end_id or not rel_type:
                        raise ValueError("Missing required parameters in create_relationship operation")
                        
                    # Convert UUIDs to strings
                    if isinstance(start_id, UUID):
                        start_id = str(start_id)
                    if isinstance(end_id, UUID):
                        end_id = str(end_id)
                        
                    # Ensure relationship has a UUID if not provided
                    if "uuid" not in properties:
                        properties["uuid"] = str(uuid4())
                        
                    # Convert properties for Neo4j
                    neo4j_props = params_to_neo4j(properties)
                    
                    # Execute query
                    query = f"""
                    MATCH (a), (b)
                    WHERE a.uuid = $start_id AND b.uuid = $end_id
                    CREATE (a)-[r:`{rel_type}` $props]->(b)
                    RETURN r, a, b
                    """
                    result = tx.run(query, start_id=start_id, end_id=end_id, props=neo4j_props)
                    record = result.single()
                    if not record:
                        # Check if both nodes exist
                        check_query = """
                        MATCH (a), (b)
                        WHERE a.uuid = $start_id AND b.uuid = $end_id
                        RETURN count(a) AS start_exists, count(b) AS end_exists
                        """
                        check_result = tx.run(check_query, start_id=start_id, end_id=end_id)
                        check_record = check_result.single()
                        
                        if check_record:
                            if check_record["start_exists"] == 0:
                                raise Neo4jError(f"Start node with UUID {start_id} not found", query)
                            if check_record["end_exists"] == 0:
                                raise Neo4jError(f"End node with UUID {end_id} not found", query)
                                
                        raise Neo4jError("Failed to create relationship", query)
                        
                    rel_dict = relationship_to_dict(record["r"])
                    rel_dict["start_node"] = node_to_dict(record["a"])
                    rel_dict["end_node"] = node_to_dict(record["b"])
                    results.append(rel_dict)
                    
                else:
                    # Unsupported operation type
                    raise ValueError(f"Unsupported operation type: {op_type}")
                    
            return results
            
        try:
            return self._execute_in_transaction(execute_batch_tx, readonly=False, max_retries=3)
        except Neo4jError as e:
            logger.error(f"Error executing batch operations: {str(e)}")
            raise