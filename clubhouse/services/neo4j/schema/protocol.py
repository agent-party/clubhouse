"""
Neo4j schema management protocol interface.

This module defines the protocol interface for Neo4j schema management,
including methods for constraint and index management, schema validation,
and migration patterns.
"""

from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union
from uuid import UUID


class Neo4jSchemaManagerProtocol(Protocol):
    """Protocol for Neo4j schema management operations.
    
    This protocol defines methods for managing Neo4j schema elements
    such as constraints, indexes, and core entity relationships.
    """
    
    def initialize(self) -> None:
        """
        Initialize the schema manager.
        
        This method is called during application startup to set up
        database connections and initialize schema components.
        """
        ...
    
    def shutdown(self) -> None:
        """
        Shut down the schema manager.
        
        This method is called during application shutdown to
        clean up resources and close connections.
        """
        ...
    
    def is_healthy(self) -> bool:
        """
        Check if the schema manager is healthy.
        
        Returns:
            True if the schema manager is functioning correctly.
        """
        ...
    
    def create_constraint(
        self,
        label: str,
        properties: List[str],
        constraint_type: str = "unique",
        constraint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a constraint on nodes with the specified label and properties.
        
        Args:
            label: Node label for the constraint
            properties: List of properties to include in the constraint
            constraint_type: Type of constraint (unique, exists, etc.)
            constraint_name: Optional custom name for the constraint
            
        Returns:
            Dictionary containing constraint information
            
        Raises:
            Neo4jError: If there is an error creating the constraint
        """
        ...
    
    def drop_constraint(
        self,
        constraint_name: str,
    ) -> bool:
        """
        Drop an existing constraint by name.
        
        Args:
            constraint_name: Name of the constraint to drop
            
        Returns:
            True if the constraint was dropped, False if it doesn't exist
            
        Raises:
            Neo4jError: If there is an error dropping the constraint
        """
        ...
    
    def get_constraints(
        self,
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all constraints or constraints for a specific label.
        
        Args:
            label: Optional node label to filter constraints
            
        Returns:
            List of constraint information dictionaries
            
        Raises:
            Neo4jError: If there is an error retrieving constraints
        """
        ...
    
    def create_index(
        self,
        label: str,
        properties: List[str],
        index_type: str = "btree",
        index_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an index on nodes with the specified label and properties.
        
        Args:
            label: Node label for the index
            properties: List of properties to include in the index
            index_type: Type of index (btree, text, etc.)
            index_name: Optional custom name for the index
            
        Returns:
            Dictionary containing index information
            
        Raises:
            Neo4jError: If there is an error creating the index
        """
        ...
    
    def drop_index(
        self,
        index_name: str,
    ) -> bool:
        """
        Drop an existing index by name.
        
        Args:
            index_name: Name of the index to drop
            
        Returns:
            True if the index was dropped, False if it doesn't exist
            
        Raises:
            Neo4jError: If there is an error dropping the index
        """
        ...
    
    def get_indexes(
        self,
        label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all indexes or indexes for a specific label.
        
        Args:
            label: Optional node label to filter indexes
            
        Returns:
            List of index information dictionaries
            
        Raises:
            Neo4jError: If there is an error retrieving indexes
        """
        ...
    
    def validate_schema(self) -> Tuple[bool, List[str]]:
        """
        Validate that the current database schema matches expected schema.
        
        Returns:
            Tuple containing:
            - Boolean indicating if validation passed
            - List of validation error messages
            
        Raises:
            Neo4jError: If there is an error accessing schema information
        """
        ...
    
    def apply_migrations(
        self,
        target_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply schema migrations to reach a target version.
        
        Args:
            target_version: Optional version to migrate to (defaults to latest)
            
        Returns:
            List of applied migration information
            
        Raises:
            Neo4jError: If there is an error during migration
        """
        ...
    
    def get_schema_version(self) -> str:
        """
        Get the current schema version.
        
        Returns:
            String representing the current schema version
            
        Raises:
            Neo4jError: If there is an error accessing schema information
        """
        ...
