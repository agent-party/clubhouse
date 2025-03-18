"""
Protocol definition for Neo4j graph database service.

This module defines the interface that Neo4j service implementations
must adhere to, following the Protocol pattern for type safety and
decoupling implementation from interface.
"""

from typing import Any, Dict, List, Optional, Protocol, Set, TypeVar, Union, runtime_checkable
from uuid import UUID

# Define our own Node and Relationship types for compatibility with Neo4j driver 5.x
# These are type aliases that represent the structure of Neo4j nodes and relationships
Node = Dict[str, Any]  # A node is essentially a dictionary of properties
Relationship = Dict[str, Any]  # A relationship is also a dictionary of properties

from clubhouse.core.service_registry import ServiceProtocol
from typing import cast, List, Dict, Any, Type


# Type variables for generic methods
T = TypeVar('T')
R = TypeVar('R')


@runtime_checkable
class Neo4jServiceProtocol(ServiceProtocol, Protocol):
    """
    Protocol defining the interface for Neo4j graph database services.
    
    This protocol extends the base ServiceProtocol with methods specific
    to Neo4j graph database operations, providing a contract that all
    Neo4j service implementations must fulfill.
    """
    
    def create_node(
        self, 
        labels: Union[str, List[str]], 
        properties: Dict[str, Any]
    ) -> UUID:
        """
        Create a node in the graph database.
        
        Args:
            labels: Label or list of labels for the node
            properties: Properties to set on the node
            
        Returns:
            UUID of the created node
        """
        ...
    
    def get_node(self, node_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get a node by its ID.
        
        Args:
            node_id: UUID of the node to retrieve
            
        Returns:
            Dictionary representation of the node, or None if not found
        """
        ...
    
    def update_node(
        self, 
        node_id: UUID, 
        properties: Dict[str, Any], 
        merge: bool = True
    ) -> bool:
        """
        Update a node's properties.
        
        Args:
            node_id: UUID of the node to update
            properties: Properties to update on the node
            merge: If True, merge properties with existing, otherwise replace all
            
        Returns:
            True if the node was updated, False if it doesn't exist
        """
        ...
    
    def delete_node(self, node_id: UUID) -> bool:
        """
        Delete a node from the graph database.
        
        Args:
            node_id: UUID of the node to delete
            
        Returns:
            True if the node was deleted, False if it doesn't exist
        """
        ...
    
    def create_relationship(
        self, 
        from_node_id: UUID, 
        to_node_id: UUID, 
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: UUID of the source node
            to_node_id: UUID of the target node
            relationship_type: Type of the relationship
            properties: Optional properties to set on the relationship
            
        Returns:
            UUID of the created relationship
        """
        ...
    
    def get_relationship(self, relationship_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get a relationship by its ID.
        
        Args:
            relationship_id: UUID of the relationship to retrieve
            
        Returns:
            Dictionary representation of the relationship, or None if not found
        """
        ...
    
    def update_relationship(
        self, 
        relationship_id: UUID, 
        properties: Dict[str, Any], 
        merge: bool = True
    ) -> bool:
        """
        Update a relationship's properties.
        
        Args:
            relationship_id: UUID of the relationship to update
            properties: Properties to update on the relationship
            merge: If True, merge properties with existing, otherwise replace all
            
        Returns:
            True if the relationship was updated, False if it doesn't exist
        """
        ...
    
    def delete_relationship(self, relationship_id: UUID) -> bool:
        """
        Delete a relationship from the graph database.
        
        Args:
            relationship_id: UUID of the relationship to delete
            
        Returns:
            True if the relationship was deleted, False if it doesn't exist
        """
        ...
    
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
        """
        ...
    
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
        """
        ...
    
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
        """
        ...
    
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
        """
        ...
    
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
        """
        ...
    
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
        """
        ...
    
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
        """
        ...
    
    def execute_batch(
        self, 
        operations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute a batch of operations in a single transaction.
        
        Args:
            operations: List of operation definitions
                Each operation should be a dict with keys:
                - "type": operation type (create_node, update_node, etc.)
                - Other parameters specific to the operation type
            
        Returns:
            List of operation results
        """
        ...
