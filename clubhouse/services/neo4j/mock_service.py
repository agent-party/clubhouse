"""
Mock Neo4j service implementation.

This module provides a mock implementation of the Neo4j service protocol
for testing and development purposes. It simulates Neo4j functionality
using in-memory data structures without requiring an actual Neo4j instance.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from uuid import UUID, uuid4
import json
import re

from clubhouse.core.config.models.database import Neo4jDatabaseConfig
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol

logger = logging.getLogger(__name__)


class MockNeo4jService(Neo4jServiceProtocol):
    """
    Mock implementation of the Neo4j service protocol.
    
    This class simulates Neo4j functionality using in-memory data structures.
    It is useful for testing and development when a real Neo4j instance
    is not available.
    """
    
    def __init__(self, config: Optional[Neo4jDatabaseConfig] = None) -> None:
        """
        Initialize the mock Neo4j service.
        
        Args:
            config: Optional Neo4j database configuration
        """
        self._config = config
        
        # In-memory storage
        self._nodes: Dict[Union[str, UUID], Dict[str, Any]] = {}                # node_id -> node data
        self._relationships: Dict[Union[str, UUID], Dict[str, Any]] = {}        # relationship_id -> relationship data 
        self._labels: Dict[str, Set[str]] = {}                     # label -> set of node_ids
        self._node_relationships: Dict[str, List[str]] = {}        # node_id -> list of relationship_ids
        self._indexes: Dict[str, Dict[str, Any]] = {}              # index_name -> index data
        self._constraints: Dict[str, Dict[str, Any]] = {}          # constraint_name -> constraint data
        
        logger.info("Mock Neo4j Service initialized")
    
    def initialize(self) -> None:
        """Initialize the service."""
        logger.info("Mock Neo4j Service initialized")
    
    def shutdown(self) -> None:
        """Shut down the service."""
        # Clear all data
        self._nodes.clear()
        self._relationships.clear()
        self._labels.clear()
        self._node_relationships.clear()
        self._indexes.clear()
        self._constraints.clear()
        
        logger.info("Mock Neo4j Service shut down")
    
    def create_node(
        self,
        labels: Union[str, List[str]],
        properties: Dict[str, Any],
        node_id: Optional[Union[str, UUID]] = None
    ) -> Union[str, UUID]:
        """Create a node with labels and properties.
        
        Args:
            labels: Label or list of labels for the node
            properties: Properties for the node
            node_id: Optional ID for the node, if None a new UUID will be generated
            
        Returns:
            ID of the created node (string or UUID depending on input type)
            
        Raises:
            ValueError: If node with the same ID already exists
            ValueError: If the node would violate a uniqueness constraint
        """
        # Normalize labels to a list
        if isinstance(labels, str):
            labels_list = [labels]
        else:
            labels_list = labels.copy()
            
        logger.debug(f"Creating node with labels: {labels_list}")
        
        # Check for uniqueness constraint violations
        for label in labels_list:
            for constraint_name, constraint in self._constraints.items():
                if constraint["label"] == label and constraint["type"] == "uniqueness":
                    prop_name = constraint["property"]
                    
                    # Skip if the property doesn't exist in the new node
                    if prop_name not in properties:
                        continue
                        
                    prop_value = properties[prop_name]
                    
                    # Check all existing nodes with this label for uniqueness violations
                    if label in self._labels:
                        for node_id_str in self._labels[label]:
                            node_data = self._nodes[node_id_str]
                            if (prop_name in node_data["properties"] and 
                                node_data["properties"][prop_name] == prop_value):
                                raise ValueError("Uniqueness constraint violation")
        
        # Track the input type
        input_is_string = isinstance(node_id, str)
        
        # Generate a UUID if not provided
        if node_id is None:
            new_id = uuid.uuid4()
            input_is_string = False
        else:
            # Convert string ID to UUID if needed
            new_id = node_id if isinstance(node_id, UUID) else UUID(str(node_id))
            
        str_id = str(new_id)
        
        # Check if the node already exists
        if str_id in self._nodes or new_id in self._nodes:
            error_msg = f"Node with ID {new_id} already exists"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Create node with properties
        node_data = {
            "labels": labels_list,
            "properties": properties.copy(),
        }
        
        # Store with both string and UUID keys for test compatibility
        self._nodes[str_id] = node_data
        self._nodes[new_id] = node_data
        
        # Add to label indices - using string ID for consistency
        for label in labels_list:
            if label not in self._labels:
                self._labels[label] = set()
            self._labels[label].add(str_id)
            
        # Initialize relationships for this node
        self._node_relationships[str_id] = []
        
        logger.info(f"Created node with ID {new_id} and labels {labels_list}")
        # Return the same type that was provided
        return str_id if input_is_string else new_id
    
    def get_node(self, node_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """Get a node by ID.
        
        Args:
            node_id: ID of the node to get
            
        Returns:
            Node properties or None if not found
        """
        # Ensure we have a string version for lookups
        str_id = str(node_id)
        
        # Try all possible ways of storing
        if node_id in self._nodes:
            return self._nodes[node_id]["properties"].copy()
        elif str_id in self._nodes:
            return self._nodes[str_id]["properties"].copy()
            
        return None
    
    def update_node(
        self, 
        node_id: Union[str, UUID], 
        properties: Dict[str, Any], 
        merge: bool = True
    ) -> bool:
        """Update a node's properties.
        
        Args:
            node_id: ID of the node to update
            properties: New properties to set
            merge: If true, merge with existing properties, else replace
            
        Returns:
            True if updated, False if node not found
        """
        # Try both string and UUID versions
        str_id = str(node_id)
        
        if node_id in self._nodes:
            node = self._nodes[node_id]
        elif str_id in self._nodes:
            node = self._nodes[str_id]
        else:
            logger.warning(f"Cannot update non-existent node: {node_id}")
            return False
            
        # Update properties based on merge flag
        if merge:
            node["properties"].update(properties)
        else:
            node["properties"] = properties.copy()
            
        logger.info(f"Updated node {node_id} with properties {properties}")
        return True
    
    def delete_node(self, node_id: Union[str, UUID]) -> bool:
        """Delete a node by ID.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if deleted, False otherwise
            
        Raises:
            ValueError: If the node has relationships
        """
        logger.debug(f"Deleting node with ID {node_id}")
        
        # Convert ID to string for lookup
        str_node_id = str(node_id)
        
        # Return false if node doesn't exist
        if str_node_id not in self._nodes and node_id not in self._nodes:
            logger.warning(f"Cannot delete non-existent node: {str_node_id}")
            return False
            
        # Check for relationships
        if str_node_id in self._node_relationships and self._node_relationships[str_node_id]:
            error_msg = f"Cannot delete node with existing relationships"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Get the node's labels before deleting
        if str_node_id in self._nodes:
            node_labels = self._nodes[str_node_id]["labels"]
        elif node_id in self._nodes:
            node_labels = self._nodes[node_id]["labels"]
        else:
            return False
        
        # Remove node from label indices
        for label in node_labels:
            if label in self._labels and str_node_id in self._labels[label]:
                self._labels[label].remove(str_node_id)
                
                # Do not delete empty label indices to maintain test expectations
                # Even when empty, we need to keep the label key in the dictionary
                
        # Remove node relationships tracking
        if str_node_id in self._node_relationships:
            del self._node_relationships[str_node_id]
            
        # Remove the node (both string and UUID keys)
        if str_node_id in self._nodes:
            del self._nodes[str_node_id]
        if node_id in self._nodes:
            del self._nodes[node_id]
        
        logger.info(f"Deleted node with ID {str_node_id}")
        return True
    
    def create_relationship(
        self, 
        start_node_id: Union[str, UUID], 
        end_node_id: Union[str, UUID], 
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a relationship between two nodes.
        
        Args:
            start_node_id: ID of the start node
            end_node_id: ID of the end node
            relationship_type: Type of relationship
            properties: Optional properties for the relationship
            
        Returns:
            String ID of the created relationship
            
        Raises:
            ValueError: If either node doesn't exist
        """
        if properties is None:
            properties = {}
            
        # Convert IDs to strings for lookup
        str_start_id = str(start_node_id)
        str_end_id = str(end_node_id)
        
        # Check if nodes exist
        start_node_exists = start_node_id in self._nodes or str_start_id in self._nodes
        end_node_exists = end_node_id in self._nodes or str_end_id in self._nodes
        
        if not start_node_exists:
            raise ValueError(f"Start node with ID {start_node_id} not found")
            
        if not end_node_exists:
            raise ValueError(f"End node with ID {end_node_id} not found")
            
        # Generate a relationship ID
        rel_id = uuid.uuid4()
        str_rel_id = str(rel_id)
        
        # Create the relationship - preserve original ID types
        rel_data = {
            "id": rel_id,  # Store UUID object for internal use
            "start_node_id": start_node_id,  # Store original type
            "end_node_id": end_node_id,      # Store original type
            "type": relationship_type,
            "properties": properties.copy()
        }
        
        # Store with both string and UUID keys for test compatibility
        self._relationships[str_rel_id] = rel_data
        self._relationships[rel_id] = rel_data
        
        # Add to node relationships - preserve original ID types
        if start_node_id not in self._node_relationships:
            self._node_relationships[start_node_id] = []
        self._node_relationships[start_node_id].append(rel_id)  # Store actual UUID
        
        if end_node_id not in self._node_relationships:
            self._node_relationships[end_node_id] = []
        self._node_relationships[end_node_id].append(rel_id)  # Store actual UUID
        
        # Also maintain string versions for backward compatibility
        if str_start_id not in self._node_relationships:
            self._node_relationships[str_start_id] = []
        if rel_id not in self._node_relationships[str_start_id]:
            self._node_relationships[str_start_id].append(rel_id)
        
        if str_end_id not in self._node_relationships:
            self._node_relationships[str_end_id] = []
        if rel_id not in self._node_relationships[str_end_id]:
            self._node_relationships[str_end_id].append(rel_id)
        
        logger.info(f"Created relationship {rel_id} from {str_start_id} to {str_end_id}")
        # Return the UUID object for tests that expect it
        return rel_id
    
    def get_relationship(
        self, 
        relationship_id: Union[str, UUID]
    ) -> Optional[Dict[str, Any]]:
        """Get a relationship by ID.
        
        Args:
            relationship_id: ID of the relationship to retrieve
            
        Returns:
            Relationship data or None if not found
        """
        # Try to get the relationship with both string and UUID versions of the ID
        str_rel_id = str(relationship_id)
        uuid_rel_id = relationship_id
        if isinstance(relationship_id, str):
            try:
                uuid_rel_id = UUID(relationship_id)
            except ValueError:
                pass
        
        # Look up with both forms of ID
        rel_data = self._relationships.get(relationship_id)
        if rel_data is None:
            # Try with string ID if original lookup failed
            if isinstance(relationship_id, UUID):
                rel_data = self._relationships.get(str_rel_id)
            # Try with UUID if original lookup with string failed
            elif uuid_rel_id != relationship_id:
                rel_data = self._relationships.get(uuid_rel_id)
                
        if rel_data is None:
            return None
            
        # Create a response dictionary
        response = {
            "id": rel_data["id"],  # Keep this as UUID for test compatibility
            "start_node_id": rel_data["start_node_id"],
            "end_node_id": rel_data["end_node_id"],
            "type": rel_data["type"]
        }
        
        # Add properties to the response
        for key, value in rel_data["properties"].items():
            response[key] = value
            
        return response
    
    def update_relationship(
        self, 
        relationship_id: Union[str, UUID], 
        properties: Dict[str, Any], 
        merge: bool = False
    ) -> bool:
        """Update a relationship's properties.
        
        Args:
            relationship_id: ID of the relationship to update
            properties: New properties or properties to merge
            merge: If True, merge with existing properties, else replace
            
        Returns:
            True if updated, False if not found
        """
        # Try both string and UUID versions
        str_id = str(relationship_id)
        
        if relationship_id in self._relationships:
            rel_data = self._relationships[relationship_id]
        elif str_id in self._relationships:
            rel_data = self._relationships[str_id]
        else:
            logger.warning(f"Cannot update non-existent relationship: {relationship_id}")
            return False
            
        # Update properties based on merge flag
        if merge:
            # Merge with existing properties
            rel_data["properties"].update(properties)
        else:
            # Replace properties
            rel_data["properties"] = properties.copy()
            
        # Also add properties to top level for consistency with get_relationship
        for key, value in properties.items():
            rel_data[key] = value
            
        logger.info(f"Updated relationship {relationship_id}")
        return True
    
    def delete_relationship(self, relationship_id: Union[str, UUID]) -> bool:
        """
        Delete a relationship from the graph database.
        
        Args:
            relationship_id: ID of the relationship
            
        Returns:
            True if deleted, False if not found
        """
        # Try both string and UUID versions
        str_id = str(relationship_id)
        
        if relationship_id in self._relationships:
            rel = self._relationships[relationship_id]
        elif str_id in self._relationships:
            rel = self._relationships[str_id]
        else:
            logger.warning(f"Cannot delete non-existent relationship: {relationship_id}")
            return False
            
        # Get the relationship data before deleting
        start_node_id = rel["start_node_id"]
        end_node_id = rel["end_node_id"]
        
        # Remove relationship from node relationships
        if start_node_id in self._node_relationships:
            self._node_relationships[start_node_id] = [
                r for r in self._node_relationships[start_node_id] 
                if str(r) != str_id
            ]
            
        if end_node_id in self._node_relationships:
            self._node_relationships[end_node_id] = [
                r for r in self._node_relationships[end_node_id] 
                if str(r) != str_id
            ]
            
        # Delete the relationship (both string and UUID keys)
        if str_id in self._relationships:
            del self._relationships[str_id]
            
        if relationship_id in self._relationships:
            del self._relationships[relationship_id]
            
        logger.info(f"Deleted relationship {relationship_id}")
        return True
    
    def run_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a Cypher query.
        
        This is a simplified implementation that recognizes certain patterns 
        in the query string for testing purposes.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of matching results
        """
        if parameters is None:
            parameters = {}
            
        logger.info(f"Running query: {query}")
        logger.debug(f"Query parameters: {parameters}")
        
        # Extract where conditions and labels from query
        query_lower = query.lower().replace("\n", " ").replace("  ", " ")
        
        # Check for agents with capability query pattern
        if "match (a:agent)" in query_lower and "'text_transform' in a.capabilities" in query_lower:
            # Find all agent nodes with the TEXT_TRANSFORM capability
            results = []
            processed_ids = set()  # Track processed IDs to avoid duplicates
            
            for node_id, node_data in self._nodes.items():
                # Skip if already processed this node (might have duplicate entries in different formats)
                str_id = str(node_id)
                if str_id in processed_ids:
                    continue
                processed_ids.add(str_id)
                
                if "Agent" in node_data["labels"] and "capabilities" in node_data["properties"]:
                    capabilities = node_data["properties"]["capabilities"]
                    if "TEXT_TRANSFORM" in capabilities:
                        results.append({
                            "a.name": node_data["properties"]["name"],
                            "a.uuid": str(node_id) if isinstance(node_id, UUID) else node_id,
                            "a.capabilities": capabilities
                        })
            return results
        
        # Check for relationships between agents
        if "match (a1:agent)-[r]->(a2:agent)" in query_lower:
            results = []
            processed_rels = set()  # Track processed relationship IDs to avoid duplicates
            
            # Find all relationships between agents
            for rel_id, rel_data in self._relationships.items():
                # Skip if already processed this relationship (might have duplicate entries)
                str_rel_id = str(rel_id)
                if str_rel_id in processed_rels:
                    continue
                processed_rels.add(str_rel_id)
                
                start_id = rel_data["start_node_id"]
                end_id = rel_data["end_node_id"]
                
                # Check if both start and end nodes exist
                start_node = None
                end_node = None
                
                # Try to get the start node
                if start_id in self._nodes:
                    start_node = self._nodes[start_id]
                elif str(start_id) in self._nodes:
                    start_node = self._nodes[str(start_id)]
                    
                # Try to get the end node
                if end_id in self._nodes:
                    end_node = self._nodes[end_id]
                elif str(end_id) in self._nodes:
                    end_node = self._nodes[str(end_id)]
                
                # Check if both nodes exist and are agents
                if (start_node and end_node and
                    "Agent" in start_node["labels"] and
                    "Agent" in end_node["labels"]):
                    
                    # Create the result record
                    results.append({
                        "a1.name": start_node["properties"]["name"],
                        "type(r)": rel_data["type"],
                        "r": rel_data["properties"] if "properties" in rel_data else {},
                        "a2.name": end_node["properties"]["name"]
                    })
            
            return results
        
        # Default: Return empty result
        logger.warning(f"Unrecognized query pattern: {query}")
        return []
    
    def get_node_relationships(
        self, 
        node_id: Union[str, UUID], 
        direction: Optional[str] = None, 
        relationship_type: Optional[str] = None,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get relationships for a node.
        
        Args:
            node_id: ID of the node
            direction: Optional direction filter ('IN', 'OUT', 'incoming', 'outgoing', or None for both)
            relationship_type: Optional relationship type filter (single type)
            relationship_types: Optional relationship type filter (list of types)
            
        Returns:
            List of relationship data
        """
        str_node_id = str(node_id)
        
        # Check if node exists using both string and UUID keys
        node_exists = node_id in self._nodes or str_node_id in self._nodes
        if not node_exists:
            logger.warning(f"Cannot get relationships for non-existent node: {node_id}")
            return []
            
        # Get all relationship IDs for this node
        rel_ids = []
        
        # Check if we have relationships stored under the original node_id
        if node_id in self._node_relationships:
            rel_ids.extend(self._node_relationships[node_id])
            
        # Also check for relationships stored under the string version of node_id
        if str_node_id in self._node_relationships:
            str_rel_ids = self._node_relationships[str_node_id]
            for rel_id in str_rel_ids:
                if rel_id not in rel_ids and UUID(str(rel_id)) not in rel_ids:
                    rel_ids.append(rel_id)
        
        results = []
        for rel_id in rel_ids:
            # Skip if relationship was deleted
            if rel_id not in self._relationships and str(rel_id) not in self._relationships:
                continue
            
            # Get relationship data
            rel_data = None
            uuid_rel_id = rel_id if isinstance(rel_id, UUID) else None
            str_rel_id = str(rel_id)
            
            if uuid_rel_id and uuid_rel_id in self._relationships:
                rel_data = self._relationships[uuid_rel_id]
            elif str_rel_id in self._relationships:
                rel_data = self._relationships[str_rel_id]
            else:
                continue  # Skip if relationship not found
            
            # Normalize direction parameter
            dir_filter = None
            if direction is not None:
                direction = direction.upper()
                if direction in ("OUT", "OUTGOING"):
                    dir_filter = "OUT"
                elif direction in ("IN", "INCOMING"):
                    dir_filter = "IN"
                    
            # Get start and end node IDs from relationship data
            start_node_id = rel_data["start_node_id"]
            end_node_id = rel_data["end_node_id"]
            
            # Check direction filter
            if dir_filter is not None:
                # Convert to string for comparison if needed
                if isinstance(start_node_id, UUID) and isinstance(node_id, str):
                    start_is_node = str(start_node_id) == node_id
                elif isinstance(start_node_id, str) and isinstance(node_id, UUID):
                    start_is_node = start_node_id == str(node_id)
                else:
                    start_is_node = start_node_id == node_id
                    
                if isinstance(end_node_id, UUID) and isinstance(node_id, str):
                    end_is_node = str(end_node_id) == node_id
                elif isinstance(end_node_id, str) and isinstance(node_id, UUID):
                    end_is_node = end_node_id == str(node_id)
                else:
                    end_is_node = end_node_id == node_id
                
                if dir_filter == "OUT" and not start_is_node:
                    continue
                if dir_filter == "IN" and not end_is_node:
                    continue
            
            # Get the relationship type
            rel_type = rel_data["type"]
                    
            # Check relationship type filter (single type)
            if relationship_type is not None and rel_type != relationship_type:
                continue
                
            # Check relationship types filter (list of types)
            if relationship_types is not None and rel_type not in relationship_types:
                continue
                
            # Format relationship for test expectations
            rel_copy = {
                "id": rel_data["id"],  # Use the UUID object
                "start_node_id": rel_data["start_node_id"],
                "end_node_id": rel_data["end_node_id"],
                "type": rel_data["type"],
            }
            
            # Add all properties to the top level
            if "properties" in rel_data:
                for key, value in rel_data["properties"].items():
                    rel_copy[key] = value
                    
            results.append(rel_copy)
            
        return results
    
    def get_connected_nodes(
        self, 
        node_id: Union[str, UUID], 
        relationship_type: Optional[str] = None,
        relationship_types: Optional[List[str]] = None,
        direction: Optional[str] = None, 
        node_labels: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Get nodes connected to a node.
        
        Args:
            node_id: ID of the node
            relationship_type: Optional relationship type filter (deprecated, use relationship_types)
            relationship_types: Optional list of relationship type filters
            direction: Optional direction filter ('IN', 'OUT', or None for both)
            node_labels: Optional label or list of labels to filter connected nodes
            
        Returns:
            List of connected node data
        """
        # For backward compatibility, handle both relationship_type and relationship_types
        if relationship_type is not None and relationship_types is None:
            relationship_types = [relationship_type]
            
        # Get relationships for the node
        relationships = self.get_node_relationships(node_id, direction, relationship_types=relationship_types)
        
        # Ensure node_id is a string for comparison
        str_node_id = str(node_id)
        connected_nodes = []
        
        for rel in relationships:
            # Determine the connected node ID based on direction
            if self._compare_ids(rel["start_node_id"], str_node_id):
                connected_id = rel["end_node_id"]
            else:
                connected_id = rel["start_node_id"]
                
            # Get the node data
            node_data = self._get_node_by_id(connected_id)
            if node_data is None:
                continue
            
            # Check label filter
            if node_labels is not None:
                labels_list = [node_labels] if isinstance(node_labels, str) else node_labels
                if not any(label in node_data["labels"] for label in labels_list):
                    continue
                    
            # Format node for test expectations
            node_result = {
                "id": str(connected_id),
                "labels": node_data["labels"].copy(),
                "properties": node_data["properties"].copy(),
                "name": node_data["properties"].get("name", ""),  # Add name for test expectations
                "relationship": {
                    "id": rel["id"],
                    "type": rel["type"],
                    "properties": rel.get("properties", {}).copy()
                }
            }
            connected_nodes.append(node_result)
            
        return connected_nodes
        
    def _get_node_by_id(self, node_id: Union[str, UUID]) -> Optional[Dict[str, Any]]:
        """Helper method to get a node by ID, handling both string and UUID formats.
        
        Args:
            node_id: ID of the node (string or UUID)
            
        Returns:
            Node data dict or None if not found
        """
        # Try direct lookup
        if node_id in self._nodes:
            return self._nodes[node_id]
        
        # Try string conversion lookup
        str_id = str(node_id)
        if str_id in self._nodes:
            return self._nodes[str_id]
        
        # Try UUID conversion lookup (if string was provided)
        if isinstance(node_id, str):
            try:
                uuid_id = UUID(node_id)
                if uuid_id in self._nodes:
                    return self._nodes[uuid_id]
            except ValueError:
                pass
                
        return None
        
    def _compare_ids(self, id1: Union[str, UUID], id2: Union[str, UUID]) -> bool:
        """Helper method to compare two IDs that might be of different types (string vs UUID).
        
        Args:
            id1: First ID to compare
            id2: Second ID to compare
            
        Returns:
            True if the IDs represent the same value, False otherwise
        """
        return str(id1) == str(id2)
    
    def find_nodes(
        self, 
        labels: Optional[Union[str, List[str]]] = None, 
        properties: Optional[Dict[str, Any]] = None, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find nodes by labels and properties.
        
        Args:
            labels: Label or list of labels to filter by
            properties: Properties to filter by
            limit: Maximum number of nodes to return
            
        Returns:
            List of matching nodes with their properties
        """
        # Convert labels to list if needed
        if labels is not None and isinstance(labels, str):
            labels_list = [labels]
        else:
            labels_list = labels
            
        # Start with all node ids
        matching_ids = set(self._nodes.keys())
        
        # Filter by labels if provided
        if labels_list:
            label_matches = set()
            for label in labels_list:
                if label in self._labels:
                    # For the first label, initialize with all matches
                    if not label_matches:
                        label_matches = self._labels[label].copy()
                    # For subsequent labels, take intersection
                    else:
                        label_matches &= self._labels[label]
            
            matching_ids = matching_ids & label_matches
        
        # Filter by properties if provided
        if properties:
            property_matches = set()
            for node_id in matching_ids:
                node_props = self._nodes[node_id]["properties"]
                # Check if all specified properties match
                if all(key in node_props and node_props[key] == value 
                       for key, value in properties.items()):
                    property_matches.add(node_id)
            
            matching_ids = property_matches
        
        # Prepare result list with node properties
        result = []
        for node_id in matching_ids:
            # Add node properties with node_id
            node_props = self._nodes[node_id]["properties"].copy()
            node_props["id"] = node_id
            result.append(node_props)
            
            # Apply limit if provided
            if limit is not None and len(result) >= limit:
                break
                
        return result
    
    def find_paths(
        self, 
        start_node_id: Union[str, UUID], 
        end_node_id: Union[str, UUID], 
        relationship_types: Optional[Union[str, List[str]]] = None, 
        max_depth: int = 1
    ) -> List[List[Dict[str, Any]]]:
        """Find paths between two nodes.
        
        Args:
            start_node_id: ID of the start node
            end_node_id: ID of the end node
            relationship_types: Optional type or list of types to filter relationships
            max_depth: Maximum path depth
            
        Returns:
            List of paths, where each path is a list of nodes and relationships
        """
        str_start_id = str(start_node_id)
        str_end_id = str(end_node_id)
        
        # Get node data using helper method
        start_node = self._get_node_by_id(start_node_id)
        end_node = self._get_node_by_id(end_node_id)
        
        if start_node is None or end_node is None:
            return []
            
        # Normalize relationship types to list
        rel_types = None
        if relationship_types is not None:
            rel_types = [relationship_types] if isinstance(relationship_types, str) else relationship_types
            
        # For simplicity in tests, return direct paths
        paths = []
        
        # Get relationships for start node
        relationships = self.get_node_relationships(start_node_id)
        
        for rel in relationships:
            # Check if this relationship connects to the end node
            if self._compare_ids(rel["end_node_id"], str_end_id):
                # Check relationship type if specified
                if rel_types is not None and rel["type"] not in rel_types:
                    continue
                    
                # Format the path as expected by the test
                path = [
                    {"node": start_node["properties"].copy()},
                    {"relationship": {"type": rel["type"]}},
                    {"node": end_node["properties"].copy()}
                ]
                paths.append(path)
                
        return paths
    
    def create_index(
        self, 
        label: str, 
        properties: List[str], 
        index_name: Optional[str] = None, 
        index_type: str = "BTREE"
    ) -> str:
        """Create an index on a label and properties.
        
        Args:
            label: Label to index
            properties: Properties to index
            index_name: Optional name for the index
            index_type: Type of index (default: BTREE)
            
        Returns:
            The name of the created index
            
        Raises:
            ValueError: If an index with the provided name already exists
        """
        # Generate an index name if not provided
        if index_name is None:
            index_name = f"{label}_{'.'.join(properties)}_idx"
            
        # Check if index already exists
        if index_name in self._indexes:
            raise ValueError(f"Index with name {index_name} already exists")
            
        # Create the index
        self._indexes[index_name] = {
            "label": label,
            "properties": properties.copy(),
            "type": index_type
        }
        
        logger.info(f"Created index {index_name} on {label}({', '.join(properties)})")
        return index_name
    
    def create_constraint(
        self, 
        label: str, 
        property_name: str, 
        constraint_name: Optional[str] = None, 
        constraint_type: str = "UNIQUE"
    ) -> str:
        """Create a constraint on a label and property.
        
        Args:
            label: Label to constrain
            property_name: Property to constrain
            constraint_name: Optional name for the constraint
            constraint_type: Type of constraint (default: UNIQUE)
            
        Returns:
            Name of the created constraint
            
        Raises:
            ValueError: If a constraint with the provided name already exists
        """
        # Generate a constraint name if not provided
        if constraint_name is None:
            constraint_name = f"{label}_{property_name}_{constraint_type.lower()}"
            
        # Check if constraint already exists
        if constraint_name in self._constraints:
            raise ValueError(f"Constraint with name {constraint_name} already exists")
            
        # Create the constraint
        self._constraints[constraint_name] = {
            "label": label,
            "property": property_name,
            "type": "uniqueness" if constraint_type.lower() == "unique" else constraint_type.lower()
        }
        
        logger.info(f"Created {constraint_type} constraint {constraint_name} on {label}.{property_name}")
        return constraint_name
    
    def execute_batch(
        self,
        operations: List[Union[Tuple[str, Dict[str, Any]], Dict[str, Any]]]
    ) -> List[Any]:
        """Execute a batch of operations.
        
        Args:
            operations: List of operations to execute, either as:
                - Tuples: (method_name, params_dict)
                - Dicts with 'operation' key
                
        Returns:
            List of results, one for each operation
        """
        results = []
        
        for operation in operations:
            try:
                # Handle tuple-style operations (method_name, params_dict)
                if isinstance(operation, tuple) and len(operation) == 2:
                    method_name, params = operation
                    
                    # Check if the method exists
                    if not hasattr(self, method_name):
                        logger.error(f"Method {method_name} not found")
                        results.append(None)
                        continue
                        
                    method = getattr(self, method_name)
                    # Call the method with the parameters
                    result = method(**params)
                    
                    # Convert UUID to string for create_relationship
                    if method_name == "create_relationship" and isinstance(result, UUID):
                        result = str(result)
                    
                    results.append(result)
                    
                # Handle dictionary-style operations
                elif isinstance(operation, dict):
                    # Get operation type
                    op_type = operation.get("operation")
                    
                    if op_type == "create_node":
                        # Extract parameters
                        labels = operation.get("labels")
                        properties = operation.get("properties", {})
                        node_id = operation.get("node_id")
                        
                        # Validate parameters
                        if labels is None:
                            raise ValueError("Missing required parameter: 'labels'")
                            
                        # Create the node
                        result = self.create_node(labels, properties, node_id)
                        results.append(result)
                        
                    elif op_type == "update_node":
                        # Extract parameters
                        node_id = operation.get("node_id")
                        properties = operation.get("properties", {})
                        merge = operation.get("merge", False)
                        
                        # Validate parameters
                        if node_id is None:
                            raise ValueError("Missing required parameter: 'node_id'")
                        if properties is None:
                            raise ValueError("Missing required parameter: 'properties'")
                            
                        # Update the node
                        result = self.update_node(node_id, properties, merge)
                        results.append(result)
                        
                    elif op_type == "delete_node":
                        # Extract parameters
                        node_id = operation.get("node_id")
                        
                        # Validate parameters
                        if node_id is None:
                            raise ValueError("Missing required parameter: 'node_id'")
                            
                        # Delete the node
                        result = self.delete_node(node_id)
                        results.append(result)
                        
                    elif op_type == "create_relationship":
                        # Extract parameters
                        start_node_id = operation.get("start_node_id")
                        end_node_id = operation.get("end_node_id")
                        relationship_type = operation.get("relationship_type")
                        properties = operation.get("properties", {})
                        
                        # Validate parameters
                        if start_node_id is None:
                            raise ValueError("Missing required parameter: 'start_node_id'")
                        if end_node_id is None:
                            raise ValueError("Missing required parameter: 'end_node_id'")
                        if relationship_type is None:
                            raise ValueError("Missing required parameter: 'relationship_type'")
                            
                        # Create the relationship
                        result = self.create_relationship(
                            start_node_id, end_node_id, relationship_type, properties
                        )
                        
                        # Convert UUID to string for consistency in batch operations
                        if isinstance(result, UUID):
                            result = str(result)
                            
                        results.append(result)
                        
                    elif op_type == "update_relationship":
                        # Extract parameters
                        relationship_id = operation.get("relationship_id")
                        properties = operation.get("properties", {})
                        merge = operation.get("merge", False)
                        
                        # Validate parameters
                        if relationship_id is None:
                            raise ValueError("Missing required parameter: 'relationship_id'")
                        if properties is None:
                            raise ValueError("Missing required parameter: 'properties'")
                            
                        # Update the relationship
                        result = self.update_relationship(relationship_id, properties, merge)
                        results.append(result)
                        
                    elif op_type == "delete_relationship":
                        # Extract parameters
                        relationship_id = operation.get("relationship_id")
                        
                        # Validate parameters
                        if relationship_id is None:
                            raise ValueError("Missing required parameter: 'relationship_id'")
                            
                        # Delete the relationship
                        result = self.delete_relationship(relationship_id)
                        results.append(result)
                        
                    elif op_type == "run_query":
                        # Extract parameters
                        query = operation.get("query")
                        parameters = operation.get("parameters", {})
                        
                        # Validate parameters
                        if query is None:
                            raise ValueError("Missing required parameter: 'query'")
                            
                        # Run the query
                        result = self.run_query(query, parameters)
                        results.append(result)
                        
                    else:
                        logger.warning(f"Unknown operation type: {op_type}")
                        results.append(None)
                else:
                    logger.warning(f"Invalid operation format: {operation}")
                    results.append(None)
                    
            except Exception as e:
                logger.error(f"Error executing operation {operation}: {e}")
                results.append(None)
                
        return results
    
    def execute(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query.
        
        This is a simplified implementation that handles basic queries for testing.
        For real Neo4j, use the real service.
        
        Args:
            query: Cypher query
            parameters: Query parameters
            
        Returns:
            List of results
        """
        if parameters is None:
            parameters = {}
            
        try:
            # Basic parser for common operations
            if "CREATE CONSTRAINT" in query or "CREATE CONSTRAINT IF NOT EXISTS" in query:
                # Extract constraint parameters from the query or use defaults
                if parameters:
                    constraint_name = parameters.get("name", f"constraint_{uuid4().hex[:8]}")
                    label = parameters.get("label", "TestLabel")
                    property_names = parameters.get("properties", ["id"])
                    if isinstance(property_names, str):
                        property_names = [property_names]
                    self.create_constraint(label, property_names, constraint_name=constraint_name)
                    
                elif "MATCH" in query:
                    # Simulate a match operation - no real implementation needed for tests
                    return []
                    
                elif "CREATE" in query:
                    # Simulate a create operation - no real implementation needed for tests
                    return []
                    
            # Return empty result as a fallback
            return []
            
        except Exception as e:
            logger.error(f"Error executing query {query}: {e}")
            return []