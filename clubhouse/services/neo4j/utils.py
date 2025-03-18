"""
Utility functions for Neo4j operations in the Clubhouse platform.

This module provides helper functions for converting between Neo4j
and Python types, handling data transformations, and other common
operations used by the Neo4j service.
"""

import datetime
import json
from typing import Any, Dict, List, Union
from uuid import UUID

from neo4j.graph import Node, Relationship
from neo4j.time import DateTime
from typing import cast, List, Dict, Any, Type


def node_to_dict(node: Node) -> Dict[str, Any]:
    """
    Convert a Neo4j Node to a Python dictionary.
    
    Args:
        node: Neo4j Node object
        
    Returns:
        Dictionary representation of the node
    """
    result = dict(node.items())
    
    # Add labels to the dictionary
    result["_labels"] = list(node.labels)
    
    # Convert Neo4j specific types to Python types
    result = convert_neo4j_types(result)
    
    return result


def relationship_to_dict(relationship: Relationship) -> Dict[str, Any]:
    """
    Convert a Neo4j Relationship to a Python dictionary.
    
    Args:
        relationship: Neo4j Relationship object
        
    Returns:
        Dictionary representation of the relationship
    """
    result = dict(relationship.items())
    
    # Add type to the dictionary
    result["_type"] = relationship.type
    
    # Add source and target node IDs (using uuid property)
    start_node = relationship.start_node
    end_node = relationship.end_node
    
    if "uuid" in start_node:
        result["_start_node_id"] = start_node["uuid"]
    
    if "uuid" in end_node:
        result["_end_node_id"] = end_node["uuid"]
    
    # Convert Neo4j specific types to Python types
    result = convert_neo4j_types(result)
    
    return result


def convert_neo4j_types(data: Any) -> Any:
    """
    Convert Neo4j specific types to Python native types.
    
    Args:
        data: Data containing Neo4j types
        
    Returns:
        Data with Neo4j types converted to Python types
    """
    if isinstance(data, DateTime):
        return datetime.datetime.fromisoformat(data.iso_format())
    elif isinstance(data, dict):
        return {k: convert_neo4j_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_neo4j_types(item) for item in data]
    elif hasattr(data, "items") and callable(getattr(data, "items")):
        # Handle Node and Relationship objects
        return {k: convert_neo4j_types(v) for k, v in data.items()}
    else:
        return data


def params_to_neo4j(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Python parameters to Neo4j compatible parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Dictionary with values converted to Neo4j compatible types
    """
    result = {}
    
    for key, value in params.items():
        result[key] = value_to_neo4j(value)
    
    return result


def value_to_neo4j(value: Any) -> Any:
    """
    Convert a Python value to a Neo4j compatible value.
    
    Args:
        value: Python value
        
    Returns:
        Neo4j compatible value
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    elif isinstance(value, UUID):
        return str(value)
    elif isinstance(value, dict):
        return {k: value_to_neo4j(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [value_to_neo4j(item) for item in value]
    elif hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return value_to_neo4j(value.to_dict())
    else:
        # Try to convert to JSON for complex objects
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)


def build_labels_string(labels: Union[str, List[str]]) -> str:
    """
    Build a Cypher-compatible labels string from a label or list of labels.
    
    Args:
        labels: Label or list of labels
        
    Returns:
        String in format ":Label1:Label2" for use in Cypher queries
    """
    if isinstance(labels, str):
        labels = [labels]
    
    return ":" + ":".join(labels)


def build_where_clause(properties: Dict[str, Any], node_var: str = "n") -> str:
    """
    Build a Cypher WHERE clause from a dictionary of properties.
    
    Args:
        properties: Dictionary of properties to match
        node_var: Variable name for the node in the Cypher query
        
    Returns:
        WHERE clause string for use in Cypher queries
    """
    if not properties:
        return ""
    
    clauses = []
    for key, value in properties.items():
        if value is None:
            clauses.append(f"{node_var}.{key} IS NULL")
        else:
            clauses.append(f"{node_var}.{key} = ${key}")
    
    return "WHERE " + " AND ".join(clauses)


def format_direction(direction: str) -> str:
    """
    Format relationship direction for use in Cypher queries.
    
    Args:
        direction: Direction string ("incoming", "outgoing", or "both")
        
    Returns:
        Cypher-compatible direction string
        
    Raises:
        ValueError: If direction is not one of the valid values
    """
    direction = direction.lower()
    
    if direction == "outgoing":
        return "-[r]->"
    elif direction == "incoming":
        return "<-[r]-"
    elif direction == "both":
        return "-[r]-"
    else:
        raise ValueError(
            f"Invalid direction: {direction}. Must be 'incoming', 'outgoing', or 'both'."
        )
