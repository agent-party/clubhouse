"""
Neo4j utility functions for parameter handling, formatting, and data transformation.

This module provides utility functions for working with Neo4j, focusing on
common operations needed across multiple services and repositories.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar, Callable
from uuid import UUID
from datetime import datetime, date

from neo4j.graph import Node, Relationship

# Type variables
T = TypeVar('T')


def params_to_neo4j(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert Python parameter values to Neo4j compatible values.
    
    Handles special types like UUID, datetime, and nested dictionaries/lists.
    
    Args:
        params: Dictionary of parameters to convert
        
    Returns:
        Dictionary with Neo4j compatible values
    """
    if params is None:
        return {}
        
    result = {}
    
    for key, value in params.items():
        result[key] = _convert_value_to_neo4j(value)
            
    return result


def format_direction(direction: str) -> str:
    """
    Format a relationship direction string for use in Cypher queries.
    
    Args:
        direction: Direction string ('outgoing', 'incoming', or 'both')
        
    Returns:
        Formatted direction string for Cypher (e.g., '->', '<-', or '-')
        
    Raises:
        ValueError: If an invalid direction is provided
    """
    direction = direction.lower()
    
    if direction == 'outgoing':
        return '->'
    elif direction == 'incoming':
        return '<-'
    elif direction == 'both':
        return '-'
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'outgoing', 'incoming', or 'both'")


def node_to_dict(node: Node) -> Dict[str, Any]:
    """
    Convert a Neo4j Node object to a dictionary.
    
    The resulting dictionary includes:
    - All node properties
    - A 'labels' field containing the node's labels
    - An 'id' field containing the node's internal ID (for reference only)
    
    Args:
        node: Neo4j Node object to convert
        
    Returns:
        Dictionary representation of the node
    """
    if node is None:
        return {}
        
    # Start with all node properties
    result = dict(node.items())
    
    # Add labels and internal ID
    result['labels'] = list(node.labels)
    result['id'] = node.id
    
    return result


def relationship_to_dict(relationship: Relationship) -> Dict[str, Any]:
    """
    Convert a Neo4j Relationship object to a dictionary.
    
    The resulting dictionary includes:
    - All relationship properties
    - A 'type' field containing the relationship type
    - 'start_node_id' and 'end_node_id' fields containing internal IDs
    - An 'id' field containing the relationship's internal ID (for reference only)
    
    Args:
        relationship: Neo4j Relationship object to convert
        
    Returns:
        Dictionary representation of the relationship
    """
    if relationship is None:
        return {}
        
    # Start with all relationship properties
    result = dict(relationship.items())
    
    # Add type and IDs
    result['type'] = relationship.type
    result['start_node_id'] = relationship.start_node.id
    result['end_node_id'] = relationship.end_node.id
    result['id'] = relationship.id
    
    return result


def _convert_value_to_neo4j(value: Any) -> Any:
    """
    Convert a single value to Neo4j compatible format.
    
    Args:
        value: Value to convert
        
    Returns:
        Neo4j compatible value
    """
    if value is None:
        return None
        
    # Handle UUIDs
    if isinstance(value, UUID):
        return str(value)
        
    # Handle datetime objects
    if isinstance(value, (datetime, date)):
        return value.isoformat()
        
    # Handle dictionaries (recursive conversion)
    if isinstance(value, dict):
        return {k: _convert_value_to_neo4j(v) for k, v in value.items()}
        
    # Handle lists (recursive conversion)
    if isinstance(value, (list, tuple)):
        return [_convert_value_to_neo4j(item) for item in value]
        
    # Return primitive types as is
    return value
