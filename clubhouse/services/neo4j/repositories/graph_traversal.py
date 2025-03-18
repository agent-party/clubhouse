"""
Graph Traversal Utilities for Neo4j repositories.

This module provides a collection of utilities for efficient graph traversal
and path-finding in Neo4j. These utilities are designed to support complex
context retrieval, relationship navigation, and knowledge graph exploration
while maintaining type safety and performance.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union
from uuid import UUID

import logging
import neo4j
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from pydantic import BaseModel

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol

logger = logging.getLogger(__name__)

# Type variables for generic result transformations
T = TypeVar('T')
U = TypeVar('U')


class PathResult(BaseModel):
    """Represents a path result from a graph traversal operation."""
    
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    length: int
    start_node: Dict[str, Any]
    end_node: Dict[str, Any]
    
    @classmethod
    def from_neo4j_path(cls, path: neo4j.graph.Path) -> "PathResult":
        """
        Convert a Neo4j Path object to a PathResult.
        
        This method extracts all properties from nodes and relationships
        in the path and converts them to dictionaries.
        
        Args:
            path: Neo4j Path object
            
        Returns:
            PathResult object containing node and relationship data
        """
        # Extract node properties
        nodes = []
        for node in path.nodes:
            node_dict = {}
            try:
                if hasattr(node, '_properties'):
                    node_dict.update(node._properties)
                else:
                    node_dict.update(dict(node.items()))
            except (AttributeError, TypeError):
                try:
                    node_dict.update({k: v for k, v in node})
                except (TypeError, ValueError):
                    node_dict.update(vars(node))
            nodes.append(node_dict)
        
        # Extract relationship properties
        relationships = []
        for rel in path.relationships:
            rel_dict = {}
            try:
                if hasattr(rel, '_properties'):
                    rel_dict.update(rel._properties)
                else:
                    rel_dict.update(dict(rel.items()))
                # Add relationship type
                rel_dict['type'] = rel.type
            except (AttributeError, TypeError):
                try:
                    rel_dict.update({k: v for k, v in rel})
                except (TypeError, ValueError):
                    rel_dict.update(vars(rel))
            relationships.append(rel_dict)
        
        # Get start and end nodes
        start_node = nodes[0] if nodes else {}
        end_node = nodes[-1] if nodes else {}
        
        return cls(
            nodes=nodes,
            relationships=relationships,
            length=len(relationships),
            start_node=start_node,
            end_node=end_node
        )


class GraphTraversalRepository:
    """
    Repository for graph traversal and path-finding operations.
    
    This repository provides methods for:
    1. Finding paths between entities
    2. Retrieving connected subgraphs
    3. Navigating relationships with filters and constraints
    4. Transforming graph results into domain models
    """
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol):
        """
        Initialize the repository with a Neo4j service.
        
        Args:
            neo4j_service: Service that provides Neo4j database connections
        """
        self.neo4j_service = neo4j_service
    
    def find_shortest_path(
        self,
        source_id: Union[str, UUID],
        target_id: Union[str, UUID],
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 5,
    ) -> Optional[PathResult]:
        """
        Find the shortest path between two nodes in the graph.
        
        Args:
            source_id: UUID of the source node
            target_id: UUID of the target node
            relationship_types: Optional list of relationship types to consider
            max_depth: Maximum path length to consider
            
        Returns:
            PathResult if a path is found, None otherwise
        """
        # Convert IDs to strings if they're UUIDs
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Build relationship constraint for the query
        rel_filter = ""
        if relationship_types:
            # Proper syntax for relationship type filtering in Cypher
            if len(relationship_types) == 1:
                rel_filter = f"[r:{relationship_types[0]}*..{max_depth}]"
            else:
                # For multiple relationship types, we need to use the | operator
                rel_types = "|".join(relationship_types)
                rel_filter = f"[r:{rel_types}*..{max_depth}]"
        else:
            rel_filter = f"[r*..{max_depth}]"
        
        # Build Cypher query to find shortest path
        query = f"""
        MATCH (source), (target), 
              p = shortestPath((source)-{rel_filter}->(target))
        WHERE source.uuid = $source_id AND target.uuid = $target_id
        RETURN p
        """
        
        logger.info(f"Executing shortest path query with filter: {rel_filter}")
        
        try:
            result = self.neo4j_service.execute_query(
                query,
                parameters={
                    "source_id": source_id_str,
                    "target_id": target_id_str
                },
                readonly=True
            )
            
            if result and len(result) > 0:
                record = result[0]
                if 'p' in record and record['p'] is not None:
                    path_result = PathResult.from_neo4j_path(record['p'])
                    return path_result
            
            logger.info(f"No path found between {source_id_str} and {target_id_str}")
            return None
                
        except Exception as e:
            logger.error(f"Error finding shortest path: {str(e)}")
            return None
    
    def find_all_paths(
        self,
        source_id: Union[str, UUID],
        target_id: Union[str, UUID],
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 3,
        limit: int = 10
    ) -> List[PathResult]:
        """
        Find all paths between two nodes in the graph, up to a limit.
        
        Args:
            source_id: UUID of the source node
            target_id: UUID of the target node
            relationship_types: Optional list of relationship types to consider
            max_depth: Maximum path length to consider
            limit: Maximum number of paths to return
            
        Returns:
            List of PathResult objects, one for each path found
        """
        # Convert IDs to strings if they're UUIDs
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Build relationship constraint
        rel_filter = ""
        if relationship_types:
            if len(relationship_types) == 1:
                rel_filter = f"[r:{relationship_types[0]}*..{max_depth}]"
            else:
                rel_types = "|".join(relationship_types)
                rel_filter = f"[r:{rel_types}*..{max_depth}]"
        else:
            rel_filter = f"[r*..{max_depth}]"
        
        # Build Cypher query to find all paths
        query = f"""
        MATCH (source), (target),
              p = (source)-{rel_filter}->(target)
        WHERE source.uuid = $source_id AND target.uuid = $target_id
        RETURN p
        LIMIT $limit
        """
        
        try:
            result = self.neo4j_service.execute_query(
                query,
                parameters={
                    "source_id": source_id_str,
                    "target_id": target_id_str,
                    "limit": limit
                },
                readonly=True
            )
            
            paths = []
            for record in result:
                if 'p' in record and record['p'] is not None:
                    path_result = PathResult.from_neo4j_path(record['p'])
                    paths.append(path_result)
            
            return paths
            
        except Exception as e:
            logger.error(f"Error finding all paths: {str(e)}")
            return []
    
    def get_subgraph(
        self,
        root_id: Union[str, UUID],
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
        direction: str = "OUTGOING"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract a subgraph starting from a root node.
        
        Args:
            root_id: UUID of the root node
            relationship_types: Optional list of relationship types to consider
            max_depth: Maximum depth of the subgraph
            direction: Direction of relationships to follow (OUTGOING, INCOMING, or BOTH)
            
        Returns:
            Dictionary with 'nodes' and 'relationships' representing the subgraph
        """
        root_id_str = str(root_id)
        
        # Define the relationship pattern based on direction parameter
        if direction == "BOTH":
            # For BOTH direction, we use an undirected pattern to capture all connections
            path_pattern = f"MATCH path = (root)-[*1..{max_depth}]-()"
        elif direction == "INCOMING":
            # For INCOMING, we look for paths ending at our root
            path_pattern = f"MATCH path = (n)-[*1..{max_depth}]->(root)"
        else:  # OUTGOING is default
            # For OUTGOING, we look for paths starting from our root
            path_pattern = f"MATCH path = (root)-[*1..{max_depth}]->()"
        
        # Build relationship type filter if needed
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
            # Adjust the path pattern to include relationship filter
            path_pattern = path_pattern.replace("[*", f"[{rel_filter}*")
        
        # Build the full Cypher query
        # This approach ensures we get all nodes regardless of direction
        query = f"""
        MATCH (root)
        WHERE root.uuid = $root_id
        {path_pattern}
        WHERE root.uuid = $root_id
        WITH collect(path) as paths
        UNWIND paths as p
        WITH p
        UNWIND nodes(p) as node
        WITH collect(DISTINCT node) as nodes
        UNWIND nodes as n
        WITH nodes, n
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE m IN nodes
        WITH nodes, collect(DISTINCT r) as relationships
        RETURN nodes, relationships
        """
        
        try:
            result = self.neo4j_service.execute_query(
                query,
                parameters={
                    "root_id": root_id_str
                },
                readonly=True
            )
            
            if result and len(result) > 0:
                record = result[0]
                
                # Convert nodes to dictionaries
                nodes = []
                for node in record['nodes']:
                    node_dict = dict(node.items())
                    nodes.append(node_dict)
                
                # Convert relationships to dictionaries
                relationships = []
                for rel in record['relationships']:
                    rel_dict = dict(rel.items())
                    rel_dict['type'] = rel.type
                    relationships.append(rel_dict)
                
                return {
                    'nodes': nodes,
                    'relationships': relationships
                }
            
            return {'nodes': [], 'relationships': []}
            
        except Exception as e:
            logger.error(f"Error getting subgraph: {str(e)}")
            return {'nodes': [], 'relationships': []}
    
    def transform_results(self, records: List[Dict[str, Any]], model_cls: type[T]) -> List[T]:
        """
        Transform Neo4j query results into domain model instances.
        
        Args:
            records: List of record dictionaries from Neo4j
            model_cls: Pydantic model class to transform records into
            
        Returns:
            List of domain model instances
        """
        try:
            return [model_cls(**record) for record in records]
        except Exception as e:
            logger.error(f"Error transforming results to {model_cls.__name__}: {str(e)}")
            return []
