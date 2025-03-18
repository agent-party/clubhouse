"""
Query builder for Neo4j Cypher queries.

This module provides a fluent interface for building Cypher queries
in a type-safe way, reducing the risk of syntax errors and making
complex queries more maintainable.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
from typing import cast, List, Dict, Any, Type


class RelationshipDirection(str, Enum):
    """Direction of a relationship in a Cypher query."""
    OUTGOING = "OUTGOING"  # -[r]->
    INCOMING = "INCOMING"  # <-[r]-
    BOTH = "BOTH"         # -[r]-


class ClauseType(str, Enum):
    """Types of Cypher query clauses."""
    MATCH = "MATCH"
    OPTIONAL_MATCH = "OPTIONAL MATCH"
    WHERE = "WHERE"
    WITH = "WITH"
    RETURN = "RETURN"
    ORDER_BY = "ORDER BY"
    SKIP = "SKIP"
    LIMIT = "LIMIT"
    CREATE = "CREATE"
    MERGE = "MERGE"
    SET = "SET"
    DELETE = "DELETE"
    DETACH_DELETE = "DETACH DELETE"
    REMOVE = "REMOVE"


class SortDirection(str, Enum):
    """Sort direction for ORDER BY clauses."""
    ASCENDING = "ASC"
    DESCENDING = "DESC"


class CypherQueryBuilder:
    """
    Fluent builder for Cypher queries.
    
    This class allows building Cypher queries step by step in a readable
    and type-safe manner. It supports all common Cypher clauses and
    operations.
    
    Example:
        >>> query = (CypherQueryBuilder()
        ...          .match("n:Agent")
        ...          .where("n.status = $status")
        ...          .return_clause("n")
        ...          .build())
        >>> params = {"status": "active"}
    """
    
    def __init__(self) -> None:
        """Initialize an empty Cypher query builder."""
        self._clauses: List[Tuple[ClauseType, str]] = []
        self._params: Dict[str, Any] = {}
        
    def match(self, pattern: str) -> 'CypherQueryBuilder':
        """
        Add a MATCH clause to the query.
        
        Args:
            pattern: Cypher pattern to match
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.MATCH, pattern))
        return self
        
    def optional_match(self, pattern: str) -> 'CypherQueryBuilder':
        """
        Add an OPTIONAL MATCH clause to the query.
        
        Args:
            pattern: Cypher pattern to optionally match
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.OPTIONAL_MATCH, pattern))
        return self
        
    def where(self, condition: str) -> 'CypherQueryBuilder':
        """
        Add a WHERE clause to the query.
        
        Args:
            condition: Condition to filter results
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.WHERE, condition))
        return self
        
    def with_clause(self, expressions: str) -> 'CypherQueryBuilder':
        """
        Add a WITH clause to the query.
        
        Args:
            expressions: Expressions to pass to the next part of the query
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.WITH, expressions))
        return self
        
    def return_clause(self, expressions: str) -> 'CypherQueryBuilder':
        """
        Add a RETURN clause to the query.
        
        Args:
            expressions: Expressions to return
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.RETURN, expressions))
        return self
        
    def order_by(self, expressions: str) -> 'CypherQueryBuilder':
        """
        Add an ORDER BY clause to the query.
        
        Args:
            expressions: Expressions to order by
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.ORDER_BY, expressions))
        return self
        
    def skip(self, count: int) -> 'CypherQueryBuilder':
        """
        Add a SKIP clause to the query.
        
        Args:
            count: Number of results to skip
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.SKIP, str(count)))
        return self
        
    def limit(self, count: int) -> 'CypherQueryBuilder':
        """
        Add a LIMIT clause to the query.
        
        Args:
            count: Maximum number of results to return
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.LIMIT, str(count)))
        return self
        
    def create(self, pattern: str) -> 'CypherQueryBuilder':
        """
        Add a CREATE clause to the query.
        
        Args:
            pattern: Pattern to create
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.CREATE, pattern))
        return self
        
    def merge(self, pattern: str) -> 'CypherQueryBuilder':
        """
        Add a MERGE clause to the query.
        
        Args:
            pattern: Pattern to merge
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.MERGE, pattern))
        return self
        
    def set(self, assignments: str) -> 'CypherQueryBuilder':
        """
        Add a SET clause to the query.
        
        Args:
            assignments: Property assignments
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.SET, assignments))
        return self
        
    def delete(self, targets: str) -> 'CypherQueryBuilder':
        """
        Add a DELETE clause to the query.
        
        Args:
            targets: Targets to delete
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.DELETE, targets))
        return self
        
    def detach_delete(self, targets: str) -> 'CypherQueryBuilder':
        """
        Add a DETACH DELETE clause to the query.
        
        Args:
            targets: Targets to detach delete
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.DETACH_DELETE, targets))
        return self
        
    def remove(self, items: str) -> 'CypherQueryBuilder':
        """
        Add a REMOVE clause to the query.
        
        Args:
            items: Items to remove (labels or properties)
            
        Returns:
            Self for method chaining
        """
        self._clauses.append((ClauseType.REMOVE, items))
        return self
        
    def with_params(self, params: Dict[str, Any]) -> 'CypherQueryBuilder':
        """
        Add parameters to the query.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            Self for method chaining
        """
        self._params.update(params)
        return self
        
    def add_param(self, name: str, value: Any) -> 'CypherQueryBuilder':
        """
        Add a single parameter to the query.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Self for method chaining
        """
        self._params[name] = value
        return self
        
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the final Cypher query and parameters.
        
        Returns:
            Tuple of (query_string, params_dict)
        """
        query_parts = []
        
        for clause_type, clause_content in self._clauses:
            query_parts.append(f"{clause_type.value} {clause_content}")
            
        return " ".join(query_parts), self._params


class NodePattern:
    """
    Helper class for building node patterns in Cypher queries.
    
    This class provides a way to build node patterns with variable
    names, labels, and properties in a type-safe way.
    
    Example:
        >>> node = NodePattern("n", ["Agent"], {"status": "active"})
        >>> str(node)
        '(n:Agent {status: $n_status})'
    """
    
    def __init__(
        self, 
        variable: str, 
        labels: Optional[Union[str, List[str]]] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a node pattern.
        
        Args:
            variable: Variable name for the node
            labels: Label or list of labels
            properties: Dictionary of properties
        """
        self.variable = variable
        self.labels = labels if labels else []
        if isinstance(self.labels, str):
            self.labels = [self.labels]
        self.properties = properties or {}
        
    def __str__(self) -> str:
        """
        Convert the node pattern to a Cypher string.
        
        Returns:
            Cypher node pattern string
        """
        pattern = f"({self.variable}"
        
        if self.labels:
            pattern += ":" + ":".join(self.labels)
            
        if self.properties:
            props = []
            for key in self.properties:
                props.append(f"{key}: ${self.variable}_{key}")
            pattern += " {" + ", ".join(props) + "}"
            
        pattern += ")"
        return pattern
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for the node's properties.
        
        Returns:
            Dictionary of parameters with prefixed keys
        """
        return {f"{self.variable}_{key}": value for key, value in self.properties.items()}


class RelationshipPattern:
    """
    Helper class for building relationship patterns in Cypher queries.
    
    This class provides a way to build relationship patterns with variable
    names, types, properties, and direction in a type-safe way.
    
    Example:
        >>> rel = RelationshipPattern("r", "KNOWS", {"since": 2020}, RelationshipDirection.OUTGOING)
        >>> str(rel)
        '-[r:KNOWS {since: $r_since}]->'
    """
    
    def __init__(
        self,
        variable: str,
        rel_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        direction: RelationshipDirection = RelationshipDirection.BOTH
    ) -> None:
        """
        Initialize a relationship pattern.
        
        Args:
            variable: Variable name for the relationship
            rel_type: Type of relationship
            properties: Dictionary of properties
            direction: Direction of the relationship
        """
        self.variable = variable
        self.rel_type = rel_type
        self.properties = properties or {}
        self.direction = direction
        
    def __str__(self) -> str:
        """
        Convert the relationship pattern to a Cypher string.
        
        Returns:
            Cypher relationship pattern string
        """
        if self.direction == RelationshipDirection.INCOMING:
            pattern = "<-"
        else:
            pattern = "-"
            
        pattern += f"[{self.variable}"
        
        if self.rel_type:
            pattern += f":{self.rel_type}"
            
        if self.properties:
            props = []
            for key in self.properties:
                props.append(f"{key}: ${self.variable}_{key}")
            pattern += " {" + ", ".join(props) + "}"
            
        pattern += "]-"
        
        if self.direction == RelationshipDirection.OUTGOING:
            pattern += ">"
            
        return pattern
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for the relationship's properties.
        
        Returns:
            Dictionary of parameters with prefixed keys
        """
        return {f"{self.variable}_{key}": value for key, value in self.properties.items()}


class PathPattern:
    """
    Helper class for building path patterns in Cypher queries.
    
    This class provides a way to build path patterns with nodes and
    relationships in a type-safe way.
    
    Example:
        >>> node1 = NodePattern("a", "Agent")
        >>> rel = RelationshipPattern("r", "KNOWS", direction=RelationshipDirection.OUTGOING)
        >>> node2 = NodePattern("b", "Agent")
        >>> path = PathPattern([node1, rel, node2])
        >>> str(path)
        '(a:Agent)-[r:KNOWS]->(b:Agent)'
    """
    
    def __init__(self, elements: List[Union[NodePattern, RelationshipPattern]]) -> None:
        """
        Initialize a path pattern.
        
        Args:
            elements: List of node and relationship patterns, must alternate
        """
        if len(elements) % 2 == 0:
            raise ValueError("Path must have an odd number of elements (alternating nodes and relationships)")
            
        for i, element in enumerate(elements):
            if i % 2 == 0 and not isinstance(element, NodePattern):
                raise ValueError(f"Element at position {i} must be a NodePattern")
            if i % 2 == 1 and not isinstance(element, RelationshipPattern):
                raise ValueError(f"Element at position {i} must be a RelationshipPattern")
                
        self.elements = elements
        
    def __str__(self) -> str:
        """
        Convert the path pattern to a Cypher string.
        
        Returns:
            Cypher path pattern string
        """
        return "".join(str(element) for element in self.elements)
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for all elements in the path.
        
        Returns:
            Dictionary of parameters with prefixed keys
        """
        params = {}
        for element in self.elements:
            params.update(element.get_params())
        return params
