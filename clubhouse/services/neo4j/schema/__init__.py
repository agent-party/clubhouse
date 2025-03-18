"""
Neo4j schema management package.

This package provides tools for managing Neo4j database schemas,
including constraints, indexes, and entity relationships.
"""

from clubhouse.services.neo4j.schema.protocol import Neo4jSchemaManagerProtocol
from clubhouse.services.neo4j.schema.manager import Neo4jSchemaManager
from clubhouse.services.neo4j.schema.factory import register_neo4j_schema_manager

__all__ = [
    "Neo4jSchemaManagerProtocol",
    "Neo4jSchemaManager",
    "register_neo4j_schema_manager",
]
