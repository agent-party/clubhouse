"""
Factory functions for Neo4j schema management service.

This module provides factory functions for creating and registering
Neo4j schema management service instances.
"""

from typing import cast

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import DatabaseConfig
from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.services.neo4j.schema.manager import Neo4jSchemaManager
from clubhouse.services.neo4j.schema.protocol import Neo4jSchemaManagerProtocol


def register_neo4j_schema_manager(
    service_registry: ServiceRegistry,
    config: ConfigProtocol[DatabaseConfig],
) -> Neo4jSchemaManagerProtocol:
    """
    Create and register a Neo4j schema manager service.
    
    Args:
        service_registry: Service registry to register with
        config: Configuration provider
        
    Returns:
        Registered Neo4j schema manager instance
    """
    schema_manager = Neo4jSchemaManager(config)
    
    # Register as a singleton service
    service_registry.register_singleton(
        Neo4jSchemaManagerProtocol,
        schema_manager,
    )
    
    return cast(Neo4jSchemaManagerProtocol, schema_manager)
