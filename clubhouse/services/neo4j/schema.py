"""
Neo4j schema management for the Clubhouse platform.

This module provides tools for initializing, validating, and migrating
the Neo4j database schema, ensuring data integrity and consistency.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from neo4j.exceptions import Neo4jError, ClientError

from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol
from clubhouse.services.neo4j.query_builder import CypherQueryBuilder
from typing import cast, List, Dict, Any, Type

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of constraints that can be applied in Neo4j."""
    UNIQUE = "UNIQUE"
    EXISTS = "EXISTS"
    NODE_KEY = "NODE_KEY"  # Enterprise only


class IndexType(str, Enum):
    """Types of indexes that can be created in Neo4j."""
    BTREE = "BTREE"
    FULLTEXT = "FULLTEXT"
    TEXT = "TEXT"  # Neo4j 4.3+
    POINT = "POINT"  # Spatial index


@dataclass
class ConstraintDefinition:
    """Definition of a Neo4j constraint."""
    name: str
    label: str
    property_keys: List[str]
    type: ConstraintType
    
    def to_cypher(self) -> str:
        """
        Convert the constraint definition to a Cypher CREATE CONSTRAINT statement.
        
        Returns:
            Cypher query string for creating the constraint
        """
        props = ", ".join(f"node.{prop}" for prop in self.property_keys)
        
        if self.type == ConstraintType.UNIQUE:
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (node:{self.label}) REQUIRE ({props}) IS UNIQUE"
        elif self.type == ConstraintType.EXISTS:
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (node:{self.label}) REQUIRE ({props}) IS NOT NULL"
        elif self.type == ConstraintType.NODE_KEY:
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (node:{self.label}) REQUIRE ({props}) IS NODE KEY"
        else:
            raise ValueError(f"Unsupported constraint type: {self.type}")


@dataclass
class IndexDefinition:
    """Definition of a Neo4j index."""
    name: str
    label: str
    property_keys: List[str]
    type: IndexType = IndexType.BTREE
    
    def to_cypher(self) -> str:
        """
        Convert the index definition to a Cypher CREATE INDEX statement.
        
        Returns:
            Cypher query string for creating the index
        """
        if self.type == IndexType.BTREE:
            props = ", ".join(f"node.{prop}" for prop in self.property_keys)
            return f"CREATE INDEX {self.name} IF NOT EXISTS FOR (node:{self.label}) ON ({props})"
        elif self.type == IndexType.FULLTEXT:
            props = ", ".join(repr(prop) for prop in self.property_keys)
            return f"CALL db.index.fulltext.createNodeIndex({repr(self.name)}, [{repr(self.label)}], [{props}])"
        elif self.type == IndexType.TEXT:
            props = ", ".join(f"node.{prop}" for prop in self.property_keys)
            return f"CREATE TEXT INDEX {self.name} IF NOT EXISTS FOR (node:{self.label}) ON ({props})"
        elif self.type == IndexType.POINT:
            props = ", ".join(f"node.{prop}" for prop in self.property_keys)
            return f"CREATE POINT INDEX {self.name} IF NOT EXISTS FOR (node:{self.label}) ON ({props})"
        else:
            raise ValueError(f"Unsupported index type: {self.type}")


class SchemaMigration:
    """Base class for schema migrations."""
    
    version: int = 0  # Must be overridden in subclasses
    description: str = ""  # Must be overridden in subclasses
    
    def up(self) -> List[str]:
        """
        Get the list of Cypher statements for applying this migration.
        
        Returns:
            List of Cypher statements
        """
        raise NotImplementedError("Subclasses must implement up()")
        
    def down(self) -> List[str]:
        """
        Get the list of Cypher statements for reverting this migration.
        
        Returns:
            List of Cypher statements
        """
        raise NotImplementedError("Subclasses must implement down()")


class SchemaManager:
    """
    Manager for Neo4j database schema.
    
    This class provides methods for initializing the database schema,
    applying migrations, and validating the schema.
    """
    
    # Default constraints for the agent data model
    DEFAULT_CONSTRAINTS = [
        ConstraintDefinition(
            name="agent_uuid_unique",
            label="Agent",
            property_keys=["uuid"],
            type=ConstraintType.UNIQUE
        ),
        ConstraintDefinition(
            name="agent_id_unique",
            label="Agent",
            property_keys=["agent_id"],
            type=ConstraintType.UNIQUE
        ),
        ConstraintDefinition(
            name="capability_uuid_unique",
            label="Capability",
            property_keys=["uuid"],
            type=ConstraintType.UNIQUE
        ),
        ConstraintDefinition(
            name="capability_name_unique",
            label="Capability",
            property_keys=["name"],
            type=ConstraintType.UNIQUE
        ),
        ConstraintDefinition(
            name="agent_group_uuid_unique",
            label="AgentGroup",
            property_keys=["uuid"],
            type=ConstraintType.UNIQUE
        ),
    ]
    
    # Default indexes for the agent data model
    DEFAULT_INDEXES = [
        IndexDefinition(
            name="agent_status_idx",
            label="Agent",
            property_keys=["status"]
        ),
        IndexDefinition(
            name="agent_type_idx",
            label="Agent",
            property_keys=["type"]
        ),
        IndexDefinition(
            name="agent_name_fulltext",
            label="Agent",
            property_keys=["name", "description"],
            type=IndexType.FULLTEXT
        ),
    ]
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol) -> None:
        """
        Initialize the schema manager.
        
        Args:
            neo4j_service: Service for interacting with Neo4j
        """
        self.neo4j_service = neo4j_service
        self._migrations: List[SchemaMigration] = []
    
    def register_migration(self, migration: SchemaMigration) -> None:
        """
        Register a migration with the schema manager.
        
        Args:
            migration: Migration to register
        """
        self._migrations.append(migration)
        # Sort migrations by version
        self._migrations.sort(key=lambda m: m.version)
    
    async def initialize_schema(self) -> None:
        """
        Initialize the database schema.
        
        This creates the migration tracking table if it doesn't exist,
        and applies any default constraints and indexes if the database is empty.
        """
        try:
            # Create the migration tracking table if it doesn't exist
            await self._create_migration_table()
            
            # Check if any migrations have been applied
            current_version = await self._get_current_version()
            
            if current_version == 0:
                # No migrations applied yet, initialize with default schema
                logger.info("Initializing Neo4j database schema with default constraints and indexes")
                await self._apply_default_schema()
            
            # Apply any pending migrations
            await self.migrate()
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    async def migrate(self, target_version: Optional[int] = None) -> None:
        """
        Apply pending migrations to reach the target version.
        
        Args:
            target_version: Optional target version, defaults to latest
        """
        current_version = await self._get_current_version()
        
        if not self._migrations:
            logger.info("No migrations registered")
            return
            
        if target_version is None:
            target_version = max(m.version for m in self._migrations)
            
        if current_version == target_version:
            logger.info(f"Database is already at version {current_version}")
            return
            
        # Determine if we're migrating up or down
        if current_version < target_version:
            # Apply migrations from current_version+1 to target_version
            migrations_to_apply = [m for m in self._migrations 
                                 if current_version < m.version <= target_version]
            
            logger.info(f"Migrating up from version {current_version} to {target_version}")
            
            for migration in migrations_to_apply:
                logger.info(f"Applying migration {migration.version}: {migration.description}")
                
                try:
                    # Get and apply the migration statements
                    statements = migration.up()
                    for statement in statements:
                        await self.neo4j_service.run_query(statement, {})
                        
                    # Record the migration
                    await self._record_migration(migration.version, migration.description)
                    
                    logger.info(f"Migration {migration.version} applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply migration {migration.version}: {e}")
                    raise
        else:
            # Apply migrations from current_version down to target_version+1
            migrations_to_apply = [m for m in self._migrations 
                                 if target_version < m.version <= current_version]
            migrations_to_apply.reverse()  # Apply in reverse order
            
            logger.info(f"Migrating down from version {current_version} to {target_version}")
            
            for migration in migrations_to_apply:
                logger.info(f"Reverting migration {migration.version}: {migration.description}")
                
                try:
                    # Get and apply the migration statements
                    statements = migration.down()
                    for statement in statements:
                        await self.neo4j_service.run_query(statement, {})
                        
                    # Record the migration reversion
                    await self._delete_migration_record(migration.version)
                    
                    logger.info(f"Migration {migration.version} reverted successfully")
                except Exception as e:
                    logger.error(f"Failed to revert migration {migration.version}: {e}")
                    raise
    
    async def validate_schema(self) -> bool:
        """
        Validate that the database schema matches the expected schema.
        
        Returns:
            True if the schema is valid, False otherwise
        """
        try:
            # Check all constraints
            constraints_result = await self.neo4j_service.run_query(
                "SHOW CONSTRAINTS", {})
            
            expected_constraint_names = set(c.name for c in self.DEFAULT_CONSTRAINTS)
            actual_constraint_names = set()
            
            for record in constraints_result:
                if 'name' in record:
                    actual_constraint_names.add(record['name'])
            
            missing_constraints = expected_constraint_names - actual_constraint_names
            if missing_constraints:
                logger.warning(f"Missing constraints: {missing_constraints}")
                return False
            
            # Check all indexes
            indexes_result = await self.neo4j_service.run_query(
                "SHOW INDEXES", {})
            
            expected_index_names = set(idx.name for idx in self.DEFAULT_INDEXES 
                                    if idx.type != IndexType.FULLTEXT)
            actual_index_names = set()
            
            for record in indexes_result:
                if 'name' in record:
                    actual_index_names.add(record['name'])
            
            missing_indexes = expected_index_names - actual_index_names
            if missing_indexes:
                logger.warning(f"Missing indexes: {missing_indexes}")
                return False
            
            # Check fulltext indexes separately
            fulltext_indexes = [idx for idx in self.DEFAULT_INDEXES 
                             if idx.type == IndexType.FULLTEXT]
            
            for idx in fulltext_indexes:
                result = await self.neo4j_service.run_query(
                    "CALL db.index.fulltext.listNodeIndexes() YIELD indexName WHERE indexName = $name RETURN count(*) as count",
                    {"name": idx.name})
                
                if not result or result[0]['count'] == 0:
                    logger.warning(f"Missing fulltext index: {idx.name}")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to validate schema: {e}")
            return False
    
    async def _create_migration_table(self) -> None:
        """Create the migration tracking table if it doesn't exist."""
        query = """
        CREATE CONSTRAINT migration_id_unique IF NOT EXISTS 
        FOR (m:SchemaMigration) REQUIRE m.version IS UNIQUE
        """
        await self.neo4j_service.run_query(query, {})
    
    async def _get_current_version(self) -> int:
        """
        Get the current schema version.
        
        Returns:
            Current schema version, 0 if no migrations have been applied
        """
        query = """
        MATCH (m:SchemaMigration)
        RETURN max(m.version) AS version
        """
        result = await self.neo4j_service.run_query(query, {})
        
        if not result or result[0]['version'] is None:
            return 0
            
        return int(result[0]['version'])
    
    async def _record_migration(self, version: int, description: str) -> None:
        """
        Record that a migration has been applied.
        
        Args:
            version: Migration version
            description: Migration description
        """
        query = """
        CREATE (m:SchemaMigration {
            version: $version,
            description: $description,
            applied_at: datetime()
        })
        """
        await self.neo4j_service.run_query(query, {
            "version": version,
            "description": description
        })
    
    async def _delete_migration_record(self, version: int) -> None:
        """
        Delete a migration record.
        
        Args:
            version: Migration version to delete
        """
        query = """
        MATCH (m:SchemaMigration {version: $version})
        DELETE m
        """
        await self.neo4j_service.run_query(query, {"version": version})
    
    async def _apply_default_schema(self) -> None:
        """Apply the default constraints and indexes."""
        # Apply default constraints
        for constraint in self.DEFAULT_CONSTRAINTS:
            try:
                await self.neo4j_service.run_query(constraint.to_cypher(), {})
                logger.info(f"Created constraint: {constraint.name}")
            except Exception as e:
                logger.error(f"Failed to create constraint {constraint.name}: {e}")
                raise
        
        # Apply default indexes
        for index in self.DEFAULT_INDEXES:
            try:
                if index.type == IndexType.FULLTEXT:
                    # Fulltext indexes use a different syntax
                    query = (
                        f"CALL db.index.fulltext.createNodeIndex("
                        f"{repr(index.name)}, [{repr(index.label)}], "
                        f"[{', '.join(repr(prop) for prop in index.property_keys)}])"
                    )
                    await self.neo4j_service.run_query(query, {})
                else:
                    await self.neo4j_service.run_query(index.to_cypher(), {})
                
                logger.info(f"Created index: {index.name}")
                
                # Wait for indexes to come online (especially important in tests)
                await self._wait_for_index(index.name)
            except Exception as e:
                logger.error(f"Failed to create index {index.name}: {e}")
                raise
    
    async def _wait_for_index(self, index_name: str, timeout_seconds: int = 30) -> None:
        """
        Wait for an index to come online.
        
        Args:
            index_name: Name of the index to wait for
            timeout_seconds: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                result = await self.neo4j_service.run_query(
                    "SHOW INDEXES WHERE name = $name", {"name": index_name})
                
                if result and len(result) > 0:
                    record = result[0]
                    if record.get('state') == 'online':
                        return
            except Exception as e:
                logger.warning(f"Error checking index state: {e}")
            
            # Sleep for a short time before checking again
            time.sleep(1)
        
        logger.warning(f"Timeout waiting for index {index_name} to come online")


class InitialMigration(SchemaMigration):
    """Initial migration that creates the core schema for agents."""
    
    version = 1
    description = "Create initial agent schema"
    
    def up(self) -> List[str]:
        """
        Get the list of Cypher statements for applying this migration.
        
        Returns:
            List of Cypher statements
        """
        return [
            # Ensure the Agent model is properly defined
            """
            CREATE CONSTRAINT agent_uuid_unique IF NOT EXISTS
            FOR (a:Agent) REQUIRE a.uuid IS UNIQUE
            """,
            
            """
            CREATE CONSTRAINT agent_id_unique IF NOT EXISTS
            FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE
            """,
            
            # Ensure the Capability model is properly defined
            """
            CREATE CONSTRAINT capability_uuid_unique IF NOT EXISTS
            FOR (c:Capability) REQUIRE c.uuid IS UNIQUE
            """,
            
            """
            CREATE CONSTRAINT capability_name_unique IF NOT EXISTS
            FOR (c:Capability) REQUIRE c.name IS UNIQUE
            """,
            
            # Ensure the AgentGroup model is properly defined
            """
            CREATE CONSTRAINT agent_group_uuid_unique IF NOT EXISTS
            FOR (g:AgentGroup) REQUIRE g.uuid IS UNIQUE
            """,
            
            # Create indexes for common query patterns
            """
            CREATE INDEX agent_status_idx IF NOT EXISTS
            FOR (a:Agent) ON (a.status)
            """,
            
            """
            CREATE INDEX agent_type_idx IF NOT EXISTS
            FOR (a:Agent) ON (a.type)
            """
        ]
    
    def down(self) -> List[str]:
        """
        Get the list of Cypher statements for reverting this migration.
        
        Returns:
            List of Cypher statements
        """
        return [
            "DROP INDEX agent_type_idx IF EXISTS",
            "DROP INDEX agent_status_idx IF EXISTS",
            "DROP CONSTRAINT agent_group_uuid_unique IF EXISTS",
            "DROP CONSTRAINT capability_name_unique IF EXISTS",
            "DROP CONSTRAINT capability_uuid_unique IF EXISTS",
            "DROP CONSTRAINT agent_id_unique IF EXISTS",
            "DROP CONSTRAINT agent_uuid_unique IF EXISTS"
        ]


class AddFullTextSearchMigration(SchemaMigration):
    """Migration to add full-text search for agents."""
    
    version = 2
    description = "Add full-text search for agents"
    
    def up(self) -> List[str]:
        """
        Get the list of Cypher statements for applying this migration.
        
        Returns:
            List of Cypher statements
        """
        return [
            """
            CALL db.index.fulltext.createNodeIndex(
                'agent_name_fulltext',
                ['Agent'],
                ['name', 'description']
            )
            """
        ]
    
    def down(self) -> List[str]:
        """
        Get the list of Cypher statements for reverting this migration.
        
        Returns:
            List of Cypher statements
        """
        return [
            """
            CALL db.index.fulltext.drop('agent_name_fulltext')
            """
        ]
