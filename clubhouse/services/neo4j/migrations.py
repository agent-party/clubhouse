"""
Neo4j schema migrations for the Clubhouse platform.

This module contains all schema migrations for the Neo4j database,
organized in chronological order by version number.
"""

import logging
from typing import Dict, List, Any

from clubhouse.services.neo4j.schema import SchemaMigration

logger = logging.getLogger(__name__)


class ConversationSchemaMigration(SchemaMigration):
    """
    Migration to add conversation and context schema.
    
    This migration adds the necessary schema elements for conversation tracking,
    message persistence, and context management in the graph database.
    """
    
    version = 3  # Building on existing migrations (1 and 2)
    description = "Add conversation and context data model"
    
    def up(self) -> List[str]:
        """
        Get the list of Cypher statements for applying this migration.
        
        Returns:
            List of Cypher statements to create conversation schema
        """
        return [
            # Conversation node constraints
            "CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS FOR (n:Conversation) REQUIRE n.conversation_id IS UNIQUE",
            
            # Message node constraints
            "CREATE CONSTRAINT message_id_unique IF NOT EXISTS FOR (n:Message) REQUIRE n.message_id IS UNIQUE",
            
            # Context node constraints
            "CREATE CONSTRAINT context_id_unique IF NOT EXISTS FOR (n:Context) REQUIRE n.context_id IS UNIQUE",
            
            # Indexes for efficient retrieval
            "CREATE INDEX conversation_created_idx IF NOT EXISTS FOR (n:Conversation) ON (n.created_at)",
            "CREATE INDEX message_timestamp_idx IF NOT EXISTS FOR (n:Message) ON (n.timestamp)",
            "CREATE INDEX context_type_idx IF NOT EXISTS FOR (n:Context) ON (n.type)",
            
            # Ensure created_at is present on conversations
            "CREATE CONSTRAINT conversation_created_at_exists IF NOT EXISTS FOR (n:Conversation) REQUIRE n.created_at IS NOT NULL",
            
            # Ensure timestamp is present on messages
            "CREATE CONSTRAINT message_timestamp_exists IF NOT EXISTS FOR (n:Message) REQUIRE n.timestamp IS NOT NULL",
            
            # Ensure type is present on contexts
            "CREATE CONSTRAINT context_type_exists IF NOT EXISTS FOR (n:Context) REQUIRE n.type IS NOT NULL"
        ]
    
    def down(self) -> List[str]:
        """
        Get the list of Cypher statements for reverting this migration.
        
        Returns:
            List of Cypher statements to revert conversation schema
        """
        return [
            # Drop constraints
            "DROP CONSTRAINT conversation_id_unique IF EXISTS",
            "DROP CONSTRAINT message_id_unique IF EXISTS",
            "DROP CONSTRAINT context_id_unique IF EXISTS",
            "DROP CONSTRAINT conversation_created_at_exists IF EXISTS",
            "DROP CONSTRAINT message_timestamp_exists IF EXISTS",
            "DROP CONSTRAINT context_type_exists IF EXISTS",
            
            # Drop indexes
            "DROP INDEX conversation_created_idx IF EXISTS",
            "DROP INDEX message_timestamp_idx IF EXISTS", 
            "DROP INDEX context_type_idx IF EXISTS"
        ]


# Register all migrations in this ordered list, which should be imported by the SchemaManager
ALL_MIGRATIONS = [
    ConversationSchemaMigration(),
]
