"""
Unit tests for the Neo4j Schema Manager.

This module contains tests for the Neo4jSchemaManager service, focusing on
constraint and index management, schema validation, and migration operations.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from neo4j.exceptions import Neo4jError, ClientError, ServiceUnavailable

from clubhouse.core.config import ConfigProtocol
from clubhouse.core.config.models.database import DatabaseConfig, Neo4jDatabaseConfig, DatabaseType
from clubhouse.services.neo4j.schema.manager import Neo4jSchemaManager


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = MagicMock(spec=ConfigProtocol)
    
    # Create Neo4j configuration
    neo4j_config = Neo4jDatabaseConfig(
        name="test-neo4j",
        hosts=["localhost:7687"],
        username="neo4j",
        password="password",
        database="neo4j",
        connection_pool={"max_size": 50, "connection_timeout_seconds": 60, "max_idle_time_seconds": 3600}
    )
    
    # Set up the config.get() method to return the Neo4j config directly
    config.get.return_value = neo4j_config
    
    return config


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    with patch("neo4j.GraphDatabase.driver") as mock:
        driver = MagicMock()
        mock.return_value = driver
        yield driver


@pytest.fixture
def schema_manager(mock_config, mock_driver):
    """Create a Neo4jSchemaManager instance with mocked dependencies."""
    with patch("clubhouse.services.neo4j.schema.manager.GraphDatabase") as mock_graph_db:
        mock_graph_db.driver.return_value = mock_driver
        manager = Neo4jSchemaManager(mock_config)
        manager.initialize()
        yield manager


class TestNeo4jSchemaManager:
    """Test suite for the Neo4jSchemaManager service."""

    def test_initialize(self, mock_config):
        """Test that initialize creates a driver and sets up correctly."""
        with patch("clubhouse.services.neo4j.schema.manager.GraphDatabase") as mock_graph_db:
            driver = MagicMock()
            mock_graph_db.driver.return_value = driver
            
            manager = Neo4jSchemaManager(mock_config)
            manager.initialize()
            
            # Verify driver was created with correct parameters
            mock_graph_db.driver.assert_called_once()
            assert manager._driver is not None
            assert manager._initialized is True

    def test_shutdown(self, schema_manager, mock_driver):
        """Test that shutdown closes the driver correctly."""
        schema_manager.shutdown()
        
        # Verify driver was closed
        mock_driver.close.assert_called_once()
        assert schema_manager._initialized is False
        assert schema_manager._driver is None

    def test_is_healthy_success(self, schema_manager, mock_driver):
        """Test health check when database is available."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Execute a simple query that returns True
        tx.run.return_value.value.return_value = True
        
        # Reset mock to clear the call from initialize()
        mock_driver.session.reset_mock()
        
        assert schema_manager.is_healthy() is True
        mock_driver.session.assert_called_once()
        tx.run.assert_called_once()

    def test_is_healthy_failure(self, schema_manager, mock_driver):
        """Test health check when database is unavailable."""
        mock_driver.session.side_effect = ServiceUnavailable("Database unavailable")
        
        assert schema_manager.is_healthy() is False

    def test_create_constraint_success(self, schema_manager, mock_driver):
        """Test creating a constraint successfully."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock result
        result = MagicMock()
        result.data.return_value = [{"name": "constraint_name", "type": "UNIQUENESS"}]
        tx.run.return_value = result
        
        # Call method
        constraint = schema_manager.create_constraint(
            label="Person", 
            properties=["email"], 
            constraint_type="unique",
            constraint_name="person_email_unique"
        )
        
        # Verify constraint was created
        assert constraint["name"] == "constraint_name"
        assert constraint["type"] == "UNIQUENESS"
        tx.run.assert_called_once()
        assert "CREATE CONSTRAINT" in tx.run.call_args[0][0]
        assert "Person" in tx.run.call_args[0][0]
        assert "email" in tx.run.call_args[0][0]

    def test_create_constraint_failure(self, schema_manager, mock_driver):
        """Test creating a constraint that fails."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock exception
        tx.run.side_effect = ClientError("Constraint already exists")
        
        # Call method and expect exception
        with pytest.raises(Neo4jError):
            schema_manager.create_constraint(
                label="Person", 
                properties=["email"]
            )

    def test_drop_constraint_success(self, schema_manager, mock_driver):
        """Test dropping a constraint successfully."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Call method
        result = schema_manager.drop_constraint("person_email_unique")
        
        # Verify constraint was dropped
        assert result is True
        tx.run.assert_called_once()
        assert "DROP CONSTRAINT" in tx.run.call_args[0][0]
        assert "person_email_unique" in tx.run.call_args[0][0]

    def test_drop_constraint_not_found(self, schema_manager, mock_driver):
        """Test dropping a constraint that doesn't exist."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock exception indicating constraint not found
        tx.run.side_effect = ClientError("Constraint does not exist")
        
        # Call method
        result = schema_manager.drop_constraint("nonexistent_constraint")
        
        # Verify result
        assert result is False

    def test_get_constraints(self, schema_manager, mock_driver):
        """Test getting all constraints."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock result
        constraints = [
            {"name": "person_email_unique", "type": "UNIQUENESS", "labelsOrTypes": ["Person"]},
            {"name": "message_id_unique", "type": "UNIQUENESS", "labelsOrTypes": ["Message"]}
        ]
        result = MagicMock()
        result.data.return_value = constraints
        tx.run.return_value = result
        
        # Call method
        result = schema_manager.get_constraints()
        
        # Verify constraints were returned
        assert len(result) == 2
        assert result[0]["name"] == "person_email_unique"
        assert result[1]["name"] == "message_id_unique"
        tx.run.assert_called_once()
        assert "SHOW CONSTRAINTS" in tx.run.call_args[0][0]

    def test_get_constraints_filtered(self, schema_manager, mock_driver):
        """Test getting constraints filtered by label."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock result
        constraints = [
            {"name": "person_email_unique", "type": "UNIQUENESS", "labelsOrTypes": ["Person"]}
        ]
        result = MagicMock()
        result.data.return_value = constraints
        tx.run.return_value = result
        
        # Call method
        result = schema_manager.get_constraints(label="Person")
        
        # Verify constraints were returned
        assert len(result) == 1
        assert result[0]["name"] == "person_email_unique"
        tx.run.assert_called_once()
        assert "SHOW CONSTRAINTS" in tx.run.call_args[0][0]
        assert "Person" in tx.run.call_args[0][0]

    def test_create_index_success(self, schema_manager, mock_driver):
        """Test creating an index successfully."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock result
        result = MagicMock()
        result.data.return_value = [{"name": "person_name_index", "type": "BTREE"}]
        tx.run.return_value = result
        
        # Call method
        index = schema_manager.create_index(
            label="Person", 
            properties=["name"], 
            index_type="btree",
            index_name="person_name_index"
        )
        
        # Verify index was created
        assert index["name"] == "person_name_index"
        assert index["type"] == "BTREE"
        tx.run.assert_called_once()
        assert "CREATE INDEX" in tx.run.call_args[0][0]
        assert "Person" in tx.run.call_args[0][0]
        assert "name" in tx.run.call_args[0][0]

    def test_drop_index_success(self, schema_manager, mock_driver):
        """Test dropping an index successfully."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Call method
        result = schema_manager.drop_index("person_name_index")
        
        # Verify index was dropped
        assert result is True
        tx.run.assert_called_once()
        assert "DROP INDEX" in tx.run.call_args[0][0]
        assert "person_name_index" in tx.run.call_args[0][0]

    def test_get_indexes(self, schema_manager, mock_driver):
        """Test getting all indexes."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock result
        indexes = [
            {"name": "person_name_index", "type": "BTREE", "labelsOrTypes": ["Person"]},
            {"name": "message_content_index", "type": "TEXT", "labelsOrTypes": ["Message"]}
        ]
        result = MagicMock()
        result.data.return_value = indexes
        tx.run.return_value = result
        
        # Call method
        result = schema_manager.get_indexes()
        
        # Verify indexes were returned
        assert len(result) == 2
        assert result[0]["name"] == "person_name_index"
        assert result[1]["name"] == "message_content_index"
        tx.run.assert_called_once()
        assert "SHOW INDEXES" in tx.run.call_args[0][0]

    def test_validate_schema_success(self, schema_manager, mock_driver):
        """Test schema validation when schema is valid."""
        # Mock methods to return expected constraints and indexes
        schema_manager.get_constraints = MagicMock(return_value=[
            {"name": "person_email_unique", "labelsOrTypes": ["Person"]}
        ])
        schema_manager.get_indexes = MagicMock(return_value=[
            {"name": "person_name_index", "labelsOrTypes": ["Person"]}
        ])
        
        # Define expected schema in manager (normally would be in config or code)
        schema_manager._expected_constraints = [
            {"label": "Person", "properties": ["email"], "type": "unique"}
        ]
        schema_manager._expected_indexes = [
            {"label": "Person", "properties": ["name"], "type": "btree"}
        ]
        
        # Call method
        is_valid, errors = schema_manager.validate_schema()
        
        # Verify schema is valid
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_schema_failure(self, schema_manager, mock_driver):
        """Test schema validation when schema is invalid."""
        # Mock methods to return unexpected constraints and indexes
        schema_manager.get_constraints = MagicMock(return_value=[])
        schema_manager.get_indexes = MagicMock(return_value=[])
        
        # Define expected schema in manager
        schema_manager._expected_constraints = [
            {"label": "Person", "properties": ["email"], "type": "unique"}
        ]
        schema_manager._expected_indexes = [
            {"label": "Person", "properties": ["name"], "type": "btree"}
        ]
        
        # Call method
        is_valid, errors = schema_manager.validate_schema()
        
        # Verify schema is invalid
        assert is_valid is False
        assert len(errors) > 0
        assert "Missing constraint" in errors[0]

    def test_get_schema_version(self, schema_manager, mock_driver):
        """Test getting the current schema version."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock result
        result = MagicMock()
        result.value.return_value = "1.0.0"
        tx.run.return_value = result
        
        # Call method
        version = schema_manager.get_schema_version()
        
        # Verify version was returned
        assert version == "1.0.0"
        tx.run.assert_called_once()
        assert "SchemaVersion" in tx.run.call_args[0][0]

    def test_apply_migrations(self, schema_manager, mock_driver):
        """Test applying migrations."""
        # Mock session and transaction
        session = MagicMock()
        tx = MagicMock()
        session.__enter__.return_value = session
        session.begin_transaction.return_value = tx
        tx.__enter__.return_value = tx
        mock_driver.session.return_value = session
        
        # Mock get_schema_version
        schema_manager.get_schema_version = MagicMock(return_value="0.9.0")
        
        # Define migrations in manager
        schema_manager._migrations = [
            {
                "version": "1.0.0",
                "description": "Initial schema",
                "queries": [
                    "CREATE CONSTRAINT person_email_unique ON (p:Person) ASSERT p.email IS UNIQUE"
                ]
            }
        ]
        
        # Call method
        migrations = schema_manager.apply_migrations()
        
        # Verify migrations were applied
        assert len(migrations) == 1
        assert migrations[0]["version"] == "1.0.0"
        assert tx.run.call_count == 2  # One for migration, one for updating version
        assert "CREATE CONSTRAINT" in tx.run.call_args_list[0][0][0]
        assert "SchemaVersion" in tx.run.call_args_list[1][0][0]
