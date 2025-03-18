"""
Tests for the SchemaRegistrator class.

This module contains unit tests for the schema registration functionality
in the Clubhouse application.
"""

import json
import logging
import unittest
from typing import Any, Dict, List, Optional, Type, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from clubhouse.messaging.schema_registrator import SchemaRegistrator
from clubhouse.services.kafka_protocol import SchemaRegistryProtocol
from scripts.kafka_cli.message_schemas import (
    BaseMessage,
    CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
    AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
    AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent
)
from scripts.kafka_cli.schema_utils import SchemaConverter


class MockSchemaRegistry(SchemaRegistryProtocol):
    """Mock implementation of the SchemaRegistryProtocol for testing."""

    def __init__(self):
        self.registered_schemas = {}
        self.schema_id_counter = 1

    def register(self, subject: str, schema: Dict[str, Any]) -> int:
        """Register a schema and return a mock ID."""
        schema_id = self.schema_id_counter
        self.schema_id_counter += 1
        self.registered_schemas[subject] = {"id": schema_id, "schema": schema}
        return schema_id

    def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """Get a schema by ID."""
        for subject, data in self.registered_schemas.items():
            if data["id"] == schema_id:
                return data["schema"]
        raise Exception(f"Schema with ID {schema_id} not found")

    def get_latest_schema(self, subject: str) -> Dict[str, Any]:
        """Get the latest version of a schema for a subject."""
        if subject in self.registered_schemas:
            return self.registered_schemas[subject]
        raise Exception(f"No schema found for subject {subject}")

    def check_schema_exists(self, subject: str, schema: Dict[str, Any]) -> Optional[int]:
        """Check if a schema exists for a subject."""
        if subject in self.registered_schemas:
            if json.dumps(self.registered_schemas[subject]["schema"]) == json.dumps(schema):
                return self.registered_schemas[subject]["id"]
        return None


class TestSchemaRegistrator:
    """Test cases for the SchemaRegistrator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_registry = MockSchemaRegistry()
        self.registrator = SchemaRegistrator(self.mock_registry, "test-topic")

    def test_register_schema_success(self):
        """Test successful registration of a schema."""
        # Arrange
        model_class = CreateAgentCommand

        # Act
        result = self.registrator._register_schema(model_class)

        # Assert
        assert result is True
        expected_subject = "test-topic-CreateAgentCommand-value"
        assert expected_subject in self.mock_registry.registered_schemas
        assert isinstance(self.mock_registry.registered_schemas[expected_subject]["schema"], dict)

    def test_register_schema_not_pydantic_model(self):
        """Test registration with a non-Pydantic model class."""
        # Arrange
        class NotPydanticModel:
            pass

        # Act
        result = self.registrator._register_schema(NotPydanticModel)  # type: ignore

        # Assert
        assert result is False
        assert len(self.mock_registry.registered_schemas) == 0

    def test_register_schema_exception(self):
        """Test handling of exceptions during schema registration."""
        # Arrange
        model_class = CreateAgentCommand
        self.mock_registry.register = Mock(side_effect=Exception("Test error"))
        
        # Act & Assert
        with pytest.raises(Exception):
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                
                try:
                    self.registrator._register_schema(model_class)
                except Exception:
                    # Verify that an error was logged
                    mock_logger.error.assert_called()
                    raise

    def test_register_all_schemas(self):
        """Test registration of all schemas."""
        # Act
        num_registered = self.registrator.register_all_schemas()

        # Assert
        # We expect at least the base models and some concrete models to be registered
        assert num_registered > 0
        assert len(self.mock_registry.registered_schemas) > 0
        
        # Check that base models are registered
        assert "test-topic-BaseMessage-value" in self.mock_registry.registered_schemas
        
        # Check that concrete models are registered
        assert "test-topic-CreateAgentCommand-value" in self.mock_registry.registered_schemas

    def test_register_schema_already_exists(self):
        """Test registration when schema already exists."""
        # Arrange
        model_class = CreateAgentCommand
        subject = "test-topic-CreateAgentCommand-value"
        avro_schema = SchemaConverter.pydantic_to_avro(model_class)
        self.mock_registry.registered_schemas[subject] = {"id": 99, "schema": avro_schema}

        # Act
        result = self.registrator._register_schema(model_class)

        # Assert
        assert result is True
        # Verify that check_schema_exists was called and returned existing schema ID
        assert self.mock_registry.registered_schemas[subject]["id"] == 99


@patch("scripts.kafka_cli.schema_utils.SchemaConverter.pydantic_to_avro")
class TestSchemaRegistratorWithMocks:
    """Test cases for SchemaRegistrator using patched mocks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_registry = MagicMock(spec=SchemaRegistryProtocol)
        self.registrator = SchemaRegistrator(self.mock_registry, "test-topic")

    def test_register_schema_calls_converter(self, mock_pydantic_to_avro):
        """Test that the schema converter is called correctly."""
        # Arrange
        mock_schema = {"type": "record", "name": "CreateAgentCommand", "fields": []}
        mock_pydantic_to_avro.return_value = mock_schema
        
        # Mock the schema registry
        self.mock_registry = MagicMock()
        self.mock_registry.check_schema_exists.return_value = None
        self.mock_registry.register.return_value = 123
        
        # Create registrator with mocked registry
        self.registrator = SchemaRegistrator(self.mock_registry, "test-topic")

        # Act
        result = self.registrator._register_schema(CreateAgentCommand)

        # Assert
        assert result is True
        mock_pydantic_to_avro.assert_called_once_with(CreateAgentCommand, include_null=True)
        self.mock_registry.register.assert_called_once_with("test-topic-CreateAgentCommand-value", mock_schema)

    def test_register_all_schemas_handles_error(self, mock_pydantic_to_avro):
        """Test that register_all_schemas continues even if one schema fails."""
        # Arrange
        self.mock_registry = MagicMock()
        self.registrator = SchemaRegistrator(self.mock_registry, "test-topic")
        self.registrator.logger = MagicMock()
        
        # Set up successful registration for most models but failure for one
        def mock_register_schema(model_class):
            if model_class == CreateAgentCommand:
                return False
            return True
            
        # Patch the internal _register_schema method
        with patch.object(self.registrator, '_register_schema', side_effect=mock_register_schema) as mock_register:
            # Act
            result = self.registrator.register_all_schemas()
            
            # Assert
            # Should return count of successful registrations (at least BaseMessage and some others)
            assert result > 0
            # Should be called for all models
            assert mock_register.call_count > 1


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
