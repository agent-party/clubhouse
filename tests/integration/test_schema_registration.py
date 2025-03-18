"""
Integration tests for schema registration with Confluent Schema Registry.

This module contains integration tests that verify the schema registration
functionality works correctly with an actual Schema Registry instance.
These tests require a running Schema Registry service.

To run these tests:
    1. Ensure Schema Registry is running (e.g., with Confluent Platform)
    2. Set SCHEMA_REGISTRY_URL environment variable (defaults to http://localhost:8081)
    3. Run pytest with the integration marker: pytest -m integration
"""

import os
import logging
import unittest
from typing import Dict, Any, List

import pytest
from confluent_kafka.schema_registry import SchemaRegistryClient
from pydantic import BaseModel

from clubhouse.messaging.schema_registrator import SchemaRegistrator
from clubhouse.services.schema_registry import ConfluentSchemaRegistry
from scripts.kafka_cli.message_schemas import (
    BaseMessage,
    CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
    AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
    AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent
)

logger = logging.getLogger(__name__)

# Skip these tests if integration testing is not enabled
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def schema_registry_url() -> str:
    """Get the Schema Registry URL from environment, with default."""
    # Always use the Docker-based Schema Registry for testing
    # This ensures we're using real infrastructure rather than mocks
    return "http://localhost:8081"


@pytest.fixture(scope="module")
def schema_registry_client(schema_registry_url: str) -> SchemaRegistryClient:
    """Create a Schema Registry client for testing."""
    logger.info(f"Attempting to connect to Schema Registry at {schema_registry_url}")
    try:
        client = SchemaRegistryClient({"url": schema_registry_url})
        # Test the connection by getting subjects
        subjects = client.get_subjects()
        logger.info(f"Successfully connected to Schema Registry. Available subjects: {subjects}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Schema Registry at {schema_registry_url}: {str(e)}")
        # Instead of skipping, we'll fail the test to enforce the use of real infrastructure
        pytest.fail(f"Cannot connect to Schema Registry at {schema_registry_url}. Ensure the Docker container is running: {str(e)}")


@pytest.fixture(scope="module")
def confluent_registry(schema_registry_url: str) -> ConfluentSchemaRegistry:
    """Create a ConfluentSchemaRegistry instance for testing."""
    logger.info(f"Initializing ConfluentSchemaRegistry with URL: {schema_registry_url}")
    try:
        # Create registry and verify connection
        registry = ConfluentSchemaRegistry(schema_registry_url)
        if not registry.check_connection():
            logger.error(f"Failed to connect to Schema Registry at {schema_registry_url}")
            # Instead of skipping, we'll fail the test to enforce the use of real infrastructure
            pytest.fail(f"Cannot connect to Schema Registry at {schema_registry_url}. Ensure the Docker container is running.")
        
        logger.info(f"Successfully connected to ConfluentSchemaRegistry at {schema_registry_url}")
        return registry
    except Exception as e:
        logger.error(f"Failed to initialize ConfluentSchemaRegistry: {str(e)}")
        # Instead of skipping, we'll fail the test to enforce the use of real infrastructure
        pytest.fail(f"Error setting up ConfluentSchemaRegistry: {str(e)}")


@pytest.fixture(scope="module")
def test_subject_prefix() -> str:
    """Get a unique subject prefix for test isolation."""
    return f"test-schema-reg-{os.getpid()}"


@pytest.fixture(scope="function")
def cleanup_subjects(schema_registry_client: SchemaRegistryClient, test_subject_prefix: str):
    """Cleanup registered schemas after each test."""
    yield
    
    # Delete all test subjects created during the test
    try:
        subjects = schema_registry_client.get_subjects()
        for subject in subjects:
            if subject.startswith(test_subject_prefix):
                try:
                    versions = schema_registry_client.get_versions(subject)
                    for version in versions:
                        schema_registry_client.delete_version(subject, version)
                    # Delete the subject itself
                    schema_registry_client.delete_subject(subject)
                    logger.info(f"Deleted test subject: {subject}")
                except Exception as e:
                    logger.warning(f"Error cleaning up subject {subject}: {e}")
    except Exception as e:
        logger.warning(f"Error listing subjects for cleanup: {e}")


class TestSchemaRegistrationIntegration:
    """Integration tests for schema registration."""

    def test_register_schema(self, confluent_registry: ConfluentSchemaRegistry, 
                            test_subject_prefix: str, cleanup_subjects):
        """Test registering a schema with a real Schema Registry."""
        # Arrange
        registrator = SchemaRegistrator(confluent_registry, test_subject_prefix)
        
        # Act
        result = registrator._register_schema(CreateAgentCommand)
        
        # Assert
        assert result is True
        
        # Verify the schema was actually registered
        subject = f"{test_subject_prefix}-CreateAgentCommand-value"
        schema_id, schema = confluent_registry.get_latest_schema(subject)
        assert schema_id > 0
        assert "type" in schema
        assert schema["type"] == "record"
        assert schema["name"] == "CreateAgentCommand"

    def test_register_all_schemas(self, confluent_registry: ConfluentSchemaRegistry, 
                               test_subject_prefix: str, cleanup_subjects):
        """Test registering all schemas with a real Schema Registry."""
        # Arrange
        registrator = SchemaRegistrator(confluent_registry, test_subject_prefix)
        
        # Act
        num_registered = registrator.register_all_schemas()
        
        # Assert
        assert num_registered > 0
        
        # Verify some key schemas were registered
        for model_name in ["CreateAgentCommand", "DeleteAgentCommand", "ProcessMessageCommand", 
                           "AgentCreatedResponse", "AgentDeletedResponse", "MessageProcessedResponse", 
                           "AgentThinkingEvent", "AgentErrorEvent", "AgentStateChangedEvent"]:
            subject = f"{test_subject_prefix}-{model_name}-value"
            schema_id, schema = confluent_registry.get_latest_schema(subject)
            assert schema_id > 0
            assert schema["name"] == model_name

    def test_idempotent_registration(self, confluent_registry: ConfluentSchemaRegistry, 
                                  test_subject_prefix: str, cleanup_subjects):
        """Test that registering the same schema twice returns the same ID."""
        # Arrange
        registrator = SchemaRegistrator(confluent_registry, test_subject_prefix)
        
        # Act
        registrator._register_schema(CreateAgentCommand)
        subject = f"{test_subject_prefix}-CreateAgentCommand-value"
        first_id, _ = confluent_registry.get_latest_schema(subject)
        
        # Register again
        registrator._register_schema(CreateAgentCommand)
        second_id, _ = confluent_registry.get_latest_schema(subject)
        
        # Assert
        assert first_id == second_id
