"""
Schema Registry service implementation.

This module provides an implementation of the Schema Registry protocol
for managing Avro schemas with Confluent Schema Registry.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from confluent_kafka.schema_registry import SchemaRegistryClient, SchemaRegistryError
from pydantic import BaseModel

from clubhouse.services.kafka_protocol import SchemaRegistryProtocol
from typing import cast, List, Dict, Any, Type
from scripts.kafka_cli.schema_utils import SchemaConverter

logger = logging.getLogger(__name__)


class ConfluentSchemaRegistry(SchemaRegistryProtocol):
    """
    Implementation of SchemaRegistryProtocol using Confluent Schema Registry.

    This class provides an interface to the Confluent Schema Registry for
    registering and retrieving Avro schemas.
    """

    def __init__(self, url: str) -> None:
        """
        Initialize the schema registry client.

        Args:
            url: URL of the schema registry
        """
        self._registry_url = url
        self._client = None  # Lazy initialization
        self._schema_converter = SchemaConverter()

    def _get_client(self) -> SchemaRegistryClient:
        """
        Get the schema registry client, initializing if needed.
        
        Returns:
            Schema registry client
        """
        if self._client is None:
            # The Schema Registry client only supports a limited set of configuration parameters
            # See: https://docs.confluent.io/platform/current/clients/confluent-kafka-python/html/index.html#schemaregistryclient
            self._client = SchemaRegistryClient({
                "url": self._registry_url
            })
        return self._client

    def check_connection(self) -> bool:
        """
        Check if the schema registry is accessible.
        
        Returns:
            True if the schema registry is accessible, False otherwise
        """
        try:
            # First try using direct HTTP request which is more reliable
            # for checking basic connectivity
            import httpx
            
            # Add timeout to avoid hanging
            url = f"{self._registry_url}/subjects"
            response = httpx.get(url, timeout=5.0)
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to Schema Registry at {self._registry_url}")
                return True
                
            logger.warning(f"Schema Registry HTTP check failed: {response.status_code} {response.text}")
            # Don't immediately fail - try the client API as well
            
        except Exception as e:
            # Log but continue to try the client API
            logger.warning(f"Schema Registry HTTP connection check failed: {str(e)}")
        
        # Try using the client API as a fallback
        try:
            client = self._get_client()
            # This is a lightweight call that should succeed if the registry is up
            subjects = client.get_subjects()
            logger.info(f"Successfully connected to Schema Registry at {self._registry_url} via client API. Subjects: {subjects}")
            return True
        except Exception as e:
            logger.warning(f"Schema Registry client API connection check failed: {str(e)}")
            return False  # Return False instead of raising to allow tests to be skipped gracefully

    def register(self, subject: str, schema_def: Dict[str, Any]) -> int:
        """
        Register a new schema with the schema registry.
        
        Args:
            subject: Subject name (typically <topic>-value or <topic>-key)
            schema_def: Schema definition as a dictionary
            
        Returns:
            Schema ID
            
        Raises:
            SchemaRegistryError: If there is an error registering the schema
        """
        try:
            # Directly use HTTP requests to interact with Schema Registry API
            # This is more reliable than client libraries with version compatibility issues
            import requests
            
            # Convert dictionary to JSON string if it's not already a string
            if isinstance(schema_def, dict):
                schema_json = json.dumps(schema_def)
            else:
                schema_json = schema_def
                
            # Prepare request data - schema needs to be a string in a JSON object
            payload = {"schema": schema_json}
            
            # Register schema using Schema Registry REST API
            url = f"{self._registry_url}/subjects/{subject}/versions"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=payload, headers=headers, timeout=10.0)
            
            if response.status_code == 200:
                schema_id = response.json().get("id")
                logger.info(f"Registered schema for subject {subject} with ID {schema_id}")
                return schema_id
            else:
                error_msg = f"Failed to register schema: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Failed to register schema for subject {subject}: {str(e)}")
            raise

    def get_schema(self, subject: str, schema_id: int) -> Dict[str, Any]:
        """
        Get a schema by subject and ID.

        Args:
            subject: Subject name
            schema_id: Schema ID

        Returns:
            Schema definition

        Raises:
            SchemaRegistryError: If there is an error retrieving the schema
        """
        try:
            client = self._get_client()
            schema = client.get_schema(schema_id)
            return json.loads(schema.schema_str)
        except Exception as e:
            logger.error(f"Failed to get schema for subject {subject}, ID {schema_id}: {e}")
            raise

    def get_latest_schema(self, subject: str) -> Tuple[int, Dict[str, Any]]:
        """
        Get the latest schema for a subject.

        Args:
            subject: Subject name

        Returns:
            Tuple of (schema_id, schema_definition)

        Raises:
            SchemaRegistryError: If there is an error retrieving the schema
        """
        try:
            client = self._get_client()
            schema_metadata = client.get_latest_version(subject)
            schema_id = schema_metadata.schema_id
            schema = client.get_schema(schema_id)
            return schema_id, json.loads(schema.schema_str)
        except Exception as e:
            logger.error(f"Failed to get latest schema for subject {subject}: {e}")
            raise

    def get_schema_from_model(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Generate Avro schema from a Pydantic model class.

        Args:
            model_class: Pydantic model class to convert to Avro schema

        Returns:
            Avro schema as a dictionary
        """
        try:
            # Use the SchemaConverter to convert Pydantic model to Avro schema
            schema_dict = self._schema_converter.pydantic_to_avro(model_class)
            logger.info(f"Generated Avro schema from {model_class.__name__}")
            return schema_dict
        except Exception as e:
            logger.error(f"Failed to generate schema from model {model_class.__name__}: {e}")
            raise

    def check_compatibility(self, subject: str, schema_def: Dict[str, Any]) -> bool:
        """
        Check if a schema is compatible with the latest version.

        Args:
            subject: Subject name
            schema_def: Schema definition to check

        Returns:
            True if the schema is compatible, False otherwise
        """
        try:
            # Ensure we have a schema definition dictionary
            if not isinstance(schema_def, dict):
                logger.error(f"Schema definition must be a dictionary, got {type(schema_def)}")
                return False
                
            # Convert dictionary to JSON string
            schema_json = json.dumps(schema_def)
            
            # Get the client
            client = self._get_client()
            
            # Check compatibility
            result = client.test_compatibility(subject, schema_json)
            logger.info(f"Schema compatibility check for {subject}: {result}")
            return result
        except Exception as e:
            logger.warning(f"Schema compatibility check failed for subject {subject}: {e}")
            # If the subject doesn't exist yet, consider it compatible
            if "Subject not found" in str(e):
                logger.info(f"Subject {subject} not found, considering schema compatible for first registration")
                return True
            return False

    def delete_subject(self, subject: str) -> List[int]:
        """
        Delete a subject from the registry.

        Args:
            subject: Subject name

        Returns:
            List of deleted schema IDs

        Raises:
            SchemaRegistryError: If there is an error deleting the subject
        """
        try:
            client = self._get_client()
            deleted_ids = client.delete_subject(subject)
            logger.info(f"Deleted subject {subject}, removed schema IDs: {deleted_ids}")
            return deleted_ids
        except Exception as e:
            logger.error(f"Failed to delete subject {subject}: {e}")
            raise