"""
Schema Registry service implementation.

This module provides an implementation of the Schema Registry protocol
for managing Avro schemas with Confluent Schema Registry.
"""

import json
import logging
from typing import Dict, Any, Optional, List, cast, Tuple

import requests
from schema_registry.client import SchemaRegistryClient, schema
from schema_registry.client.errors import ClientError

from project_name.services.kafka_protocol import SchemaRegistryProtocol

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
        self._client = SchemaRegistryClient(url=url)
    
    def register(self, subject: str, schema_def: Dict[str, Any]) -> int:
        """
        Register a new schema with the schema registry.
        
        Args:
            subject: Subject name (typically <topic>-value or <topic>-key)
            schema_def: Schema definition as a dictionary
            
        Returns:
            Schema ID
            
        Raises:
            ClientError: If there is an error registering the schema
        """
        try:
            avro_schema = schema.AvroSchema(json.dumps(schema_def))
            return self._client.register(subject, avro_schema)
        except ClientError as e:
            logger.error(f"Failed to register schema for subject {subject}: {e}")
            raise
    
    def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """
        Get a schema by ID.
        
        Args:
            schema_id: Schema ID
            
        Returns:
            Schema definition
            
        Raises:
            ClientError: If there is an error retrieving the schema
        """
        try:
            registered_schema = self._client.get_by_id(schema_id)
            return cast(Dict[str, Any], json.loads(registered_schema.schema_str))
        except ClientError as e:
            logger.error(f"Failed to get schema for ID {schema_id}: {e}")
            raise
    
    def get_latest_schema(self, subject: str) -> tuple[int, Dict[str, Any]]:
        """
        Get the latest schema for a subject.
        
        Args:
            subject: Subject name
            
        Returns:
            Tuple of (schema_id, schema_definition)
            
        Raises:
            ClientError: If there is an error retrieving the schema
        """
        try:
            schema_metadata = self._client.get_latest_version(subject)
            schema_id = schema_metadata.schema_id
            schema_str = schema_metadata.schema.schema_str
            return schema_id, cast(Dict[str, Any], json.loads(schema_str))
        except ClientError as e:
            logger.error(f"Failed to get latest schema for subject {subject}: {e}")
            raise
    
    def get_subjects(self) -> List[str]:
        """
        Get all subjects.
        
        Returns:
            List of subject names
            
        Raises:
            ClientError: If there is an error retrieving the subjects
        """
        try:
            return self._client.get_subjects()
        except ClientError as e:
            logger.error(f"Failed to get subjects: {e}")
            raise
    
    def check_schema_exists(self, subject: str, schema_def: Dict[str, Any]) -> Optional[int]:
        """
        Check if a schema already exists for a subject.
        
        Args:
            subject: Subject name
            schema_def: Schema definition
            
        Returns:
            Schema ID if the schema exists, None otherwise
            
        Raises:
            ClientError: If there is an error checking the schema
        """
        try:
            avro_schema = schema.AvroSchema(json.dumps(schema_def))
            return self._client.check_version(subject, avro_schema)
        except ClientError as e:
            if e.http_code == 404:
                return None
            logger.error(f"Failed to check schema for subject {subject}: {e}")
            raise
