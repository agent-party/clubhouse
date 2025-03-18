#!/usr/bin/env python3
"""
Simple test script to debug schema conversion and registration.
"""

import json
import logging
import os
import requests
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from clubhouse.services.schema_registry import ConfluentSchemaRegistry
from scripts.kafka_cli.schema_utils import SchemaConverter

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a simple test model
class TestModel(BaseModel):
    """A simple test model for schema conversion."""
    id: str
    name: str
    age: int = 30
    metadata: Dict[str, str] = Field(default_factory=dict)  # Simplified to use only string values

def register_schema_manually(url: str, subject: str, schema: Dict[str, Any]) -> int:
    """Register a schema directly using the REST API."""
    schema_json = json.dumps(schema)
    
    # The schema registry REST API expects the schema as a JSON string inside a JSON object
    payload = json.dumps({"schema": schema_json})
    
    headers = {
        "Content-Type": "application/vnd.schemaregistry.v1+json",
    }
    
    response = requests.post(
        f"{url}/subjects/{subject}/versions",
        data=payload,
        headers=headers
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to register schema: {response.text}")
        response.raise_for_status()
    
    result = response.json()
    return result["id"]

def main():
    """Main function to test schema conversion and registration."""
    logger.info("Starting schema conversion test")
    
    # Get schema registry URL
    schema_registry_url = os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    logger.info(f"Using schema registry URL: {schema_registry_url}")
    
    try:
        # Convert model to Avro schema
        avro_schema = SchemaConverter.pydantic_to_avro(TestModel)
        logger.info(f"Converted schema: {json.dumps(avro_schema, indent=2)}")
        
        # Validate the schema
        is_valid = SchemaConverter.validate_avro_schema(avro_schema)
        logger.info(f"Schema validation result: {is_valid}")
        
        # Create registry client for testing connection
        registry = ConfluentSchemaRegistry(schema_registry_url)
        
        # Test connection
        subjects = registry.get_subjects()
        logger.info(f"Existing subjects: {subjects}")
        
        # Register the schema manually using REST API
        subject = "test-simple-model-value"
        schema_id = register_schema_manually(schema_registry_url, subject, avro_schema)
        logger.info(f"Registered schema with ID: {schema_id}")
        
        # Retrieve the schema to verify
        _, retrieved_schema = registry.get_latest_schema(subject)
        logger.info(f"Retrieved schema: {json.dumps(retrieved_schema, indent=2)}")
        
        logger.info("Schema conversion test completed successfully")
        return 0
    except Exception as e:
        logger.exception(f"Error during schema conversion test: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
