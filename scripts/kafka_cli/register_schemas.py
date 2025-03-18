#!/usr/bin/env python
"""
Schema Registration Utility for Clubhouse Messaging

This script registers Avro schemas for all message types used in the Clubhouse messaging system.
It is designed to be run as a separate administrative tool during deployment or setup,
rather than having schema registration embedded in the client.

Usage:
    python register_schemas.py [--bootstrap-servers SERVERS] [--schema-registry-url URL] [--topic-prefix PREFIX]

Example:
    python register_schemas.py --schema-registry-url http://localhost:8081 --topic-prefix clubhouse
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('schema_registry')

try:
    from confluent_kafka.schema_registry import SchemaRegistryClient
    SCHEMA_REGISTRY_AVAILABLE = True
except ImportError:
    SCHEMA_REGISTRY_AVAILABLE = False
    logger.error("Schema Registry client not available. Install confluent-kafka[avro]")

# Import local modules
from scripts.kafka_cli.formatter import CLIFormatter
from scripts.kafka_cli.schema_utils import SchemaConverter, register_all_schemas

# Import message models from CLI
from scripts.kafka_cli.cli import (
    Command, Response, Event,
    CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
    AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
    AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent,
)

class SchemaRegistrationTool:
    """Utility for registering message schemas with the Schema Registry."""
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        schema_registry_url: str = None,
        topic_prefix: str = "clubhouse"
    ):
        """Initialize the schema registration tool.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers (optional)
            schema_registry_url: Schema Registry URL (required)
            topic_prefix: Topic prefix for all message subjects
        """
        # Set default environment variables if not provided
        self.bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.schema_registry_url = schema_registry_url or os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
        self.topic_prefix = topic_prefix
        
        # Set up formatter for output
        self.formatter = CLIFormatter()
        
    async def register_schemas(self) -> bool:
        """Register all message schemas with the Schema Registry.
        
        Returns:
            bool: True if all schemas were registered successfully, False otherwise
        """
        if not SCHEMA_REGISTRY_AVAILABLE:
            self.formatter.print_error("Schema Registry client not available. Install confluent-kafka[avro]")
            return False
            
        if not self.schema_registry_url:
            self.formatter.print_error("Schema Registry URL not provided")
            return False
            
        self.formatter.print_info(f"Registering schemas with Schema Registry at {self.schema_registry_url}...")
        
        # Create Schema Registry client
        registry_client = SchemaRegistryClient({"url": self.schema_registry_url})
        
        # Register base message classes first
        base_classes = [Command, Response, Event]
        registered_base = 0
        
        self.formatter.print_info("Registering base message classes...")
        try:
            for base_class in base_classes:
                subject = f"{self.topic_prefix}-{base_class.__name__}-value"
                avro_schema = SchemaConverter.pydantic_to_avro(base_class)
                avro_schema_str = json.dumps(avro_schema)
                try:
                    schema_id = SchemaConverter.register_schema(registry_client, subject, avro_schema_str)
                    if schema_id:
                        logger.info(f"Registered base schema for {subject} with ID {schema_id}")
                        registered_base += 1
                        self.formatter.print_success(f"Registered {base_class.__name__} schema with ID {schema_id}")
                except Exception as e:
                    logger.warning(f"Could not register base schema for {subject}: {e}")
                    self.formatter.print_warning(f"Could not register {base_class.__name__} schema: {e}")
        except Exception as e:
            logger.error(f"Error registering base schemas: {e}")
            self.formatter.print_error(f"Error registering base schemas: {e}")
            return False
            
        # Now register specific message classes
        message_classes = [
            CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
            AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
            AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent
        ]
        
        self.formatter.print_info("Registering specific message classes...")
        try:
            num_registered = register_all_schemas(
                registry_client, 
                message_classes, 
                topic_prefix=self.topic_prefix,
                include_null=True
            )
                
            self.formatter.print_success(f"Registered {num_registered} message schemas")
            
            # Return success if at least some schemas were registered
            return num_registered > 0 or registered_base > 0
        except Exception as e:
            logger.error(f"Error registering specific schemas: {e}")
            self.formatter.print_error(f"Failed to register specific schemas: {e}")
            return False
            
async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Schema Registration Tool')
    parser.add_argument('--bootstrap-servers', help='Kafka bootstrap servers')
    parser.add_argument('--schema-registry-url', help='Schema Registry URL')
    parser.add_argument('--topic-prefix', help='Topic prefix', default='clubhouse')
    parser.add_argument('--verbose', help='Enable verbose logging', action='store_true')
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create and run schema registration tool
    tool = SchemaRegistrationTool(
        bootstrap_servers=args.bootstrap_servers,
        schema_registry_url=args.schema_registry_url,
        topic_prefix=args.topic_prefix
    )
    
    success = await tool.register_schemas()
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    asyncio.run(main())
