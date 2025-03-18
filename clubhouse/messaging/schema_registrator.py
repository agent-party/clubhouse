"""
Schema registration for Clubhouse message schemas.

This module provides functionality to register all message schemas with
the Schema Registry on application startup. It handles the conversion
of Pydantic models to Avro schemas and registers them with a Schema Registry
service following SOLID principles with robust error handling.

The SchemaRegistrator follows the Single Responsibility Principle by focusing
solely on schema registration and uses dependency injection to accept any
implementation of the SchemaRegistryProtocol.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type, Tuple, Union, cast

from pydantic import BaseModel

from clubhouse.services.kafka_protocol import SchemaRegistryProtocol
from scripts.kafka_cli.schema_utils import SchemaConverter

# Import all message models that need registration
from scripts.kafka_cli.message_schemas import (
    BaseMessage,
    CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
    AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
    AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent,
)

logger = logging.getLogger(__name__)


class SchemaRegistrationError(Exception):
    """Exception raised for errors during schema registration."""
    
    def __init__(self, message: str, subject: str, details: Optional[Exception] = None):
        """
        Initialize a schema registration error.
        
        Args:
            message: Error message
            subject: Subject that failed registration
            details: Optional underlying exception
        """
        self.subject = subject
        self.details = details
        super().__init__(f"{message} for subject {subject}: {details}")


class SchemaRegistrator:
    """Registers all message schemas with the Schema Registry on startup.
    
    This class follows the Single Responsibility Principle by focusing solely
    on registering schemas with the Schema Registry. It uses dependency injection
    to accept any implementation of the SchemaRegistryProtocol.
    
    The class provides both individual schema registration and batch registration
    of all message schemas. It handles errors gracefully, continuing with registration
    even if some schemas fail, and provides detailed logging for debugging.
    """

    def __init__(
        self, 
        schema_registry: SchemaRegistryProtocol,
        topic_prefix: str = "clubhouse",
        include_null: bool = True
    ) -> None:
        """
        Initialize the schema registrator.

        Args:
            schema_registry: Schema Registry client implementing SchemaRegistryProtocol
            topic_prefix: Topic prefix for all subjects
            include_null: Whether to include null type in schema fields for optional fields
        """
        self._schema_registry = schema_registry
        self._topic_prefix = topic_prefix
        self._include_null = include_null

    def register_all_schemas(self) -> int:
        """
        Register all message schemas with the Schema Registry.
        
        This method attempts to register all base and concrete message models.
        It continues even if some registrations fail, logging errors for debugging.

        Returns:
            Number of schemas successfully registered
        """
        # Define all message classes to register
        base_models = [BaseMessage]
        concrete_models = [
            CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
            AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
            AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent,
        ]

        total_registered = 0
        registration_errors = []
        
        # Register base models first to ensure proper schema evolution
        logger.info(f"Registering {len(base_models)} base message models...")
        for model_class in base_models:
            try:
                if self._register_schema(model_class):
                    total_registered += 1
            except SchemaRegistrationError as e:
                logger.error(f"Failed to register base model {model_class.__name__}: {e}")
                registration_errors.append(e)
                
        # Then register concrete models
        logger.info(f"Registering {len(concrete_models)} concrete message models...")
        for model_class in concrete_models:
            try:
                if self._register_schema(model_class):
                    total_registered += 1
            except SchemaRegistrationError as e:
                logger.error(f"Failed to register concrete model {model_class.__name__}: {e}")
                registration_errors.append(e)

        # Summarize results
        if registration_errors:
            logger.warning(
                f"Completed schema registration with {len(registration_errors)} errors. "
                f"Successfully registered {total_registered} schemas."
            )
        else:
            logger.info(f"Successfully registered all {total_registered} schemas.")
            
        return total_registered

    def _register_schema(self, model_class: Type[BaseModel]) -> bool:
        """
        Register a single schema with the Schema Registry.
        
        This method handles the conversion of a Pydantic model to an Avro schema
        and registers it with the Schema Registry. It checks if the schema already
        exists before registering to avoid unnecessary operations.

        Args:
            model_class: Pydantic model class to register

        Returns:
            True if registration was successful, False otherwise
            
        Raises:
            SchemaRegistrationError: If there is an error during schema registration
        """
        if not issubclass(model_class, BaseModel):
            logger.warning(f"{model_class.__name__} is not a Pydantic model, skipping")
            return False

        # Create subject name using topic prefix and class name
        subject = f"{self._topic_prefix}-{model_class.__name__}-value"
        
        try:
            # Convert Pydantic model to Avro schema
            avro_schema = SchemaConverter.pydantic_to_avro(
                model_class, include_null=self._include_null
            )
            
            # Check if schema already exists
            try:
                schema_id = self._schema_registry.check_schema_exists(subject, avro_schema)
                
                if schema_id is not None:
                    logger.info(f"Schema for {subject} already exists with ID {schema_id}")
                    return True
            except Exception as e:
                # If check_schema_exists fails, log warning and continue to registration
                logger.warning(f"Error checking if schema exists for {subject}: {e}")
                
            # Register schema
            try:
                schema_id = self._schema_registry.register(subject, avro_schema)
                logger.info(f"Registered schema for {subject} with ID {schema_id}")
                return True
            except Exception as e:
                raise SchemaRegistrationError(
                    "Failed to register schema", subject, e
                )
        except Exception as e:
            if isinstance(e, SchemaRegistrationError):
                raise
            else:
                raise SchemaRegistrationError(
                    "Error processing schema", subject, e
                )
