"""
Schema utilities for Kafka CLI.

This module provides utilities for Pydantic to Avro schema conversion and
integration with the Schema Registry.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin, get_type_hints
import uuid

try:
    import fastavro
    FASTAVRO_AVAILABLE = True
except ImportError:
    FASTAVRO_AVAILABLE = False
    
try:
    from pydantic import BaseModel
    
    # Check Pydantic version
    import pydantic
    PYDANTIC_V2 = hasattr(pydantic, "__version__") and pydantic.__version__.startswith("2")
    
    if PYDANTIC_V2:
        from pydantic_core import PydanticCustomError
    else:
        from pydantic.error_wrappers import ValidationError
        
except ImportError:
    BaseModel = None
    PYDANTIC_V2 = False

logger = logging.getLogger(__name__)

# Pydantic to Avro type mapping
PYDANTIC_TO_AVRO_TYPES = {
    str: "string",
    int: "int",
    float: "float",
    bool: "boolean",
    bytes: "bytes",
    dict: "record",
    list: "array",
    datetime: {"type": "string", "logicalType": "iso-datetime"},
    None: "null",
}

class SchemaConverter:
    """
    Converter for Pydantic models to Avro schemas.
    
    This class provides utilities to convert Pydantic models to Avro schemas
    compatible with the Schema Registry.
    """
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitize a name to be valid in Avro schema (must match [A-Za-z_][A-Za-z0-9_]*)
        
        Args:
            name: The name to sanitize
            
        Returns:
            Sanitized name compatible with Avro
        """
        # Replace special characters with underscores
        first_char = re.sub(r'[^A-Za-z_]', '_', name[0]) if name else '_'
        
        # Replace remaining invalid characters with underscores
        rest = re.sub(r'[^A-Za-z0-9_]', '_', name[1:]) if len(name) > 1 else ''
        
        return first_char + rest
    
    @classmethod
    def pydantic_to_avro(
        cls, 
        model_class: Type[BaseModel], 
        namespace: str = "com.clubhouse", 
        include_null: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a Pydantic model to an Avro schema.
        
        Args:
            model_class: Pydantic model class
            namespace: Namespace for the schema
            include_null: Whether to include null type for optional fields
            
        Returns:
            Avro schema as a dict
        """
        try:
            # Get model name
            model_name = model_class.__name__
            
            # Clean name to be Avro-compatible
            avro_name = cls._sanitize_name(model_name)
            
            # Initialize schema with name and type
            schema = {
                "type": "record",
                "name": avro_name,
                "namespace": namespace,
                "fields": []
            }
            
            # Collect fields from the model and all its parent classes
            all_fields = {}
            
            # Start with this model's fields - use model_fields for v2 compatibility
            if PYDANTIC_V2 and hasattr(model_class, "model_fields"):
                all_fields.update(model_class.model_fields)
            elif hasattr(model_class, "__fields__"):  # Pydantic v1
                all_fields.update(model_class.__fields__)
            
            # Collect fields from parent classes that are BaseModel
            for base in model_class.__mro__[1:]:  # Skip the class itself
                if issubclass(base, BaseModel) and base != BaseModel:
                    if PYDANTIC_V2 and hasattr(base, "model_fields"):
                        # Don't override fields already defined in child classes
                        for field_name, field in base.model_fields.items():
                            if field_name not in all_fields:
                                all_fields[field_name] = field
                    elif hasattr(base, "__fields__"):  # Pydantic v1
                        # Don't override fields already defined in child classes
                        for field_name, field in base.__fields__.items():
                            if field_name not in all_fields:
                                all_fields[field_name] = field
            
            # Convert fields to Avro
            for field_name, field in all_fields.items():
                try:
                    # Get field type and default value
                    if hasattr(field, "annotation"):  # Pydantic v2 approach
                        field_type = field.annotation
                        # In Pydantic v2, we need to check if default is set differently
                        if hasattr(field, "is_required") and not field.is_required:
                            # Non-required fields have defaults
                            default_value = field.default
                        else:
                            # Required fields don't have defaults
                            default_value = ...
                    else:  # Pydantic v1 approach
                        field_type = field.type_
                        default_value = field.default if field.default is not ... else ...
                    
                    # Convert field type to Avro
                    avro_field = cls._convert_field_type(
                        field_type, 
                        field_name, 
                        default_value, 
                        include_null
                    )
                    
                    # Add field to schema
                    schema["fields"].append(avro_field)
                except Exception as e:
                    logger.error(f"Error converting field {field_name} to Avro: {str(e)}")
                    # Skip this field rather than failing completely
                    continue
            
            return schema
        except Exception as e:
            logger.error(f"Error converting model {model_class.__name__} to Avro: {str(e)}")
            raise
            
    @classmethod
    def _get_avro_type(cls, field_type, field_info, include_null: bool = True):
        """
        Get the Avro type for a field type.
        
        Args:
            field_type: Type of the field
            field_info: ModelField object for the field
            include_null: Whether to include None as a valid type for Optional fields
            
        Returns:
            Avro type as a string or dictionary
        """
        # Handle the case where field_type is a string (happens with forward refs)
        if isinstance(field_type, str):
            return "string"
            
        origin = get_origin(field_type)
        args = get_args(field_type)
        
        # Check for Optional fields
        if origin is Union and type(None) in args:
            # Find the non-None type in the Union
            inner_types = [arg for arg in args if arg is not type(None)]
            if not inner_types:
                return "null"
                
            # Use the first non-None type
            inner_type = inner_types[0]
            avro_type = cls._get_avro_type(inner_type, field_info, False)
            return ["null", avro_type] if include_null else avro_type
        
        # Handle regular Union types (not Optional)
        if origin is Union:
            # For Union types in Avro, we need to specify all possible types
            # Using string as a fallback for compatibility
            return "string"
        
        # Check for Dict fields - always use string as values to avoid Union compatibility issues
        if origin is dict:
            return {
                "type": "map",
                "values": "string"
            }
        
        # Check for List fields
        if origin is list:
            inner_type = args[0] if args else Any
            return {
                "type": "array",
                "items": cls._get_avro_type(inner_type, field_info, include_null)
            }
        
        # Handle basic types
        if field_type is str:
            return "string"
        elif field_type is int:
            return "int"
        elif field_type is float:
            return "double"
        elif field_type is bool:
            return "boolean"
        elif field_type is bytes:
            return "bytes"
        elif field_type is uuid.UUID:
            return "string"
        elif field_type is datetime or field_type is date:
            return "string"  # Store dates as ISO format strings
        
        # Handle Pydantic models (nested records)
        if BaseModel and isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return cls.pydantic_to_avro(field_type)
        
        # Default to string for other types
        return "string"

    @classmethod
    def _convert_field_type(
            cls, 
            field_type: Any, 
            field_name: str, 
            default_value: Any = ...,
            include_null: bool = True
        ) -> Dict[str, Any]:
        """
        Convert a Pydantic field type to an Avro field type.
        
        Args:
            field_type: Pydantic field type
            field_name: Field name
            default_value: Default value of the field
            include_null: Whether to include null as a possible type for Optional fields
            
        Returns:
            Avro field type definition
        """
        # Create basic field structure
        avro_field = {
            "name": field_name
        }
        
        # Handle type conversion
        avro_type = cls._get_avro_type(field_type, field_name, include_null)
        avro_field["type"] = avro_type
        
        # Add default value if provided and not "..."
        if default_value is not ... and default_value is not None:
            # Handle dict and list defaults by converting to JSON-compatible types
            if isinstance(default_value, dict) or isinstance(default_value, list):
                avro_field["default"] = json.loads(json.dumps(default_value))
            # Handle datetime defaults by converting to string
            elif isinstance(default_value, (datetime, date)):
                avro_field["default"] = default_value.isoformat()
            # Handle UUID defaults by converting to string
            elif isinstance(default_value, uuid.UUID):
                avro_field["default"] = str(default_value)
            # Handle enum defaults
            elif isinstance(default_value, Enum):
                avro_field["default"] = default_value.value
            # Handle other primitive types
            else:
                avro_field["default"] = default_value
        # For optional fields with no default, add null as default if include_null is True
        elif include_null and isinstance(avro_type, list) and "null" in avro_type:
            avro_field["default"] = None
            
        return avro_field

    @staticmethod
    def validate_avro_schema(schema: Dict[str, Any]) -> bool:
        """
        Validate an Avro schema.
        
        Args:
            schema: Avro schema to validate
            
        Returns:
            True if schema is valid, False otherwise
        """
        try:
            fastavro.parse_schema(schema)
            return True
        except Exception as e:
            logger.error(f"Invalid Avro schema: {e}")
            return False

    @staticmethod
    def register_schema(client, subject: str, schema: Union[Dict[str, Any], str], avro_schema_strategy="TOPIC_NAME") -> Optional[int]:
        """
        Register schema with Schema Registry.
        
        Args:
            client: Schema Registry client
            subject: Subject name
            schema: Avro schema to register (either a dictionary or a JSON string)
            avro_schema_strategy: Schema Registry naming strategy
            
        Returns:
            Schema ID if registration was successful, None otherwise
        """
        try:
            # Convert schema to string if it's a dictionary
            if isinstance(schema, dict):
                schema_str = json.dumps(schema)
            else:
                # Assume it's already a string
                schema_str = schema
            
            # Register schema
            result = client.register_schema(subject, schema_str)
            
            # Return schema ID
            return result
            
        except Exception as e:
            logger.error(f"Error registering schema for {subject}: {str(e)}")
            return None
            
    @classmethod
    def register_pydantic_schemas(
        cls, 
        client, 
        model_classes: List[Type[BaseModel]], 
        topic_prefix: str = "clubhouse",
        include_null: bool = True
    ) -> Dict[str, int]:
        """
        Register Pydantic models as Avro schemas.
        
        Args:
            client: Schema Registry client
            model_classes: List of Pydantic model classes
            topic_prefix: Prefix for Kafka topics
            include_null: Whether to include null as a possible type for Optional fields
            
        Returns:
            Dictionary mapping model names to schema IDs
        """
        registered_schemas = {}
        
        for model_class in model_classes:
            try:
                # Check if model_class is a proper Pydantic model
                if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
                    logger.error(f"Invalid model class: {model_class}, skipping")
                    continue
                    
                # Convert model to Avro schema
                avro_schema = cls.pydantic_to_avro(model_class, include_null=include_null)
                
                # Convert to JSON string for Schema Registry
                avro_schema_str = json.dumps(avro_schema)
                
                # Schema subject name
                subject = f"{topic_prefix}-{model_class.__name__}-value"
                
                # Register schema with Schema Registry
                schema_id = cls.register_schema(client, subject, avro_schema_str)
                
                if schema_id is not None:
                    registered_schemas[model_class.__name__] = schema_id
                    logger.info(f"Registered schema for {subject} with ID {schema_id}")
                else:
                    logger.error(f"Failed to register schema for {subject}")
                    
            except Exception as e:
                logger.error(f"Failed to register schema for {model_class.__name__}-value: {str(e)}")
                continue
                
        return registered_schemas

def register_all_schemas(
    client,
    model_classes: List[Type[BaseModel]],
    topic_prefix: str = "clubhouse",
    include_null: bool = True
) -> int:
    """
    Register all schema models with Schema Registry.
    
    Args:
        client: Schema Registry client
        model_classes: List of Pydantic models
        topic_prefix: Prefix for Kafka topics
        include_null: Whether to include null as a possible type for Optional fields
        
    Returns:
        Number of successfully registered schemas
    """
    try:
        registered_schemas = SchemaConverter.register_pydantic_schemas(
            client, 
            model_classes, 
            topic_prefix=topic_prefix,
            include_null=include_null
        )
        
        return len(registered_schemas)
    except Exception as e:
        logger.error(f"Error registering schemas: {str(e)}")
        return 0
