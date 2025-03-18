# Schema Registry Implementation

## Overview

The Schema Registry Implementation provides a centralized schema management system for the Agent Orchestration Platform's event-driven architecture. This document outlines how the Schema Registry integrates with Kafka to ensure data compatibility, enable schema evolution, and maintain consistent message formats across the system.

## Core Principles

1. **Schema Evolution**: Support schema changes while maintaining compatibility
2. **Data Validation**: Ensure all messages conform to registered schemas
3. **Centralized Management**: Single source of truth for all message schemas
4. **Backward Compatibility**: New schema versions can read old data
5. **Forward Compatibility**: Optional fields and careful versioning

## Architecture Components

### 1. Schema Registry Service

```
┌────────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                    │       │                 │       │                 │
│ Kafka Producer     │──────▶│ Schema Registry │──────▶│ Schema Store    │
│                    │       │ Client          │       │                 │
└────────────────────┘       └─────────────────┘       └─────────────────┘
```

The Schema Registry Service provides schema management:

- **Schema Registry Client**: Interfaces with the Schema Registry
- **Schema Store**: Persists schemas and versions
- **Compatibility Checker**: Enforces compatibility rules

### 2. Serialization Framework

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│ Pydantic Models │────▶│  Schema Generator │────▶│   Avro/JSON       │
│                 │     │                   │     │   Serializers     │
└─────────────────┘     └───────────────────┘     └───────────────────┘
```

The Serialization Framework converts between Python and wire formats:

- **Pydantic Models**: Define message structures in Python
- **Schema Generator**: Converts models to Avro/JSON schemas
- **Serializers/Deserializers**: Handle conversion between formats

### 3. Schema Management

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Schema Versioning │────▶│ Compatibility     │────▶│ Schema Evolution  │
│                   │     │ Rules             │     │ Strategy          │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Schema Management system ensures data integrity:

- **Schema Versioning**: Tracks schema changes over time
- **Compatibility Rules**: Enforces compatibility policies
- **Schema Evolution Strategy**: Guides schema changes

### 4. Integration Points

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Kafka Producers   │────▶│ Event Bus         │────▶│ Kafka Consumers   │
│                   │     │ Abstraction       │     │                   │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Integration Points connect Schema Registry to the system:

- **Kafka Producers**: Register schemas before publishing
- **Event Bus Abstraction**: Handles schema management
- **Kafka Consumers**: Validate and deserialize messages

## Implementation Details

### Data Models

```python
class SchemaRegistryConfig(BaseModel):
    """Configuration for Schema Registry."""
    url: str
    basic_auth_user_info: Optional[str] = None
    
class SchemaDefinition(BaseModel):
    """Definition of a schema in the registry."""
    subject: str
    schema: Dict[str, Any]
    version: int
    id: int
    
class CompatibilityLevel(str, Enum):
    """Compatibility levels for schemas."""
    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"
    FULL = "FULL"
    NONE = "NONE"
    
class SchemaVersion(BaseModel):
    """Version information for a schema."""
    subject: str
    version: int
    schema_id: int
    schema: Dict[str, Any]
```

### Schema Registry Client

```python
class SchemaRegistryClient:
    """Client for interacting with the Schema Registry."""
    
    def __init__(self, config: SchemaRegistryConfig):
        """Initialize the Schema Registry client."""
        self.config = config
        self.client = CachedSchemaRegistryClient({
            'url': config.url,
            'basic.auth.user.info': config.basic_auth_user_info,
        })
        self._schema_cache: Dict[int, Dict[str, Any]] = {}
    
    async def register_schema(self, subject: str, schema: Dict[str, Any]) -> int:
        """
        Register a schema with the Schema Registry.
        
        Args:
            subject: The subject to register the schema under
            schema: The schema to register
            
        Returns:
            The schema ID
        """
        try:
            schema_id = self.client.register(subject, schema)
            self._schema_cache[schema_id] = schema
            return schema_id
        except Exception as e:
            logger.error(f"Error registering schema for {subject}: {e}")
            raise SchemaRegistryError(f"Failed to register schema: {e}") from e
    
    async def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """
        Get a schema by ID.
        
        Args:
            schema_id: The schema ID
            
        Returns:
            The schema
        """
        if schema_id in self._schema_cache:
            return self._schema_cache[schema_id]
        
        try:
            schema = self.client.get_schema(schema_id)
            self._schema_cache[schema_id] = schema
            return schema
        except Exception as e:
            logger.error(f"Error getting schema {schema_id}: {e}")
            raise SchemaRegistryError(f"Failed to get schema: {e}") from e
    
    async def get_latest_version(self, subject: str) -> SchemaVersion:
        """
        Get the latest version of a schema.
        
        Args:
            subject: The subject to get the schema for
            
        Returns:
            The schema version
        """
        try:
            metadata = self.client.get_latest_version(subject)
            schema = self.client.get_schema(metadata.schema_id)
            
            return SchemaVersion(
                subject=subject,
                version=metadata.version,
                schema_id=metadata.schema_id,
                schema=schema
            )
        except Exception as e:
            logger.error(f"Error getting latest version for {subject}: {e}")
            raise SchemaRegistryError(f"Failed to get latest version: {e}") from e
    
    async def check_compatibility(self, subject: str, schema: Dict[str, Any]) -> bool:
        """
        Check if a schema is compatible with the latest version.
        
        Args:
            subject: The subject to check compatibility for
            schema: The schema to check
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            return self.client.test_compatibility(subject, schema)
        except Exception as e:
            logger.error(f"Error checking compatibility for {subject}: {e}")
            raise SchemaRegistryError(f"Failed to check compatibility: {e}") from e
    
    async def set_compatibility(self, subject: str, level: CompatibilityLevel) -> None:
        """
        Set the compatibility level for a subject.
        
        Args:
            subject: The subject to set compatibility for
            level: The compatibility level
        """
        try:
            self.client.update_compatibility(level.value, subject)
        except Exception as e:
            logger.error(f"Error setting compatibility for {subject}: {e}")
            raise SchemaRegistryError(f"Failed to set compatibility: {e}") from e
```

### Schema Generation from Pydantic Models

```python
class SchemaGenerator:
    """Generates Avro schemas from Pydantic models."""
    
    @staticmethod
    def generate_avro_schema(model_class: Type[BaseModel], namespace: str) -> Dict[str, Any]:
        """
        Generate an Avro schema from a Pydantic model.
        
        Args:
            model_class: The Pydantic model class
            namespace: The namespace for the schema
            
        Returns:
            The Avro schema
        """
        schema = {
            "type": "record",
            "name": model_class.__name__,
            "namespace": namespace,
            "fields": []
        }
        
        for field_name, field in model_class.__fields__.items():
            avro_field = {
                "name": field_name,
                "type": SchemaGenerator._pydantic_type_to_avro_type(field.type_, field.default),
            }
            
            if field.default is not None and field.default is not ...:
                avro_field["default"] = field.default
            
            schema["fields"].append(avro_field)
        
        return schema
    
    @staticmethod
    def _pydantic_type_to_avro_type(type_: Any, default: Any = None) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Convert a Pydantic type to an Avro type.
        
        Args:
            type_: The Pydantic type
            default: The default value
            
        Returns:
            The Avro type
        """
        if type_ == int:
            return "int"
        elif type_ == float:
            return "double"
        elif type_ == bool:
            return "boolean"
        elif type_ == str:
            return "string"
        elif type_ == bytes:
            return "bytes"
        elif type_ == dict:
            return {"type": "map", "values": "string"}
        elif type_ == list:
            return {"type": "array", "items": "string"}
        elif hasattr(type_, "__origin__") and type_.__origin__ == list:
            item_type = SchemaGenerator._pydantic_type_to_avro_type(type_.__args__[0])
            return {"type": "array", "items": item_type}
        elif hasattr(type_, "__origin__") and type_.__origin__ == dict:
            value_type = SchemaGenerator._pydantic_type_to_avro_type(type_.__args__[1])
            return {"type": "map", "values": value_type}
        elif hasattr(type_, "__origin__") and type_.__origin__ == Union:
            # Handle Optional (Union[T, None])
            if len(type_.__args__) == 2 and type_.__args__[1] == type(None):
                inner_type = SchemaGenerator._pydantic_type_to_avro_type(type_.__args__[0])
                return ["null", inner_type]
            else:
                # General union
                return [SchemaGenerator._pydantic_type_to_avro_type(t) for t in type_.__args__]
        elif issubclass(type_, BaseModel):
            # Nested model
            return SchemaGenerator.generate_avro_schema(type_, f"{type_.__module__}.{type_.__name__}")
        elif issubclass(type_, Enum):
            # Enum
            return {"type": "enum", "name": type_.__name__, "symbols": [e.name for e in type_]}
        else:
            # Default to string for unknown types
            logger.warning(f"Unknown type {type_}, defaulting to string")
            return "string"
```

### Serialization and Deserialization

```python
class MessageSerializer:
    """Serializes messages using Avro and Schema Registry."""
    
    def __init__(self, schema_registry_client: SchemaRegistryClient):
        """Initialize the serializer."""
        self.schema_registry = schema_registry_client
        self.schema_generator = SchemaGenerator()
        self.serializers: Dict[str, AvroSerializer] = {}
        
    async def serialize(self, message: BaseModel, topic: str) -> Tuple[bytes, int]:
        """
        Serialize a message using Avro and Schema Registry.
        
        Args:
            message: The message to serialize
            topic: The Kafka topic
            
        Returns:
            The serialized message and schema ID
        """
        # Generate subject name
        subject = f"{topic}-value"
        
        # Generate schema
        schema = self.schema_generator.generate_avro_schema(
            type(message), f"{type(message).__module__}.{type(message).__name__}"
        )
        
        # Register schema
        schema_id = await self.schema_registry.register_schema(subject, schema)
        
        # Get or create serializer
        if subject not in self.serializers:
            self.serializers[subject] = AvroSerializer(
                schema_registry_client=self.schema_registry.client,
                schema_str=json.dumps(schema)
            )
        
        # Convert Pydantic model to dict
        message_dict = message.dict()
        
        # Serialize message
        serialized_data = self.serializers[subject](message_dict)
        
        return serialized_data, schema_id
        
class MessageDeserializer:
    """Deserializes messages using Avro and Schema Registry."""
    
    def __init__(self, schema_registry_client: SchemaRegistryClient):
        """Initialize the deserializer."""
        self.schema_registry = schema_registry_client
        self.deserializers: Dict[int, AvroDeserializer] = {}
        
    async def deserialize(self, data: bytes, model_class: Type[T]) -> T:
        """
        Deserialize a message using Avro and Schema Registry.
        
        Args:
            data: The serialized message
            model_class: The Pydantic model class
            
        Returns:
            The deserialized message
        """
        # Extract schema ID from message
        if len(data) <= 5:
            raise ValueError("Invalid Avro message format")
        
        schema_id = struct.unpack('>I', data[1:5])[0]
        
        # Get or create deserializer
        if schema_id not in self.deserializers:
            schema = await self.schema_registry.get_schema(schema_id)
            self.deserializers[schema_id] = AvroDeserializer(
                schema_registry_client=self.schema_registry.client,
                schema_str=json.dumps(schema)
            )
        
        # Deserialize message
        message_dict = self.deserializers[schema_id](data)
        
        # Convert dict to Pydantic model
        return model_class(**message_dict)
```

## Schema Evolution Strategy

### Compatibility Rules

```python
class CompatibilityRules:
    """Rules for schema compatibility."""
    
    @staticmethod
    def check_backward_compatibility(old_schema: Dict[str, Any], new_schema: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a new schema is backward compatible with an old schema.
        
        Args:
            old_schema: The old schema
            new_schema: The new schema
            
        Returns:
            A tuple of (is_compatible, reason)
        """
        # Fields can be added if they have defaults
        old_fields = {f["name"]: f for f in old_schema["fields"]}
        new_fields = {f["name"]: f for f in new_schema["fields"]}
        
        # Check for removed fields
        for name, field in old_fields.items():
            if name not in new_fields:
                return False, f"Field '{name}' was removed"
        
        # Check for changed field types
        for name, new_field in new_fields.items():
            if name in old_fields:
                old_field = old_fields[name]
                
                # Check if type changed
                if new_field["type"] != old_field["type"]:
                    # Allow type to change to union with the old type
                    if isinstance(new_field["type"], list) and old_field["type"] in new_field["type"]:
                        continue
                    
                    return False, f"Field '{name}' type changed from {old_field['type']} to {new_field['type']}"
        
        return True, ""
    
    @staticmethod
    def check_forward_compatibility(old_schema: Dict[str, Any], new_schema: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a new schema is forward compatible with an old schema.
        
        Args:
            old_schema: The old schema
            new_schema: The new schema
            
        Returns:
            A tuple of (is_compatible, reason)
        """
        # Added fields must be optional or have defaults
        old_fields = {f["name"]: f for f in old_schema["fields"]}
        new_fields = {f["name"]: f for f in new_schema["fields"]}
        
        # Check for added required fields
        for name, new_field in new_fields.items():
            if name not in old_fields:
                if "default" not in new_field:
                    if not (isinstance(new_field["type"], list) and "null" in new_field["type"]):
                        return False, f"Field '{name}' was added without a default or null type"
        
        return True, ""
    
    @staticmethod
    def check_full_compatibility(old_schema: Dict[str, Any], new_schema: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a new schema is fully compatible with an old schema.
        
        Args:
            old_schema: The old schema
            new_schema: The new schema
            
        Returns:
            A tuple of (is_compatible, reason)
        """
        backward_compatible, backward_reason = CompatibilityRules.check_backward_compatibility(
            old_schema, new_schema
        )
        
        if not backward_compatible:
            return False, backward_reason
        
        forward_compatible, forward_reason = CompatibilityRules.check_forward_compatibility(
            old_schema, new_schema
        )
        
        if not forward_compatible:
            return False, forward_reason
        
        return True, ""
```

### Schema Evolution Guidelines

1. **Backward Compatibility (Default)**:
   - Always add new fields with defaults
   - Never remove fields
   - Never change field types

2. **Forward Compatibility**:
   - Make all new fields optional (Union with null)
   - Use type unions to support multiple types

3. **Full Compatibility (Both)**:
   - Add new fields with defaults AND make them optional
   - Never remove fields
   - Use unions for type changes

4. **Version Transitions**:
   - Two-phase schema evolution: add field as optional, then make required
   - Deprecation period before removing fields

## Integration with Kafka Service

```python
class SchemaRegistryKafkaService:
    """Kafka service with Schema Registry integration."""
    
    def __init__(
        self,
        producer: KafkaProducerProtocol,
        consumer: KafkaConsumerProtocol,
        schema_registry_client: SchemaRegistryClient,
    ):
        """Initialize the Kafka service."""
        self.producer = producer
        self.consumer = consumer
        self.schema_registry = schema_registry_client
        self.serializer = MessageSerializer(schema_registry_client)
        self.deserializer = MessageDeserializer(schema_registry_client)
        self._running = False
    
    async def produce_message(self, message: BaseModel, topic: str, key: Optional[str] = None) -> None:
        """
        Produce a message to Kafka.
        
        Args:
            message: The message to produce
            topic: The topic to produce to
            key: The message key
        """
        try:
            # Serialize message
            serialized_data, schema_id = await self.serializer.serialize(message, topic)
            
            # Produce to Kafka
            self.producer.produce(
                topic=topic,
                value=serialized_data,
                key=key.encode('utf-8') if key else None,
                headers={"schema_id": str(schema_id)}
            )
            
            # Flush to ensure delivery
            self.producer.flush()
            
        except Exception as e:
            logger.error(f"Error producing message to {topic}: {e}")
            raise MessageProducerError(f"Failed to produce message: {e}") from e
    
    async def consume_messages(
        self,
        topics: List[str],
        model_class: Type[BaseModel],
        handler: Callable[[BaseModel, Optional[str]], None],
        timeout: float = 1.0,
    ) -> None:
        """
        Consume messages from Kafka.
        
        Args:
            topics: The topics to consume from
            model_class: The Pydantic model class for messages
            handler: The message handler
            timeout: The poll timeout in seconds
        """
        if not topics:
            raise ValueError("At least one topic is required")
        
        try:
            # Subscribe to topics
            self.consumer.subscribe(topics)
            logger.info(f"Subscribed to topics: {topics}")
            
            # Set running flag
            self._running = True
            
            # Consume messages
            while self._running:
                # Poll for messages
                msg = self.consumer.poll(timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    match msg.error().code():
                        case KafkaError._PARTITION_EOF:
                            # End of partition event - not an error
                            logger.debug("Reached end of partition")
                        case _:
                            # Error
                            logger.error(f"Consumer error: {msg.error()}")
                    continue
                
                # Process the message
                try:
                    # Deserialize the message
                    value = await self.deserializer.deserialize(msg.value(), model_class)
                    key = msg.key().decode('utf-8') if msg.key() else None
                    
                    # Call the handler
                    handler(value, key)
                    
                    # Commit offset
                    self.consumer.commit(msg)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
            raise MessageConsumerError(f"Failed to consume messages: {e}") from e
        
        finally:
            # Clean up
            self._running = False
```

## Topic-Specific Schema Strategies

### Event Bus Topic Schemas

```
Event Bus topics follow a standardized schema based on event type:

1. Lifecycle Events: agent.lifecycle.*
   {
     "agent_id": string,
     "timestamp": string (ISO 8601),
     "event_type": string,
     "metadata": map<string, string>
   }

2. Capability Events: capability.*.*
   {
     "agent_id": string,
     "operation_id": string,
     "capability_type": string,
     "timestamp": string (ISO 8601),
     "params": map<string, any>,  # For "started" events
     "result": map<string, any>,  # For "completed" events
     "error": string              # For "failed" events
   }

3. Task Events: task.*.*
   {
     "task_id": string,
     "agent_id": string,
     "timestamp": string (ISO 8601),
     "status": string,
     "metadata": map<string, string>
   }
```

### Schema Naming Conventions

```
All schemas follow a consistent naming convention:

1. Subject Names: {topic}-{key|value}
   - Example: "agent.lifecycle.created-value"

2. Schema Names: Matches the Pydantic model name
   - Example: "AgentLifecycleEvent"

3. Namespace: Reflects package structure
   - Example: "clubhouse.events.agent.lifecycle"
```

## Testing Strategy

Following our test-driven development approach, we implement:

1. **Unit Tests**:
   - Test schema generation from Pydantic models
   - Validate serialization/deserialization
   - Test compatibility checks

2. **Integration Tests**:
   - Test end-to-end schema registration
   - Verify producer/consumer with Schema Registry
   - Test compatibility enforcement

3. **Evolution Tests**:
   - Test schema evolution scenarios
   - Verify backward/forward compatibility
   - Test schema versioning

## Schema Registry Administration

### Initial Setup

```python
class SchemaRegistryAdmin:
    """Administrative functions for Schema Registry."""
    
    def __init__(self, client: SchemaRegistryClient):
        """Initialize the admin client."""
        self.client = client
    
    async def initialize_global_compatibility(self, level: CompatibilityLevel = CompatibilityLevel.BACKWARD) -> None:
        """
        Initialize global compatibility setting.
        
        Args:
            level: The compatibility level
        """
        try:
            await self.client.set_compatibility("", level)
            logger.info(f"Set global compatibility to {level.value}")
        except Exception as e:
            logger.error(f"Error setting global compatibility: {e}")
            raise
    
    async def create_subject_if_not_exists(
        self,
        subject: str,
        initial_schema: Dict[str, Any],
        compatibility: Optional[CompatibilityLevel] = None
    ) -> int:
        """
        Create a subject if it doesn't exist.
        
        Args:
            subject: The subject name
            initial_schema: The initial schema
            compatibility: The compatibility level
            
        Returns:
            The schema ID
        """
        try:
            # Try to get latest version
            await self.client.get_latest_version(subject)
            logger.info(f"Subject {subject} already exists")
            
            # Register schema (will create a new version if different)
            return await self.client.register_schema(subject, initial_schema)
        except SchemaRegistryError:
            # Subject doesn't exist, create it
            schema_id = await self.client.register_schema(subject, initial_schema)
            logger.info(f"Created subject {subject} with schema ID {schema_id}")
            
            # Set compatibility if specified
            if compatibility:
                await self.client.set_compatibility(subject, compatibility)
                logger.info(f"Set compatibility for {subject} to {compatibility.value}")
            
            return schema_id
```

### Monitoring and Management

```python
class SchemaRegistryMonitor:
    """Monitors Schema Registry."""
    
    def __init__(self, client: SchemaRegistryClient):
        """Initialize the monitor."""
        self.client = client
    
    async def list_subjects(self) -> List[str]:
        """
        List all subjects in the Schema Registry.
        
        Returns:
            The list of subjects
        """
        try:
            return self.client.client.get_subjects()
        except Exception as e:
            logger.error(f"Error listing subjects: {e}")
            raise
    
    async def list_versions(self, subject: str) -> List[int]:
        """
        List all versions for a subject.
        
        Args:
            subject: The subject name
            
        Returns:
            The list of versions
        """
        try:
            return self.client.client.get_versions(subject)
        except Exception as e:
            logger.error(f"Error listing versions for {subject}: {e}")
            raise
    
    async def get_compatibility(self, subject: Optional[str] = None) -> CompatibilityLevel:
        """
        Get the compatibility level for a subject or global.
        
        Args:
            subject: The subject name, or None for global
            
        Returns:
            The compatibility level
        """
        try:
            compat = self.client.client.get_compatibility(subject or "")
            return CompatibilityLevel(compat)
        except Exception as e:
            logger.error(f"Error getting compatibility for {subject or 'global'}: {e}")
            raise
```

## Future Enhancements

1. **Schema Evolution UI**:
   - Web interface for managing schema evolution
   - Visual comparison of schema versions
   - Schema dependency analysis

2. **Schema Validation Hooks**:
   - Pre-commit hooks for schema validation
   - Integration with CI/CD pipelines
   - Automatic compatibility testing

3. **Schema Documentation Generation**:
   - Automatic documentation from schemas
   - Example message generation
   - Integration with API documentation

## Conclusion

The Schema Registry Implementation provides a robust foundation for managing message schemas in the Agent Orchestration Platform. By enforcing schema compatibility, enabling controlled evolution, and integrating with the Kafka messaging system, it ensures reliable communication between components while allowing the system to evolve over time.
