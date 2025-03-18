# Kafka Implementation

## Overview

The Kafka implementation provides a robust event bus for the Agent Orchestration Platform, enabling reliable, scalable, and asynchronous communication between components. This document details the architecture, implementation, and integration patterns for the Kafka messaging system, including the Schema Registry for maintaining message compatibility.

## Core Principles

1. **Event-Driven Architecture**: All significant state changes are published as events
2. **Schema Evolution**: Message schemas evolve while maintaining backward compatibility
3. **Exactly-Once Semantics**: Critical operations guarantee exactly-once processing
4. **Scalable Throughput**: System scales horizontally to handle increasing event volumes
5. **Error Resilience**: Robust error handling and dead-letter queues for message failures

## Architecture Components

### 1. Kafka Service Layer

```
┌────────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                    │       │                 │       │                 │
│ Platform Component │──────▶│ Kafka Service   │──────▶│ Kafka Cluster   │
│                    │       │ Protocol        │       │                 │
└────────────────────┘       └─────────────────┘       └─────────────────┘
```

The Kafka Service Layer provides a clean abstraction for producing and consuming events:

- **Kafka Service Protocol**: Defines the interface for event production and consumption
- **Kafka Service Implementation**: Handles connection management and error handling
- **Message Serialization/Deserialization**: Converts between domain objects and Kafka messages

### 2. Schema Registry Integration

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│ Message Models  │────▶│  Schema Registry  │────▶│   Compatibility   │
│                 │     │  Client           │     │   Validation      │
└─────────────────┘     └───────────────────┘     └───────────────────┘
```

The Schema Registry Integration ensures message compatibility:

- **Message Models**: Pydantic models that define event structures
- **Schema Registry Client**: Interfaces with the Schema Registry service
- **Compatibility Validation**: Ensures schema changes maintain compatibility

### 3. Topic Management

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Topic Naming      │────▶│ Partition         │────▶│ Retention         │
│ Convention        │     │ Strategy          │     │ Policies          │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Topic Management system organizes message streams:

- **Topic Naming Convention**: Structured approach to topic names
- **Partition Strategy**: Determines message distribution across partitions
- **Retention Policies**: Configures message retention periods

### 4. Consumer Group Management

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Consumer Group    │────▶│ Offset Management │────▶│ Rebalance         │
│ Naming            │     │                   │     │ Strategies        │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Consumer Group Management system coordinates message consumption:

- **Consumer Group Naming**: Organizes consumers by function
- **Offset Management**: Tracks message processing progress
- **Rebalance Strategies**: Handles consumer addition and removal

## Implementation Details

### Data Models

```python
class KafkaMessage(BaseModel):
    """Base model for all Kafka messages."""
    topic: str
    value: Dict[str, Any]
    key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    
class KafkaConfig(BaseModel):
    """Configuration for Kafka connection."""
    bootstrap_servers: List[str]
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    
class SchemaRegistryConfig(BaseModel):
    """Configuration for Schema Registry."""
    url: str
    basic_auth_user_info: Optional[str] = None
```

### Service Interfaces

```python
class KafkaProducerProtocol(Protocol):
    """Protocol for Kafka producer operations."""
    
    def produce(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Produce a message to Kafka."""
        ...
    
    def flush(self, timeout: Optional[float] = None) -> int:
        """Flush the producer's message queue."""
        ...
    
class KafkaConsumerProtocol(Protocol):
    """Protocol for Kafka consumer operations."""
    
    def subscribe(self, topics: List[str]) -> None:
        """Subscribe to topics."""
        ...
    
    def poll(self, timeout: float) -> Any:
        """Poll for messages."""
        ...
    
    def commit(self, message: Optional[Any] = None) -> None:
        """Commit offsets."""
        ...
    
    def close(self) -> None:
        """Close the consumer."""
        ...
    
class MessageHandlerProtocol(Protocol):
    """Protocol for message handlers."""
    
    def process_message(
        self, value: Dict[str, Any], key: Optional[str], headers: Optional[Dict[str, str]]
    ) -> None:
        """Process a message."""
        ...
```

### Schema Registry Integration

```python
class SchemaRegistryClient:
    """Client for interacting with Schema Registry."""
    
    def __init__(self, config: SchemaRegistryConfig):
        """Initialize the Schema Registry client."""
        self.config = config
        self.client = CachedSchemaRegistryClient({
            'url': config.url,
            'basic.auth.user.info': config.basic_auth_user_info,
        })
        
    def register_schema(self, subject: str, schema: Dict[str, Any]) -> int:
        """Register a schema with the Schema Registry."""
        return self.client.register(subject, schema)
        
    def get_schema(self, schema_id: int) -> Dict[str, Any]:
        """Get a schema by ID."""
        return self.client.get_schema(schema_id)
        
    def check_compatibility(self, subject: str, schema: Dict[str, Any]) -> bool:
        """Check if a schema is compatible with the latest version."""
        return self.client.test_compatibility(subject, schema)
```

### Topic Naming Convention

The platform follows a structured topic naming convention:

```
{domain}.{entity}.{action}
```

Examples:
- `agent.lifecycle.created`
- `task.execution.started`
- `capability.search.completed`
- `evolution.generation.evaluated`

This convention provides clear topic organization and enables fine-grained subscription patterns.

## Integration with Existing Components

### 1. Event Bus Integration

The Kafka service implements the event bus interface:

```python
class KafkaEventBus(EventBusProtocol):
    """Event bus implementation using Kafka."""
    
    def __init__(
        self,
        kafka_service: KafkaService,
        schema_registry_client: SchemaRegistryClient,
    ):
        """Initialize the Kafka event bus."""
        self.kafka_service = kafka_service
        self.schema_registry_client = schema_registry_client
        
    async def publish(self, topic: str, value: Dict[str, Any], key: Optional[str] = None) -> None:
        """Publish an event to the event bus."""
        try:
            # Validate and register schema if needed
            schema = self._extract_schema(value)
            schema_id = self.schema_registry_client.register_schema(f"{topic}-value", schema)
            
            # Create message
            message = KafkaMessage(
                topic=topic,
                value=value,
                key=key,
                headers={"schema_id": str(schema_id)}
            )
            
            # Publish message
            self.kafka_service.produce_message(message)
            
        except Exception as e:
            logger.error(f"Error publishing event to {topic}: {e}")
            raise EventBusError(f"Failed to publish event: {e}") from e
            
    async def subscribe(
        self, topics: List[str], handler: Callable[[Dict[str, Any], Optional[str]], None]
    ) -> None:
        """Subscribe to topics on the event bus."""
        try:
            # Create message handler adapter
            message_handler = KafkaMessageHandlerAdapter(handler, self.schema_registry_client)
            
            # Start consuming
            self.kafka_service.consume_messages(topics, message_handler.process_message)
            
        except Exception as e:
            logger.error(f"Error subscribing to topics {topics}: {e}")
            raise EventBusError(f"Failed to subscribe to topics: {e}") from e
```

### 2. Agent Lifecycle Integration

Agent lifecycle events are published to the event bus:

```python
# Event types
AGENT_CREATED = "agent.lifecycle.created"
AGENT_UPDATED = "agent.lifecycle.updated"
AGENT_DELETED = "agent.lifecycle.deleted"
AGENT_EXECUTION_STARTED = "agent.execution.started"
AGENT_EXECUTION_COMPLETED = "agent.execution.completed"

# Publishing an agent creation event
event_bus.publish(
    topic=AGENT_CREATED,
    value={
        "agent_id": agent.id,
        "agent_type": agent.type,
        "created_at": agent.created_at.isoformat(),
        "capabilities": [str(cap) for cap in agent.capabilities],
        "creator_id": user_id,
    },
    key=agent.id
)
```

### 3. Capability Integration

Capabilities publish execution events to the event bus:

```python
# Publishing capability execution events
async def execute_with_lifecycle(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the capability with lifecycle events."""
    # Generate operation ID
    operation_id = str(uuid.uuid4())
    
    # Publish start event
    await self.event_bus.publish(
        topic=f"capability.{self.capability_type}.started",
        value={
            "agent_id": self.agent_id,
            "capability_type": self.capability_type,
            "operation_id": operation_id,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        },
        key=operation_id
    )
    
    try:
        # Execute capability
        result = await self.execute(params)
        
        # Publish completion event
        await self.event_bus.publish(
            topic=f"capability.{self.capability_type}.completed",
            value={
                "agent_id": self.agent_id,
                "capability_type": self.capability_type,
                "operation_id": operation_id,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            },
            key=operation_id
        )
        
        return result
    except Exception as e:
        # Publish error event
        await self.event_bus.publish(
            topic=f"capability.{self.capability_type}.failed",
            value={
                "agent_id": self.agent_id,
                "capability_type": self.capability_type,
                "operation_id": operation_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
            key=operation_id
        )
        raise
```

### 4. Cost Management Integration

Cost events are published to the event bus:

```python
# Publishing cost tracking events
async def record_cost(self, cost_entry: CostEntry) -> None:
    """Record a cost entry and publish event."""
    # Store cost entry
    await self.cost_repository.create(cost_entry)
    
    # Publish cost recorded event
    await self.event_bus.publish(
        topic="cost.tracking.recorded",
        value={
            "token_usage_id": cost_entry.token_usage_id,
            "prompt_cost": float(cost_entry.prompt_cost),
            "completion_cost": float(cost_entry.completion_cost),
            "total_cost": float(cost_entry.total_cost),
            "currency": cost_entry.currency,
            "timestamp": datetime.now().isoformat(),
        },
        key=cost_entry.token_usage_id
    )
```

## Schema Registry Implementation

### Schema Evolution

The Schema Registry enforces compatibility rules for schema evolution:

1. **Backward Compatibility**: New schema can read old data
2. **Forward Compatibility**: Old schema can read new data
3. **Full Compatibility**: Both backward and forward compatibility

Default compatibility setting: `BACKWARD`

### Schema Subject Naming

Schema subjects follow a consistent naming pattern:

```
{topic_name}-{key|value}
```

Examples:
- `agent.lifecycle.created-value`
- `task.execution.started-key`

### Avro Schema Generation

Pydantic models are automatically converted to Avro schemas:

```python
def pydantic_to_avro_schema(model_class: Type[BaseModel], namespace: str) -> Dict[str, Any]:
    """Convert a Pydantic model to an Avro schema."""
    schema = {
        "type": "record",
        "name": model_class.__name__,
        "namespace": namespace,
        "fields": []
    }
    
    for field_name, field in model_class.__fields__.items():
        avro_field = {
            "name": field_name,
            "type": _pydantic_type_to_avro_type(field.type_),
        }
        
        if field.default is not None:
            avro_field["default"] = field.default
        
        schema["fields"].append(avro_field)
    
    return schema
```

## Error Handling and Resilience

### Retry Mechanism

The Kafka service implements smart retries for transient failures:

```python
class RetryPolicy:
    """Defines retry behavior for Kafka operations."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 0.1,
        backoff_multiplier: float = 2.0,
    ):
        """Initialize the retry policy."""
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
    
    async def execute_with_retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Execute an operation with retries."""
        last_exception = None
        backoff = self.initial_backoff
        
        for retry_count in range(self.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e) or retry_count == self.max_retries:
                    break
                
                # Apply backoff
                await asyncio.sleep(backoff)
                backoff *= self.backoff_multiplier
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Retry failed with unknown error")
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Determine if an exception is retryable."""
        if isinstance(exception, KafkaException):
            error_code = exception.args[0].code() if exception.args else None
            return error_code in [
                KafkaError.BROKER_NOT_AVAILABLE,
                KafkaError.NETWORK_EXCEPTION,
                KafkaError.LEADER_NOT_AVAILABLE,
                KafkaError.REQUEST_TIMED_OUT,
            ]
        return False
```

### Dead Letter Queue

Failed messages are sent to dead letter queues for later analysis:

```python
# Dead letter queue topic naming convention
DLQ_TOPIC_PREFIX = "dlq"

# Dead letter queue handler
async def handle_failed_message(
    topic: str,
    value: Dict[str, Any],
    key: Optional[str],
    exception: Exception,
) -> None:
    """Handle a failed message by sending it to a dead letter queue."""
    dlq_topic = f"{DLQ_TOPIC_PREFIX}.{topic}"
    
    # Create dead letter message
    dlq_message = KafkaMessage(
        topic=dlq_topic,
        value={
            "original_topic": topic,
            "original_value": value,
            "error": str(exception),
            "error_type": type(exception).__name__,
            "timestamp": datetime.now().isoformat(),
        },
        key=key,
        headers={"error": type(exception).__name__}
    )
    
    # Send to dead letter queue
    await kafka_service.produce_message(dlq_message)
    logger.warning(f"Message sent to DLQ {dlq_topic}: {exception}")
```

## Monitoring and Metrics

### Kafka Metrics Collection

```python
class KafkaMetricsCollector:
    """Collects and publishes Kafka metrics."""
    
    def __init__(self, producer: KafkaProducerProtocol, consumer: KafkaConsumerProtocol):
        """Initialize the metrics collector."""
        self.producer = producer
        self.consumer = consumer
        self.metrics = {}
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect Kafka metrics."""
        # Producer metrics
        producer_metrics = self.producer._producer.metrics()
        
        # Consumer metrics
        consumer_metrics = self.consumer._consumer.metrics()
        
        # Combine metrics
        self.metrics = {
            "producer": self._extract_relevant_metrics(producer_metrics),
            "consumer": self._extract_relevant_metrics(consumer_metrics),
            "timestamp": datetime.now().isoformat(),
        }
        
        return self.metrics
    
    def _extract_relevant_metrics(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from raw metrics."""
        relevant_metrics = {}
        
        # Extract throughput metrics
        if "producer-metrics" in raw_metrics:
            producer_data = raw_metrics["producer-metrics"]
            relevant_metrics["outgoing_byte_rate"] = producer_data.get("outgoing-byte-rate", 0)
            relevant_metrics["request_rate"] = producer_data.get("request-rate", 0)
            relevant_metrics["response_rate"] = producer_data.get("response-rate", 0)
            
        # Extract consumer metrics
        if "consumer-metrics" in raw_metrics:
            consumer_data = raw_metrics["consumer-metrics"]
            relevant_metrics["incoming_byte_rate"] = consumer_data.get("incoming-byte-rate", 0)
            relevant_metrics["records_consumed_rate"] = consumer_data.get("records-consumed-rate", 0)
            
        return relevant_metrics
```

### Lag Monitoring

```python
class ConsumerLagMonitor:
    """Monitors consumer lag."""
    
    def __init__(self, admin_client: AdminClient, consumer_group_id: str):
        """Initialize the lag monitor."""
        self.admin_client = admin_client
        self.consumer_group_id = consumer_group_id
    
    async def get_consumer_lag(self) -> Dict[str, int]:
        """Get consumer lag by topic-partition."""
        # Get consumer group offsets
        consumer_offsets = self._get_consumer_offsets()
        
        # Get topic end offsets
        end_offsets = self._get_topic_end_offsets(list(consumer_offsets.keys()))
        
        # Calculate lag
        lag_by_partition = {}
        for tp, consumer_offset in consumer_offsets.items():
            topic_partition_str = f"{tp.topic}-{tp.partition}"
            end_offset = end_offsets.get(tp, 0)
            lag = max(0, end_offset - consumer_offset)
            lag_by_partition[topic_partition_str] = lag
            
        return lag_by_partition
```

## Testing Strategy

Following our test-driven development approach, we implement:

1. **Unit Tests**:
   - Test message serialization/deserialization
   - Test error handling and retry logic
   - Validate schema compatibility checks

2. **Integration Tests**:
   - Test against embedded Kafka and Schema Registry
   - Verify end-to-end event publication and consumption
   - Test dead letter queue functionality

3. **Performance Tests**:
   - Measure throughput and latency
   - Test under high message volumes
   - Verify scalability with multiple consumers

## Kafka Configuration Best Practices

### Producer Configuration

```python
producer_config = {
    'bootstrap.servers': ','.join(config.bootstrap_servers),
    'security.protocol': config.security_protocol,
    'sasl.mechanisms': config.sasl_mechanism,
    'sasl.username': config.sasl_username,
    'sasl.password': config.sasl_password,
    
    # Performance settings
    'linger.ms': 5,  # Batch messages for 5ms to improve throughput
    'batch.size': 16384,  # 16KB batch size
    'compression.type': 'snappy',  # Use Snappy compression
    
    # Reliability settings
    'acks': 'all',  # Wait for all replicas to acknowledge
    'retries': 3,  # Retry up to 3 times
    'retry.backoff.ms': 100,  # 100ms between retries
    
    # Schema Registry settings (if using Confluent's Python client)
    'schema.registry.url': config.schema_registry.url,
    'auto.register.schemas': True,
}
```

### Consumer Configuration

```python
consumer_config = {
    'bootstrap.servers': ','.join(config.bootstrap_servers),
    'security.protocol': config.security_protocol,
    'sasl.mechanisms': config.sasl_mechanism,
    'sasl.username': config.sasl_username,
    'sasl.password': config.sasl_password,
    
    # Consumer group settings
    'group.id': config.consumer_group_id,
    'auto.offset.reset': 'earliest',  # Start from earliest offset when no committed offset exists
    
    # Commit settings
    'enable.auto.commit': False,  # Manual commits for better control
    
    # Performance settings
    'fetch.min.bytes': 1,  # Minimum bytes to fetch
    'fetch.max.wait.ms': 500,  # Maximum time to wait for fetch.min.bytes
    
    # Schema Registry settings (if using Confluent's Python client)
    'schema.registry.url': config.schema_registry.url,
}
```

## Future Enhancements

1. **Kafka Streams Integration**:
   - Implement stream processing for complex event patterns
   - Develop stateful stream processors for aggregations
   - Create real-time analytics pipelines

2. **Advanced Partitioning Strategies**:
   - Custom partitioners for optimized message distribution
   - Sticky partitioning for related messages
   - Dynamic partition rebalancing based on load

3. **Multi-Cluster Support**:
   - Kafka MirrorMaker2 integration for cross-cluster replication
   - Active-active cluster configuration
   - Disaster recovery procedures

## Conclusion

The Kafka implementation provides a robust, scalable, and reliable messaging backbone for the Agent Orchestration Platform. By leveraging Kafka's streaming capabilities and the Schema Registry for message compatibility, the platform can evolve while maintaining reliable and consistent communication between components.
