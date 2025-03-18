# Communication Patterns

## Overview

Effective agent collaboration requires standardized communication patterns. The Agent Orchestration Platform implements robust message formats, event systems, and interaction protocols to enable clear and consistent communication between agents, services, and human users.

## Core Communication Principles

1. **Standardized Message Formats**: All communications use well-defined structures
2. **Event-Driven Architecture**: Components interact through events
3. **Asynchronous Communication**: Non-blocking message passing
4. **Validated Payloads**: Message content validation using Pydantic
5. **Clear Routing**: Explicit addressing of message targets

## Message Structure

All messages in the system follow a standardized format:

```python
class AgentMessage(BaseModel):
    """Standard message format for agent communication."""
    
    message_id: str  # Unique identifier for the message
    type: str  # Message type identifier
    sender: str  # Identifier of the sending agent/component
    recipient: Optional[str] = None  # Identifier of the target agent/component
    timestamp: datetime = Field(default_factory=datetime.now)  # When the message was created
    content: Dict[str, Any] = {}  # Message payload/content
    metadata: Dict[str, Any] = {}  # Additional metadata about the message
    parent_id: Optional[str] = None  # For threaded conversations
    trace_id: Optional[str] = None  # For request tracing
```

### Message Types

The platform defines several standard message types:

1. **Command Messages**: Instructions to execute a capability
2. **Event Messages**: Notifications about system events
3. **Response Messages**: Results from capability executions
4. **Error Messages**: Information about errors
5. **Status Messages**: Updates about agent/system status

### Example Message Flow

A typical message flow for capability execution:

```
User → AssistantAgent: CommandMessage(type="summarize", content={...})
AssistantAgent → NLPService: ServiceRequest(operation="summarize", parameters={...})
NLPService → AssistantAgent: ServiceResponse(result={...})
AssistantAgent → User: ResponseMessage(content={...})
```

## Event System

The event system enables loose coupling between components:

```python
class EventType(str, Enum):
    """Standard event types in the system."""
    
    # Lifecycle events
    AGENT_INITIALIZED = "agent_initialized"
    AGENT_SHUTDOWN = "agent_shutdown"
    
    # State events
    STATE_CHANGED = "state_changed"
    STATE_TRANSITION_REJECTED = "state_transition_rejected"
    
    # Capability events
    CAPABILITY_STARTED = "capability_started"
    CAPABILITY_COMPLETED = "capability_completed"
    CAPABILITY_ERROR = "capability_error"
    
    # Message events
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_PROCESSED = "message_processed"
    MESSAGE_SENT = "message_sent"
    
    # Human integration events
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RECEIVED = "approval_received"
    APPROVAL_TIMEOUT = "approval_timeout"
```

### Event Handler Registration

Components register for events they're interested in:

```python
class EventHandlerRegistry:
    """Registry for event handlers."""
    
    def __init__(self):
        """Initialize the event handler registry."""
        self.handlers: Dict[str, List[Callable]] = defaultdict(list)
        
    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type."""
        self.handlers[event_type].append(handler)
        
    def unregister_handler(self, event_type: str, handler: Callable) -> None:
        """Unregister a handler for an event type."""
        if event_type in self.handlers and handler in self.handlers[event_type]:
            self.handlers[event_type].remove(handler)
            
    async def trigger_event(self, event_type: str, **kwargs) -> None:
        """Trigger an event and call all registered handlers."""
        # Create event object
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": kwargs
        }
        
        # Call all handlers for this event type
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    # Log error but continue with other handlers
                    logger.error(f"Error in event handler: {str(e)}")
```

## Kafka Integration

For reliable, scalable message passing, the platform integrates with Kafka:

```python
class KafkaProducer:
    """Producer for sending messages to Kafka."""
    
    def __init__(self, bootstrap_servers: str, schema_registry_url: Optional[str] = None):
        """Initialize the Kafka producer."""
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': socket.gethostname()
        }
        
        if schema_registry_url:
            self.schema_registry_client = SchemaRegistryClient({'url': schema_registry_url})
            self.avro_serializer = AvroSerializer(
                schema_registry_client=self.schema_registry_client,
                schema_str=self._get_message_schema()
            )
            self.config['value.serializer'] = self.avro_serializer
        else:
            self.config['value.serializer'] = lambda v: json.dumps(v).encode('utf-8')
            
        # Create the producer
        self.producer = Producer(self.config)
        
    def _get_message_schema(self) -> str:
        """Get the Avro schema for messages."""
        # Implementation of schema retrieval
        # ...
        
    def produce(self, topic: str, message: Dict[str, Any], key: Optional[str] = None) -> None:
        """Produce a message to a Kafka topic."""
        # Produce the message
        self.producer.produce(
            topic=topic,
            value=message,
            key=key.encode('utf-8') if key else None,
            on_delivery=self._delivery_callback
        )
        
        # Flush to ensure message is sent
        self.producer.flush()
        
    def _delivery_callback(self, err, msg) -> None:
        """Callback for message delivery."""
        if err:
            logger.error(f'Message delivery failed: {err}')
        else:
            logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
```

```python
class KafkaConsumer:
    """Consumer for receiving messages from Kafka."""
    
    def __init__(
        self, 
        bootstrap_servers: str, 
        group_id: str,
        topics: List[str],
        schema_registry_url: Optional[str] = None
    ):
        """Initialize the Kafka consumer."""
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }
        
        if schema_registry_url:
            self.schema_registry_client = SchemaRegistryClient({'url': schema_registry_url})
            self.avro_deserializer = AvroDeserializer(
                schema_registry_client=self.schema_registry_client
            )
            self.config['value.deserializer'] = self.avro_deserializer
        else:
            self.config['value.deserializer'] = lambda v: json.loads(v.decode('utf-8'))
            
        # Create the consumer
        self.consumer = Consumer(self.config)
        self.consumer.subscribe(topics)
        self.running = False
        self.message_handlers = []
        
    def add_message_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add a handler for received messages."""
        self.message_handlers.append(handler)
        
    def start(self) -> None:
        """Start consuming messages."""
        self.running = True
        thread = threading.Thread(target=self._consume_loop)
        thread.daemon = True
        thread.start()
        
    def stop(self) -> None:
        """Stop consuming messages."""
        self.running = False
        
    def _consume_loop(self) -> None:
        """Main consumption loop."""
        try:
            while self.running:
                msg = self.consumer.poll(1.0)
                
                if msg is None:
                    continue
                    
                if msg.error():
                    logger.error(f'Consumer error: {msg.error()}')
                    continue
                    
                # Process the message
                message = msg.value()
                for handler in self.message_handlers:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f'Error in message handler: {str(e)}')
                        
        except Exception as e:
            logger.error(f'Error in consume loop: {str(e)}')
            
        finally:
            # Close the consumer when done
            self.consumer.close()
```

## Message Adapters

Message adapters translate between different communication formats:

```python
class MessageAdapter(Protocol):
    """Protocol for message adapters."""
    
    def convert_to_internal(self, external_message: Any) -> AgentMessage:
        """Convert external message format to internal AgentMessage."""
        ...
        
    def convert_to_external(self, internal_message: AgentMessage) -> Any:
        """Convert internal AgentMessage to external format."""
        ...
```

```python
class KafkaMessageAdapter:
    """Adapter for Kafka messages."""
    
    def convert_to_internal(self, kafka_message: Dict[str, Any]) -> AgentMessage:
        """Convert Kafka message to internal AgentMessage."""
        return AgentMessage(
            message_id=kafka_message.get('message_id', str(uuid.uuid4())),
            type=kafka_message.get('type', ''),
            sender=kafka_message.get('sender', ''),
            recipient=kafka_message.get('recipient'),
            timestamp=self._parse_timestamp(kafka_message.get('timestamp')),
            content=kafka_message.get('content', {}),
            metadata=kafka_message.get('metadata', {}),
            parent_id=kafka_message.get('parent_id'),
            trace_id=kafka_message.get('trace_id')
        )
        
    def convert_to_external(self, internal_message: AgentMessage) -> Dict[str, Any]:
        """Convert internal AgentMessage to Kafka message."""
        return {
            'message_id': internal_message.message_id,
            'type': internal_message.type,
            'sender': internal_message.sender,
            'recipient': internal_message.recipient,
            'timestamp': internal_message.timestamp.isoformat(),
            'content': internal_message.content,
            'metadata': internal_message.metadata,
            'parent_id': internal_message.parent_id,
            'trace_id': internal_message.trace_id
        }
        
    def _parse_timestamp(self, timestamp_str: Optional[str]) -> datetime:
        """Parse timestamp string to datetime."""
        if not timestamp_str:
            return datetime.now()
            
        try:
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            return datetime.now()
```

## Human-Agent Communication

The platform implements specialized patterns for human-agent interaction:

```python
class HumanInteractionManager:
    """Manages communication between agents and humans."""
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        notification_service: NotificationServiceProtocol
    ):
        """Initialize the human interaction manager."""
        self.service_registry = service_registry
        self.notification_service = notification_service
        self.pending_interactions: Dict[str, Dict[str, Any]] = {}
        
    async def request_human_input(
        self,
        interaction_id: str,
        agent_id: str,
        prompt: str,
        options: Optional[List[str]] = None,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Request input from a human."""
        # Create interaction record
        interaction = {
            'interaction_id': interaction_id,
            'agent_id': agent_id,
            'prompt': prompt,
            'options': options,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'timeout_at': (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat()
        }
        
        # Store in pending interactions
        self.pending_interactions[interaction_id] = interaction
        
        # Send notification to user
        await self.notification_service.send_notification(
            recipient='human_user',  # This would be more specific in a real implementation
            notification_type='input_request',
            content={
                'interaction_id': interaction_id,
                'agent_id': agent_id,
                'prompt': prompt,
                'options': options,
                'timeout_at': interaction['timeout_at']
            }
        )
        
        # Wait for response or timeout
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check if interaction has been updated
            current = self.pending_interactions[interaction_id]
            if current['status'] != 'pending':
                return current
                
            # Wait a bit before checking again
            await asyncio.sleep(1)
            
        # If we got here, the interaction timed out
        self.pending_interactions[interaction_id]['status'] = 'timeout'
        return self.pending_interactions[interaction_id]
        
    def record_human_response(
        self,
        interaction_id: str,
        response: Any
    ) -> None:
        """Record a human response to an interaction."""
        if interaction_id not in self.pending_interactions:
            raise ValueError(f"Interaction {interaction_id} not found")
            
        # Update interaction record
        interaction = self.pending_interactions[interaction_id]
        interaction['status'] = 'completed'
        interaction['response'] = response
        interaction['completed_at'] = datetime.now().isoformat()
```

## Evolutionary Communication Flow

In the evolutionary framework, the communication flow follows a specific pattern:

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│  Generator │     │   Critic   │     │  Refiner   │     │ Evaluator  │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
       │                 │                  │                  │
       │ Generate        │                  │                  │
       │ Solutions       │                  │                  │
       ├────────────────>│ Critique        │                  │
       │                 │ Solutions        │                  │
       │                 ├─────────────────>│ Refine          │
       │                 │                  │ Solutions        │
       │                 │                  ├─────────────────>│
       │                 │                  │                  │ Evaluate
       │                 │                  │                  │ Solutions
       │<───────────────┼──────────────────┼──────────────────┤
       │                 │                  │                  │
       │ Generate New    │                  │                  │
       │ Solutions       │                  │                  │
       └────────────────>│                  │                  │
                         │                  │                  │
                         └──────────────────┘                  │
                                                               │
                                                               │
                                                               └
```

The workflow orchestrates this communication automatically, passing messages between specialized agents to enable the evolutionary process.

## Message Observability

For monitoring and debugging, the platform implements message tracing:

```python
class MessageTracer:
    """Traces message flow through the system."""
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol):
        """Initialize the message tracer."""
        self.neo4j_service = neo4j_service
        
    async def trace_message(
        self,
        message: AgentMessage,
        direction: str,  # "sent" or "received"
        component_id: str
    ) -> None:
        """Record a message trace."""
        # Create a message record in Neo4j
        query = """
        MERGE (m:Message {message_id: $message_id})
        ON CREATE SET
            m.type = $type,
            m.sender = $sender,
            m.recipient = $recipient,
            m.timestamp = datetime($timestamp),
            m.content = $content,
            m.parent_id = $parent_id,
            m.trace_id = $trace_id
            
        WITH m
        
        MATCH (c:Component {component_id: $component_id})
        MERGE (c)-[:PROCESSED {
            direction: $direction,
            timestamp: datetime()
        }]->(m)
        
        RETURN m
        """
        
        params = {
            'message_id': message.message_id,
            'type': message.type,
            'sender': message.sender,
            'recipient': message.recipient,
            'timestamp': message.timestamp.isoformat(),
            'content': json.dumps(message.content),
            'parent_id': message.parent_id,
            'trace_id': message.trace_id or message.message_id,
            'component_id': component_id,
            'direction': direction
        }
        
        await self.neo4j_service.execute_query(query, params)
        
    async def get_message_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get the complete trace for a message flow."""
        query = """
        MATCH (m:Message {trace_id: $trace_id})
        OPTIONAL MATCH (c:Component)-[p:PROCESSED]->(m)
        RETURN m, c, p
        ORDER BY p.timestamp
        """
        
        params = {'trace_id': trace_id}
        
        result = await self.neo4j_service.execute_query(query, params)
        
        # Process and return the trace
        # ...
```

## Security and Access Control

Message security is managed through authentication and authorization:

```python
class MessageSecurity:
    """Handles message security and access control."""
    
    def __init__(self, auth_service: AuthServiceProtocol):
        """Initialize the message security service."""
        self.auth_service = auth_service
        
    async def authenticate_message(self, message: AgentMessage) -> bool:
        """Authenticate a message to verify its origin."""
        # Implementation of message authentication
        # ...
        
    async def authorize_message(
        self,
        message: AgentMessage,
        recipient_id: str
    ) -> bool:
        """Authorize a message to verify recipient can access it."""
        # Implementation of message authorization
        # ...
```

## Conclusion

The communication patterns in the Agent Orchestration Platform provide a robust foundation for agent collaboration:

1. Standardized message formats ensure consistent communication
2. Event-driven architecture enables loose coupling between components
3. Kafka integration provides reliable, scalable message delivery
4. Message adapters support integration with external systems
5. Human-agent interaction patterns enable effective collaboration
6. Message tracing supports observability and debugging

These patterns support the evolutionary framework by enabling structured communication between specialized agents, fostering a collaborative environment for solving complex problems.
