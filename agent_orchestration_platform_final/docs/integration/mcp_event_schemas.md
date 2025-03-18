# MCP Event Schemas

## Overview

This document defines the Avro schemas for Model Context Protocol (MCP) events in the Agent Orchestration Platform. These schemas ensure type safety, enable schema evolution, and facilitate interoperability between platform components.

## Schema Structure

### Namespace Convention

All MCP event schemas use the namespace `com.agent_platform.events.mcp` with sub-namespaces organized by domain:

- `com.agent_platform.events.mcp.conversation` - Conversation-related events
- `com.agent_platform.events.mcp.message` - Message-related events
- `com.agent_platform.events.mcp.capability` - Capability-related events
- `com.agent_platform.events.mcp.agent` - Agent-related events

## Core Event Schemas

### Base Event Schema

```json
{
  "type": "record",
  "name": "BaseEvent",
  "namespace": "com.agent_platform.events.mcp",
  "doc": "Base schema for all MCP events",
  "fields": [
    {
      "name": "event_id",
      "type": "string",
      "doc": "Unique identifier for the event"
    },
    {
      "name": "event_type",
      "type": "string",
      "doc": "Type of the event"
    },
    {
      "name": "version",
      "type": "string",
      "doc": "Schema version in semantic versioning format"
    },
    {
      "name": "timestamp",
      "type": "long",
      "doc": "Event creation timestamp in milliseconds since epoch"
    },
    {
      "name": "source",
      "type": "string",
      "doc": "Source service that generated the event"
    },
    {
      "name": "trace_id",
      "type": ["null", "string"],
      "default": null,
      "doc": "Distributed tracing ID for request correlation"
    },
    {
      "name": "metadata",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null"]
      }],
      "default": null,
      "doc": "Additional metadata for the event"
    }
  ]
}
```

### Conversation Schemas

#### Conversation Created Event

```json
{
  "type": "record",
  "name": "ConversationCreatedEvent",
  "namespace": "com.agent_platform.events.mcp.conversation",
  "doc": "Event emitted when a new conversation is created",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Unique identifier for the conversation"
    },
    {
      "name": "user_id",
      "type": "string",
      "doc": "Identifier of the user who initiated the conversation"
    },
    {
      "name": "agent_id",
      "type": "string",
      "doc": "Identifier of the agent associated with the conversation"
    },
    {
      "name": "title",
      "type": ["null", "string"],
      "default": null,
      "doc": "Optional title for the conversation"
    },
    {
      "name": "system_prompt",
      "type": ["null", "string"],
      "default": null,
      "doc": "System prompt for the conversation"
    },
    {
      "name": "conversation_metadata",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null"]
      }],
      "default": null,
      "doc": "Additional metadata for the conversation"
    }
  ]
}
```

#### Conversation Updated Event

```json
{
  "type": "record",
  "name": "ConversationUpdatedEvent",
  "namespace": "com.agent_platform.events.mcp.conversation",
  "doc": "Event emitted when a conversation is updated",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Unique identifier for the conversation"
    },
    {
      "name": "updated_fields",
      "type": {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null"]
      },
      "doc": "Fields that were updated and their new values"
    }
  ]
}
```

#### Conversation Completed Event

```json
{
  "type": "record",
  "name": "ConversationCompletedEvent",
  "namespace": "com.agent_platform.events.mcp.conversation",
  "doc": "Event emitted when a conversation is completed",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Unique identifier for the conversation"
    },
    {
      "name": "completion_reason",
      "type": ["null", "string"],
      "default": null,
      "doc": "Reason for conversation completion"
    },
    {
      "name": "summary",
      "type": ["null", "string"],
      "default": null,
      "doc": "Optional summary of the conversation"
    },
    {
      "name": "metrics",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null"]
      }],
      "default": null,
      "doc": "Metrics collected during the conversation"
    }
  ]
}
```

### Message Schemas

#### Message Created Event

```json
{
  "type": "record",
  "name": "MessageCreatedEvent",
  "namespace": "com.agent_platform.events.mcp.message",
  "doc": "Event emitted when a new message is created",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "message_id",
      "type": "string",
      "doc": "Unique identifier for the message"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Identifier of the conversation the message belongs to"
    },
    {
      "name": "role",
      "type": {
        "type": "enum",
        "name": "MessageRole",
        "symbols": ["SYSTEM", "USER", "ASSISTANT", "FUNCTION", "TOOL"]
      },
      "doc": "Role of the message sender"
    },
    {
      "name": "content",
      "type": "string",
      "doc": "Content of the message"
    },
    {
      "name": "parent_message_id",
      "type": ["null", "string"],
      "default": null,
      "doc": "Identifier of the parent message"
    },
    {
      "name": "message_type",
      "type": {
        "type": "enum",
        "name": "MessageType",
        "symbols": ["TEXT", "TOOL_CALL", "TOOL_RESULT", "FUNCTION_CALL", "FUNCTION_RESULT", "IMAGE", "AUDIO", "VIDEO", "FILE"]
      },
      "doc": "Type of the message content"
    },
    {
      "name": "metadata",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null"]
      }],
      "default": null,
      "doc": "Additional metadata for the message"
    }
  ]
}
```

#### Tool Call Schema

```json
{
  "type": "record",
  "name": "ToolCall",
  "namespace": "com.agent_platform.events.mcp.message",
  "doc": "Schema for a tool call within a message",
  "fields": [
    {
      "name": "tool_call_id",
      "type": "string",
      "doc": "Unique identifier for the tool call"
    },
    {
      "name": "tool_name",
      "type": "string",
      "doc": "Name of the tool being called"
    },
    {
      "name": "parameters",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null", {
          "type": "array",
          "items": ["string", "int", "long", "float", "double", "boolean", "null"]
        }, {
          "type": "map",
          "values": ["string", "int", "long", "float", "double", "boolean", "null"]
        }]
      }],
      "default": null,
      "doc": "Parameters for the tool call"
    }
  ]
}
```

#### Tool Call Message Event

```json
{
  "type": "record",
  "name": "ToolCallMessageEvent",
  "namespace": "com.agent_platform.events.mcp.message",
  "doc": "Event emitted when a message contains tool calls",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "message_id",
      "type": "string",
      "doc": "Unique identifier for the message"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Identifier of the conversation the message belongs to"
    },
    {
      "name": "tool_calls",
      "type": {
        "type": "array",
        "items": "com.agent_platform.events.mcp.message.ToolCall"
      },
      "doc": "List of tool calls in the message"
    }
  ]
}
```

#### Generation Started Event

```json
{
  "type": "record",
  "name": "GenerationStartedEvent",
  "namespace": "com.agent_platform.events.mcp.message",
  "doc": "Event emitted when message generation starts",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Identifier of the conversation"
    },
    {
      "name": "generation_id",
      "type": "string",
      "doc": "Unique identifier for this generation"
    },
    {
      "name": "model",
      "type": "string",
      "doc": "Model being used for generation"
    },
    {
      "name": "parameters",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null"]
      }],
      "default": null,
      "doc": "Generation parameters"
    }
  ]
}
```

#### Generation Completed Event

```json
{
  "type": "record",
  "name": "GenerationCompletedEvent",
  "namespace": "com.agent_platform.events.mcp.message",
  "doc": "Event emitted when message generation completes",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Identifier of the conversation"
    },
    {
      "name": "generation_id",
      "type": "string",
      "doc": "Unique identifier for this generation"
    },
    {
      "name": "message_id",
      "type": "string",
      "doc": "Identifier of the generated message"
    },
    {
      "name": "usage",
      "type": ["null", {
        "type": "record",
        "name": "TokenUsage",
        "fields": [
          {
            "name": "prompt_tokens",
            "type": "int",
            "doc": "Number of tokens in the prompt"
          },
          {
            "name": "completion_tokens",
            "type": "int",
            "doc": "Number of tokens in the completion"
          },
          {
            "name": "total_tokens",
            "type": "int",
            "doc": "Total number of tokens used"
          }
        ]
      }],
      "default": null,
      "doc": "Token usage information"
    }
  ]
}
```

### Capability Schemas

#### Capability Invoked Event

```json
{
  "type": "record",
  "name": "CapabilityInvokedEvent",
  "namespace": "com.agent_platform.events.mcp.capability",
  "doc": "Event emitted when a capability is invoked",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Identifier of the conversation"
    },
    {
      "name": "message_id",
      "type": "string",
      "doc": "Identifier of the message containing the capability invocation"
    },
    {
      "name": "capability_id",
      "type": "string",
      "doc": "Identifier of the capability being invoked"
    },
    {
      "name": "capability_name",
      "type": "string",
      "doc": "Name of the capability"
    },
    {
      "name": "parameters",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null", {
          "type": "array",
          "items": ["string", "int", "long", "float", "double", "boolean", "null"]
        }, {
          "type": "map",
          "values": ["string", "int", "long", "float", "double", "boolean", "null"]
        }]
      }],
      "default": null,
      "doc": "Parameters for the capability invocation"
    }
  ]
}
```

#### Capability Completed Event

```json
{
  "type": "record",
  "name": "CapabilityCompletedEvent",
  "namespace": "com.agent_platform.events.mcp.capability",
  "doc": "Event emitted when a capability execution completes",
  "fields": [
    {
      "name": "base",
      "type": "com.agent_platform.events.mcp.BaseEvent",
      "doc": "Base event fields"
    },
    {
      "name": "conversation_id",
      "type": "string",
      "doc": "Identifier of the conversation"
    },
    {
      "name": "message_id",
      "type": "string",
      "doc": "Identifier of the message containing the capability invocation"
    },
    {
      "name": "capability_id",
      "type": "string",
      "doc": "Identifier of the capability"
    },
    {
      "name": "result",
      "type": ["null", {
        "type": "map",
        "values": ["string", "int", "long", "float", "double", "boolean", "null", {
          "type": "array",
          "items": ["string", "int", "long", "float", "double", "boolean", "null"]
        }, {
          "type": "map",
          "values": ["string", "int", "long", "float", "double", "boolean", "null"]
        }]
      }],
      "default": null,
      "doc": "Result of the capability execution"
    },
    {
      "name": "status",
      "type": {
        "type": "enum",
        "name": "CapabilityStatus",
        "symbols": ["SUCCESS", "ERROR", "PARTIAL_SUCCESS"]
      },
      "doc": "Status of the capability execution"
    },
    {
      "name": "duration_ms",
      "type": "long",
      "doc": "Duration of capability execution in milliseconds"
    },
    {
      "name": "error",
      "type": ["null", {
        "type": "record",
        "name": "CapabilityError",
        "fields": [
          {
            "name": "error_code",
            "type": "string",
            "doc": "Error code"
          },
          {
            "name": "error_message",
            "type": "string",
            "doc": "Error message"
          },
          {
            "name": "error_details",
            "type": ["null", {
              "type": "map",
              "values": ["string", "int", "long", "float", "double", "boolean", "null"]
            }],
            "default": null,
            "doc": "Additional error details"
          }
        ]
      }],
      "default": null,
      "doc": "Error information if capability execution failed"
    }
  ]
}
```

## Subject Naming Convention

When registering schemas with the Schema Registry, use the following subject naming convention:

- For key schemas: `{topic_name}-key`
- For value schemas: `{topic_name}-value`

Common MCP topic patterns:

- `mcp.conversation.events` - All conversation events
- `mcp.message.events` - All message events
- `mcp.capability.events` - All capability events
- `mcp.agent.events` - All agent events

## Schema Compatibility Settings

The Schema Registry is configured with the following compatibility settings for MCP events:

| Subject | Compatibility Level | Notes |
|---------|---------------------|-------|
| Default | BACKWARD | Ensures consumers can read new data with old schemas |
| `mcp.*.events-value` | BACKWARD | Standard for all MCP event values |
| `mcp.*.events-key` | FULL | Stricter compatibility for event keys |

## Schema Registry Roles

1. **Schema Development**
   - **Who**: Platform developers
   - **Responsibility**: Creating and evolving schemas
   - **Tools**: IDE with Avro plugins, schema registry UI

2. **Schema Review**
   - **Who**: Platform architects
   - **Responsibility**: Reviewing schema changes for compatibility
   - **Tools**: Schema Registry REST API, compatibility testing tools

3. **Schema Deployment**
   - **Who**: DevOps team
   - **Responsibility**: Deploying schema changes to production
   - **Tools**: CI/CD pipelines, schema registry CLI

## Integration with MCP Service

```python
from agent_orchestration.schemas.schema_registry import SchemaRegistryService
from agent_orchestration.mcp.models import Conversation, Message
from agent_orchestration.events.publisher import EventPublisher
import uuid
import time

class MCPService:
    """Service for interacting with the Model Context Protocol."""
    
    def __init__(
        self,
        config_service,
        schema_registry_service: SchemaRegistryService,
        event_publisher: EventPublisher
    ):
        """Initialize the MCP service.
        
        Args:
            config_service: Configuration service
            schema_registry_service: Schema Registry service
            event_publisher: Event publisher
        """
        self.config_service = config_service
        self.schema_registry = schema_registry_service
        self.event_publisher = event_publisher
    
    async def create_conversation(self, user_id: str, agent_id: str, system_prompt: str, metadata=None):
        """Create a new conversation.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            system_prompt: System prompt for the conversation
            metadata: Optional metadata
            
        Returns:
            Conversation identifier
        """
        # Create conversation in MCP
        conversation_id = str(uuid.uuid4())
        
        # Build event data
        event_data = {
            "base": {
                "event_id": str(uuid.uuid4()),
                "event_type": "conversation.created",
                "version": "1.0.0",
                "timestamp": int(time.time() * 1000),
                "source": "mcp_service",
                "trace_id": None,
                "metadata": None
            },
            "conversation_id": conversation_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "title": None,
            "system_prompt": system_prompt,
            "conversation_metadata": metadata
        }
        
        # Serialize and publish event
        try:
            serialized_data = await self.schema_registry.serialize(
                "mcp.conversation.events-value",
                event_data
            )
            
            await self.event_publisher.publish(
                topic="mcp.conversation.events",
                key=conversation_id,
                value=serialized_data
            )
            
            return conversation_id
            
        except Exception as e:
            from agent_orchestration.errors.mcp_errors import ConversationCreationError
            raise ConversationCreationError(f"Failed to create conversation: {str(e)}")
```

## Testing Schema Integration

```python
import unittest
from unittest.mock import AsyncMock, MagicMock
from agent_orchestration.schemas.schema_registry import SchemaRegistryService
from agent_orchestration.mcp.service import MCPService
from agent_orchestration.events.publisher import EventPublisher

class TestMCPSchemaIntegration(unittest.TestCase):
    """Test MCP Schema Registry integration."""
    
    def setUp(self):
        """Set up test case."""
        # Mock dependencies
        self.config_service = AsyncMock()
        self.schema_registry = AsyncMock(spec=SchemaRegistryService)
        self.event_publisher = AsyncMock(spec=EventPublisher)
        
        # Create MCP service
        self.mcp_service = MCPService(
            self.config_service,
            self.schema_registry,
            self.event_publisher
        )
    
    async def test_create_conversation_publishes_valid_event(self):
        """Test that create_conversation publishes a valid event."""
        # Arrange
        user_id = "user123"
        agent_id = "agent456"
        system_prompt = "You are a helpful assistant"
        
        # Mock schema registry
        self.schema_registry.serialize.return_value = b"serialized_data"
        
        # Act
        conversation_id = await self.mcp_service.create_conversation(
            user_id, agent_id, system_prompt
        )
        
        # Assert
        self.schema_registry.serialize.assert_called_once()
        self.event_publisher.publish.assert_called_once_with(
            topic="mcp.conversation.events",
            key=conversation_id,
            value=b"serialized_data"
        )
    
    async def test_handle_schema_validation_error(self):
        """Test handling of schema validation errors."""
        # Arrange
        user_id = "user123"
        agent_id = "agent456"
        system_prompt = "You are a helpful assistant"
        
        # Mock schema registry to raise error
        from agent_orchestration.errors.schema_errors import SerializationError
        self.schema_registry.serialize.side_effect = SerializationError("Invalid schema")
        
        # Act/Assert
        from agent_orchestration.errors.mcp_errors import ConversationCreationError
        with self.assertRaises(ConversationCreationError):
            await self.mcp_service.create_conversation(
                user_id, agent_id, system_prompt
            )
```
