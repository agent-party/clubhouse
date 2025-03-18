# MCP Tools Integration

## Overview

This document outlines how the Agent Orchestration Platform leverages the Multi-agent Capabilities Platform (MCP) and Kafka Schema Registry to implement a consistent, tool-based approach to agent capabilities. By treating capabilities as standardized tools, we enable better interoperability, observability, and evolvability across the agent ecosystem.

## Core Principles

1. **Capabilities as Tools**: Every agent capability is implemented as a well-defined tool with standardized interfaces
2. **Schema-First Development**: All messages and data structures are defined in the Schema Registry before implementation
3. **Event-Driven Communications**: All tool invocations and responses follow the event-driven pattern
4. **Consistent Validation**: Schemas enforce consistent validation across all components
5. **Traceable Operations**: All tool invocations are tracked for observability and debugging

## MCP Integration Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      Agent Orchestration Platform                           │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────────────┐ │
│  │                 │   │                 │   │                           │ │
│  │  Agent Layer    │◄──┤  Capability     │◄──┤  MCP Tools Registry       │ │
│  │                 │   │  Layer          │   │                           │ │
│  └─────────────────┘   └─────────────────┘   └───────────────────────────┘ │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────────────┐ │
│  │                 │   │                 │   │                           │ │
│  │  Kafka Message  │◄──┤  Schema         │◄──┤  Tool Schema Definitions  │ │
│  │  Bus            │   │  Registry       │   │                           │ │
│  └─────────────────┘   └─────────────────┘   └───────────────────────────┘ │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────────────┐ │
│  │                 │   │                 │   │                           │ │
│  │  Neo4j          │◄──┤  Collective     │◄──┤  Tool Usage Patterns      │ │
│  │  Graph DB       │   │  Unconscious    │   │                           │ │
│  └─────────────────┘   └─────────────────┘   └───────────────────────────┘ │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Tool-Based Capability Implementation

### 1. Tool Schema Definition with Kafka Schema Registry

All tool schemas are defined in Avro format and registered with the Schema Registry:

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime

# Base tool invocation message
class ToolInvocationSchema(BaseModel):
    """Schema for tool invocation messages."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    parameters: Dict[str, Any]
    trace_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "schema_name": "tool_invocation",
            "namespace": "com.agentorchestration.tools",
            "doc": "Schema for agent tool invocation messages"
        }

# Tool response message
class ToolResponseSchema(BaseModel):
    """Schema for tool response messages."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    invocation_id: str
    tool_name: str
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str  # "success", "error", "partial"
    result: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "schema_name": "tool_response",
            "namespace": "com.agentorchestration.tools",
            "doc": "Schema for agent tool response messages"
        }
```

### 2. Schema Registry Service

The system uses a SchemaRegistryService to register and validate all schemas:

```python
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
from typing import Type, Dict, Any, Optional

class SchemaRegistryService:
    """Service for interacting with the Kafka Schema Registry."""
    
    def __init__(self, config: Dict[str, str]):
        """Initialize with Schema Registry configuration."""
        self.schema_registry_client = SchemaRegistryClient(config)
        self.serializers: Dict[str, AvroSerializer] = {}
        self.deserializers: Dict[str, AvroDeserializer] = {}
        
    def register_schema(self, schema_model: Type[BaseModel]) -> str:
        """Register a Pydantic model schema with the Schema Registry."""
        # Extract schema information
        config = schema_model.Config.schema_extra
        schema_name = config["schema_name"]
        namespace = config.get("namespace", "com.agentorchestration")
        
        # Convert Pydantic schema to Avro schema
        avro_schema = self._convert_to_avro_schema(schema_model)
        
        # Register schema
        schema_id = self.schema_registry_client.register_schema(
            f"{namespace}.{schema_name}",
            avro_schema
        )
        
        # Create and cache serializer/deserializer
        self.serializers[schema_name] = AvroSerializer(
            avro_schema,
            self.schema_registry_client,
            to_dict=lambda obj, ctx: obj.dict()
        )
        
        self.deserializers[schema_name] = AvroDeserializer(
            avro_schema,
            self.schema_registry_client,
            from_dict=lambda obj, ctx: schema_model(**obj)
        )
        
        return schema_id
        
    def serialize(self, schema_name: str, data: BaseModel) -> bytes:
        """Serialize data according to registered schema."""
        if schema_name not in self.serializers:
            raise ValueError(f"Schema {schema_name} not registered")
            
        return self.serializers[schema_name](data)
        
    def deserialize(self, schema_name: str, data: bytes) -> BaseModel:
        """Deserialize data according to registered schema."""
        if schema_name not in self.deserializers:
            raise ValueError(f"Schema {schema_name} not registered")
            
        return self.deserializers[schema_name](data)
        
    def _convert_to_avro_schema(self, schema_model: Type[BaseModel]) -> str:
        """Convert Pydantic model to Avro schema."""
        # Implementation for converting Pydantic to Avro
        # ...
```

### 3. Tool Registry

The MCP Tools Registry manages all available tools and their invocation patterns:

```python
from typing import Dict, List, Type, Callable, Awaitable, Any
from pydantic import BaseModel

class ToolDefinition:
    """Definition of a tool in the MCP Tools Registry."""
    
    def __init__(
        self,
        tool_name: str,
        description: str,
        parameter_schema: Type[BaseModel],
        result_schema: Type[BaseModel],
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        required_capabilities: List[str] = None,
        cost: Optional[Dict[str, float]] = None
    ):
        """Initialize tool definition."""
        self.tool_name = tool_name
        self.description = description
        self.parameter_schema = parameter_schema
        self.result_schema = result_schema
        self.handler = handler
        self.required_capabilities = required_capabilities or []
        self.cost = cost or {"compute": 0.0, "storage": 0.0, "network": 0.0}

class MCPToolsRegistry:
    """Registry for MCP tools."""
    
    def __init__(self, schema_registry_service: SchemaRegistryService):
        """Initialize with schema registry service."""
        self.schema_registry = schema_registry_service
        self.tools: Dict[str, ToolDefinition] = {}
        
    def register_tool(self, tool_definition: ToolDefinition) -> None:
        """Register a tool with the registry."""
        # Register parameter and result schemas
        self.schema_registry.register_schema(tool_definition.parameter_schema)
        self.schema_registry.register_schema(tool_definition.result_schema)
        
        # Store tool definition
        self.tools[tool_definition.tool_name] = tool_definition
        
    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: str,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Invoke a tool with parameters."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
            
        tool = self.tools[tool_name]
        
        # Validate parameters
        validated_parameters = tool.parameter_schema(**parameters)
        
        # Create invocation message
        invocation = ToolInvocationSchema(
            tool_name=tool_name,
            agent_id=agent_id,
            parameters=validated_parameters.dict(),
            trace_id=trace_id
        )
        
        # Log invocation
        await self._log_tool_invocation(invocation)
        
        try:
            # Execute tool handler
            result = await tool.handler(validated_parameters.dict())
            
            # Validate result
            validated_result = tool.result_schema(**result)
            
            # Create response message
            response = ToolResponseSchema(
                invocation_id=invocation.message_id,
                tool_name=tool_name,
                agent_id=agent_id,
                status="success",
                result=validated_result.dict(),
                trace_id=trace_id
            )
            
            # Log response
            await self._log_tool_response(response)
            
            return validated_result.dict()
            
        except Exception as e:
            # Create error response
            error_response = ToolResponseSchema(
                invocation_id=invocation.message_id,
                tool_name=tool_name,
                agent_id=agent_id,
                status="error",
                result={},
                error={"type": type(e).__name__, "message": str(e)},
                trace_id=trace_id
            )
            
            # Log error response
            await self._log_tool_response(error_response)
            
            # Re-raise exception
            raise
            
    async def _log_tool_invocation(self, invocation: ToolInvocationSchema) -> None:
        """Log tool invocation to observability system."""
        # Implementation for logging invocations
        # ...
        
    async def _log_tool_response(self, response: ToolResponseSchema) -> None:
        """Log tool response to observability system."""
        # Implementation for logging responses
        # ...
        
    def get_tool_description(self, tool_name: str) -> Dict[str, Any]:
        """Get tool description for agent prompting."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
            
        tool = self.tools[tool_name]
        
        # Generate parameter descriptions from schema
        parameter_schema = self._get_schema_description(tool.parameter_schema)
        result_schema = self._get_schema_description(tool.result_schema)
        
        return {
            "tool_name": tool.tool_name,
            "description": tool.description,
            "parameters": parameter_schema,
            "result": result_schema,
            "required_capabilities": tool.required_capabilities,
            "cost": tool.cost
        }
        
    def _get_schema_description(self, schema: Type[BaseModel]) -> Dict[str, Any]:
        """Generate a description of schema fields for agent prompting."""
        # Implementation for schema description generation
        # ...
```

## Refactoring SummarizeCapability as a Tool

Following our memory about the SummarizeCapability needing refactoring, here's how it would be implemented as a tool:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Parameter schema
class SummarizeParameters(BaseModel):
    """Parameters for summarize capability."""
    
    text: str = Field(..., description="Text to summarize")
    max_length: Optional[int] = Field(None, description="Maximum length of summary")
    focus_areas: Optional[List[str]] = Field(None, description="Areas to focus on in summary")
    format: Optional[str] = Field("paragraph", description="Format of the summary")
    
    class Config:
        schema_extra = {
            "schema_name": "summarize_parameters",
            "namespace": "com.agentorchestration.capabilities",
            "doc": "Parameters for the summarize capability"
        }

# Result schema
class SummarizeResult(BaseModel):
    """Result of summarize capability."""
    
    summary: str = Field(..., description="Generated summary")
    length: int = Field(..., description="Length of summary in characters")
    focus_coverage: Optional[Dict[str, float]] = Field(None, description="Coverage score for each focus area")
    
    class Config:
        schema_extra = {
            "schema_name": "summarize_result",
            "namespace": "com.agentorchestration.capabilities",
            "doc": "Result of the summarize capability"
        }

# Handler implementation
class SummarizeCapability(BaseCapability):
    """Capability for generating summaries of text."""
    
    def __init__(self, agent: AgentProtocol, llm_service: LLMServiceProtocol):
        """Initialize with agent and LLM service."""
        self.agent = agent
        self.llm_service = llm_service
        
        # Register with MCP Tool Registry
        tool_registry = ServiceRegistry.get_service(MCPToolsRegistry)
        tool_registry.register_tool(ToolDefinition(
            tool_name="summarize",
            description="Generate a concise summary of provided text",
            parameter_schema=SummarizeParameters,
            result_schema=SummarizeResult,
            handler=self.execute,
            required_capabilities=["text_processing"],
            cost={"compute": 0.01, "storage": 0.0, "network": 0.0}
        ))
        
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the summarize capability."""
        # For compatibility with existing tests that expect these events
        await self.agent.emit_event("summarize_started", {
            "text_length": len(parameters["text"])
        })
        
        # Validate parameters
        params = SummarizeParameters(**parameters)
        
        # Apply system prompt based on parameters
        system_prompt = self._create_system_prompt(params)
        
        # Generate summary using LLM
        summary_text = await self.llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=params.text,
            max_tokens=params.max_length if params.max_length else 500,
            temperature=0.5
        )
        
        # Process focus areas if specified
        focus_coverage = None
        if params.focus_areas:
            focus_coverage = await self._analyze_focus_coverage(summary_text, params.focus_areas)
            
        # Create result
        result = SummarizeResult(
            summary=summary_text,
            length=len(summary_text),
            focus_coverage=focus_coverage
        )
        
        # For compatibility with existing tests that expect these events
        await self.agent.emit_event("summarize_completed", {
            "summary_length": len(summary_text)
        })
        
        return result.dict()
        
    def _create_system_prompt(self, params: SummarizeParameters) -> str:
        """Create system prompt based on parameters."""
        prompt = "You are an expert summarizer. Create a concise summary of the text."
        
        if params.max_length:
            prompt += f" The summary should be no longer than {params.max_length} characters."
            
        if params.focus_areas:
            areas = ", ".join(params.focus_areas)
            prompt += f" Focus particularly on these aspects: {areas}."
            
        if params.format:
            prompt += f" Format the summary as a {params.format}."
            
        return prompt
        
    async def _analyze_focus_coverage(
        self,
        summary: str,
        focus_areas: List[str]
    ) -> Dict[str, float]:
        """Analyze how well the summary covers each focus area."""
        # Implementation of focus coverage analysis
        # ...
        return {area: 0.8 for area in focus_areas}  # Placeholder
```

## Kafka Integration for Tool Communications

Tool communications are handled through Kafka:

```python
from confluent_kafka import Producer, Consumer
from typing import Callable, Dict, Any, Optional

class ToolCommunicationService:
    """Service for tool-based communication via Kafka."""
    
    def __init__(
        self,
        kafka_config: Dict[str, str],
        schema_registry_service: SchemaRegistryService
    ):
        """Initialize with Kafka configuration and schema registry."""
        self.producer = Producer(kafka_config)
        self.consumer = Consumer(kafka_config)
        self.schema_registry = schema_registry_service
        
        # Subscribe to tool topics
        self.consumer.subscribe([
            "tools.invocations", 
            "tools.responses"
        ])
        
    async def invoke_remote_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: str,
        trace_id: Optional[str] = None
    ) -> str:
        """Invoke a tool remotely via Kafka."""
        # Create invocation message
        invocation = ToolInvocationSchema(
            tool_name=tool_name,
            agent_id=agent_id,
            parameters=parameters,
            trace_id=trace_id
        )
        
        # Serialize message
        serialized = self.schema_registry.serialize("tool_invocation", invocation)
        
        # Produce to Kafka
        self.producer.produce(
            topic="tools.invocations",
            key=invocation.message_id,
            value=serialized
        )
        self.producer.flush()
        
        return invocation.message_id
        
    def register_response_handler(
        self,
        handler: Callable[[ToolResponseSchema], None]
    ) -> None:
        """Register a handler for tool responses."""
        self._response_handler = handler
        
    def start_consuming(self) -> None:
        """Start consuming tool messages."""
        # Implementation of Kafka message consumption
        # ...
        
    def _handle_message(self, message: Any) -> None:
        """Handle incoming Kafka message."""
        # Implementation of message handling
        # ...
```

## Integration with Collective Unconscious

The tool-based approach integrates with our collective unconscious:

```python
class KnowledgeToolIntegration:
    """Integration between MCP tools and the collective unconscious."""
    
    def __init__(
        self,
        tool_registry: MCPToolsRegistry,
        neo4j_service: Neo4jServiceProtocol
    ):
        """Initialize with tool registry and Neo4j service."""
        self.tool_registry = tool_registry
        self.neo4j_service = neo4j_service
        
    async def register_tool_patterns(self) -> None:
        """Register tool usage patterns in collective unconscious."""
        for tool_name, tool in self.tool_registry.tools.items():
            await self._register_tool_knowledge(tool)
            
    async def _register_tool_knowledge(self, tool: ToolDefinition) -> None:
        """Register knowledge about a tool in Neo4j."""
        # Create tool knowledge node
        query = """
        MERGE (t:Tool {name: $name})
        SET t.description = $description,
            t.updated_at = datetime()
        RETURN t
        """
        
        await self.neo4j_service.execute_query(query, {
            "name": tool.tool_name,
            "description": tool.description
        })
        
        # Connect to relevant knowledge domains
        for capability in tool.required_capabilities:
            query = """
            MATCH (t:Tool {name: $tool_name})
            MATCH (c:Capability {name: $capability})
            MERGE (t)-[:REQUIRES]->(c)
            """
            
            await self.neo4j_service.execute_query(query, {
                "tool_name": tool.tool_name,
                "capability": capability
            })
            
    async def track_tool_usage(
        self,
        tool_name: str,
        agent_id: str,
        success: bool,
        parameters: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track tool usage in the collective unconscious."""
        # Record tool usage
        query = """
        MATCH (t:Tool {name: $tool_name})
        MATCH (a:Agent {agent_id: $agent_id})
        CREATE (a)-[:USED {
            timestamp: datetime(),
            success: $success,
            parameters: $parameters,
            result: $result
        }]->(t)
        """
        
        await self.neo4j_service.execute_query(query, {
            "tool_name": tool_name,
            "agent_id": agent_id,
            "success": success,
            "parameters": str(parameters),  # Convert to string for Neo4j
            "result": str(result) if result else None
        })
        
        # Update usage statistics
        query = """
        MATCH (t:Tool {name: $tool_name})
        SET t.usage_count = COALESCE(t.usage_count, 0) + 1,
            t.success_count = COALESCE(t.success_count, 0) + $success_increment
        """
        
        await self.neo4j_service.execute_query(query, {
            "tool_name": tool_name,
            "success_increment": 1 if success else 0
        })
        
    async def get_tool_suggestions(
        self,
        agent_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get tool suggestions based on collective knowledge."""
        # Query for relevant tools based on context
        # ...
        
        return []  # Placeholder
```

## Agent Prompt Integration

Tool descriptions are integrated into agent system prompts:

```python
class ToolPromptIntegration:
    """Integrates tool descriptions into agent prompts."""
    
    def __init__(self, tool_registry: MCPToolsRegistry):
        """Initialize with tool registry."""
        self.tool_registry = tool_registry
        
    def generate_tool_prompt_section(
        self,
        agent_id: str,
        available_tools: List[str]
    ) -> str:
        """Generate a tool section for an agent system prompt."""
        tool_descriptions = []
        
        for tool_name in available_tools:
            if tool_name in self.tool_registry.tools:
                desc = self.tool_registry.get_tool_description(tool_name)
                tool_descriptions.append(self._format_tool_description(desc))
                
        if not tool_descriptions:
            return ""
            
        return f"""
        You have access to the following tools:
        
        {' '.join(tool_descriptions)}
        
        To use a tool, respond with:
        {{
            "tool": "<tool_name>",
            "parameters": {{
                // tool parameters
            }}
        }}
        """
        
    def _format_tool_description(self, description: Dict[str, Any]) -> str:
        """Format a tool description for inclusion in a prompt."""
        params = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in description["parameters"].items()
        ])
        
        return f"""
        Tool: {description["tool_name"]}
        Description: {description["description"]}
        Parameters:
        {params}
        """
```

## Testing Strategy

Following our test-driven development guidelines, here's a testing approach for the tool system:

```python
import unittest
from unittest.mock import MagicMock, patch
import asyncio

class TestMCPToolsRegistry(unittest.TestCase):
    """Tests for the MCP Tools Registry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.schema_registry = MagicMock(spec=SchemaRegistryService)
        self.tool_registry = MCPToolsRegistry(self.schema_registry)
        
        # Create a dummy tool handler
        async def dummy_handler(params):
            return {"result": "Success"}
            
        # Create a tool definition
        self.tool_def = ToolDefinition(
            tool_name="test_tool",
            description="A test tool",
            parameter_schema=SummarizeParameters,
            result_schema=SummarizeResult,
            handler=dummy_handler
        )
        
    def test_register_tool(self):
        """Test tool registration."""
        # Register the tool
        self.tool_registry.register_tool(self.tool_def)
        
        # Verify schema registration
        self.schema_registry.register_schema.assert_any_call(SummarizeParameters)
        self.schema_registry.register_schema.assert_any_call(SummarizeResult)
        
        # Verify tool is registered
        self.assertIn("test_tool", self.tool_registry.tools)
        self.assertEqual(self.tool_registry.tools["test_tool"], self.tool_def)
        
    async def test_invoke_tool(self):
        """Test tool invocation."""
        # Register the tool
        self.tool_registry.register_tool(self.tool_def)
        
        # Mock the log methods
        self.tool_registry._log_tool_invocation = MagicMock()
        self.tool_registry._log_tool_response = MagicMock()
        
        # Invoke the tool
        result = await self.tool_registry.invoke_tool(
            "test_tool",
            {"text": "Test text"},
            "agent-123"
        )
        
        # Verify result
        self.assertEqual(result, {"result": "Success"})
        
        # Verify logging
        self.tool_registry._log_tool_invocation.assert_called_once()
        self.tool_registry._log_tool_response.assert_called_once()
```

## Evolving New Tools

The system supports evolving new tools through the collective unconscious:

```python
class ToolEvolution:
    """Supports evolution of new tools from usage patterns."""
    
    async def discover_tool_patterns(self) -> List[Dict[str, Any]]:
        """Discover potential new tool patterns from usage."""
        # Implementation of pattern discovery
        # ...
        
    async def suggest_tool_improvements(
        self,
        tool_name: str
    ) -> Dict[str, Any]:
        """Suggest improvements for an existing tool."""
        # Implementation of improvement suggestions
        # ...
        
    async def generate_tool_prototype(
        self,
        pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a prototype for a new tool based on a pattern."""
        # Implementation of tool prototype generation
        # ...
```

## Conclusion

The MCP Tools Integration provides a comprehensive approach to implementing capability-based agents:

1. **Schema-First Validation**: All tools use Kafka Schema Registry for consistent validation
2. **Standardized Interfaces**: Uniform invocation and response patterns for all capabilities
3. **Observability**: Complete tracking of tool usage for monitoring and improvement
4. **Collective Knowledge**: Integration with the shared knowledge structure
5. **Evolvability**: Framework for discovering and evolving new tools

This approach ensures that all capabilities across the Agent Orchestration Platform follow consistent patterns while allowing for specialized behaviors based on agent personalities and roles.
