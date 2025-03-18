# Model Context Protocol Integration Plan

## Overview

This document outlines the strategic approach for integrating the Model Context Protocol (MCP) into the Clubhouse platform. The integration will standardize how AI agents communicate with human users and access system capabilities, following our core principles of real infrastructure usage, human-AI partnership, and test-driven development.

## Current Architecture Analysis

### Key Components

1. **Messaging Architecture**
   - Kafka-based message bus for commands and events
   - Command/Response pattern with standardized message formats
   - Pydantic models for message validation

2. **Service Architecture**
   - Service Registry for dependency management
   - Repository pattern for data access
   - Neo4j for graph database operations

3. **Agent Capabilities**
   - Graph traversal for knowledge representation
   - Conversation management
   - Command processing

## MCP Integration Strategy

### 1. Core Architecture

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│                 │     │               │     │                 │
│  MCP Server     │<--->│ Kafka         │<--->│ Clubhouse       │
│  (FastMCP)      │     │ Message Bus   │     │ Services        │
│                 │     │               │     │                 │
└─────────────────┘     └───────────────┘     └─────────────────┘
```

### 2. Implementation Components

#### A. MCP Server Implementation

Create a dedicated `MCPServerService` that implements the MCP protocol while integrating with our existing Kafka infrastructure.

```python
# clubhouse/services/mcp/server.py

from mcp.server.fastmcp import FastMCP, Context
from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.services.kafka_protocol import KafkaProducerProtocol

class ClubhouseMCPServer:
    """MCP Server implementation for the Clubhouse platform."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.server = FastMCP(
            name="ClubhouseServer",
            instructions="Interact with the Agent Clubhouse platform"
        )
        
        # Register MCP capabilities
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        """Register MCP tools that map to Clubhouse capabilities."""
        
        @self.server.tool(
            name="create_agent",
            description="Create a new agent with specified configuration"
        )
        async def create_agent(name: str, config: dict, ctx: Context) -> dict:
            # Get Kafka producer from service registry
            producer = self.service_registry.get_protocol(KafkaProducerProtocol)
            
            # Create command message
            command = {
                "message_type": "CreateAgentCommand",
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "payload": {
                    "name": name,
                    "config": config
                }
            }
            
            # Report progress to client
            ctx.report_progress(25, 100)
            ctx.info(f"Creating agent '{name}'...")
            
            # Send command and wait for response
            response = await producer.produce_and_wait_for_response(
                topic="clubhouse-commands",
                value=command,
                response_topic="clubhouse-responses",
                timeout=30
            )
            
            ctx.report_progress(100, 100)
            return response
```

#### B. Kafka-MCP Bridge

Extend the existing Kafka event system to support MCP messaging patterns.

```python
# clubhouse/messaging/mcp_bridge.py

from typing import Dict, Any, Optional
import asyncio
from mcp.types import Request, Response
from clubhouse.messaging.event_publisher import EventPublisher
from clubhouse.messaging.message_router import MessageRouter

class MCPKafkaBridge:
    """Bridge between MCP protocol and Kafka messaging."""
    
    def __init__(
        self,
        event_publisher: EventPublisher,
        message_router: MessageRouter
    ):
        self.event_publisher = event_publisher
        self.message_router = message_router
        self.response_futures: Dict[str, asyncio.Future] = {}
    
    async def handle_mcp_request(self, request: Request) -> Response:
        """Convert MCP request to Kafka message and wait for response."""
        request_id = str(uuid.uuid4())
        
        # Create future for response
        response_future = asyncio.Future()
        self.response_futures[request_id] = response_future
        
        # Convert MCP request to Kafka message
        kafka_message = {
            "message_type": f"{request.method}Command",
            "message_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": request.params
        }
        
        # Publish message
        self.event_publisher.publish_event(
            kafka_message, 
            topic="clubhouse-commands"
        )
        
        # Wait for response
        try:
            response = await asyncio.wait_for(response_future, timeout=60)
            return Response(
                id=request_id,
                result=response["payload"]
            )
        except asyncio.TimeoutError:
            return Response(
                id=request_id,
                error={
                    "code": -32000,
                    "message": "Request timed out"
                }
            )
    
    def handle_kafka_response(self, response: Dict[str, Any]) -> None:
        """Handle Kafka response and resolve corresponding future."""
        request_id = response.get("in_response_to")
        if request_id in self.response_futures:
            future = self.response_futures.pop(request_id)
            if not future.done():
                future.set_result(response)
```

#### C. Neo4j Resource Access

Implement MCP resources that provide access to graph data:

```python
# clubhouse/services/mcp/resources.py

from mcp.server.fastmcp import FastMCP
from clubhouse.services.neo4j.repositories.graph_traversal import GraphTraversalRepository

def register_neo4j_resources(server: FastMCP, graph_repo: GraphTraversalRepository):
    """Register Neo4j resources with the MCP server."""
    
    @server.resource("resource://graph/{node_id}")
    async def get_node(node_id: str) -> dict:
        """Get a node from the graph by ID."""
        # This will be exposed as a resource that can be referenced by AI models
        node = await graph_repo.get_node(node_id)
        return {
            "id": node_id,
            "properties": node,
            "resource_type": "node"
        }
    
    @server.resource("resource://subgraph/{root_id}")
    async def get_subgraph(root_id: str, max_depth: int = 2) -> dict:
        """Get a subgraph starting from a root node."""
        subgraph = await graph_repo.get_subgraph(
            root_id=root_id,
            max_depth=max_depth
        )
        return {
            "root_id": root_id,
            "nodes": subgraph["nodes"],
            "relationships": subgraph["relationships"],
            "resource_type": "subgraph"
        }
```

### 3. Schema Definitions

#### A. MCP Tool Schemas

```python
# clubhouse/schemas/mcp_tools.py

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class CreateAgentParams(BaseModel):
    """Parameters for the create_agent tool."""
    name: str = Field(..., description="Name of the agent")
    config: Dict[str, Any] = Field(..., description="Agent configuration")

class SendMessageParams(BaseModel):
    """Parameters for the send_message tool."""
    content: str = Field(..., description="Message content")
    agent_id: Optional[str] = Field(None, description="ID of the agent to send to")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation")

class GetSubgraphParams(BaseModel):
    """Parameters for the get_subgraph tool."""
    root_id: str = Field(..., description="ID of the root node")
    max_depth: int = Field(2, description="Maximum depth of the subgraph")
    relationship_types: Optional[List[str]] = Field(None, description="Types of relationships to include")
```

## Integration Testing Strategy

### 1. MCP Server Tests

```python
# tests/integration/test_mcp_server_integration.py

import pytest
import asyncio
from mcp.client.lowlevel.client import Client
from mcp.client.stdio import StdioClientTransport

from clubhouse.services.mcp.server import ClubhouseMCPServer
from clubhouse.core.service_registry import ServiceRegistry

@pytest.fixture
async def mcp_client():
    """Create an MCP client connected to a running MCP server."""
    # Start MCP server in background
    # In a real test, this would use a real server connection
    
    # Create client
    transport = StdioClientTransport()
    client = Client(transport)
    await client.initialize()
    
    yield client
    await client.shutdown()

@pytest.mark.asyncio
async def test_list_tools(mcp_client):
    """Test that the MCP server correctly lists available tools."""
    # Get list of tools
    tools = await mcp_client.list_tools()
    
    # Verify tools
    tool_names = [tool.name for tool in tools]
    assert "create_agent" in tool_names
    assert "send_message" in tool_names
    assert "get_subgraph" in tool_names

@pytest.mark.asyncio
async def test_create_agent(mcp_client):
    """Test agent creation via MCP."""
    # Call create_agent tool
    result = await mcp_client.call_tool(
        "create_agent",
        {
            "name": "test-agent",
            "config": {"capability": "test"}
        }
    )
    
    # Verify result
    assert "agent_id" in result
    assert result["name"] == "test-agent"
    assert result["status"] == "created"
```

### 2. Neo4j Resource Tests

```python
# tests/integration/test_mcp_neo4j_resources.py

import pytest
import asyncio
from mcp.client.lowlevel.client import Client

@pytest.mark.asyncio
async def test_node_resource(mcp_client, complex_graph):
    """Test accessing Neo4j nodes as MCP resources."""
    # Get node ID from test graph
    node_id = complex_graph["A"]  # From the complex_graph fixture
    
    # Read resource
    resource = await mcp_client.read_resource(f"resource://graph/{node_id}")
    
    # Verify resource content
    assert resource["id"] == node_id
    assert "properties" in resource
    assert resource["resource_type"] == "node"

@pytest.mark.asyncio
async def test_subgraph_resource(mcp_client, complex_graph):
    """Test accessing Neo4j subgraphs as MCP resources."""
    # Get root node ID from test graph
    root_id = complex_graph["A"]  # From the complex_graph fixture
    
    # Read resource
    resource = await mcp_client.read_resource(f"resource://subgraph/{root_id}")
    
    # Verify resource content
    assert resource["root_id"] == root_id
    assert "nodes" in resource
    assert "relationships" in resource
    assert resource["resource_type"] == "subgraph"
    
    # Verify that at least some nodes were returned
    assert len(resource["nodes"]) > 0
```

## Implementation Plan

### Phase 1: MCP Core Services (2 weeks)

1. **Infrastructure Setup**
   - Add MCP dependencies to project
   - Configure Docker containers for MCP server testing
   - Create service interfaces for MCP components

2. **Basic MCP Server**
   - Implement ClubhouseMCPServer class
   - Register core tools (create_agent, send_message)
   - Integrate with Service Registry

3. **Kafka Integration**
   - Implement MCPKafkaBridge
   - Add message conversion between MCP and Kafka formats
   - Set up request/response tracking

### Phase 2: Neo4j Resource Access (2 weeks)

1. **Graph Data Resources**
   - Implement node and subgraph resources
   - Create resource handlers
   - Add caching for resource access

2. **Testing Infrastructure**
   - Create integration tests for MCP-Neo4j resources
   - Set up test fixtures with real Neo4j database
   - Implement test helpers for MCP clients

### Phase 3: Enhanced Capabilities (2 weeks)

1. **Prompt Templates**
   - Define graph-aware prompt templates
   - Implement prompt registration
   - Add context-aware prompt generation

2. **UI Integration**
   - Create MCP-aware CLI client
   - Implement progress reporting and streaming
   - Add formatted output for graph data

### Phase 4: Production Readiness (2 weeks)

1. **Performance Optimization**
   - Add monitoring for MCP operations
   - Implement resource caching
   - Optimize graph traversal operations

2. **Documentation**
   - Create API documentation for MCP tools
   - Update developer guides
   - Add example code for common operations

## Service Architecture Updates

### New Components

1. **MCPServerService**
   - Manages MCP server lifecycle
   - Registers with Service Registry
   - Provides service interface for MCP operations

2. **MCPMessageHandler**
   - Handles MCP protocol messages
   - Translates between MCP and Kafka formats
   - Manages request/response tracking

3. **MCPResourceManager**
   - Manages Neo4j resources
   - Handles resource caching
   - Provides type-safe resource interfaces

## Conclusion

This integration plan provides a clear path for incorporating the Model Context Protocol into the Clubhouse platform while adhering to our core principles of using real infrastructure, focusing on human-AI partnership, and following test-driven development. By standardizing agent interactions through MCP, we enable more consistent, reliable, and extensible AI capabilities while maintaining our existing Kafka-based architecture.
