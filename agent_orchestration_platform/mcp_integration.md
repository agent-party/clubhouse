# MCP Integration Documentation

## Overview

This document describes the integration strategy for the Model Context Protocol (MCP) with the Agent Orchestration Platform. MCP is a standardized protocol for LLMs to interact with external tools, resources, and context, providing a consistent interface between agents and their environment. By implementing MCP support, we enable our agents to connect with any MCP-compatible client and expose our platform's capabilities through a standardized interface.

## Core Principles

1. **Protocol Abstraction**: Abstract MCP details behind clean interfaces to allow for protocol evolution
2. **Capability Exposure**: Expose existing agent capabilities as MCP tools and resources
3. **Bi-directional Integration**: Support both MCP server and client modes for maximum flexibility
4. **Modular Implementation**: Allow incremental adoption of MCP across the platform
5. **Type Safety**: Ensure strong typing and validation throughout the integration

## Architecture Components

### 1. MCP Server Adapter

```
┌────────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                    │       │                 │       │                 │
│ Agent Capabilities │──────▶│ MCP Server      │──────▶│ MCP Clients     │
│                    │       │ Adapter         │       │                 │
└────────────────────┘       └─────────────────┘       └─────────────────┘
```

The MCP Server Adapter exposes our platform's capabilities to external MCP clients:

- **Capability Mapping**: Maps agent capabilities to MCP tools and resources
- **Protocol Translation**: Handles MCP protocol messages and lifecycle
- **Session Management**: Manages client connections and request context
- **Security Boundary**: Enforces access control and capability restrictions

### 2. MCP Client Adapter

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│ Agent Workflow  │────▶│  MCP Client       │────▶│   External MCP    │
│                 │     │  Adapter          │     │   Servers         │
└─────────────────┘     └───────────────────┘     └───────────────────┘
```

The MCP Client Adapter allows agents to connect to external MCP servers:

- **Tool Discovery**: Discovers and catalogs tools from external MCP servers
- **Request Handling**: Manages requests to external MCP servers
- **Result Processing**: Processes and integrates results into agent workflows
- **Connection Management**: Handles connection pooling and lifecycle

### 3. Capability Mapping System

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Capability Registry│────▶│ MCP Mapping      │────▶│ Schema Generation │
│                   │     │ Rules             │     │                   │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Capability Mapping System handles the translation between our platform's capability model and MCP:

- **Capability Registry**: Tracks available capabilities for MCP exposure
- **MCP Mapping Rules**: Defines how capabilities map to MCP concepts
- **Schema Generation**: Generates MCP-compatible schemas from capabilities

### 4. Integration Layer

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Event Bus         │────▶│ MCP Event         │────▶│ Deployment        │
│ Integration       │     │ Translations      │     │ Support           │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Integration Layer connects MCP components with the rest of the platform:

- **Event Bus Integration**: Maps MCP events to our event bus
- **MCP Event Translations**: Translates between MCP notifications and our events
- **Deployment Support**: Provides utilities for deploying MCP servers and clients

## Implementation Details

### MCP Server Implementation

We'll use the FastMCP library to expose our platform's capabilities through an MCP server:

```python
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any, Optional
from clubhouse.capabilities.base import BaseCapability, CapabilityResult
from clubhouse.service_registry import ServiceRegistry 

class MCPServerAdapter:
    """Adapts our platform's capabilities to an MCP server."""
    
    def __init__(
        self, 
        service_registry: ServiceRegistry,
        name: str = "Agent Orchestration Platform",
        instructions: Optional[str] = None
    ):
        """Initialize the MCP server adapter."""
        self.service_registry = service_registry
        self.mcp = FastMCP(name=name, instructions=instructions)
        self.capability_registry = service_registry.get(CapabilityRegistry)
        
        # Register capabilities as tools and resources
        self._register_capabilities()
    
    def _register_capabilities(self) -> None:
        """Register all available capabilities as MCP tools and resources."""
        capabilities = self.capability_registry.get_all_capabilities()
        
        for capability in capabilities:
            if self._is_tool_compatible(capability):
                self._register_capability_as_tool(capability)
            
            if self._is_resource_compatible(capability):
                self._register_capability_as_resource(capability)
    
    def _is_tool_compatible(self, capability: BaseCapability) -> bool:
        """Check if a capability can be exposed as an MCP tool."""
        # Logic to determine if capability produces side effects (tool)
        return hasattr(capability, 'execute') and callable(getattr(capability, 'execute'))
    
    def _is_resource_compatible(self, capability: BaseCapability) -> bool:
        """Check if a capability can be exposed as an MCP resource."""
        # Logic to determine if capability is read-only (resource)
        return hasattr(capability, 'get_data') and callable(getattr(capability, 'get_data'))
    
    def _register_capability_as_tool(self, capability: BaseCapability) -> None:
        """Register a capability as an MCP tool."""
        # Create a tool wrapper for the capability
        @self.mcp.tool()
        async def capability_tool(params: Dict[str, Any], ctx: "Context") -> Any:
            """Tool for executing capability."""
            ctx.info(f"Executing capability: {capability.capability_type}")
            
            # Execute the capability
            result = await capability.execute(params)
            
            # Convert result to MCP-compatible format
            return self._convert_result(result)
        
        # Rename the tool to match the capability
        capability_tool.__name__ = f"{capability.capability_type}"
        capability_tool.__doc__ = capability.get_description()
    
    def _register_capability_as_resource(self, capability: BaseCapability) -> None:
        """Register a capability as an MCP resource."""
        @self.mcp.resource(f"capability://{capability.capability_type}/{{resource_id}}")
        async def capability_resource(resource_id: str) -> str:
            """Resource for accessing capability data."""
            # Get data from the capability
            data = await capability.get_data(resource_id)
            
            # Convert data to string representation
            return json.dumps(data)
    
    def _convert_result(self, result: CapabilityResult) -> Dict[str, Any]:
        """Convert a capability result to MCP-compatible format."""
        if isinstance(result, dict):
            return result
        
        # If it's a custom object, convert to dict
        return result.dict() if hasattr(result, 'dict') else vars(result)
    
    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server with the specified transport."""
        self.mcp.run(transport=transport)
```

### MCP Client Implementation

The MCP Client Adapter allows our agents to consume external MCP server tools:

```python
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client
from typing import Dict, Any, List, Optional, Tuple
from clubhouse.agents.base import BaseAgent
from clubhouse.event_bus import EventBusProtocol

class MCPClientAdapter:
    """Adapts external MCP servers to our platform's agents."""
    
    def __init__(
        self, 
        event_bus: EventBusProtocol,
        server_command: Optional[str] = None,
        server_url: Optional[str] = None
    ):
        """Initialize the MCP client adapter."""
        self.event_bus = event_bus
        self.server_command = server_command
        self.server_url = server_url
        self.tools_cache: Dict[str, Dict[str, Any]] = {}
        self.resources_cache: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self.server_command:
            # Connect via stdio to a local process
            self.session = await self._connect_via_stdio()
        elif self.server_url:
            # Connect via HTTP to a remote server
            self.session = await self._connect_via_http()
        else:
            raise ValueError("Either server_command or server_url must be provided")
        
        # Cache available tools and resources
        await self._cache_server_capabilities()
    
    async def _connect_via_stdio(self) -> ClientSession:
        """Connect to an MCP server via stdio."""
        async with stdio_client(self.server_command) as (read, write):
            session = ClientSession(read, write)
            await session.initialize()
            return session
    
    async def _connect_via_http(self) -> ClientSession:
        """Connect to an MCP server via HTTP."""
        # Implement HTTP connection to MCP server
        raise NotImplementedError("HTTP connection not yet implemented")
    
    async def _cache_server_capabilities(self) -> None:
        """Cache available tools and resources from the server."""
        # Get available tools
        tools_result = await self.session.send_request("tools/list")
        self.tools_cache = {tool["name"]: tool for tool in tools_result["tools"]}
        
        # Get available resources
        resources_result = await self.session.send_request("resources/list")
        self.resources_cache = {
            resource["uri"]: resource for resource in resources_result["resources"]
        }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server."""
        return list(self.tools_cache.values())
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if tool_name not in self.tools_cache:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        result = await self.session.send_request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )
        
        # Publish event to notify tool call completion
        await self.event_bus.publish(
            topic="mcp.tool.called",
            value={
                "tool_name": tool_name, 
                "arguments": arguments,
                "result": result
            }
        )
        
        return result
    
    async def read_resource(self, uri: str) -> Any:
        """Read a resource from the MCP server."""
        if uri not in self.resources_cache:
            raise ValueError(f"Unknown resource: {uri}")
        
        result = await self.session.send_request("resources/read", {"uri": uri})
        
        # Process contents based on type
        contents = result["contents"]
        if not contents:
            return None
        
        # Return the first content item's text or blob
        if "text" in contents[0]:
            return contents[0]["text"]
        elif "blob" in contents[0]:
            return contents[0]["blob"]
        
        return None
```

### Capability-to-MCP Mapping

To expose our capabilities through MCP, we need a mapping system:

```python
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Type, Union
from clubhouse.capabilities.base import BaseCapability, CapabilityParams

class MCPToolSchema(BaseModel):
    """Schema for an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]

class MCPResourceSchema(BaseModel):
    """Schema for an MCP resource."""
    uri_template: str
    name: str
    description: str

class CapabilityMCPMapper:
    """Maps platform capabilities to MCP tools and resources."""
    
    def create_tool_schema(
        self, capability: BaseCapability, params_model: Type[BaseModel]
    ) -> MCPToolSchema:
        """Create a tool schema for a capability."""
        # Generate name from capability type
        name = capability.capability_type.lower().replace("_", "-")
        
        # Use capability description or generate one
        description = getattr(
            capability, "description", f"Execute {capability.capability_type} capability"
        )
        
        # Generate JSON schema from params model
        input_schema = params_model.model_json_schema()
        
        return MCPToolSchema(
            name=name,
            description=description,
            input_schema=input_schema
        )
    
    def create_resource_schema(
        self, capability: BaseCapability
    ) -> MCPResourceSchema:
        """Create a resource schema for a capability."""
        # Generate URI template
        uri_template = f"capability://{capability.capability_type.lower()}/"
        if hasattr(capability, "resource_id_pattern"):
            uri_template += f"{{{capability.resource_id_pattern}}}"
        else:
            uri_template += "{id}"
        
        # Use capability description or generate one
        description = getattr(
            capability, "description", f"Access {capability.capability_type} data"
        )
        
        return MCPResourceSchema(
            uri_template=uri_template,
            name=f"{capability.capability_type} Resource",
            description=description
        )
    
    def params_to_mcp_arguments(
        self, params: Union[Dict[str, Any], BaseModel]
    ) -> Dict[str, Any]:
        """Convert capability params to MCP tool arguments."""
        if isinstance(params, BaseModel):
            return params.model_dump()
        return params
    
    def mcp_result_to_capability_result(
        self, mcp_result: Dict[str, Any], result_model: Type[BaseModel]
    ) -> BaseModel:
        """Convert MCP tool result to capability result."""
        return result_model.model_validate(mcp_result)
```

## Integration with Event Bus

MCP operations will be integrated with our event bus for observability:

```python
from clubhouse.event_bus import EventBusProtocol
from typing import Dict, Any, Callable, Awaitable
from datetime import datetime

class MCPEventAdapter:
    """Adapts MCP events to platform event bus."""
    
    def __init__(self, event_bus: EventBusProtocol):
        """Initialize the MCP event adapter."""
        self.event_bus = event_bus
    
    async def handle_tool_call(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        result: Any = None,
        error: Exception = None
    ) -> None:
        """Handle a tool call event."""
        event_data = {
            "tool_name": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat(),
        }
        
        if result is not None:
            event_data["result"] = result
            await self.event_bus.publish("mcp.tool.called.success", event_data)
        
        if error is not None:
            event_data["error"] = str(error)
            event_data["error_type"] = type(error).__name__
            await self.event_bus.publish("mcp.tool.called.error", event_data)
    
    async def handle_resource_read(
        self,
        uri: str,
        result: Any = None,
        error: Exception = None
    ) -> None:
        """Handle a resource read event."""
        event_data = {
            "uri": uri,
            "timestamp": datetime.now().isoformat(),
        }
        
        if result is not None:
            # Don't include potentially large resource content in the event
            event_data["content_size"] = len(str(result))
            await self.event_bus.publish("mcp.resource.read.success", event_data)
        
        if error is not None:
            event_data["error"] = str(error)
            event_data["error_type"] = type(error).__name__
            await self.event_bus.publish("mcp.resource.read.error", event_data)
    
    def register_event_handlers(self) -> None:
        """Register event handlers for MCP events."""
        # Subscribe to relevant events
        self.event_bus.subscribe("mcp.server.started", self._handle_server_started)
        self.event_bus.subscribe("mcp.client.connected", self._handle_client_connected)
        # Add more handlers as needed
    
    async def _handle_server_started(self, event_data: Dict[str, Any]) -> None:
        """Handle MCP server started event."""
        # Log and potentially notify other components
        pass
    
    async def _handle_client_connected(self, event_data: Dict[str, Any]) -> None:
        """Handle MCP client connected event."""
        # Log and potentially notify other components
        pass
```

## Security Considerations

MCP integration must carefully consider security aspects:

1. **Authentication**: MCP clients/servers need proper authentication mechanisms
2. **Authorization**: Implement capability-based security for MCP tools and resources
3. **Input Validation**: Rigorously validate all inputs from MCP clients
4. **Sandboxing**: Execute MCP tool calls in isolated environments
5. **Rate Limiting**: Prevent abuse through proper rate limiting
6. **Audit Logging**: Log all MCP operations for security audit

## Agent Integration

MCP will be integrated with our agents to leverage external capabilities:

```python
from clubhouse.agents.base import BaseAgent
from typing import Dict, Any, List, Optional
from clubhouse.service_registry import ServiceRegistry

class MCPEnabledAgent(BaseAgent):
    """Agent that can leverage MCP capabilities."""
    
    def __init__(
        self,
        agent_id: str,
        service_registry: ServiceRegistry,
        **kwargs: Any
    ):
        """Initialize the MCP-enabled agent."""
        super().__init__(agent_id, service_registry, **kwargs)
        self.mcp_client_adapter = service_registry.get(MCPClientAdapter)
    
    async def discover_mcp_tools(self) -> List[Dict[str, Any]]:
        """Discover available MCP tools."""
        return await self.mcp_client_adapter.list_tools()
    
    async def call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Call an MCP tool."""
        return await self.mcp_client_adapter.call_tool(tool_name, arguments)
    
    async def read_mcp_resource(self, uri: str) -> Any:
        """Read an MCP resource."""
        return await self.mcp_client_adapter.read_resource(uri)
    
    async def execute_with_mcp_tools(
        self, 
        task_description: str,
        available_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a task using MCP tools."""
        # Example implementation - in reality, this would use LLM to decide which tools to use
        result = {
            "task": task_description,
            "tool_calls": []
        }
        
        # For demonstration purposes - would be replaced by actual LLM logic
        for tool in available_tools[:2]:  # Just use first two tools as an example
            try:
                tool_name = tool["name"]
                # Generate simple arguments - in reality, LLM would determine these
                tool_result = await self.call_mcp_tool(tool_name, {})
                result["tool_calls"].append({
                    "tool": tool_name,
                    "result": tool_result
                })
            except Exception as e:
                result["tool_calls"].append({
                    "tool": tool["name"],
                    "error": str(e)
                })
        
        return result
```

## Deployment Strategies

We'll provide several deployment options for MCP integration:

1. **Embedded MCP Server**: Run MCP server within the main platform process
2. **Sidecar MCP Server**: Deploy MCP server as a separate process/container
3. **Standalone MCP Gateway**: Central MCP gateway for multiple agent instances
4. **Multi-protocol Support**: Support stdio, HTTP and WebSocket transports

## Testing Strategy

Following our test-driven development approach, we'll implement:

1. **Unit Tests**:
   - Test protocol message serialization/deserialization
   - Validate capability-to-MCP mapping
   - Test error handling and retries

2. **Integration Tests**:
   - Test end-to-end MCP tool calls
   - Verify correct event bus integration
   - Test with different transports (stdio, HTTP)

3. **Compatibility Tests**:
   - Test with multiple MCP client implementations
   - Verify protocol version compatibility
   - Test with external MCP servers

## Future Enhancements

1. **Protocol Version Management**:
   - Support multiple MCP protocol versions
   - Handle protocol upgrades gracefully
   - Maintain backward compatibility

2. **Advanced Tool Composition**:
   - Enable chaining of MCP tools for complex workflows
   - Support tool recommendation based on task description
   - Implement tool usage analytics

3. **Enhanced Security**:
   - Add fine-grained permission model for MCP operations
   - Implement token-based authentication
   - Add scope-based authorization

## Implementation Roadmap

1. **Phase 1: Core MCP Protocol Support**
   - Implement basic MCP server and client adapters
   - Add mapping for essential capabilities
   - Create test suite for protocol compliance

2. **Phase 2: Event Bus Integration**
   - Connect MCP operations to event bus
   - Implement observability for MCP operations
   - Add metrics and monitoring

3. **Phase 3: Agent Integration**
   - Enable agents to discover and use MCP tools
   - Implement smart tool selection logic
   - Add support for MCP resource integration

4. **Phase 4: Security and Scaling**
   - Enhance security model for MCP operations
   - Optimize for high throughput
   - Support clustering and load balancing

## Conclusion

The MCP integration provides a standardized way for our Agent Orchestration Platform to interact with external tools and resources. By supporting the MCP protocol, we enable seamless interoperability with a growing ecosystem of AI tools and services, while maintaining our platform's security, performance, and scalability requirements.
