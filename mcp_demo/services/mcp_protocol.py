"""
MCP Integration Protocol definitions.

This module defines the protocols for services that can integrate with
the Model Context Protocol (MCP).
"""
from typing import Protocol, Any, runtime_checkable


# Define a protocol for the FastMCP server to avoid direct dependency
@runtime_checkable
class MCPServerProtocol(Protocol):
    """Protocol defining the interface of an MCP server."""
    
    def tool(self, *args, **kwargs):
        """Register a tool with the MCP server."""
        ...
    
    def resource(self, uri_template, *args, **kwargs):
        """Register a resource with the MCP server."""
        ...
    
    def middleware(self, func):
        """Register middleware with the MCP server."""
        ...
    
    async def start(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the MCP server."""
        ...
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        ...


@runtime_checkable
class MCPIntegrationProtocol(Protocol):
    """
    Protocol for services that can integrate with MCP.
    
    Services implementing this protocol can register their
    tools and resources with an MCP server.
    """
    
    def register_with_mcp(self, server: MCPServerProtocol) -> None:
        """
        Register service capabilities with an MCP server.
        
        This method should register all tools, resources, and middleware
        that the service provides to the MCP server.
        
        Args:
            server: The MCP server to register with
        """
        ...
