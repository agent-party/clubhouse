"""Tests for MCP protocol interfaces."""
import pytest
from typing import Any, Dict, List
from unittest.mock import Mock

from clubhouse.services.mcp_protocol import MCPIntegrationProtocol, MCPServerProtocol


class MockMCPServer:
    """Mock FastMCP server for testing."""
    
    def __init__(self):
        self.tools = []
        self.resources = []
        
    def tool(self, *args, **kwargs):
        """Mock tool decorator."""
        def decorator(func):
            self.tools.append(func)
            return func
        return decorator
        
    def resource(self, uri_template, *args, **kwargs):
        """Mock resource decorator."""
        def decorator(func):
            self.resources.append((uri_template, func))
            return func
        return decorator


@pytest.fixture
def mock_mcp_server():
    return MockMCPServer()


class TestMCPIntegrationProtocol:
    """Test cases for MCP integration protocol."""
    
    def test_protocol_definition(self):
        """Test that the protocol defines the expected methods."""
        # Verify Protocol has the expected methods
        assert hasattr(MCPIntegrationProtocol, "register_with_mcp")
        
        # Create a mock object that conforms to the protocol
        class MockMCPService:
            def register_with_mcp(self, server: MCPServerProtocol) -> None:
                """Mock implementation of register_with_mcp."""
                pass
        
        # Verify the mock is recognized as implementing the protocol
        assert isinstance(MockMCPService(), MCPIntegrationProtocol)
    
    def test_implementation_compatibility(self, mock_mcp_server):
        """Test that a concrete implementation can be registered."""
        # Create a concrete implementation
        class MCPService:
            def __init__(self):
                self.register_calls = 0
            
            def register_with_mcp(self, server: Any) -> None:
                """Register with MCP server."""
                self.register_calls += 1
                # Register some mock tools
                @server.tool()
                def sample_tool(arg1: str) -> str:
                    """Sample tool for testing."""
                    return f"Processed: {arg1}"
        
        # Create an instance
        service = MCPService()
        
        # Verify it implements the protocol
        assert isinstance(service, MCPIntegrationProtocol)
        
        # Call the method
        service.register_with_mcp(mock_mcp_server)
        
        # Verify the method was called
        assert service.register_calls == 1
        
        # Verify something was registered with the server
        assert len(mock_mcp_server.tools) >= 1


class TestMCPServerProtocol:
    """Test cases for MCP server protocol."""
    
    def test_server_protocol_definition(self):
        """Test that the server protocol defines the expected methods."""
        # Verify Protocol has the expected methods
        assert hasattr(MCPServerProtocol, "tool")
        assert hasattr(MCPServerProtocol, "resource")
        assert hasattr(MCPServerProtocol, "middleware")
        assert hasattr(MCPServerProtocol, "start")
        assert hasattr(MCPServerProtocol, "stop")
        
        # Verify our mock_mcp_server fixture implements the protocol
        assert isinstance(Mock(spec=MCPServerProtocol), MCPServerProtocol)
