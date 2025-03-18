"""Tests for MCP service registry."""
import pytest
from typing import Any, Dict
from unittest.mock import Mock

from clubhouse.core.mcp_service_registry import MCPServiceRegistry
from clubhouse.services.mcp_protocol import MCPIntegrationProtocol, MCPServerProtocol


class TestMCPServiceRegistry:
    """Test cases for MCP service registry."""
    
    def test_registry_creation(self):
        """Test that the registry can be created."""
        registry = MCPServiceRegistry()
        assert isinstance(registry, MCPServiceRegistry)
        assert not registry.has_service("test")
    
    def test_register_service(self, mock_mcp_integrated_service):
        """Test that a service can be registered."""
        registry = MCPServiceRegistry()
        
        # Register a service
        registry.register_service("test_service", mock_mcp_integrated_service)
        
        # Verify service was registered
        assert registry.has_service("test_service")
        assert registry.get_service("test_service") == mock_mcp_integrated_service
    
    def test_register_with_mcp(self, mock_mcp_server, mock_mcp_integrated_service):
        """Test that services can be registered with an MCP server."""
        registry = MCPServiceRegistry()
        
        # Register services
        registry.register_service("test_service_1", mock_mcp_integrated_service)
        
        # Create another mock service
        another_service = Mock()
        another_service.register_with_mcp = Mock()
        registry.register_service("test_service_2", another_service)
        
        # Register all services with MCP
        registry.register_with_mcp(mock_mcp_server)
        
        # Verify each service's register_with_mcp was called
        mock_mcp_integrated_service.register_with_mcp.assert_called_once_with(mock_mcp_server)
        another_service.register_with_mcp.assert_called_once_with(mock_mcp_server)
    
    def test_get_nonexistent_service(self):
        """Test that getting a nonexistent service raises KeyError."""
        registry = MCPServiceRegistry()
        
        with pytest.raises(KeyError):
            registry.get_service("nonexistent")
    
    def test_register_non_mcp_service(self):
        """Test that registering a non-MCP service raises TypeError."""
        registry = MCPServiceRegistry()
        
        # Create a service that doesn't implement MCPIntegrationProtocol
        non_mcp_service = object()
        
        # Attempt to register it
        with pytest.raises(TypeError):
            registry.register_service("invalid", non_mcp_service)
    
    def test_get_service_names(self, mock_mcp_integrated_service):
        """Test that service names can be retrieved."""
        registry = MCPServiceRegistry()
        
        # Register services
        registry.register_service("test_service_1", mock_mcp_integrated_service)
        registry.register_service("test_service_2", mock_mcp_integrated_service)
        
        # Get service names
        names = registry.get_service_names()
        
        # Verify service names
        assert sorted(names) == ["test_service_1", "test_service_2"]
