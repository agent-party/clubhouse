"""Integration tests for MCP server."""
import pytest
import asyncio
from typing import Dict, Any, List, AsyncGenerator
from unittest.mock import Mock, patch, AsyncMock

from clubhouse.core.mcp_service_registry import MCPServiceRegistry
from clubhouse.core.config import MCPConfig  
from clubhouse.core.lifecycle import MCPServerLifecycle


class TestMCPIntegration:
    """Integration tests for MCP server."""
    
    @pytest.mark.asyncio
    async def test_mcp_service_registration(self, mcp_service_registry, mock_mcp_integrated_service, mock_mcp_server):
        """Test registering services with MCP."""
        # Register a test service
        service_name = "test_service"
        mcp_service_registry.register_service(service_name, mock_mcp_integrated_service)
        
        # Register all services with MCP
        mcp_service_registry.register_with_mcp(mock_mcp_server)
        
        # Verify the service's register_with_mcp was called
        mock_mcp_integrated_service.register_with_mcp.assert_called_once_with(mock_mcp_server)
    
    @pytest.mark.asyncio
    async def test_lifecycle_context_manager(self, mock_mcp_server):
        """Test the lifecycle context manager for MCP server."""
        # Patch async methods on mock_mcp_server
        with patch.object(mock_mcp_server, 'start', AsyncMock()) as mock_start, \
             patch.object(mock_mcp_server, 'stop', AsyncMock()) as mock_stop:
            
            # Create service registry
            registry = MCPServiceRegistry()
            
            # Create lifecycle manager with correct parameter order
            config = MCPConfig()  
            lifecycle = MCPServerLifecycle(
                config=config,
                service_registry=registry,
                server=mock_mcp_server
            )
            
            # Use as async context manager
            async with lifecycle:
                # Verify server was started
                mock_start.assert_called_once()
            
            # Verify server was stopped
            mock_stop.assert_called_once()
