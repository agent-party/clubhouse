"""Tests for MCP lifecycle management."""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, call

from clubhouse.core.config import MCPConfig
from clubhouse.core.lifecycle import MCPServerLifecycle
from clubhouse.services.mcp_protocol import MCPServerProtocol


# Remove class-level asyncio mark and apply it to individual async test methods
class TestMCPLifecycle:
    """Test cases for MCP lifecycle management."""
    
    def test_constructor(self, mock_mcp_server):
        """Test that the lifecycle manager can be properly constructed."""
        config = MCPConfig()
        service_registry = Mock()
        
        lifecycle = MCPServerLifecycle(
            config=config,
            service_registry=service_registry,
            server=mock_mcp_server
        )
        
        assert lifecycle is not None
        assert lifecycle.config == config
        assert lifecycle.service_registry == service_registry
        assert lifecycle.server == mock_mcp_server
        assert hasattr(lifecycle, 'startup')
        assert hasattr(lifecycle, 'shutdown')
    
    @pytest.mark.asyncio
    async def test_mcp_server_lifecycle_context_manager(self, mock_mcp_server):
        """Test that the lifecycle manager can be used as a context manager."""
        # Setup
        config = MCPConfig()
        service_registry = Mock()
        
        # Make sure start/stop are AsyncMock objects
        mock_mcp_server.start = AsyncMock()
        mock_mcp_server.stop = AsyncMock()
        
        # Use in a context manager
        async with MCPServerLifecycle(
            config=config,
            service_registry=service_registry,
            server=mock_mcp_server
        ) as lifecycle:
            assert lifecycle is not None
            mock_mcp_server.start.assert_called_once()
            service_registry.register_with_mcp.assert_called_once_with(mock_mcp_server)
        
        # Verify shutdown was called after context exit
        mock_mcp_server.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_shutdown_hooks(self, mock_mcp_server):
        """Test that startup and shutdown hooks are executed correctly."""
        # Setup
        config = MCPConfig()
        service_registry = Mock()
        
        # Make sure start/stop are AsyncMock objects
        mock_mcp_server.start = AsyncMock()
        mock_mcp_server.stop = AsyncMock()
        
        # Create the lifecycle manager
        lifecycle = MCPServerLifecycle(
            config=config,
            service_registry=service_registry,
            server=mock_mcp_server
        )
        
        # Create mock hooks
        startup_hook1 = AsyncMock(name="startup_hook1")
        startup_hook2 = AsyncMock(name="startup_hook2")
        shutdown_hook1 = AsyncMock(name="shutdown_hook1")
        shutdown_hook2 = AsyncMock(name="shutdown_hook2")
        
        # Register hooks
        lifecycle.register_startup_hook(startup_hook1)
        lifecycle.register_startup_hook(startup_hook2)
        lifecycle.register_shutdown_hook(shutdown_hook1)
        lifecycle.register_shutdown_hook(shutdown_hook2)
        
        # Startup should call hooks
        await lifecycle.startup()
        startup_hook1.assert_called_once()
        startup_hook2.assert_called_once()
        service_registry.register_with_mcp.assert_called_once_with(mock_mcp_server)
        mock_mcp_server.start.assert_called_once()
        
        # Shutdown should call hooks in reverse order
        await lifecycle.shutdown()
        shutdown_hook1.assert_called_once()
        shutdown_hook2.assert_called_once()
        mock_mcp_server.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, mock_mcp_server):
        """Test that exceptions in hooks are properly handled."""
        # Setup
        config = MCPConfig()
        service_registry = Mock()
        
        # Create the lifecycle manager
        lifecycle = MCPServerLifecycle(
            config=config,
            service_registry=service_registry,
            server=mock_mcp_server
        )
        
        # Mock server methods
        mock_mcp_server.start = AsyncMock()
        mock_mcp_server.stop = AsyncMock(side_effect=Exception("Stop error"))
        
        # Register a failing shutdown hook
        failing_hook = AsyncMock(side_effect=Exception("Hook error"))
        normal_hook = AsyncMock()
        
        lifecycle.register_shutdown_hook(failing_hook)
        lifecycle.register_shutdown_hook(normal_hook)
        
        # Shutdown should continue despite hook error
        await lifecycle.shutdown()
        
        # Both hooks should have been called
        failing_hook.assert_called_once()
        normal_hook.assert_called_once()
        
        # Server stop should have been called
        mock_mcp_server.stop.assert_called_once()
