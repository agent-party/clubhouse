"""
Lifecycle management for the MCP server.

This module provides classes for managing the lifecycle of the MCP server,
including startup, shutdown, and registration of services.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from clubhouse.core.config import MCPConfig
from clubhouse.core.mcp_service_registry import MCPServiceRegistry
from clubhouse.services.mcp_protocol import MCPServerProtocol

# Type definition for lifecycle hooks
LifecycleHook = Callable[[], Awaitable[None]]


class MCPServerLifecycle:
    """
    Lifecycle manager for the MCP server.

    This class manages the startup and shutdown of the MCP server,
    including the registration of services and execution of hooks.

    Attributes:
        config: Configuration for the MCP server
        service_registry: Registry of services to be registered with MCP
        server: The MCP server instance
        logger: Logger for this class
        startup_hooks: List of hooks to execute during startup
        shutdown_hooks: List of hooks to execute during shutdown
    """

    def __init__(
        self,
        config: MCPConfig,
        service_registry: MCPServiceRegistry,
        server: MCPServerProtocol,
    ) -> None:
        """
        Initialize the MCP server lifecycle manager.

        Args:
            config: Configuration for the MCP server
            service_registry: Registry of services to register with MCP
            server: The MCP server instance
        """
        self.config = config
        self.service_registry = service_registry
        self.server = server
        self.logger = logging.getLogger(__name__)

        self.startup_hooks: List[LifecycleHook] = []
        self.shutdown_hooks: List[LifecycleHook] = []

    def register_startup_hook(self, hook: LifecycleHook) -> None:
        """
        Register a hook to be executed during startup.

        Args:
            hook: Async function to execute during startup
        """
        self.startup_hooks.append(hook)
        self.logger.debug(f"Registered startup hook: {hook.__name__}")

    def register_shutdown_hook(self, hook: LifecycleHook) -> None:
        """
        Register a hook to be executed during shutdown.

        Args:
            hook: Async function to execute during shutdown
        """
        self.shutdown_hooks.append(hook)
        self.logger.debug(f"Registered shutdown hook: {hook.__name__}")

    async def startup(self) -> None:
        """
        Start the MCP server and run startup hooks.

        This method:
        1. Runs all registered startup hooks
        2. Registers all services with the MCP server
        3. Starts the MCP server

        Raises:
            Exception: If any startup hook fails
        """
        self.logger.info("Starting MCP server lifecycle")

        # Run startup hooks
        for hook in self.startup_hooks:
            self.logger.debug(f"Running startup hook: {hook.__name__}")
            await hook()

        # Register services with MCP
        self.logger.info("Registering services with MCP server")
        self.service_registry.register_with_mcp(self.server)

        # Start MCP server
        self.logger.info(
            f"Starting MCP server on {self.config.host}:{self.config.port}"
        )
        await self.server.start(host=self.config.host, port=self.config.port)
        self.logger.info("MCP server started successfully")

    async def shutdown(self) -> None:
        """
        Stop the MCP server and run shutdown hooks.

        This method:
        1. Stops the MCP server
        2. Runs all registered shutdown hooks

        Hooks are executed in reverse order from registration.

        Raises:
            Exception: If any shutdown hook fails
        """
        self.logger.info("Shutting down MCP server")

        # Stop MCP server
        try:
            await self.server.stop()
            self.logger.info("MCP server stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping MCP server: {str(e)}")

        # Run shutdown hooks in reverse order
        for hook in reversed(self.shutdown_hooks):
            self.logger.debug(f"Running shutdown hook: {hook.__name__}")
            try:
                await hook()
            except Exception as e:
                self.logger.error(f"Error in shutdown hook {hook.__name__}: {str(e)}")

        self.logger.info("MCP server lifecycle shutdown complete")

    async def __aenter__(self) -> "MCPServerLifecycle":
        """
        Enter async context manager.

        This allows the lifecycle manager to be used with an async with statement.

        Returns:
            The lifecycle manager instance
        """
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager.

        This ensures the MCP server is properly shut down even if an exception occurs.

        Args:
            exc_type: The exception type, if any
            exc_val: The exception value, if any
            exc_tb: The exception traceback, if any
        """
        await self.shutdown()
