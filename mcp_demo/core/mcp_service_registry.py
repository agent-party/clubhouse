"""
MCP Service Registry implementation.

This module provides a registry for services that implement the MCP integration
protocol, allowing centralized management of services that can be registered
with an MCP server.
"""
from typing import Dict, Any, Optional, TypeVar, Type, cast
import logging

from mcp_demo.services.mcp_protocol import MCPIntegrationProtocol, MCPServerProtocol


class MCPServiceRegistry:
    """
    Registry for services that implement the MCP integration protocol.
    
    This registry maintains a mapping of service names to service instances,
    allowing for centralized management of services that can be registered
    with an MCP server.
    
    Attributes:
        services: Dictionary mapping service names to service instances
    """
    
    def __init__(self) -> None:
        """Initialize an empty service registry."""
        self._services: Dict[str, MCPIntegrationProtocol] = {}
        self._logger = logging.getLogger(__name__)
    
    def register_service(self, name: str, service: Any) -> None:
        """
        Register a service with the registry.
        
        Args:
            name: Unique name for the service
            service: Service instance that implements MCPIntegrationProtocol
            
        Raises:
            TypeError: If the service does not implement MCPIntegrationProtocol
        """
        if not isinstance(service, MCPIntegrationProtocol):
            raise TypeError(
                f"Service '{name}' does not implement MCPIntegrationProtocol"
            )
        
        self._services[name] = service
        self._logger.info(f"Registered MCP service: {name}")
    
    def get_service(self, name: str) -> MCPIntegrationProtocol:
        """
        Get a service by name.
        
        Args:
            name: Name of the service to retrieve
            
        Returns:
            The service instance
            
        Raises:
            KeyError: If the service is not registered
        """
        if name not in self._services:
            raise KeyError(f"MCP service not found: {name}")
        
        return self._services[name]
    
    def register_with_mcp(self, server: MCPServerProtocol) -> None:
        """
        Register all services with an MCP server.
        
        This calls register_with_mcp on each registered service,
        passing the MCP server instance.
        
        Args:
            server: The MCP server to register services with
        """
        for name, service in self._services.items():
            self._logger.info(f"Registering service with MCP: {name}")
            service.register_with_mcp(server)
    
    def has_service(self, name: str) -> bool:
        """
        Check if a service is registered.
        
        Args:
            name: Name of the service to check
            
        Returns:
            True if the service is registered, False otherwise
        """
        return name in self._services
    
    def get_service_names(self) -> list[str]:
        """
        Get the names of all registered services.
        
        Returns:
            List of service names
        """
        return list(self._services.keys())
