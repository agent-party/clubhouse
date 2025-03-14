"""
Structured logging system for the MCP framework.

This module provides a protocol-based approach to structured logging
with context propagation and JSON formatting.
"""

# Export public API
from mcp_demo.core.logging.protocol import LoggingProtocol, LogLevel, LogHandlerProtocol, LogEntryProtocol
from mcp_demo.core.logging.config import LoggingConfig, LogHandlerConfig
from mcp_demo.core.logging.factory import get_logger, configure_logging, LoggerFactory
from mcp_demo.core.logging.logger import StructuredLogger
from mcp_demo.core.logging.handlers import ConsoleHandler, FileHandler

__all__ = [
    # Protocol interfaces
    "LoggingProtocol",
    "LogHandlerProtocol", 
    "LogEntryProtocol",
    "LogLevel",
    
    # Configuration
    "LoggingConfig",
    "LogHandlerConfig",
    
    # Factory functions
    "get_logger",
    "configure_logging",
    
    # Implementation classes
    "LoggerFactory",
    "StructuredLogger",
    "ConsoleHandler",
    "FileHandler",
]
