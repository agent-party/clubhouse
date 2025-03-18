"""
Structured logging system for the MCP framework.

This module provides a protocol-based approach to structured logging
with context propagation and JSON formatting.
"""

from clubhouse.core.logging.config import LoggingConfig, LogHandlerConfig
from clubhouse.core.logging.factory import LoggerFactory, configure_logging, get_logger
from clubhouse.core.logging.handlers import ConsoleHandler, FileHandler
from clubhouse.core.logging.logger import StructuredLogger
from typing import cast, List, Dict, Any, Type

# Export public API
from clubhouse.core.logging.protocol import (
    LogEntryProtocol,
    LoggingProtocol,
    LogHandlerProtocol,
    LogLevel,
)

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
