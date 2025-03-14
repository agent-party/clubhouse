"""
Log level enum for MCP configuration.
"""

from enum import Enum


class LogLevel(str, Enum):
    """Valid log levels for the application."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
