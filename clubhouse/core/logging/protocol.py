"""
Protocol definitions for the structured logging system.

This module defines the core interfaces that all logging implementations
must adhere to, ensuring consistency across the framework.
"""

import enum
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, TypeVar
from typing import cast, List, Dict, Any, Type


class LogLevel(enum.IntEnum):
    """Log levels with numeric values matching standard Python logging."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, level_name: str) -> "LogLevel":
        """Convert a string level name to a LogLevel enum."""
        level_map = {
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
        }
        normalized = level_name.lower()
        if normalized not in level_map:
            raise ValueError(f"Unknown log level: {level_name}")
        return level_map[normalized]


T = TypeVar("T", bound="LoggingProtocol")


class LoggingProtocol(Protocol):
    """Protocol defining the interface for structured loggers."""

    @property
    def context(self) -> Dict[str, Any]:
        """
        Get the current logger context.

        Returns:
            A copy of the current context dictionary.
        """
        ...

    def with_context(self: T, **context_updates: Any) -> T:
        """
        Create a new logger with updated context.

        This method returns a new logger instance with the specified
        context updates applied on top of the current context. The
        original logger is not modified.

        Args:
            **context_updates: Key-value pairs to add to the context.

        Returns:
            A new logger instance with the updated context.
        """
        ...

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log at DEBUG level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log at INFO level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log at WARNING level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log at ERROR level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...

    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log at CRITICAL level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        ...


class LogEntryProtocol(Protocol):
    """Protocol defining the structure of a log entry."""

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp when the log entry was created."""
        ...

    @property
    def level(self) -> LogLevel:
        """Get the log level."""
        ...

    @property
    def message(self) -> str:
        """Get the log message."""
        ...

    @property
    def context(self) -> Dict[str, Any]:
        """Get the context associated with this log entry."""
        ...

    @property
    def extra(self) -> Dict[str, Any]:
        """Get additional data associated with this log entry."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert the log entry to a dictionary representation."""
        ...


class LogHandlerProtocol(Protocol):
    """Protocol defining the interface for log handlers."""

    def handle(self, entry: LogEntryProtocol) -> None:
        """
        Handle a log entry.

        Args:
            entry: The log entry to handle.
        """
        ...

    def shutdown(self) -> None:
        """
        Shut down the handler, releasing any resources.

        This method should be called when the handler is no longer needed.
        """
        ...
