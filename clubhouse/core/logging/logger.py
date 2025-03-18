"""
Structured logger implementation for the MCP framework.

This module provides the core implementation of the LoggingProtocol
with support for context propagation and multiple handlers.
"""

import asyncio
import contextvars
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Type, TypeVar

from clubhouse.core.logging.config import LoggingConfig
from clubhouse.core.logging.model import LogEntry
from clubhouse.core.logging.protocol import (
    LoggingProtocol,
    LogHandlerProtocol,
    LogLevel,
)

# Context variable for propagating logging context
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

T = TypeVar("T", bound="StructuredLogger")


class StructuredLogger:
    """
    Implementation of the LoggingProtocol that supports structured logging.

    This logger creates structured log entries with context and supports
    multiple handlers for different output formats and destinations.
    """

    def __init__(
        self,
        name: str,
        config: LoggingConfig,
        handlers: Optional[List[LogHandlerProtocol]] = None,
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a new structured logger.

        Args:
            name: Logger name, typically the module name
            config: Logging configuration
            handlers: Optional list of log handlers
            parent_context: Optional parent context to inherit
        """
        self._name = name
        self._config = config
        self._handlers = handlers or []

        # Initialize context from parent and global config
        self._local_context: Dict[str, Any] = {}

        if parent_context:
            self._local_context.update(parent_context)

        if config.global_context:
            self._local_context.update(config.global_context)

        # Add standard context fields
        self._local_context.update(
            {
                "logger": name,
            }
        )

    @property
    def context(self) -> Dict[str, Any]:
        """
        Get the current logger context.

        Returns:
            A merged dictionary containing the local context and the
            context from the context variable (if propagation is enabled).
        """
        # Start with local context
        result = self._local_context.copy()

        # Add propagated context if enabled
        if self._config.propagate_context:
            propagated_context = _log_context.get()
            if propagated_context:
                result.update(propagated_context)

        return result

    def with_context(self: T, **context_updates: Any) -> T:
        """
        Create a new logger with updated context.

        Args:
            **context_updates: Key-value pairs to add to the context.

        Returns:
            A new logger instance with the updated context.
        """
        # Create new logger with same config and handlers
        new_logger = self.__class__(
            name=self._name,
            config=self._config,
            handlers=self._handlers,
            parent_context=self._local_context,
        )

        # Apply context updates
        new_logger._local_context.update(context_updates)

        return new_logger

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """
        Create and handle a log entry.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional data for the log entry
        """
        # Skip if level is below threshold
        min_level = LogLevel.from_string(self._config.default_level)
        if level < min_level:
            return

        # Extract exception if provided
        exception = kwargs.pop("exc_info", None)
        if exception is True:
            exception = sys.exc_info()[1]

        # Create log entry
        entry = LogEntry(
            message=message,
            level=level,
            context=self.context,
            extra=kwargs,
            exception=exception,
            include_traceback=self._config.include_traceback,
        )

        # Pass to all handlers
        for handler in self._handlers:
            try:
                handler.handle(entry)
            except Exception as e:
                # Log handler failure to stderr
                print(f"ERROR: Log handler failed: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log at DEBUG level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log at INFO level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log at WARNING level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log at ERROR level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log at CRITICAL level.

        Args:
            message: The log message.
            **kwargs: Additional key-value pairs to include in the log entry.
        """
        self._log(LogLevel.CRITICAL, message, **kwargs)

    @staticmethod
    def update_context(**context_updates: Any) -> contextvars.Token:
        """
        Update the global context for all loggers.

        This updates the context for the current async task and all
        child tasks that inherit this context.

        Args:
            **context_updates: Key-value pairs to add to the context.

        Returns:
            A token that can be used to restore the previous context.
        """
        current = _log_context.get()
        updated = {**current, **context_updates}
        return _log_context.set(updated)

    @staticmethod
    def reset_context(token: contextvars.Token) -> None:
        """
        Reset the global context using a token from update_context.

        Args:
            token: Token from a previous update_context call.
        """
        _log_context.reset(token)

    @staticmethod
    async def with_async_context(context_updates: Dict[str, Any], coro) -> None:
        """
        Run a coroutine with updated context.

        This is a context manager that ensures the context is properly
        restored even if the coroutine raises an exception.

        Args:
            context_updates: Context updates to apply
            coro: Coroutine to run with the updated context

        Returns:
            The result of the coroutine
        """
        token = StructuredLogger.update_context(**context_updates)
        try:
            return await coro  # type: ignore[any_return]
        finally:
            StructuredLogger.reset_context(token)