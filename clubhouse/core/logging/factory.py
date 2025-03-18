"""
Factory for creating and managing loggers.

This module provides a centralized factory for creating structured loggers
with the appropriate configuration and handlers.
"""

import os
import sys
import threading
from typing import Any, ClassVar, Dict, List, Optional, Type, cast

from clubhouse.core.logging.config import LoggingConfig, LogHandlerConfig
from clubhouse.core.logging.handlers import ConsoleHandler, FileHandler
from clubhouse.core.logging.logger import StructuredLogger
from clubhouse.core.logging.protocol import (
    LoggingProtocol,
    LogHandlerProtocol,
    LogLevel,
)


class LoggerFactory:
    """
    Factory for creating and managing structured loggers.

    This class provides a centralized way to create and configure loggers
    with consistent settings across the application.
    """

    # Singleton instance
    _instance: ClassVar[Optional["LoggerFactory"]] = None

    # Lock for thread-safe singleton access
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "LoggerFactory":
        """
        Get the singleton instance of the logger factory.

        Returns:
            The LoggerFactory singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = LoggerFactory()
            return cls._instance

    def __init__(self) -> None:
        """Initialize a new logger factory."""
        self._config = LoggingConfig()
        self._handlers: List[LogHandlerProtocol] = []
        self._loggers: Dict[str, StructuredLogger] = {}

        # Initialize with default handlers from config
        self._initialize_handlers()

    def configure(self, config: LoggingConfig) -> None:
        """
        Configure the logger factory.

        This method updates the configuration and reinitializes all handlers.
        Existing loggers will automatically use the new handlers.

        Args:
            config: New logging configuration
        """
        # Save new configuration
        self._config = config

        # Shutdown existing handlers
        for handler in self._handlers:
            handler.shutdown()

        # Initialize new handlers
        self._handlers = []
        self._initialize_handlers()

        # Update existing loggers with new handlers
        for logger in self._loggers.values():
            # We need to access the protected _handlers attribute
            # This is generally not recommended but acceptable within the same package
            logger._handlers = self._handlers

    def get_logger(self, name: str) -> LoggingProtocol:
        """
        Get a logger with the specified name.

        If a logger with the same name already exists, it will be returned.
        Otherwise, a new logger will be created.

        Args:
            name: Logger name, typically the module name

        Returns:
            A structured logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(
                name=name,
                config=self._config,
                handlers=self._handlers,
            )

        return self._loggers[name]

    def _initialize_handlers(self) -> None:
        """
        Initialize handlers based on configuration.
        """
        for handler_config in self._config.handlers:
            if handler_config.type == "console":
                handler = ConsoleHandler(
                    level=handler_config.level,
                    format=handler_config.format,
                    # Extract handler-specific options
                    use_colors=handler_config.options.get("use_colors", True),
                    output_stream=(
                        sys.stderr
                        if handler_config.options.get("use_stderr", False)
                        else None
                    ),
                )
                self._handlers.append(handler)

            elif handler_config.type == "file":
                filename = handler_config.options.get("filename", "logs/app.log")
                handler = FileHandler(  # type: ignore[type_assignment]
                    filename=filename,
                    level=handler_config.level,
                    format=handler_config.format,
                    max_size=handler_config.options.get("max_size", 10 * 1024 * 1024),
                    backup_count=handler_config.options.get("backup_count", 5),
                )
                self._handlers.append(handler)

            # Additional handler types can be added here


# Convenience functions for getting loggers


def get_logger(name: str) -> LoggingProtocol:
    """
    Get a logger with the specified name.

    This is a convenience function that uses the singleton LoggerFactory.

    Args:
        name: Logger name, typically the module name

    Returns:
        A structured logger instance
    """
    return LoggerFactory.get_instance().get_logger(name)


def configure_logging(config: LoggingConfig) -> None:
    """
    Configure the logging system.

    This is a convenience function that configures the singleton LoggerFactory.

    Args:
        config: Logging configuration
    """
    LoggerFactory.get_instance().configure(config)