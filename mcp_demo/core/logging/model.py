"""
Data models for the structured logging system.

This module defines the core data structures used by the logging system,
including the LogEntry class that implements the LogEntryProtocol.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
import traceback

from mcp_demo.core.logging.protocol import LogEntryProtocol, LogLevel


class LogEntry:
    """
    A structured log entry implementation.
    
    This class implements the LogEntryProtocol and provides a concrete
    implementation of a log entry with all required properties.
    """
    
    def __init__(
        self,
        message: str,
        level: LogLevel,
        context: Dict[str, Any],
        extra: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        exception: Optional[Exception] = None,
        include_traceback: bool = True,
    ):
        """
        Initialize a new log entry.
        
        Args:
            message: The log message
            level: The log level
            context: Contextual information for the log entry
            extra: Additional key-value pairs for the log entry
            timestamp: Optional timestamp (defaults to now)
            exception: Optional exception that triggered this log entry
            include_traceback: Whether to include traceback for exceptions
        """
        self._timestamp = timestamp or datetime.utcnow()
        self._level = level
        self._message = message
        self._context = context.copy()
        self._extra = extra.copy()
        self._exception = exception
        self._include_traceback = include_traceback
    
    @property
    def timestamp(self) -> datetime:
        """Get the timestamp when the log entry was created."""
        return self._timestamp
    
    @property
    def level(self) -> LogLevel:
        """Get the log level."""
        return self._level
    
    @property
    def message(self) -> str:
        """Get the log message."""
        return self._message
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get the context associated with this log entry."""
        return self._context.copy()
    
    @property
    def extra(self) -> Dict[str, Any]:
        """Get additional data associated with this log entry."""
        return self._extra.copy()
    
    @property
    def exception(self) -> Optional[Exception]:
        """Get the exception that triggered this log entry, if any."""
        return self._exception
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the log entry to a dictionary representation.
        
        Returns:
            A dictionary representation of the log entry suitable for
            serialization to JSON.
        """
        result = {
            "timestamp": self._timestamp.isoformat(),
            "level": self._level.name,
            "level_value": int(self._level),
            "message": self._message,
        }
        
        # Include context if not empty
        if self._context:
            result["context"] = _sanitize_for_json(self._context)
        
        # Include extra data if not empty
        if self._extra:
            result["data"] = _sanitize_for_json(self._extra)
        
        # Include exception information if available
        if self._exception:
            result["exception"] = {
                "type": type(self._exception).__name__,
                "message": str(self._exception),
            }
            
            # Include traceback if configured
            if self._include_traceback:
                result["exception"]["traceback"] = _format_traceback(self._exception)
        
        return result


def _sanitize_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a dictionary to ensure all values are JSON serializable.
    
    Args:
        data: The dictionary to sanitize
        
    Returns:
        A new dictionary with all values converted to JSON-serializable types
    """
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _sanitize_for_json(value)
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            result[key] = value.to_dict()
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
        else:
            # For other types, convert to string
            result[key] = str(value)
    
    return result


def _format_traceback(exception: Exception) -> str:
    """
    Format a traceback for an exception.
    
    Args:
        exception: The exception to format
        
    Returns:
        A formatted traceback string
    """
    return "".join(traceback.format_exception(
        type(exception), exception, exception.__traceback__
    ))
