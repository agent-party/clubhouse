"""
Configuration models for the structured logging system.

This module defines Pydantic models for configuring the logging system.
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class LogHandlerConfig(BaseModel):
    """Configuration for a log handler."""
    
    type: str = Field(
        ..., 
        description="Handler type (console, file, json_file)"
    )
    
    level: str = Field(
        "INFO", 
        description="Minimum log level to process"
    )
    
    format: str = Field(
        "json", 
        description="Log format (json or text)"
    )
    
    # Handler-specific configuration options
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Handler-specific options"
    )


class LoggingConfig(BaseModel):
    """Configuration for the logging system."""
    
    default_level: str = Field(
        "INFO", 
        description="Default log level"
    )
    
    include_timestamps: bool = Field(
        True, 
        description="Include timestamps in log entries"
    )
    
    propagate_context: bool = Field(
        True, 
        description="Propagate context across async boundaries"
    )
    
    include_traceback: bool = Field(
        True, 
        description="Include traceback information for errors"
    )
    
    handlers: List[LogHandlerConfig] = Field(
        default_factory=lambda: [
            LogHandlerConfig(type="console", level="INFO", format="json")
        ],
        description="List of log handlers"
    )
    
    # Global context to apply to all log entries
    global_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context to include in all log entries"
    )
