"""
MCP Server configuration model.
"""

from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clubhouse.core.config.models.log_level import LogLevel
from typing import cast, List, Dict, Any, Type


class MCPConfig(BaseModel):
    """
    Configuration for the MCP server.

    Attributes:
        host: The host to bind the MCP server to
        port: The port to bind the MCP server to
        log_level: The log level for the MCP server
        timeout_seconds: Timeout for MCP operations in seconds
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )

    host: str = Field(default="127.0.0.1", description="Host to bind MCP server to")
    port: int = Field(
        default=8000, description="Port to bind MCP server to", ge=1, le=65535
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Log level for MCP server"
    )
    timeout_seconds: float = Field(
        default=10.0, description="Timeout for MCP operations in seconds", gt=0
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> LogLevel:
        """Ensure log_level is a valid LogLevel enum value.

        Args:
            v: The value to validate.

        Returns:
            LogLevel: The validated log level.

        Raises:
            ValueError: If the log level is not valid.
        """
        if isinstance(v, str):
            try:
                return LogLevel(v.lower())
            except ValueError:
                valid_values = [level.value for level in LogLevel]
                raise ValueError(
                    f"Invalid log level. Must be one of: {', '.join(valid_values)}"
                )
        # If already a LogLevel enum, return it directly
        if isinstance(v, LogLevel):
            return v
        # Ensure we always return a LogLevel
        raise ValueError(
            f"Invalid log level type: {type(v)}. Must be string or LogLevel enum."
        )
