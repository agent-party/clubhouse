"""
Log handlers for the structured logging system.

This module provides implementations of the LogHandlerProtocol
for different output destinations and formats.
"""
import sys
import json
import os
from typing import Dict, Any, Optional, TextIO, Union
import logging
from datetime import datetime
import threading

from clubhouse.core.logging.protocol import LogHandlerProtocol, LogEntryProtocol, LogLevel


class ConsoleHandler:
    """
    Log handler that writes to the console (stdout or stderr).
    
    This handler supports both JSON and text formats.
    """
    
    def __init__(
        self,
        level: Union[LogLevel, str] = LogLevel.INFO,
        format: str = "json",
        output_stream: Optional[TextIO] = None,
        use_colors: bool = True,
    ):
        """
        Initialize a new console handler.
        
        Args:
            level: Minimum log level to output
            format: Output format ("json" or "text")
            output_stream: Stream to write to (defaults to stdout)
            use_colors: Whether to use ANSI colors in text format
        """
        self._level = level if isinstance(level, LogLevel) else LogLevel.from_string(level)
        self._format = format.lower()
        self._output = output_stream or sys.stdout
        self._use_colors = use_colors and self._output.isatty()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        if self._format not in ("json", "text"):
            raise ValueError(f"Unsupported format: {format}")
    
    def handle(self, entry: LogEntryProtocol) -> None:
        """
        Handle a log entry by writing to the console.
        
        Args:
            entry: The log entry to handle
        """
        # Skip if entry level is below handler level
        if entry.level < self._level:
            return
        
        # Format the entry
        if self._format == "json":
            output = json.dumps(entry.to_dict())
        else:
            output = self._format_text(entry)
        
        # Write to output stream with thread safety
        with self._lock:
            self._output.write(output + "\n")
            self._output.flush()
    
    def _format_text(self, entry: LogEntryProtocol) -> str:
        """
        Format an entry as text.
        
        Args:
            entry: The log entry to format
            
        Returns:
            Formatted text string
        """
        # Color codes for different levels
        colors = {
            LogLevel.DEBUG: "\033[36m",    # Cyan
            LogLevel.INFO: "\033[32m",     # Green
            LogLevel.WARNING: "\033[33m",  # Yellow
            LogLevel.ERROR: "\033[31m",    # Red
            LogLevel.CRITICAL: "\033[35m", # Magenta
        }
        reset = "\033[0m"
        
        # Format timestamp
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Format log level with color if enabled
        if self._use_colors:
            level = f"{colors[entry.level]}{entry.level.name:<8}{reset}"
        else:
            level = f"{entry.level.name:<8}"
        
        # Base message with timestamp and level
        result = f"{timestamp} {level} {entry.message}"
        
        # Add context if available
        if entry.context:
            ctx_str = " ".join(f"{k}={v}" for k, v in entry.context.items())
            result += f" [{ctx_str}]"
        
        # Add extra data if available
        if entry.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in entry.extra.items())
            result += f" {extra_str}"
        
        return result
    
    def shutdown(self) -> None:
        """
        Shut down the handler, releasing any resources.
        """
        # Console handler doesn't need to release resources
        pass


class FileHandler:
    """
    Log handler that writes to a file.
    
    This handler supports both JSON and text formats and handles
    file rotation based on size.
    """
    
    def __init__(
        self,
        filename: str,
        level: Union[LogLevel, str] = LogLevel.INFO,
        format: str = "json",
        max_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        """
        Initialize a new file handler.
        
        Args:
            filename: Path to the log file
            level: Minimum log level to output
            format: Output format ("json" or "text")
            max_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        self._level = level if isinstance(level, LogLevel) else LogLevel.from_string(level)
        self._format = format.lower()
        self._filename = filename
        self._max_size = max_size
        self._backup_count = backup_count
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Open file for writing
        self._file = open(filename, "a", encoding="utf-8")
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        if self._format not in ("json", "text"):
            raise ValueError(f"Unsupported format: {format}")
    
    def handle(self, entry: LogEntryProtocol) -> None:
        """
        Handle a log entry by writing to the file.
        
        Args:
            entry: The log entry to handle
        """
        # Skip if entry level is below handler level
        if entry.level < self._level:
            return
        
        # Format the entry
        if self._format == "json":
            output = json.dumps(entry.to_dict())
        else:
            # Simple text format without colors
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            level = f"{entry.level.name:<8}"
            output = f"{timestamp} {level} {entry.message}"
            
            # Add context if available
            if entry.context:
                ctx_str = " ".join(f"{k}={v}" for k, v in entry.context.items())
                output += f" [{ctx_str}]"
            
            # Add extra data if available
            if entry.extra:
                extra_str = " ".join(f"{k}={v}" for k, v in entry.extra.items())
                output += f" {extra_str}"
        
        # Write to file with thread safety
        with self._lock:
            # Check if rotation is needed
            self._check_rotation()
            
            # Write the log entry
            self._file.write(output + "\n")
            self._file.flush()
    
    def _check_rotation(self) -> None:
        """
        Check if file rotation is needed and rotate if necessary.
        """
        if self._file.tell() >= self._max_size:
            self._rotate_files()
    
    def _rotate_files(self) -> None:
        """
        Rotate log files.
        """
        # Close current file
        self._file.close()
        
        # Delete oldest backup if it exists
        oldest = f"{self._filename}.{self._backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)
        
        # Shift existing backups
        for i in range(self._backup_count - 1, 0, -1):
            src = f"{self._filename}.{i}"
            dst = f"{self._filename}.{i + 1}"
            if os.path.exists(src):
                os.rename(src, dst)
        
        # Rename current file to .1
        if os.path.exists(self._filename):
            os.rename(self._filename, f"{self._filename}.1")
        
        # Open new file
        self._file = open(self._filename, "a", encoding="utf-8")
    
    def shutdown(self) -> None:
        """
        Shut down the handler, releasing any resources.
        """
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None
