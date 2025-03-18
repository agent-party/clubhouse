"""Tests for logging protocol interfaces."""
import pytest
from typing import Dict, Any

from clubhouse.core.logging.protocol import LoggingProtocol, LogLevel


class MockLogger:
    """Mock implementation of the LoggingProtocol for testing."""
    
    def __init__(self):
        self._context: Dict[str, Any] = {}
        self.logs = []
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get the current logger context."""
        return self._context.copy()
    
    def with_context(self, **context_updates: Any) -> "MockLogger":
        """Create a new logger with updated context."""
        new_logger = MockLogger()
        new_logger._context = {**self._context, **context_updates}
        return new_logger
    
    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """Record a log entry."""
        log_entry = {
            "level": level,
            "message": message,
            "context": self._context.copy(),
            "extra": kwargs
        }
        self.logs.append(log_entry)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, **kwargs)


def test_logging_protocol_implementation():
    """Test that MockLogger correctly implements the LoggingProtocol."""
    # This test will fail at type-checking time if MockLogger doesn't implement LoggingProtocol
    logger: LoggingProtocol = MockLogger()
    assert hasattr(logger, "context")
    assert callable(logger.with_context)
    assert callable(logger.debug)
    assert callable(logger.info)
    assert callable(logger.warning)
    assert callable(logger.error)
    assert callable(logger.critical)


def test_context_propagation():
    """Test that context is properly propagated and updated."""
    # Create a base logger
    base_logger = MockLogger()
    
    # Create a derived logger with added context
    request_logger = base_logger.with_context(request_id="req-123")
    
    # Add more context
    user_logger = request_logger.with_context(user_id="user-456")
    
    # Log with the most specific logger
    user_logger.info("User logged in", device="mobile")
    
    # Check the log entry
    assert len(user_logger.logs) == 1
    log_entry = user_logger.logs[0]
    assert log_entry["level"] == LogLevel.INFO
    assert log_entry["message"] == "User logged in"
    assert log_entry["context"] == {
        "request_id": "req-123",
        "user_id": "user-456"
    }
    assert log_entry["extra"] == {"device": "mobile"}
    
    # Verify the original loggers weren't modified
    assert len(base_logger.logs) == 0
    assert len(request_logger.logs) == 0


def test_log_levels():
    """Test all log levels work correctly."""
    logger = MockLogger()
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    assert len(logger.logs) == 5
    assert logger.logs[0]["level"] == LogLevel.DEBUG
    assert logger.logs[1]["level"] == LogLevel.INFO
    assert logger.logs[2]["level"] == LogLevel.WARNING
    assert logger.logs[3]["level"] == LogLevel.ERROR
    assert logger.logs[4]["level"] == LogLevel.CRITICAL
