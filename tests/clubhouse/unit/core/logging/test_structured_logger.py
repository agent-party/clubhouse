"""Tests for the structured logger implementation."""
import pytest
import json
import asyncio
import tempfile
import os
from typing import Dict, Any, List
import contextlib
import io

from clubhouse.core.logging import (
    LoggingConfig, 
    LogHandlerConfig, 
    get_logger, 
    configure_logging,
    LogLevel,
    StructuredLogger,
    ConsoleHandler,
    LoggerFactory
)
from clubhouse.core.utils.datetime_utils import utc_now


class CaptureHandler:
    """Test handler that captures log entries for assertion."""
    
    def __init__(self):
        self.entries = []
    
    def handle(self, entry):
        """Capture the entry."""
        self.entries.append(entry.to_dict())
    
    def shutdown(self):
        """No-op for testing."""
        pass


@pytest.fixture
def capture_handler():
    """Fixture to create a capture handler."""
    return CaptureHandler()


@pytest.fixture
def configured_logger(capture_handler):
    """Fixture to create a configured logger with a capture handler."""
    # Configure logging with a test handler
    config = LoggingConfig(
        default_level="DEBUG",
        include_timestamps=True,
        propagate_context=True
    )
    
    logger = StructuredLogger(
        name="test_logger",
        config=config,
        handlers=[capture_handler]
    )
    
    return logger, capture_handler


def test_basic_logging(configured_logger):
    """Test basic logging functionality."""
    logger, handler = configured_logger
    
    logger.info("Test message")
    
    assert len(handler.entries) == 1
    entry = handler.entries[0]
    
    assert entry["message"] == "Test message"
    assert entry["level"] == "INFO"
    assert "timestamp" in entry
    assert "logger" in entry["context"]
    assert entry["context"]["logger"] == "test_logger"


def test_log_levels(configured_logger):
    """Test that log levels work correctly."""
    logger, handler = configured_logger
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    assert len(handler.entries) == 5
    
    levels = [entry["level"] for entry in handler.entries]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    messages = [entry["message"] for entry in handler.entries]
    assert messages == [
        "Debug message",
        "Info message",
        "Warning message",
        "Error message",
        "Critical message"
    ]


def test_context_propagation(configured_logger):
    """Test that context is properly propagated."""
    logger, handler = configured_logger
    
    # Base logger
    logger.info("Base message")
    
    # Create a derived logger with context
    request_logger = logger.with_context(request_id="req-123")
    request_logger.info("Request message")
    
    # Add more context
    user_logger = request_logger.with_context(user_id="user-456")
    user_logger.info("User message")
    
    # Original context remains unchanged
    request_logger.info("Another request message")
    
    assert len(handler.entries) == 4
    
    # Base logger has no added context
    assert "request_id" not in handler.entries[0]["context"]
    
    # Request logger has request_id
    assert handler.entries[1]["context"]["request_id"] == "req-123"
    assert "user_id" not in handler.entries[1]["context"]
    
    # User logger has both contexts
    assert handler.entries[2]["context"]["request_id"] == "req-123"
    assert handler.entries[2]["context"]["user_id"] == "user-456"
    
    # Request logger still has only request_id
    assert handler.entries[3]["context"]["request_id"] == "req-123"
    assert "user_id" not in handler.entries[3]["context"]


def test_extra_data(configured_logger):
    """Test that extra data is included in log entries."""
    logger, handler = configured_logger
    
    logger.info(
        "User action", 
        action="login", 
        duration_ms=150, 
        success=True
    )
    
    assert len(handler.entries) == 1
    entry = handler.entries[0]
    
    assert entry["data"]["action"] == "login"
    assert entry["data"]["duration_ms"] == 150
    assert entry["data"]["success"] is True


def test_exception_logging(configured_logger):
    """Test logging with exception information."""
    logger, handler = configured_logger
    
    try:
        # Raise an exception
        raise ValueError("Test exception")
    except ValueError as e:
        # Log with the exception
        logger.error("An error occurred", exc_info=e)
    
    assert len(handler.entries) == 1
    entry = handler.entries[0]
    
    assert entry["message"] == "An error occurred"
    assert entry["level"] == "ERROR"
    assert "exception" in entry
    assert entry["exception"]["type"] == "ValueError"
    assert entry["exception"]["message"] == "Test exception"
    assert "traceback" in entry["exception"]
    assert "ValueError" in entry["exception"]["traceback"]


def test_console_handler():
    """Test the console handler output."""
    # Use StringIO to capture output
    output = io.StringIO()
    
    # Create a console handler that writes JSON to our StringIO
    handler = ConsoleHandler(
        level=LogLevel.INFO,
        format="json",
        output_stream=output,
        use_colors=False
    )
    
    # Create a logger with this handler
    config = LoggingConfig(default_level="INFO")
    logger = StructuredLogger(name="console_test", config=config, handlers=[handler])
    
    # Log a message
    logger.info("Console test", test_id=123)
    
    # Get the output and parse the JSON
    output_str = output.getvalue().strip()
    log_entry = json.loads(output_str)
    
    # Verify the content
    assert log_entry["message"] == "Console test"
    assert log_entry["level"] == "INFO"
    assert log_entry["context"]["logger"] == "console_test"
    assert log_entry["data"]["test_id"] == 123


def test_file_handler():
    """Test the file handler output."""
    # Create a temporary directory for log files
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test.log")
        
        # Configure a logger with the file handler
        config = LoggingConfig(
            handlers=[
                LogHandlerConfig(
                    type="file",
                    level="INFO",
                    format="json",
                    options={"filename": log_file}
                )
            ]
        )
        
        # Apply the configuration
        configure_logging(config)
        
        # Get a logger and log something
        logger = get_logger("file_test")
        logger.info("File test", test_id=456)
        
        # Read the log file
        with open(log_file, "r") as f:
            log_content = f.read().strip()
        
        # Parse the JSON and verify
        log_entry = json.loads(log_content)
        assert log_entry["message"] == "File test"
        assert log_entry["level"] == "INFO"
        assert log_entry["context"]["logger"] == "file_test"
        assert log_entry["data"]["test_id"] == 456


@pytest.mark.asyncio
async def test_async_context():
    """Test context propagation across async boundaries."""
    # Configure a logger with a capture handler
    capture_handler = CaptureHandler()
    config = LoggingConfig(default_level="INFO")
    
    # Set up the factory with our capture handler to intercept all logs
    factory = LoggerFactory.get_instance()
    factory._config = config
    factory._handlers = [capture_handler]
    
    # Define an async function that logs using the factory logger
    async def do_work():
        # This should have the request_id in context
        get_logger("async_test").info("Inside async function")
        return "done"
    
    # Run with context
    await StructuredLogger.with_async_context(
        {"request_id": "async-123"},
        do_work()
    )
    
    # Verify the context was propagated
    assert len(capture_handler.entries) == 1
    entry = capture_handler.entries[0]
    assert entry["message"] == "Inside async function"
    assert entry["context"]["request_id"] == "async-123"


def test_logger_factory():
    """Test that the logger factory creates consistent loggers."""
    # Configure logging
    capture_handler = CaptureHandler()
    config = LoggingConfig(default_level="INFO")
    
    # Create a logger directly
    logger1 = StructuredLogger(name="factory_test", config=config, handlers=[capture_handler])
    
    # Create loggers through the factory
    from clubhouse.core.logging.factory import LoggerFactory
    factory = LoggerFactory()
    factory._config = config
    factory._handlers = [capture_handler]
    
    logger2 = factory.get_logger("factory_test")
    logger3 = factory.get_logger("factory_test")
    
    # All loggers with the same name should share the same behavior
    logger1.info("Logger 1")
    logger2.info("Logger 2")
    logger3.info("Logger 3")
    
    assert len(capture_handler.entries) == 3
    
    # Factory should return the same instance for the same name
    assert logger2 is logger3
