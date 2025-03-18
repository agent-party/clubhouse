"""
Tests for the agent_cli.py script.

This module contains tests for the Agent CLI script, focusing on:
1. Command parsing functionality
2. Event handling
3. Capability execution
4. Result formatting
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Add the project root to the Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the CLI components we want to test
from scripts.agent_cli import (
    CommandParser,
    OutputFormatter,
    EventHandler,
    AgentCLI
)
from clubhouse.agents.capability import CapabilityResult, BaseCapability


class TestCommandParser:
    """Tests for the CommandParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = MagicMock()
        self.parser = CommandParser(self.formatter)
    
    def test_parse_simple_command(self):
        """Test parsing a simple command with no parameters."""
        capability_name, params = self.parser.parse_command("/memory")
        assert capability_name == "memory"
        assert params == {}
    
    def test_parse_command_with_basic_params(self):
        """Test parsing a command with basic key=value parameters."""
        capability_name, params = self.parser.parse_command("/memory operation=create content=test")
        assert capability_name == "memory"
        assert params == {"operation": "create", "content": "test"}
    
    def test_parse_command_with_quoted_params(self):
        """Test parsing a command with quoted parameter values."""
        capability_name, params = self.parser.parse_command('/memory operation=create content="This is a test"')
        assert capability_name == "memory"
        assert params == {"operation": "create", "content": "This is a test"}
    
    def test_parse_command_with_json_params(self):
        """Test parsing a command with JSON parameters."""
        capability_name, params = self.parser.parse_command('/memory {"operation": "create", "content": "Test content"}')
        assert capability_name == "memory"
        assert params == {"operation": "create", "content": "Test content"}
    
    def test_convert_value_types(self):
        """Test conversion of parameter value types."""
        assert self.parser._convert_value_type("42") == 42
        assert self.parser._convert_value_type("3.14") == 3.14
        assert self.parser._convert_value_type("true") is True
        assert self.parser._convert_value_type("false") is False
        assert self.parser._convert_value_type("null") is None
        assert self.parser._convert_value_type("test") == "test"
    
    def test_invalid_command_format(self):
        """Test handling of invalid command format."""
        capability_name, params = self.parser.parse_command("memory create")
        assert capability_name is None
        assert params is None
        self.formatter.print_error.assert_called_once()


class TestOutputFormatter:
    """Tests for the OutputFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = OutputFormatter(use_color=False)
    
    def test_format_dict_result(self):
        """Test formatting a dictionary result."""
        result = {"key1": "value1", "key2": 42}
        formatted = self.formatter.format_result(result)
        # Should be pretty-printed JSON
        parsed = json.loads(formatted)
        assert parsed == result
    
    def test_format_non_dict_result(self):
        """Test formatting a non-dictionary result."""
        result = "Test result"
        formatted = self.formatter.format_result(result)
        assert formatted == result
    
    def test_display_capability_result(self):
        """Test displaying a CapabilityResult object."""
        # Create a mock for print to capture output
        with patch("builtins.print") as mock_print:
            result = CapabilityResult(
                result={"data": "test"},
                metadata={"cost": 0.01}
            )
            self.formatter.display_result(result)
            # Check that both the result and metadata were printed
            assert mock_print.call_count > 0
            # Extract calls that contain our test data
            result_calls = [call for call in mock_print.call_args_list if "test" in str(call)]
            metadata_calls = [call for call in mock_print.call_args_list if "0.01" in str(call)]
            assert len(result_calls) > 0
            assert len(metadata_calls) > 0


class TestEventHandler:
    """Tests for the EventHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = MagicMock()
        self.handler = EventHandler(self.formatter)
    
    @pytest.mark.asyncio
    async def test_handle_event(self):
        """Test handling of capability events."""
        # Test with events enabled
        self.handler.events_enabled = True
        await self.handler.handle_event("test_event", param1="value1", param2=42)
        self.formatter.display_event.assert_called_with("test_event", {"param1": "value1", "param2": 42})
        
        # Test with events disabled
        self.formatter.reset_mock()
        self.handler.events_enabled = False
        await self.handler.handle_event("test_event", param1="value1", param2=42)
        self.formatter.display_event.assert_not_called()


class MockCapability(BaseCapability):
    """Mock capability class for testing."""
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def description(self) -> str:
        return "Mock capability for testing"
    
    @property
    def parameters(self) -> dict:
        return {
            "param1": {"description": "Test parameter", "required": True},
            "param2": {"description": "Optional parameter", "required": False}
        }
    
    async def execute(self, **kwargs):
        """Mock execution method."""
        return CapabilityResult(
            result={"status": "success", "params": kwargs},
            metadata={"cost": 0.01}
        )


class TestAgentCLI:
    """Tests for the AgentCLI class."""
    
    @pytest.mark.asyncio
    async def test_process_help_command(self):
        """Test processing of the /help command."""
        # Create a mock CLI instance
        cli = MagicMock()
        cli.display_help = AsyncMock()
        
        # Create an instance method from the AgentCLI.process_command
        process_command = AgentCLI.process_command.__get__(cli, AgentCLI)
        
        # Call the method with a help command
        result = await process_command("/help")
        
        # Verify the help command was processed
        cli.display_help.assert_called_once()
        assert result is True  # Should continue the REPL
    
    @pytest.mark.asyncio
    async def test_process_exit_command(self):
        """Test processing of the /exit command."""
        # Create a mock CLI instance
        cli = MagicMock()
        
        # Create an instance method from the AgentCLI.process_command
        process_command = AgentCLI.process_command.__get__(cli, AgentCLI)
        
        # Call the method with an exit command
        result = await process_command("/exit")
        
        # Verify the exit command was processed
        assert result is False  # Should exit the REPL
    
    @pytest.mark.asyncio
    async def test_process_capability_command(self):
        """Test processing of a capability command."""
        # Create a mock CLI instance
        cli = MagicMock()
        cli.parser.parse_command.return_value = ("memory", {"operation": "create"})
        cli.execute_capability = AsyncMock()
        
        # Create an instance method from the AgentCLI.process_command
        process_command = AgentCLI.process_command.__get__(cli, AgentCLI)
        
        # Call the method with a capability command
        result = await process_command("/memory operation=create")
        
        # Verify the capability command was processed
        cli.execute_capability.assert_called_with("memory", {"operation": "create"})
        assert result is True  # Should continue the REPL
    
    @pytest.mark.asyncio
    async def test_execute_capability(self):
        """Test execution of a capability."""
        # Create a mock agent
        agent = MagicMock()
        capability = AsyncMock()
        capability.execute.return_value = CapabilityResult(
            result={"status": "success"},
            metadata={"cost": 0.01}
        )
        agent.get_capability.return_value = capability
        
        # Create a mock CLI instance
        cli = MagicMock()
        cli.agent = agent
        cli.store_command_in_history = AsyncMock()
        
        # Create an instance method from the AgentCLI.execute_capability
        execute_capability = AgentCLI.execute_capability.__get__(cli, AgentCLI)
        
        # Call the method with capability parameters
        await execute_capability("memory", {"operation": "create"})
        
        # Verify the capability was executed
        agent.get_capability.assert_called_with("memory")
        capability.execute.assert_called_with(operation="create")
        cli.store_command_in_history.assert_called_once()
        cli.formatter.display_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_capabilities(self):
        """Test listing of available capabilities."""
        # Create mock capabilities
        capability1 = MockCapability()
        capability2 = MagicMock()
        capability2.name = "test"
        capability2.description = "Test capability"
        capability2.parameters = {"param": "Test param"}
        
        # Create a mock agent
        agent = MagicMock()
        agent._capabilities = {
            "mock": capability1,
            "test": capability2
        }
        
        # Create a mock CLI instance with the mock agent
        cli = MagicMock()
        cli.agent = agent
        cli.formatter = OutputFormatter(use_color=False)
        
        # Create a spy for print to capture output
        with patch("builtins.print") as mock_print:
            # Create an instance method from the AgentCLI.list_capabilities
            list_capabilities = AgentCLI.list_capabilities.__get__(cli, AgentCLI)
            
            # Call the method
            await list_capabilities()
            
            # Verify the capabilities were listed
            capability1_calls = [call for call in mock_print.call_args_list if "mock" in str(call)]
            capability2_calls = [call for call in mock_print.call_args_list if "test" in str(call)]
            assert len(capability1_calls) > 0
            assert len(capability2_calls) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
