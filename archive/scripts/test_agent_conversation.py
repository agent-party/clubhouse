"""
Tests for the conversational features of agent_cli.py.

This module contains tests specifically focused on the agent conversation
functionality, including:
1. Multi-agent conversation setup
2. Message handling between agents
3. Tool usage within conversations
4. Conversation context tracking
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from uuid import uuid4
from datetime import datetime

# Add the project root to the Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the CLI components we want to test
from scripts.agent_cli import (
    AgentCLI, 
    OutputFormatter,
    EventHandler,
    CommandParser
)
from clubhouse.agents.capability import CapabilityResult
from clubhouse.agents.examples.assistant_agent import AssistantAgent
from clubhouse.agents.capabilities.conversation_capability import ConversationCapability


class TestAgentConversation:
    """Tests for the agent conversation functionality."""
    
    @pytest.fixture
    async def setup_cli(self):
        """Set up a CLI instance for testing."""
        # Create mock formatter and event handler
        formatter = MagicMock(spec=OutputFormatter)
        event_handler = MagicMock(spec=EventHandler)
        parser = MagicMock(spec=CommandParser)
        
        # Create a patched AgentCLI instance
        with patch('scripts.agent_cli.AgentCLI._create_agent') as mock_create_agent:
            # Mock the agent creation
            primary_agent = MagicMock(spec=AssistantAgent)
            secondary_agent = MagicMock(spec=AssistantAgent)
            
            # Setup agent capabilities
            primary_agent._capabilities = {
                'conversation': AsyncMock(spec=ConversationCapability),
                'memory': AsyncMock(),
                'llm': AsyncMock()
            }
            secondary_agent._capabilities = {
                'conversation': AsyncMock(spec=ConversationCapability),
                'memory': AsyncMock(),
                'llm': AsyncMock()
            }
            
            # Set agent properties
            primary_agent.name = "Primary Agent"
            primary_agent.description = "Primary test agent"
            secondary_agent.name = "Secondary Agent"
            secondary_agent.description = "Secondary test agent"
            
            # Configure get_capability to return appropriate capability
            primary_agent.get_capability.side_effect = lambda name: primary_agent._capabilities.get(name)
            secondary_agent.get_capability.side_effect = lambda name: secondary_agent._capabilities.get(name)
            
            # Setup mock agent creation to return our mock agents
            def create_agent_side_effect(agent_id):
                if agent_id == "primary":
                    return primary_agent
                else:
                    return secondary_agent
            
            mock_create_agent.side_effect = create_agent_side_effect
            
            # Create CLI instance with patches
            cli = AgentCLI()
            
            # Replace some properties with our mocks
            cli.formatter = formatter
            cli.event_handler = event_handler
            cli.parser = parser
            
            # Override agents dict to ensure we have our mock agents
            cli.agents = {
                "primary": primary_agent,
                "secondary": secondary_agent
            }
            
            return cli
    
    @pytest.mark.asyncio
    async def test_create_secondary_agent(self, setup_cli):
        """Test creating a secondary agent."""
        cli = await setup_cli
        
        # Ensure both agents are created
        assert "primary" in cli.agents
        assert len(cli.agents) >= 1
        
        # Create a secondary agent
        await cli.create_secondary_agent("test_agent", "Test Agent", "Test description")
        
        # Verify the agent was created with proper properties
        cli.formatter.print_success.assert_called_with("Created secondary agent 'test_agent'")
        
        # Check agent was added to the dict
        assert "test_agent" in cli.agents
    
    @pytest.mark.asyncio
    async def test_switch_active_agent(self, setup_cli):
        """Test switching between agents."""
        cli = await setup_cli
        
        # Create a secondary agent
        await cli.create_secondary_agent("test_agent")
        
        # Switch to the secondary agent
        await cli.switch_active_agent("test_agent")
        
        # Verify the active agent was switched
        assert cli.active_agent_id == "test_agent"
        assert cli.agent == cli.agents["test_agent"]
        cli.formatter.print_success.assert_called_with("Switched to agent 'test_agent'")
    
    @pytest.mark.asyncio
    async def test_start_conversation_mode(self, setup_cli):
        """Test starting conversation mode."""
        cli = await setup_cli
        
        # Start conversation mode
        await cli.start_conversation_mode()
        
        # Verify conversation mode is started
        assert cli.in_conversation_mode is True
        assert cli.conversation_id is not None
        
        # Verify conversation is initialized in both agents
        for agent_id, agent in cli.agents.items():
            conversation_capability = agent.get_capability("conversation")
            assert conversation_capability.execute.called
            
            # Check that execute was called with proper parameters
            call_args = conversation_capability.execute.call_args_list[0][1]
            assert call_args["operation"] == "initialize"
            assert call_args["conversation_id"] == cli.conversation_id
            assert "context" in call_args
            
        # Verify success message
        cli.formatter.print_success.assert_called_with("Started conversation mode")
    
    @pytest.mark.asyncio
    async def test_process_conversation_message_from_user(self, setup_cli):
        """Test processing a message in conversation mode from the user."""
        cli = await setup_cli
        cli.in_conversation_mode = True
        cli.conversation_id = "test-convo-id"
        
        # Set up the secondary agent's LLM capability response
        secondary_agent = cli.agents["secondary"]
        secondary_llm = secondary_agent.get_capability("llm")
        secondary_convo = secondary_agent.get_capability("conversation")
        
        # Mock conversation history result
        history_result = MagicMock()
        history_result.result = {"history": []}
        secondary_convo.execute.return_value = history_result
        
        # Mock LLM response
        llm_result = MagicMock()
        llm_result.result = {"data": {"response": "This is a test response."}}
        secondary_llm.execute.return_value = llm_result
        
        # Process a user message
        test_message = "Hello, this is a test message"
        await cli.process_conversation_message(test_message, "user")
        
        # Verify the user message was displayed
        cli.formatter.print_info.assert_called_with(f"You: {test_message}")
        
        # Verify the message was added to the conversation
        secondary_convo.execute.assert_called()
        add_message_call = [
            call for call in secondary_convo.execute.call_args_list 
            if call[1].get("operation") == "add_message"
        ][0]
        assert add_message_call[1]["message"] == test_message
        assert add_message_call[1]["conversation_id"] == cli.conversation_id
        assert add_message_call[1]["metadata"]["sender"] == "user"
        
        # Verify the LLM was called to generate a response
        secondary_llm.execute.assert_called()
        assert "prompt" in secondary_llm.execute.call_args[1]
        assert "system_prompt" in secondary_llm.execute.call_args[1]
        
        # Verify the agent response was processed
        cli.formatter.print_success.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_conversation_message_with_tool_usage(self, setup_cli):
        """Test processing a message in conversation mode that triggers tool usage."""
        cli = await setup_cli
        cli.in_conversation_mode = True
        cli.conversation_id = "test-convo-id"
        
        # Set up the secondary agent's capabilities
        secondary_agent = cli.agents["secondary"]
        secondary_llm = secondary_agent.get_capability("llm")
        secondary_convo = secondary_agent.get_capability("conversation")
        secondary_memory = secondary_agent.get_capability("memory")
        
        # Mock conversation history result
        history_result = MagicMock()
        history_result.result = {"history": []}
        
        # Mock memory capability result
        memory_result = MagicMock()
        memory_result.result = {"status": "success", "data": "Test memory data"}
        secondary_memory.execute.return_value = memory_result
        
        # Configure response that contains a tool command
        tool_response = "/memory operation=retrieve query=test"
        llm_result = MagicMock()
        llm_result.result = {"data": {"response": tool_response}}
        
        # Mock the parser to return the correct capability and parameters
        cli.parser.parse_command.return_value = ("memory", {"operation": "retrieve", "query": "test"})
        
        # Create side effect to return different mock results for different calls
        def convo_side_effect(**kwargs):
            if kwargs.get("operation") == "get_history":
                return history_result
            return MagicMock()
        
        secondary_convo.execute.side_effect = convo_side_effect
        secondary_llm.execute.return_value = llm_result
        
        # Process a user message that should trigger tool usage
        test_message = "Can you retrieve my test memory?"
        await cli.process_conversation_message(test_message, "user")
        
        # Verify the LLM was called
        secondary_llm.execute.assert_called()
        
        # Verify memory capability was called due to the tool command
        secondary_memory.execute.assert_called_with(operation="retrieve", query="test")
        
        # Verify the tool result was part of the agent's response
        # The response should be processed, which calls formatter.print_success
        assert cli.formatter.print_success.called
    
    @pytest.mark.asyncio
    async def test_end_conversation_mode(self, setup_cli):
        """Test ending conversation mode."""
        cli = await setup_cli
        
        # Start conversation mode
        cli.in_conversation_mode = True
        cli.conversation_id = "test-convo-id"
        
        # End conversation mode
        await cli.end_conversation_mode()
        
        # Verify conversation mode is ended
        assert cli.in_conversation_mode is False
        cli.formatter.print_success.assert_called_with("Ended conversation mode")
    
    @pytest.mark.asyncio
    async def test_command_processing_in_conversation_mode(self, setup_cli):
        """Test processing commands while in conversation mode."""
        cli = await setup_cli
        
        # Start conversation mode
        cli.in_conversation_mode = True
        
        # Mock process_conversation_message to verify it's called for non-command inputs
        cli.process_conversation_message = AsyncMock()
        
        # Process a non-command input
        continue_loop = await cli.process_command("Hello agents!")
        
        # Verify message was passed to process_conversation_message
        cli.process_conversation_message.assert_called_with("Hello agents!")
        assert continue_loop is True
        
        # Test processing a command
        cli.end_conversation_mode = AsyncMock()
        continue_loop = await cli.process_command("/endconvo")
        
        # Verify the command was processed
        cli.end_conversation_mode.assert_called_once()
        assert continue_loop is True
        
        # Test exit command in conversation mode
        continue_loop = await cli.process_command("/exit")
        assert continue_loop is False


if __name__ == "__main__":
    pytest.main(["-v", __file__])
