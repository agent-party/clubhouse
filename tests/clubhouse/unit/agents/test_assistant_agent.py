"""
Tests for the AssistantAgent implementation.
"""
import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from uuid import UUID

from clubhouse.agents.agent_protocol import AgentMessage, AgentResponse, ApprovalStatus
from clubhouse.agents.state import AgentStateManager, AgentState
from clubhouse.agents.communication import AgentCommunicationService, EnhancedAgentMessage, RoutingStrategy
from clubhouse.agents.examples.assistant_agent import (
    AssistantAgent,
    SearchCapability,
    SummarizeCapability
)


class TestSearchCapability:
    """Test cases for the SearchCapability."""
    
    @pytest.fixture(scope="function")
    def capability(self):
        """Set up the test capability."""
        capability = SearchCapability(requires_human_approval=False)
        return capability
    
    @pytest.fixture(scope="function")
    def event_handler(self, capability):
        """Set up the event handler and register it."""
        handler = Mock()
        capability.register_event_handler("search_started", handler)
        capability.register_event_handler("search_completed", handler)
        return handler
    
    @pytest.mark.asyncio
    async def test_capability_properties(self, capability):
        """Test the capability's metadata properties."""
        assert capability.name == "search"
        assert "Search for information" in capability.description
        assert "query" in capability.parameters
        assert "max_results" in capability.parameters
        assert not capability.requires_human_approval()
    
    @pytest.mark.asyncio
    async def test_execute_with_valid_parameters(self, capability, event_handler):
        """Test executing the capability with valid parameters."""
        # Execute the capability
        result = await capability.execute(query="test query", max_results=3)
        
        # Verify result structure
        assert result["status"] == "success"
        assert "data" in result
        assert result["data"]["query"] == "test query"
        assert len(result["data"]["results"]) == 3
        
        # Verify events were emitted
        assert event_handler.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_parameters(self, capability, event_handler):
        """Test executing the capability with missing parameters."""
        # Execute without required query parameter
        result = await capability.execute(max_results=3)
        
        # Verify error response
        assert result["status"] == "error"
        assert "Missing required parameter" in result["error"]
        
        # No events should be emitted
        assert event_handler.call_count == 0


class TestSummarizeCapability:
    """Test cases for the SummarizeCapability."""
    
    @pytest.fixture(scope="function")
    def capability(self):
        """Set up the test capability."""
        capability = SummarizeCapability(requires_human_approval=True)
        return capability
    
    @pytest.fixture(scope="function")
    def event_handler(self, capability):
        """Set up the event handler and register it."""
        handler = Mock()
        capability.register_event_handler("summarize_started", handler)
        capability.register_event_handler("summarize_completed", handler)
        return handler
    
    @pytest.mark.asyncio
    async def test_capability_properties(self, capability):
        """Test the capability's metadata properties."""
        assert capability.name == "summarize"
        assert "Summarize the given content" in capability.description
        assert "content" in capability.parameters
        assert "max_length" in capability.parameters
        assert capability.requires_human_approval()
    
    @pytest.mark.asyncio
    async def test_execute_with_valid_parameters(self, capability, event_handler):
        """Test executing the capability with valid parameters."""
        # Execute the capability
        content = "This is a test content that needs to be summarized. " * 10
        result = await capability.execute(content=content, max_length=20)
        
        # Verify result structure
        assert result["status"] == "success"
        assert "data" in result
        assert "summary" in result["data"]
        assert len(result["data"]["summary"].split()) <= 21  # account for ellipsis
        assert "token_usage" in result["data"]
        
        # Verify events were emitted
        assert event_handler.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_missing_parameters(self, capability, event_handler):
        """Test executing the capability with missing parameters."""
        # Execute without required content parameter
        result = await capability.execute(max_length=20)
        
        # Verify error response
        assert result["status"] == "error"
        assert "Missing required parameter" in result["error"]
        
        # No events should be emitted
        assert event_handler.call_count == 0


class TestAssistantAgent:
    """Test cases for the AssistantAgent."""
    
    @pytest.fixture(scope="function")
    def mock_state_manager(self):
        """Create a mock agent state manager."""
        manager = MagicMock(spec=AgentStateManager)
        
        # Set up common return values
        manager.initialize_agent_state.return_value = UUID("12345678-1234-5678-1234-567812345678")
        manager.get_agent_state.return_value = AgentState.READY
        manager.update_agent_state.return_value = True
        
        return manager
    
    @pytest.fixture(scope="function")
    def mock_communication_service(self):
        """Create a mock agent communication service."""
        service = MagicMock(spec=AgentCommunicationService)
        
        # Set up mock methods
        service.register_handler.return_value = None
        service.unregister_handler.return_value = None
        
        return service
    
    @pytest.fixture(scope="function")
    def assistant_agent(self, mock_state_manager, mock_communication_service):
        """Create an assistant agent with mock services."""
        agent = AssistantAgent(
            agent_id="test-assistant",
            name="Test Assistant",
            description="A test assistant agent",
            state_manager=mock_state_manager,
            communication_service=mock_communication_service
        )
        return agent
    
    @pytest.mark.asyncio
    async def test_initialization(self, assistant_agent, mock_state_manager, mock_communication_service):
        """Test agent initialization."""
        # Initialize the agent
        success = await assistant_agent.initialize()
        
        # Verify initialization
        assert success is True
        assert mock_state_manager.initialize_agent_state.called
        assert mock_state_manager.update_agent_state.call_count == 2
        assert mock_communication_service.register_handler.called
    
    @pytest.mark.asyncio
    async def test_shutdown(self, assistant_agent, mock_state_manager, mock_communication_service):
        """Test agent shutdown."""
        # Initialize first
        await assistant_agent.initialize()
        
        # Shutdown the agent
        success = await assistant_agent.shutdown()
        
        # Verify shutdown
        assert success is True
        assert mock_state_manager.update_agent_state.called
        assert mock_communication_service.unregister_handler.called
    
    @pytest.mark.asyncio
    async def test_process_message_ping(self, assistant_agent, mock_state_manager):
        """Test processing a ping message."""
        # Create a ping message
        message = {
            "message_id": "test-msg-1",
            "sender": "test-user",
            "content": {
                "command": "ping"
            }
        }
        
        # Process the message
        response = await assistant_agent.process_message(message)
        
        # Verify state updates
        assert mock_state_manager.update_agent_state.call_count >= 2
        
        # Verify response
        assert response["in_response_to"] == "test-msg-1"
        assert response["sender"] == "test-assistant"
        assert response["status"] == "success"
        assert response["result"]["reply"] == "pong"
    
    @pytest.mark.asyncio
    async def test_process_message_capabilities(self, assistant_agent):
        """Test processing a capabilities query message."""
        # Create a capabilities message
        message = {
            "message_id": "test-msg-2",
            "sender": "test-user",
            "content": {
                "command": "capabilities"
            }
        }
        
        # Process the message
        response = await assistant_agent.process_message(message)
        
        # Verify response
        assert response["status"] == "success"
        assert "capabilities" in response["result"]
        assert len(response["result"]["capabilities"]) == 2
        
        # Check capabilities in the response
        capabilities = {cap["name"] for cap in response["result"]["capabilities"]}
        assert "search" in capabilities
        assert "summarize" in capabilities
    
    @pytest.mark.asyncio
    async def test_process_message_search(self, assistant_agent):
        """Test processing a search message."""
        # Create a search message
        message = {
            "message_id": "test-msg-3",
            "sender": "test-user",
            "content": {
                "command": "search",
                "query": "test search",
                "max_results": 2
            }
        }
        
        # Process the message
        response = await assistant_agent.process_message(message)
        
        # Verify response
        assert response["status"] == "success"
        assert response["result"]["status"] == "success"
        assert "data" in response["result"]
        assert response["result"]["data"]["query"] == "test search"
        assert len(response["result"]["data"]["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_process_message_summarize(self, assistant_agent):
        """Test processing a summarize message."""
        # Create a summarize message
        message = {
            "message_id": "test-msg-4",
            "sender": "test-user",
            "content": {
                "command": "summarize",
                "content": "This is a test content that needs to be summarized.",
                "max_length": 5
            }
        }
        
        # Process the message
        response = await assistant_agent.process_message(message)
        
        # Verify response
        assert response["status"] == "success"
        assert response["result"]["status"] == "success"
        assert "data" in response["result"]
        assert "summary" in response["result"]["data"]
        assert "token_usage" in response["result"]["data"]
    
    @pytest.mark.asyncio
    async def test_process_message_unknown_command(self, assistant_agent):
        """Test processing a message with unknown command."""
        # Create a message with unknown command
        message = {
            "message_id": "test-msg-5",
            "sender": "test-user",
            "content": {
                "command": "unknown_command"
            }
        }
        
        # Process the message
        response = await assistant_agent.process_message(message)
        
        # Verify error response
        assert response["status"] == "success"  # The processing itself succeeded
        assert response["result"]["status"] == "error"
        assert "Unknown command" in response["result"]["error"]
    
    @pytest.mark.asyncio
    async def test_health_check(self, assistant_agent, mock_state_manager):
        """Test the agent health check."""
        # Run health check
        health_info = await assistant_agent.health_check()
        
        # Verify health information
        assert health_info["agent_id"] == "test-assistant"
        assert health_info["status"] == "healthy"
        assert "capabilities" in health_info
        assert len(health_info["capabilities"]) == 2
        assert "total_cost" in health_info
        assert "timestamp" in health_info
    
    @pytest.mark.asyncio
    async def test_reset(self, assistant_agent, mock_state_manager):
        """Test resetting the agent."""
        # Initialize the agent first
        await assistant_agent.initialize()
        
        # Record some costs
        assistant_agent.record_operation_cost(0.1, "test_operation")
        assert assistant_agent.get_total_cost() > 0
        
        # Reset the agent
        success = await assistant_agent.reset()
        
        # Verify reset
        assert success is True
        assert mock_state_manager.update_agent_state.call_count >= 4
        assert assistant_agent.get_total_cost() == 0
    
    @pytest.mark.asyncio
    async def test_handle_message(self, assistant_agent):
        """Test handling a message via the MessageHandlerProtocol."""
        # Create an enhanced agent message
        message = EnhancedAgentMessage.create(
            sender="test-user",
            content={"command": "ping"},
            routing_strategy=RoutingStrategy.DIRECT,  # Use enum instead of string
            recipient="test-assistant"
        )
        
        # Handle the message
        response = await assistant_agent.handle_message(message)
        
        # Verify the response
        assert response is not None
        assert response["sender"] == "test-assistant"
        assert response["routing"]["recipient"] == "test-user"
        assert "content" in response
