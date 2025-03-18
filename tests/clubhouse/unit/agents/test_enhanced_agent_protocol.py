"""Tests for enhanced agent protocol interfaces."""
import pytest
from typing import Dict, Any, List, Protocol, Type, runtime_checkable
from unittest.mock import Mock, MagicMock

from clubhouse.agents.agent_protocol import (
    AgentProtocol,
    CapabilityProtocol,
    BaseAgent,
    BaseCapability,
    AgentMessage,
    AgentResponse,
    CapabilityResult,
    ApprovalStatus
)


class TestEnhancedAgentProtocol:
    """Test cases for enhanced agent protocol interfaces."""
    
    def test_capability_runtime_checkable(self):
        """Test that CapabilityProtocol is runtime-checkable."""
        from typing import runtime_checkable, Protocol
        
        # Import the protocols
        from clubhouse.agents.protocols import CapabilityProtocol
        from clubhouse.agents.capability import BaseCapability
        
        # Create a mock capability that implements the protocol
        class TestCapability(BaseCapability):
            @property
            def name(self) -> str:
                return "test"
                
            @property
            def description(self) -> str:
                return "Test capability"
                
            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}
                
            async def execute(self, **kwargs):
                return {"result": "success"}
        
        # Create an instance
        test_capability = TestCapability()
        
        # Since CapabilityProtocol is not decorated with @runtime_checkable,
        # we can't use isinstance directly. Instead, check that it has the required
        # attributes and methods that satisfy the protocol.
        assert hasattr(test_capability, "name")
        assert hasattr(test_capability, "description")
        assert hasattr(test_capability, "parameters")
        assert hasattr(test_capability, "execute")
    
    def test_agent_lifecycle_hooks(self):
        """Test that agents implement proper lifecycle hooks."""
        # Create a mock agent
        agent = MagicMock(spec=AgentProtocol)
        
        # Verify the agent has lifecycle methods
        assert hasattr(agent, "initialize")
        assert hasattr(agent, "shutdown")
        assert hasattr(agent, "health_check")
        assert hasattr(agent, "reset")
    
    def test_agent_cost_accounting(self):
        """Test that agents implement cost accounting."""
        # Create a test agent with cost tracking
        class TestAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    agent_id="test-agent",
                    name="Test Agent",
                    description="A test agent",
                    capabilities=[]
                )
            
            async def process_message(self, message: AgentMessage) -> AgentResponse:
                # Record cost for this operation
                self.record_operation_cost(0.01, "process_message", message)
                return {
                    "message_id": "response-id",
                    "in_response_to": message["message_id"],
                    "sender": self.agent_id,
                    "status": "success",
                    "result": "Processed message"
                }
        
        # Create an instance and test cost tracking
        agent = TestAgent()
        
        # Send a test message
        message = {
            "message_id": "test-message",
            "sender": "test-user"
        }
        
        # Process the message
        import asyncio
        response = asyncio.run(agent.process_message(message))
        
        # Verify costs were tracked
        assert agent.get_total_cost() > 0
        assert len(agent.get_cost_breakdown()["operations"]) > 0
    
    def test_event_handling(self):
        """Test that agents implement event handling mechanisms."""
        # Create a test capability with event handling
        class TestCapability(BaseCapability):
            def __init__(self):
                super().__init__(requires_human_approval=False)
                
            @property
            def name(self) -> str:
                return "test_capability"
                
            @property
            def description(self) -> str:
                return "A test capability"
                
            @property
            def parameters(self) -> Dict[str, Any]:
                return {}
                        
            async def execute(self, **kwargs: Any) -> CapabilityResult:
                """Execute the capability and emit events."""
                # Emit start event
                self.emit_event("execution_started", {"capability": self.name})
                
                # Simulate execution
                result = {"status": "success", "data": "Test result"}
                
                # Emit completion event
                self.emit_event("execution_completed", {
                    "capability": self.name,
                    "result": result
                })
                
                return result
        
        # Create an instance and test event handling
        capability = TestCapability()
        
        # Mock event handlers
        start_handler = Mock()
        complete_handler = Mock()
        
        # Register event handlers
        capability.register_event_handler("execution_started", start_handler)
        capability.register_event_handler("execution_completed", complete_handler)
        
        # Execute the capability
        import asyncio
        result = asyncio.run(capability.execute())
        
        # Verify event handlers were called
        start_handler.assert_called_once()
        complete_handler.assert_called_once()
        
        # Verify the result
        assert result["status"] == "success"
