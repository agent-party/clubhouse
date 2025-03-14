"""Tests for agent protocol interfaces."""
import pytest
from typing import Dict, Any, List, Protocol
from unittest.mock import Mock

# Mock for capability-related types
class MockCapability:
    def __init__(self, name: str, permissions: List[str]):
        self.name = name
        self.permissions = permissions


class TestAgentProtocol:
    """Test cases for agent protocol interfaces."""
    
    def test_agent_protocol_definition(self):
        """Test that the agent protocol defines the expected methods."""
        from clubhouse.agents.agent_protocol import AgentProtocol
        
        # Verify Protocol has the expected properties and methods
        assert hasattr(AgentProtocol, "agent_id")
        assert hasattr(AgentProtocol, "name")
        assert hasattr(AgentProtocol, "description")
        assert hasattr(AgentProtocol, "get_capabilities")
        assert hasattr(AgentProtocol, "process_message")
        
        # Check that it's properly defined as a Protocol
        assert isinstance(AgentProtocol, type(Protocol))
    
    def test_capability_protocol_definition(self):
        """Test that the capability protocol defines the expected methods and properties."""
        from clubhouse.agents.agent_protocol import CapabilityProtocol
        
        # Verify Protocol has the expected attributes
        assert hasattr(CapabilityProtocol, "name")
        assert hasattr(CapabilityProtocol, "description")
        assert hasattr(CapabilityProtocol, "parameters")
        
        # Check that it's properly defined as a Protocol
        assert isinstance(CapabilityProtocol, type(Protocol))
    
    def test_implementation_compatibility(self):
        """Test that a concrete implementation satisfies the protocols."""
        from clubhouse.agents.agent_protocol import AgentProtocol, CapabilityProtocol
        
        # Create a mock capability that implements CapabilityProtocol
        class MockCapabilityImpl:
            @property
            def name(self) -> str:
                return "test_capability"
                
            @property
            def description(self) -> str:
                return "A test capability"
                
            @property
            def parameters(self) -> Dict[str, Any]:
                return {"param1": "value1"}
        
        # Create a mock agent that implements AgentProtocol
        class MockAgentImpl:
            @property
            def agent_id(self) -> str:
                return "test_agent"
                
            @property
            def name(self) -> str:
                return "Test Agent"
                
            @property
            def description(self) -> str:
                return "A test agent implementation"
                
            def get_capabilities(self) -> List[CapabilityProtocol]:
                return [MockCapabilityImpl()]
                
            async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
                return {"response": "processed"}
                
        # These should not raise exceptions if the implementations satisfy the protocols
        CapabilityProtocol.register(MockCapabilityImpl)
        AgentProtocol.register(MockAgentImpl)
