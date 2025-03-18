"""
Tests for the simple agent implementation.

This module contains tests for the SimpleAgent class, which provides
a basic agent implementation for the Kafka CLI integration.
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from clubhouse.agents.simple_agent import SimpleAgent


@pytest.fixture
def simple_agent():
    """Fixture for a simple agent."""
    return SimpleAgent(
        agent_id="test-agent",
        personality_type="assistant",
        name="Test Agent",
        description="A test agent for unit tests",
        metadata={"key": "value"}
    )


def test_agent_initialization():
    """Test that a simple agent initializes correctly."""
    # Create an agent
    agent = SimpleAgent(
        agent_id="test-agent",
        personality_type="researcher"
    )
    
    # Verify the agent was initialized correctly
    assert agent.agent_id() == "test-agent"
    assert agent.personality_type() == "researcher"
    assert agent.name() == "Researcher Agent"  # Default name
    assert "researcher" in agent.description().lower()  # Default description


def test_agent_custom_name_description():
    """Test that a simple agent with custom name and description."""
    # Create an agent with custom name and description
    agent = SimpleAgent(
        agent_id="custom-agent",
        personality_type="teacher",
        name="Professor Smith",
        description="An experienced teacher who specializes in mathematics."
    )
    
    # Verify the custom values were used
    assert agent.name() == "Professor Smith"
    assert agent.description() == "An experienced teacher who specializes in mathematics."


def test_agent_initialize_shutdown(simple_agent):
    """Test initializing and shutting down an agent."""
    # Initially the agent is not initialized
    assert simple_agent._is_initialized is False
    
    # Initialize the agent
    simple_agent.initialize()
    
    # Verify the agent was initialized
    assert simple_agent._is_initialized is True
    
    # Shutdown the agent
    simple_agent.shutdown()
    
    # No assertions for shutdown as it doesn't change any state in our simple implementation


def test_process_message(simple_agent):
    """Test processing a message."""
    # Initialize the agent
    simple_agent.initialize()
    
    # Create a message
    message = {
        "content": "Hello agent",
        "conversation_id": "conv-123"
    }
    
    # Process the message
    response = simple_agent.process_message(message)
    
    # Verify the response structure
    assert "content" in response
    assert "conversation_id" in response
    assert "message_id" in response
    assert "timestamp" in response
    assert "metadata" in response
    
    # Verify the conversation ID was preserved
    assert response["conversation_id"] == "conv-123"
    
    # Verify a response was generated
    assert len(response["content"]) > 0
    assert "Hello agent" in response["content"]


def test_process_message_generated_conversation_id(simple_agent):
    """Test processing a message without a conversation ID."""
    # Initialize the agent
    simple_agent.initialize()
    
    # Create a message without a conversation ID
    message = {
        "content": "Hello again"
    }
    
    # Process the message
    response = simple_agent.process_message(message)
    
    # Verify a conversation ID was generated
    assert "conversation_id" in response
    assert response["conversation_id"] is not None
    assert len(response["conversation_id"]) > 0


def test_conversation_history(simple_agent):
    """Test maintaining conversation history."""
    # Initialize the agent
    simple_agent.initialize()
    
    # Create messages in the same conversation
    conversation_id = "history-test"
    messages = [
        {"content": "First message", "conversation_id": conversation_id},
        {"content": "Second message", "conversation_id": conversation_id},
        {"content": "Third message", "conversation_id": conversation_id}
    ]
    
    # Process each message
    for message in messages:
        simple_agent.process_message(message)
    
    # Get the conversation history
    history = simple_agent.get_conversation_history(conversation_id)
    
    # Verify the history contains all messages and responses
    assert len(history) == 6  # 3 user messages + 3 agent responses
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "First message"
    assert history[1]["role"] == "assistant"
    assert "First message" in history[1]["content"]
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "Second message"


def test_conversation_history_with_limit(simple_agent):
    """Test retrieving limited conversation history."""
    # Initialize the agent
    simple_agent.initialize()
    
    # Create messages in the same conversation
    conversation_id = "limit-test"
    messages = [
        {"content": "First message", "conversation_id": conversation_id},
        {"content": "Second message", "conversation_id": conversation_id},
        {"content": "Third message", "conversation_id": conversation_id}
    ]
    
    # Process each message
    for message in messages:
        simple_agent.process_message(message)
    
    # Get limited conversation history
    limited_history = simple_agent.get_conversation_history(conversation_id, limit=2)
    
    # Verify the limited history contains only the most recent messages
    assert len(limited_history) == 2
    assert limited_history[0]["role"] == "user"
    assert limited_history[0]["content"] == "Third message"
    assert limited_history[1]["role"] == "assistant"


def test_different_personality_responses():
    """Test that different personality types generate different responses."""
    # Create agents with different personality types
    assistant = SimpleAgent(agent_id="assistant", personality_type="assistant")
    researcher = SimpleAgent(agent_id="researcher", personality_type="researcher")
    teacher = SimpleAgent(agent_id="teacher", personality_type="teacher")
    generic = SimpleAgent(agent_id="generic", personality_type="generic")
    
    # Initialize the agents
    assistant.initialize()
    researcher.initialize()
    teacher.initialize()
    generic.initialize()
    
    # Create a message
    message = {
        "content": "Tell me about quantum physics",
        "conversation_id": "test-conversation"
    }
    
    # Process the message with each agent
    assistant_response = assistant.process_message(message)
    researcher_response = researcher.process_message(message)
    teacher_response = teacher.process_message(message)
    generic_response = generic.process_message(message)
    
    # Verify that each agent's response is different
    responses = [
        assistant_response["content"],
        researcher_response["content"],
        teacher_response["content"],
        generic_response["content"]
    ]
    
    # Check that all responses are unique
    unique_responses = set(responses)
    assert len(unique_responses) == 4  # All responses should be different
