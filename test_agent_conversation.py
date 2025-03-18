#!/usr/bin/env python3
"""
Test script for multi-agent conversation with problem statement feature.
This script creates multiple agents and initiates a conversation with a problem statement.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

from clubhouse.agents.examples.assistant_agent import AssistantAgent
from clubhouse.agents.capabilities.llm_capability import LLMCapability
from clubhouse.agents.capabilities.search_capability import SearchCapability
from clubhouse.agents.capabilities.conversation_capability import ConversationCapability
from clubhouse.agents.capabilities.memory_capability import MemoryCapability
from clubhouse.agents.capabilities.summarize_capability import SummarizeCapability

def create_agent(agent_id: str, name: str, description: str = None) -> AssistantAgent:
    """Create an agent with all needed capabilities."""
    if not description:
        description = f"An AI agent designed to assist with problem-solving and reasoning tasks"
    
    # Create the agent
    agent = AssistantAgent(
        agent_id=agent_id,
        name=name,
        description=description
    )
    
    # Add core capabilities
    agent.register_capability(LLMCapability())
    agent.register_capability(SearchCapability())
    agent.register_capability(ConversationCapability())
    agent.register_capability(MemoryCapability())
    agent.register_capability(SummarizeCapability())
    
    return agent

async def setup_conversation(agents: List[AssistantAgent], problem_statement: str = None):
    """
    Set up a conversation between multiple agents with an optional problem statement.
    
    Args:
        agents: List of agents to include in the conversation
        problem_statement: Optional problem for agents to discuss
    """
    print(f"Setting up conversation with {len(agents)} agents")
    
    # Define conversation context
    conversation_id = "test-convo-1"
    
    # Initialize conversation for each agent
    for agent in agents:
        # Get the conversation capability
        conversation_capability = agent._capabilities.get("conversation")
        if not conversation_capability:
            print(f"Agent {agent.agent_id} doesn't have conversation capability!")
            continue
        
        # Create context with participants and problem statement
        context = {
            "conversation_id": conversation_id,
            "participants": [a.agent_id for a in agents],
            "message_history": [],
        }
        
        if problem_statement:
            context["problem_statement"] = problem_statement
            print(f"Added problem statement to conversation: {problem_statement}")
        
        # Initialize conversation by adding a system message
        # This satisfies the requirement for a 'message' parameter
        system_message = "Conversation initialized."
        if problem_statement:
            system_message = f"Conversation initialized with problem statement: {problem_statement}"
        
        # Add initialization message to start the conversation
        await conversation_capability.execute(
            message=system_message,
            conversation_id=conversation_id,
            metadata={"type": "system", "action": "initialize"},
            context=context
        )
    
    return conversation_id

async def send_message(agent: AssistantAgent, conversation_id: str, content: str):
    """Send a message from an agent to the conversation."""
    # Get the conversation capability
    conversation_capability = agent._capabilities.get("conversation")
    if not conversation_capability:
        print(f"Agent {agent.agent_id} doesn't have conversation capability!")
        return None
    
    # Create message metadata
    metadata = {
        "sender_id": agent.agent_id,
        "sender_name": agent.name,
        "timestamp": "2023-01-01T00:00:00Z",  # Placeholder timestamp
        "message_type": "text"
    }
    
    # Add the message to the conversation using the expected parameters
    result = await conversation_capability.execute(
        message=content,  # This is the required 'message' parameter
        conversation_id=conversation_id,
        metadata=metadata,
        resolve_references=False
    )
    
    return result

async def generate_response(agent: AssistantAgent, conversation_id: str):
    """Generate a response from an agent based on the conversation."""
    # Get the conversation capability
    conversation_capability = agent._capabilities.get("conversation")
    if not conversation_capability:
        print(f"Agent {agent.agent_id} doesn't have conversation capability!")
        return None
    
    # Get the conversation context
    conversation = conversation_capability._conversations.get(conversation_id)
    if not conversation:
        print(f"No conversation found with ID {conversation_id}")
        return None
    
    # Generate a response using the LLM capability
    llm_capability = agent._capabilities.get("llm")
    if not llm_capability:
        print(f"Agent {agent.agent_id} doesn't have LLM capability!")
        return None
    
    # Extract problem statement (if any) from the conversation context
    problem_statement = conversation.attributes.get("problem_statement", "")
    problem_context = f"Problem statement: {problem_statement}\n\n" if problem_statement else ""
    
    # Format messages for the LLM
    messages = []
    for msg in conversation.message_history:
        # Extract sender and content information
        sender_info = msg.get("metadata", {}).get("sender_name", msg.get("metadata", {}).get("sender_id", "Unknown"))
        content = msg.get("message", "")
        messages.append(f"{sender_info}: {content}")
    
    # Create system prompt
    system_prompt = f"""You are {agent.name}, an AI assistant participating in a multi-agent conversation.
{problem_context}The following is a transcript of the conversation so far.
Your task is to respond to the conversation based on the previous messages and your role.
"""
    
    # Execute LLM capability to generate a response
    result = await llm_capability.execute(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(messages) + f"\n\n{agent.name}'s response:"}
        ]
    )
    
    return result.result.get("content", "")

async def main():
    """Run the test for multi-agent conversation with problem statement."""
    # Create agents
    agent_alice = create_agent("alice", "Alice", "A creative problem solver")
    agent_bob = create_agent("bob", "Bob", "A logical thinker")
    agent_charlie = create_agent("charlie", "Charlie", "A critical analyst")
    
    agents = [agent_alice, agent_bob, agent_charlie]
    
    # Set up a conversation with a problem statement
    problem_statement = "How can we design a sustainable urban environment that balances technological advancement with environmental preservation?"
    conversation_id = await setup_conversation(agents, problem_statement)
    
    # Start the conversation with Alice
    initial_message = "I think we should start by identifying the key challenges of urban sustainability."
    print(f"\nAlice: {initial_message}")
    await send_message(agent_alice, conversation_id, initial_message)
    
    # Generate responses from Bob and Charlie
    bob_response = await generate_response(agent_bob, conversation_id)
    print(f"\nBob: {bob_response}")
    await send_message(agent_bob, conversation_id, bob_response)
    
    charlie_response = await generate_response(agent_charlie, conversation_id)
    print(f"\nCharlie: {charlie_response}")
    await send_message(agent_charlie, conversation_id, charlie_response)
    
    # Alice responds to both
    alice_response = await generate_response(agent_alice, conversation_id)
    print(f"\nAlice: {alice_response}")
    await send_message(agent_alice, conversation_id, alice_response)
    
    print("\nSuccessfully tested multi-agent conversation with problem statement!")

if __name__ == "__main__":
    asyncio.run(main())
