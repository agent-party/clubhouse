"""
Agent implementation for the Kafka CLI.

This module provides an Agent class that uses capabilities to generate responses
and interact with users through a Kafka-based messaging system.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.capabilities.conversation_capability import (
    ConversationCapability, 
    ConversationContext
)
from clubhouse.agents.capabilities.llm_capability import LLMCapability
from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.personality import AgentPersonality

from .kafka_client import KafkaMessage

logger = logging.getLogger("agent")

class AgentState(BaseModel):
    """Model for the state of an agent."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Name of the agent")
    role: str = Field(..., description="Role of the agent")
    status: str = Field("idle", description="Current status of the agent")
    last_activity: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), 
                              description="Timestamp of last activity")
    capabilities: List[str] = Field(default_factory=list, description="List of capability names")
    personality_type: str = Field(..., description="Type of personality")
    memory: Dict[str, Any] = Field(default_factory=dict, description="Agent memory")

class Agent:
    """Represents an agent with a personality that can generate responses using Kafka messaging."""
    
    def __init__(self, agent_id: str, personality: AgentPersonality):
        """
        Initialize an agent with a personality.
        
        Args:
            agent_id: Unique identifier for this agent
            personality: The personality profile for this agent
        """
        self.id = agent_id
        self.personality = personality
        
        # Extract key attributes from personality
        self.name = personality.name
        self.role = personality.role
        self.description = personality.short_description
        
        # Initialize capabilities
        self.llm_capability = LLMCapability()
        self.conversation_capability = ConversationCapability()
        self.system_prompt = ""
        
        # Initialize memory
        self.memory = {
            "discussions": [],  # Previous discussions
            "insights": [],     # Key insights gained
            "preferences": {},  # User preferences learned
        }
        
        # Initialize conversation context
        self.conversation_id = str(uuid.uuid4())
        self.conversation_context = self._initialize_conversation()
        
        logger.info(f"Initialized agent {self.name} ({self.id}) with role: {self.role}")
    
    def _initialize_conversation(self) -> ConversationContext:
        """
        Initialize a conversation context for this agent.
        
        Returns:
            ConversationContext: The initialized conversation context
        """
        # Create initial context with agent information
        context = {
            "agent_id": self.id,
            "agent_name": self.name,
            "agent_role": self.role,
            "personality_type": self.personality.__class__.__name__,
            "topic": f"Conversation with {self.name}"
        }
        
        # Initialize conversation through the conversation capability
        return self.conversation_capability._initialize_conversation(
            self.conversation_id, 
            context
        )
    
    def get_state(self) -> AgentState:
        """
        Get the current state of the agent.
        
        Returns:
            AgentState: The agent's state
        """
        return AgentState(
            agent_id=self.id,
            name=self.name,
            role=self.role,
            status="active",
            last_activity=datetime.now(timezone.utc).isoformat(),
            capabilities=["llm", "conversation"],
            personality_type=self.personality.__class__.__name__,
            memory={
                "discussion_count": len(self.memory.get("discussions", [])),
                "insights_count": len(self.memory.get("insights", [])),
            }
        )
    
    def _generate_enhanced_system_prompt(self, conversation_context: List[Dict[str, str]] = None) -> str:
        """
        Generate an enhanced system prompt that includes the personality prompt, the role, and conversation context.
        
        Args:
            conversation_context: Optional list of previous messages for context
            
        Returns:
            The enhanced system prompt
        """
        # Start with the base personality prompt
        base_prompt = self.personality.system_prompt
        
        # Add meta-cognitive guidance
        meta_cognitive_guidance = f"""
        You are {self.name}, {self.role}. {self.description}
        
        Your task is to provide helpful, accurate, and engaging responses based on your expertise.
        Before responding, consider:
        1. What specific information or guidance would be most helpful?
        2. Are there multiple perspectives or approaches to consider?
        3. What assumptions am I making and are they justified?
        4. How can I make my response clear and actionable?
        """
        
        # Add conversation context if available
        conversation_context_prompt = ""
        if conversation_context and len(conversation_context) > 0:
            conversation_context_prompt = "\n\nRecent conversation context:"
            for msg in conversation_context[-5:]:  # Include only the last 5 messages
                if "name" in msg and "content" in msg and msg["name"] != self.name:
                    # Extract a brief summary to keep context concise
                    content = msg["content"]
                    summary = content[:150] + "..." if len(content) > 150 else content
                    conversation_context_prompt += f"\n- {msg['name']}: {summary}"
        
        # Combine all parts into the final system prompt
        enhanced_prompt = f"{base_prompt}\n\n{meta_cognitive_guidance}{conversation_context_prompt}"
        
        return enhanced_prompt
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format the conversation history into a structure suitable for the LLM.
        
        Args:
            conversation_history: The raw conversation history
            
        Returns:
            Formatted conversation history for the LLM
        """
        if not conversation_history:
            return []
        
        formatted_history = []
        
        # Process each message in the history
        for message in conversation_history:
            role = message.get("role", "assistant")
            name = message.get("name", "")
            content = message.get("content", "")
            
            # Skip empty messages
            if not content.strip():
                continue
                
            # Skip system messages
            if role == "system":
                continue
                
            # Format based on role
            if role == "user":
                formatted_history.append({
                    "role": "user", 
                    "content": content
                })
            else:  # assistant or any other role
                # For other agents, format as an assistant message with name prefix
                if name and name != self.name:
                    formatted_history.append({
                        "role": "assistant", 
                        "content": f"{name}: {content}"
                    })
                # For messages from this agent, format as regular assistant messages
                elif name == self.name:
                    formatted_history.append({
                        "role": "assistant", 
                        "content": content
                    })
                # For unnamed messages that aren't system, treat as user
                else:
                    formatted_history.append({
                        "role": "user", 
                        "content": content
                    })
        
        return formatted_history
    
    async def generate_response(self, prompt: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response to a prompt, using the agent's personality and LLM capability.
        Includes a reflection stage to improve response quality.
        
        Args:
            prompt: The prompt to respond to
            conversation_history: Optional list of previous messages for context
            
        Returns:
            The generated response
        """
        try:
            # Generate an enhanced system prompt
            system_prompt = self._generate_enhanced_system_prompt(conversation_history)
            
            # Format conversation history for the LLM
            formatted_history = []
            if conversation_history:
                formatted_history = self._format_conversation_history(conversation_history)
            
            # Generate initial response
            initial_result = await self.llm_capability.execute(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,  # Add some creativity
                max_tokens=800,   # Limit token length for more concise responses
                conversation_history=formatted_history
            )
            
            # Extract the initial response text
            initial_response = ""
            if isinstance(initial_result, dict) and "content" in initial_result:
                initial_response = initial_result["content"]
            elif isinstance(initial_result, str):
                initial_response = initial_result
            else:
                # Handle response format from CapabilityResult
                result_data = initial_result.result if hasattr(initial_result, 'result') else initial_result
                if isinstance(result_data, dict) and "data" in result_data and "response" in result_data["data"]:
                    initial_response = result_data["data"]["response"]
                else:
                    # Handle unexpected response format
                    initial_response = str(initial_result)
            
            # Apply reflection to improve the response
            reflection_prompt = f"""
            You drafted this response to the question: "{prompt}"
            
            Your draft response: 
            {initial_response}
            
            REFLECT on your response using these criteria:
            1. PRACTICALITY: Is your advice directly actionable in the real world?
            2. SPECIFICITY: Did you provide concrete examples rather than just theories?
            3. CONTEXT AWARENESS: Did you consider implied constraints (privacy, surprise, feasibility)?
            4. COLLABORATION: Did you build upon ideas from other participants?
            5. CONCISENESS: Is your response clear and efficiently expressed?
            
            Based on this reflection, rewrite your response to address any shortcomings.
            Keep it concise but complete (150-200 words maximum).
            """
            
            # Generate refined response after reflection
            refined_result = await self.llm_capability.execute(
                prompt=reflection_prompt,
                temperature=0.5,  # Lower temperature for more focused response
                max_tokens=800
            )
            
            # Extract the refined response
            response = ""
            if isinstance(refined_result, dict) and "content" in refined_result:
                response = refined_result["content"]
            elif isinstance(refined_result, str):
                response = refined_result
            else:
                # Handle response format from CapabilityResult
                result_data = refined_result.result if hasattr(refined_result, 'result') else refined_result
                if isinstance(result_data, dict) and "data" in result_data and "response" in result_data["data"]:
                    response = result_data["data"]["response"]
                else:
                    # If reflection fails, use the initial response
                    response = initial_response
            
            # Update agent memory for future context
            self._update_memory(prompt, response, conversation_history)
            
            # Record the message in conversation context
            await self._record_message(prompt, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Generate a fallback response based on personality
            fallback = f"As {self.name}, {self.role}, I would analyze this situation carefully. "
            fallback += f"Based on my expertise, I think we should consider several factors. "
            fallback += f"My recommendation would be to gather more information and approach this systematically."
            
            return fallback
    
    async def _record_message(self, prompt: str, response: str) -> None:
        """
        Record a message in the conversation context using the conversation capability.
        
        Args:
            prompt: The user prompt
            response: The agent's response
        """
        try:
            # Record user message
            user_message_params = {
                "message": prompt,
                "conversation_id": self.conversation_id,
                "metadata": {
                    "sender_type": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            await self.conversation_capability.execute(**user_message_params)
            
            # Record agent response
            agent_message_params = {
                "message": response,
                "conversation_id": self.conversation_id,
                "metadata": {
                    "sender_type": "agent",
                    "agent_id": self.id,
                    "agent_name": self.name,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            await self.conversation_capability.execute(**agent_message_params)
            
        except Exception as e:
            logger.error(f"Error recording message: {str(e)}")
    
    def _update_memory(self, prompt: str, response: str, conversation_history: List[Dict[str, str]] = None) -> None:
        """
        Update the agent's memory with new information from the interaction.
        
        Args:
            prompt: The prompt that was given to the agent
            response: The response generated by the agent
            conversation_history: The conversation history provided
        """
        # Store the discussion in memory
        discussion_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response_snippet": response[:100] + "..." if len(response) > 100 else response,
            "participants": []
        }
        
        # Track participants from conversation history
        if conversation_history:
            participants = set()
            for msg in conversation_history:
                if "name" in msg and msg["name"]:
                    participants.add(msg["name"])
            discussion_entry["participants"] = list(participants)
        
        # Add to discussions memory
        self.memory["discussions"].append(discussion_entry)
        
        # Limit memory size to prevent excessive growth
        if len(self.memory["discussions"]) > 10:
            self.memory["discussions"] = self.memory["discussions"][-10:]
