#!/usr/bin/env python
"""
Agent CLI

A command-line interface for interacting with agents, enabling:
- Agent creation and management
- Capability execution
- Multi-agent conversations and problem-solving

Usage:
    python agent_cli.py [--debug]

Commands:
    /help - Show help information
    /exit - Exit the CLI
    /list - List available agents
    /createagent <agent_id> [personality_type] - Create a new agent with optional personality
    /solve <problem_statement> - Use a team of agents to solve a problem
    /startconvo [problem_statement] - Start a multi-agent conversation
    /endconvo - End the current conversation

Examples:
    /createagent expert analytical
    /solve How can we improve team collaboration?
"""

import asyncio
import json
import os
import re
import sys
import time
import uuid  # Add uuid import for unique identifiers
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

# Add the project root to the Python path for imports - IMPORTANT!
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False
    print("colorama not installed. Running without color support.")
    
    # Define placeholders for colorama
    class DummyColorama:
        def __getattr__(self, name):
            return ""
    
    Fore = Style = DummyColorama()

# Import personality functions
from clubhouse.agents.personality import (
    AgentPersonality, 
    get_analytical_expert,
    get_creative_innovator, 
    get_mediator_collaborator,
    get_practical_implementer,
    get_critical_evaluator
)

# Import LLM capability
from clubhouse.agents.capabilities.llm_capability import LLMCapability

# Load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Import our new modules
try:
    from clubhouse.agents.repositories.agent_repository import AgentRepository, AgentPerformanceMetrics
    from clubhouse.agents.evaluation.agent_evaluator import AgentEvaluator
    AGENT_STORAGE_AVAILABLE = True
except ImportError:
    print("Agent storage and evaluation modules not available or not properly installed")
    AGENT_STORAGE_AVAILABLE = False

class OutputFormatter:
    """
    Formats output for the CLI.
    """
    
    def __init__(self, use_color=True):
        """Initialize the formatter with color settings."""
        self.use_color = use_color
        self.colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "gray": "\033[90m"
        }
        
        # Store agent colors for consistent coloring
        self.agent_colors = {}
        self.available_colors = ["cyan", "green", "yellow", "magenta", "blue"]
        self.color_index = 0
    
    def _get_agent_color(self, agent_name: str) -> str:
        """Get a consistent color for an agent."""
        if agent_name not in self.agent_colors:
            color = self.available_colors[self.color_index % len(self.available_colors)]
            self.agent_colors[agent_name] = color
            self.color_index += 1
        
        return self.agent_colors[agent_name]
    
    def _format_text(self, text: str, style: str) -> str:
        """Format text with the specified style if color is enabled."""
        if not self.use_color:
            return text
            
        if style in self.colors:
            return f"{self.colors[style]}{text}{self.colors['reset']}"
        
        return text
    
    def print_header(self, text: str) -> None:
        """Print a header."""
        formatted_text = self._format_text(f"\n=== {text} ===", "bold")
        print(formatted_text)
    
    def print_subheader(self, text: str) -> None:
        """Print a subheader."""
        formatted_text = self._format_text(f"\n--- {text} ---", "bold")
        print(formatted_text)
    
    def print_error(self, text: str) -> None:
        """Print an error message."""
        formatted_text = self._format_text(f"ERROR: {text}", "red")
        print(formatted_text)
    
    def print_warning(self, text: str) -> None:
        """Print a warning message."""
        formatted_text = self._format_text(f"WARNING: {text}", "yellow")
        print(formatted_text)
    
    def print_success(self, text: str) -> None:
        """Print a success message."""
        formatted_text = self._format_text(f"SUCCESS: {text}", "green")
        print(formatted_text)
    
    def print_info(self, text: str) -> None:
        """Print an informational message."""
        print(text)
    
    def format_message(self, sender: str, message: str) -> str:
        """
        Format a message with sender information for display.
        
        Args:
            sender: The sender of the message
            message: The message content
            
        Returns:
            Formatted message string
        """
        # Get color for the sender if it's an agent
        if sender.lower() != "system":
            color = self._get_agent_color(sender)
            sender_formatted = self._format_text(sender, color)
        else:
            sender_formatted = self._format_text("SYSTEM", "gray")
        
        # Format and return the full message
        return f"{sender_formatted}: {message}"
    
    def print_agent_message(self, agent_name: str, message: str) -> None:
        """
        Print a message from an agent with appropriate formatting.
        
        Args:
            agent_name: Name of the agent
            message: The message to print
        """
        # Format the message with the agent's name
        formatted_message = self.format_message(agent_name, message)
        print(formatted_message)
    
    def print_thinking(self, agent_name: str) -> None:
        """
        Print an indicator that an agent is thinking.
        
        Args:
            agent_name: Name of the agent that is thinking
        """
        agent_color = self._get_agent_color(agent_name)
        formatted_name = self._format_text(f"{agent_name}", agent_color)
        thinking_text = self._format_text("thinking...", "gray")
        
        print(f"{formatted_name} {thinking_text}", end="\r")
        sys.stdout.flush()
    
    def print_system_message(self, message: str) -> None:
        """
        Print a system message.
        
        Args:
            message: The message to print
        """
        formatted_text = self._format_text(f"SYSTEM: {message}", "bold")
        print(formatted_text)
    
    def print_command_help(self, command: str, description: str) -> None:
        """
        Print help information for a command.
        
        Args:
            command: The command
            description: Description of the command
        """
        formatted_command = self._format_text(command, "cyan")
        print(f"{formatted_command} - {description}")
    
    def print_divider(self) -> None:
        """Print a divider line."""
        print("\n" + "-" * 80)
    
    def print_conversation_summary(self, conversation_history: List[Dict[str, str]], max_entries: int = 3) -> None:
        """
        Print a summary of the recent conversation.
        
        Args:
            conversation_history: The conversation history
            max_entries: Maximum number of entries to show
        """
        if not conversation_history:
            return
            
        self.print_subheader("Recent Conversation")
        
        # Get the last few entries
        recent_entries = conversation_history[-min(max_entries, len(conversation_history)):]
        
        # Print each entry
        for entry in recent_entries:
            name = entry.get("name", "Unknown")
            content = entry.get("content", "")
            
            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..." if len(content) > 100 else content
            elif len(content) > 50:
                content = content[:47] + "..." if len(content) > 50 else content
            
            # Format the name with color
            agent_color = self._get_agent_color(name)
            formatted_name = self._format_text(name, agent_color)
            
            print(f"{formatted_name}: {content}")
            
        print("")  # Add spacing


class Agent:
    """
    Represents an agent with a personality that can generate responses.
    """
    
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
        
        # Initialize the LLM capability
        self.llm_capability = LLMCapability()
        self.system_prompt = ""
        
        # Initialize memory
        self.memory = {
            "discussions": [],  # Previous discussions
            "insights": [],     # Key insights gained
            "preferences": {},  # User preferences learned
        }
    
    def _generate_enhanced_system_prompt(self, conversation_context: List[Dict[str, str]] = None) -> str:
        """
        Generate an enhanced system prompt that includes the personality prompt, the role, and conversation context.
        
        Args:
            conversation_context: Optional list of previous messages for context
            
        Returns:
            The enhanced system prompt
        """
        # Start with the basic personality-based system prompt
        base_prompt = self.personality.generate_system_prompt()
        
        # Add meta-cognitive guidance to encourage thoughtful responses
        meta_cognitive_guidance = f"""
        As {self.name} with expertise in {self.role}, follow these principles:
        
        1. PRACTICAL & SPECIFIC: Provide concrete, actionable advice with specific examples, not just abstract frameworks
        2. CONTEXTUALLY AWARE: Consider unstated but implied constraints (e.g., privacy, surprise, feasibility)
        3. COLLABORATIVE: Reference and build on contributions from other participants by name
        4. CONCISE & STRUCTURED: Express your ideas clearly and efficiently (aim for 150-200 words)
        5. REFLECTIVE: Consider multiple perspectives and alternatives before settling on recommendations
        
        Before finalizing your response, ensure you've:
        - Provided specific, actionable guidance that considers real-world constraints
        - Considered what might be implied but not explicitly stated in the problem
        - Built meaningfully on the conversation rather than repeating information
        - Made reference to at least one point raised by another participant (when applicable)
        - Presented your ideas in a clear, concise, and structured way
        """
        
        # Add context from the conversation history
        conversation_context_prompt = ""
        if conversation_context:
            # Identify participants in the conversation
            participants = set()
            for msg in conversation_context:
                if "name" in msg and msg["name"] and msg["name"] != self.name:
                    participants.add(msg["name"])
            
            # Format the participants information
            if participants:
                conversation_context_prompt += f"\n\nYou are discussing with: {', '.join(participants)}"
            
            # Add excerpts from recent messages to provide context
            if len(conversation_context) > 0:
                conversation_context_prompt += "\n\nRecent discussion points to consider:"
                # Include the last few messages for context
                for i in range(max(0, len(conversation_context) - 3), len(conversation_context)):
                    msg = conversation_context[i]
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
            
            return response
            
        except Exception as e:
            # If there's an error with the LLM, fall back to a simple response
            if os.environ.get("DEBUG"):
                print(f"Error generating response: {str(e)}")
                
            # Generate a fallback response based on personality
            fallback = f"As {self.name}, {self.role}, I would analyze this situation carefully. "
            fallback += f"Based on my expertise, I think we should consider several factors. "
            fallback += f"My recommendation would be to gather more information and approach this systematically."
            
            return fallback
    
    def _update_memory(self, prompt: str, response: str, conversation_history: List[Dict[str, str]] = None):
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


class ProblemSolver:
    """
    A class to solve problems with multiple agents.
    """
    
    def __init__(self, agents: List[Agent]):
        """
        Initialize the problem solver with a list of agents.
        
        Args:
            agents: A list of agents
        """
        self.agents = agents
        self.output_formatter = OutputFormatter()
        
        # Initialize agent repository and evaluator if available
        self.repository = None
        self.evaluator = None
        if AGENT_STORAGE_AVAILABLE:
            try:
                self.repository = AgentRepository()
                self.evaluator = AgentEvaluator(self.repository)
                
                # Store agent configurations in the repository
                for agent in self.agents:
                    self._store_agent_configuration(agent)
            except Exception as e:
                print(f"Could not initialize agent repository or evaluator: {str(e)}")
    
    def _store_agent_configuration(self, agent: Agent) -> str:
        """
        Store an agent's configuration in the repository.
        
        Args:
            agent: The agent to store
            
        Returns:
            The agent's ID
        """
        if not self.repository:
            return str(uuid.uuid4())
            
        # Extract agent personality from prompts
        personality = {
            "name": agent.name,
            "role": self._extract_role(agent.system_prompt),
            "short_description": self._extract_description(agent.system_prompt),
            "system_prompt": agent.system_prompt
        }
        
        # Generate or use existing agent ID
        agent_id = getattr(agent, "agent_id", str(uuid.uuid4()))
        agent.agent_id = agent_id  # Store ID on agent
        
        # Store in repository
        stored_id = self.repository.store_agent(agent_id, personality)
        return stored_id
    
    def _extract_role(self, system_prompt: str) -> str:
        """Extract role information from system prompt."""
        role_match = re.search(r'you are(?: a)? ([^.!]+)', system_prompt, re.IGNORECASE)
        if role_match:
            return role_match.group(1).strip()
        return "Assistant"
    
    def _extract_description(self, system_prompt: str) -> str:
        """Extract description from system prompt."""
        # Take first two sentences as description
        sentences = re.split(r'[.!?]', system_prompt)
        if len(sentences) >= 2:
            return f"{sentences[0].strip()}.{sentences[1].strip()}."
        return system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
    
    async def _track_context_elements(self, text: str) -> Dict[str, str]:
        """
        Extract and track key context elements from text to ensure preservation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of key elements and their types
        """
        elements = {}
        
        # Extract gender references
        if re.search(r'\bson\b', text, re.IGNORECASE):
            elements["gender"] = "son"
        elif re.search(r'\bdaughter\b', text, re.IGNORECASE):
            elements["gender"] = "daughter"
        
        # Extract age references
        age_match = re.search(r'(\d+)[- ](?:year|yr)(?:s)?[- ]old', text, re.IGNORECASE)
        if age_match:
            elements["age"] = age_match.group(1)
        
        # Extract name references
        name_match = re.search(r'(?:my|for) (?:son|daughter|child) (?:named |called )?([A-Z][a-z]+)', text, re.IGNORECASE)
        if name_match:
            elements["name"] = name_match.group(1)
        
        return elements
    
    async def _evaluate_solutions(self, problem: str, discussion: List[Dict], solution: str) -> Dict:
        """
        Evaluate the quality of solutions provided in a discussion.
        
        Args:
            problem: Original problem statement
            discussion: Full discussion history
            solution: Final synthesized solution
            
        Returns:
            Evaluation results
        """
        if not self.evaluator:
            return {}
            
        # Evaluate the team performance
        team_eval = self.evaluator.evaluate_team(problem, discussion, solution)
        
        # Evaluate individual agents
        agent_evals = {}
        for agent in self.agents:
            agent_id = getattr(agent, "agent_id", None)
            if agent_id:
                agent_eval = self.evaluator.evaluate_agent(
                    agent_id, agent.name, problem, discussion, solution
                )
                agent_evals[agent.name] = agent_eval
        
        return {
            "team": team_eval,
            "agents": agent_evals
        }
    
    async def solve(self, problem_statement: str) -> str:
        """
        Solve a problem with multiple agents.
        
        Args:
            problem_statement: The problem statement
            
        Returns:
            The solution
        """
        # Track key context elements to ensure preservation
        context_elements = await self._track_context_elements(problem_statement)
        
        # Initialize stages
        stages = [
            "problem_analysis",
            "solution_ideation",
            "critical_evaluation",
            "solution_refinement"
        ]
        
        full_discussion = []
        
        # Problem introduction 
        print("\n=== Problem Introduction ===")
        intro_message = await self._introduce_problem(problem_statement)
        full_discussion.append(intro_message)
        
        # Round-robin discussion for each stage
        for stage in stages:
            stage_responses = await self._run_discussion_round(problem_statement, stage, full_discussion)
            full_discussion.extend(stage_responses)
        
        # Generate solution synthesis with special care for context preservation
        solution = await self._generate_solution_synthesis(
            problem_statement, full_discussion, context_elements
        )
        
        # Evaluate solutions if evaluator is available
        if self.evaluator:
            evaluation = await self._evaluate_solutions(
                problem_statement, full_discussion, solution
            )
            if evaluation:
                print("\n=== Solution Evaluation ===")
                team_score = evaluation.get("team", {}).get("metrics", {}).get("overall_score", 0)
                print(f"Team Performance Score: {team_score:.2f}/1.00")
                
                # Print context preservation issues if any
                context_explanations = evaluation.get("team", {}).get("explanations", {}).get("context", [])
                context_issues = [exp for exp in context_explanations if "Issue" in exp]
                if context_issues:
                    print("Context Issues Detected:")
                    for issue in context_issues:
                        print(f"  - {issue}")
        
        return solution
        
    async def _introduce_problem(self, problem_statement: str) -> Dict:
        """
        Introduce the problem to the first agent.
        
        Args:
            problem_statement: The problem statement
            
        Returns:
            The introduction message
        """
        # Select the first agent to introduce the problem
        introducer = self.agents[0]
        
        # Generate the introduction prompt
        introduction_prompt = f"""
        You are participating in a collaborative discussion to solve the following problem:
        
        "{problem_statement}"
        
        As the first contributor, your task is to introduce this problem to the group.
        
        In your response:
        1. Provide a brief introduction to the problem
        2. Explain your role in guiding the discussion
        3. Set clear expectations for what a good solution looks like
        4. Invite the next participant to share their thoughts
        
        Keep your response focused and under 200 words.
        """
        
        # Generate the introduction
        introduction = await introducer.generate_response(introduction_prompt)
        
        # Print the introduction
        print(self.output_formatter.format_message(introducer.name, introduction))
        
        # Return the introduction for the discussion history
        return {
            "stage": "introduction",
            "sender": introducer.name,
            "content": introduction
        }
    
    async def _run_discussion_round(self, problem_statement: str, 
                                   stage: str, discussion_history: List[Dict]) -> List[Dict]:
        """
        Run a round of discussion for a specific stage.
        
        Args:
            problem_statement: The problem statement
            stage: The current discussion stage
            discussion_history: The discussion history so far
            
        Returns:
            The responses from this round
        """
        # Print the stage header
        print(f"\n=== {stage.replace('_', ' ').title()} ===")
        
        stage_responses = []
        
        # Generate prompts for each stage
        if stage == "problem_analysis":
            prompt_template = """
            You are participating in a collaborative problem-solving discussion to address:
            
            "{problem_statement}"
            
            Current stage: Problem Analysis
            
            Review the discussion so far and analyze the problem by:
            1. Identifying key constraints and requirements
            2. Clarifying any ambiguities or assumptions
            3. Breaking down the problem into manageable components
            
            Your analysis should be insightful and help the team develop effective solutions.
            Keep your response concise (100-150 words).
            """
        elif stage == "solution_ideation":
            prompt_template = """
            You are participating in a collaborative problem-solving discussion to address:
            
            "{problem_statement}"
            
            Current stage: Solution Ideation
            
            Based on the problem analysis in the previous discussion:
            1. Propose 1-2 specific solutions that address the key requirements
            2. Explain your reasoning for each proposed solution
            3. Consider how your ideas complement or build upon others' contributions
            
            Be creative yet practical, and keep your response concise (100-150 words).
            """
        elif stage == "critical_evaluation":
            prompt_template = """
            You are participating in a collaborative problem-solving discussion to address:
            
            "{problem_statement}"
            
            Current stage: Critical Evaluation
            
            Evaluate the solutions proposed in the discussion so far:
            1. Identify strengths and potential weaknesses of at least one proposed solution
            2. Consider feasibility, effectiveness, and potential unintended consequences
            3. Suggest how a promising solution could be improved or adapted
            
            Be constructive and specific in your evaluation.
            Keep your response concise (100-150 words).
            """
        elif stage == "solution_refinement":
            prompt_template = """
            You are participating in a collaborative problem-solving discussion to address:
            
            "{problem_statement}"
            
            Current stage: Solution Refinement
            
            Based on all previous discussion and critical evaluation:
            1. Present your refined solution recommendation
            2. Address the key concerns or improvements identified during evaluation
            3. Provide specific, actionable details for implementation
            
            Your refined solution should be practical, comprehensive and ready to implement.
            Keep your response concise (150-200 words).
            """
        else:
            prompt_template = """
            You are participating in a collaborative problem-solving discussion to address:
            
            "{problem_statement}"
            
            Current stage: {stage}
            
            Review the discussion so far and contribute your perspective.
            Keep your response concise (100-150 words).
            """
        
        # Format the discussion history for inclusion in the prompt
        formatted_history = ""
        for entry in discussion_history:
            if "sender" in entry and "content" in entry:
                sender = entry["sender"]
                content = entry["content"]
                formatted_history += f"{sender}: {content}\n\n"
        
        # Generate responses from each agent
        for agent in self.agents:
            # Format the prompt with the problem statement and discussion history
            full_prompt = prompt_template.format(
                problem_statement=problem_statement,
                stage=stage
            )
            
            if formatted_history:
                full_prompt += f"\n\nDiscussion so far:\n\n{formatted_history}"
            
            # Generate the response
            response = await agent.generate_response(full_prompt)
            
            # Print the response
            print(self.output_formatter.format_message(agent.name, response))
            
            # Add the response to the discussion history
            response_entry = {
                "stage": stage,
                "sender": agent.name,
                "content": response
            }
            stage_responses.append(response_entry)
            
            # Update the formatted history for the next agent
            formatted_history += f"{agent.name}: {response}\n\n"
        
        return stage_responses
    
    async def _generate_solution_synthesis(self, problem_statement: str, 
                                       discussion: List[Dict], 
                                       context_elements: Dict[str, str] = None) -> str:
        """
        Generate a synthesis of the solution based on the discussion.
        
        Args:
            problem_statement: The original problem statement
            discussion: The full discussion history
            context_elements: Key context elements to preserve
            
        Returns:
            The solution synthesis
        """
        # Create a mediator agent to synthesize the solution
        mediator = self.agents[0] if self.agents else Agent(
            "mediator", 
            get_mediator_collaborator()
        )
        
        # Extract solutions from the refinement stage
        refinement_solutions = [
            entry["content"] for entry in discussion 
            if "stage" in entry and entry["stage"] == "solution_refinement"
        ]
        
        # Generate context-specific instructions based on identified elements
        context_instructions = ""
        if context_elements:
            context_instructions = "CRITICAL CONTEXT TO PRESERVE:\n"
            for element_type, value in context_elements.items():
                context_instructions += f"- {element_type.upper()}: '{value}'\n"
        
        # Generate the synthesis prompt with explicit context preservation instructions
        synthesis_prompt = f"""
        You are synthesizing the final solution for the problem: "{problem_statement}"
        
        The team has proposed these refined solutions:
        
        {chr(10).join([f"- {solution}" for solution in refinement_solutions])}
        
        {context_instructions}
        
        IMPORTANT: Ensure your response maintains perfect consistency with the original problem statement. 
        Pay special attention to details like gender references (son vs daughter), age, and any other 
        specific characteristics mentioned in the original problem statement: "{problem_statement}".
        
        Please provide a final, cohesive solution that incorporates the best elements from each contribution.
        Focus on being specific, practical, and actionable.
        Your response should be 2-3 paragraphs and ready to implement without further clarification.
        """
        
        # Generate the synthesis
        synthesis = await mediator.generate_response(synthesis_prompt)
        
        # Verify context preservation and fix if needed
        synthesis = await self._verify_context_preservation(
            problem_statement, synthesis, context_elements
        )
        
        # Print the synthesis
        print(self.output_formatter.format_message(mediator.name, synthesis))
        
        return synthesis
    
    async def _verify_context_preservation(self, problem: str, solution: str, 
                                      context_elements: Dict[str, str]) -> str:
        """
        Verify and fix context preservation issues in the solution.
        
        Args:
            problem: Original problem statement
            solution: Generated solution
            context_elements: Key context elements to preserve
            
        Returns:
            Corrected solution
        """
        if not context_elements:
            return solution
            
        corrected = solution
        
        # Check gender consistency
        if "gender" in context_elements:
            correct_gender = context_elements["gender"]
            opposite_gender = "daughter" if correct_gender == "son" else "son"
            
            # Fix gender references
            if opposite_gender in solution.lower() and correct_gender not in solution.lower():
                corrected = re.sub(
                    rf'\b{opposite_gender}\b', 
                    correct_gender, 
                    corrected, 
                    flags=re.IGNORECASE
                )
        
        # Check age consistency
        if "age" in context_elements:
            correct_age = context_elements["age"]
            # Find incorrect age references (different numbers)
            age_references = re.findall(r'\b(\d+)[- ](?:year|yr)(?:s)?[- ]old', corrected, re.IGNORECASE)
            for age in age_references:
                if age != correct_age:
                    corrected = re.sub(
                        rf'\b{age}[- ](?:year|yr)(?:s)?[- ]old\b', 
                        f"{correct_age}-year-old", 
                        corrected, 
                        flags=re.IGNORECASE
                    )
        
        # If changes were made, log the correction
        if corrected != solution:
            print("\n=== Context Correction Applied ===")
            print("The solution was modified to maintain consistency with the original problem.")
            
        return corrected


class AgentCLI:
    """
    Command-line interface for interacting with agents.
    """
    
    def __init__(self, debug=False):
        """Initialize the CLI."""
        self.debug = debug
        self.running = True
        self.current_agent = None
        self.agents = {}
        self.formatter = OutputFormatter()
        self.problem_solver = ProblemSolver([])
        self.in_conversation = False
        self.multi_line_input = False
        self.multi_line_buffer = []
        
        # Display welcome message
        self.formatter.print_header("Agent CLI")
        self.formatter.print_info("Type /help for available commands.")
        
        # Check if LLM API keys are set
        if not self._check_llm_api_keys():
            self.formatter.print_warning("No LLM API keys found. Agent responses will use fallback mode.")
            self.formatter.print_info("Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or HUGGINGFACE_API_KEY environment variables.")
    
    def _check_llm_api_keys(self) -> bool:
        """
        Check if any LLM API keys are set in the environment.
        
        Returns:
            True if at least one API key is set
        """
        return (
            os.environ.get("OPENAI_API_KEY") is not None or 
            os.environ.get("ANTHROPIC_API_KEY") is not None or
            os.environ.get("ANTHROPIC") is not None or  # Alternative naming
            os.environ.get("HUGGINGFACE_API_KEY") is not None
        )
    
    async def start(self):
        """Start the CLI event loop."""
        while self.running:
            try:
                # Get user input
                user_input = await self._get_user_input()
                
                if not user_input:
                    continue
                
                # Handle multi-line input
                if self.multi_line_input:
                    if user_input == "/end":
                        self.multi_line_input = False
                        user_input = "\n".join(self.multi_line_buffer)
                        self.multi_line_buffer = []
                    else:
                        self.multi_line_buffer.append(user_input)
                        continue
                
                # Check if the input is a command
                if user_input.startswith("/"):
                    await self._process_command(user_input)
                else:
                    # If we're in a conversation, treat as a message to the current agent
                    if self.in_conversation:
                        await self._handle_conversation_message(user_input)
                    else:
                        self.formatter.print_error("Not in a conversation. Use /startconvo to begin or /help for commands.")
            
            except KeyboardInterrupt:
                print("\n")
                self.formatter.print_info("Exiting...")
                self.running = False
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                self.formatter.print_error(f"Error: {str(e)}")
    
    async def _get_user_input(self) -> str:
        """
        Get input from the user with appropriate prompt.
        
        Returns:
            The user's input
        """
        if self.multi_line_input:
            prompt = "... "
        elif self.in_conversation:
            prompt = "You: "
        else:
            prompt = "> "
        
        return input(prompt).strip()
    
    async def _process_command(self, command: str):
        """
        Process a command entered by the user.
        
        Args:
            command: The command to process
        """
        # Split the command into parts
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Handle commands
        if cmd == "/exit":
            self.formatter.print_info("Exiting...")
            self.running = False
        
        elif cmd == "/help":
            await self._show_help()
        
        elif cmd == "/list":
            await self._list_agents()
            
        elif cmd == "/createagent":
            await self._create_agent(args)
        
        elif cmd == "/solve":
            if not args:
                self.formatter.print_error("Please provide a problem statement.")
                self.formatter.print_info("Usage: /solve <problem_statement>")
                return
            
            await self._solve_problem(args)
        
        elif cmd == "/startconvo":
            await self._start_conversation(args)
        
        elif cmd == "/endconvo":
            await self._end_conversation()
        
        elif cmd == "/multiline":
            self.multi_line_input = True
            self.formatter.print_info("Entering multi-line input mode. Type /end when finished.")
        
        elif cmd == "/personalities":
            await self._list_personalities()
        
        else:
            self.formatter.print_error(f"Unknown command: {cmd}")
            self.formatter.print_info("Type /help for available commands.")
    
    async def _solve_problem(self, args: str):
        """
        Solve a problem using multiple agents in collaboration.
        
        Args:
            args: The problem statement
        """
        if not args:
            self.formatter.print_error("Please provide a problem statement.")
            return
            
        # Get the problem statement
        problem_statement = args.strip()
        
        print(f"\n=== Problem Solving Session ===")
        print(f"Problem: {problem_statement}")
        
        try:
            # Create standard agent personalities for the problem solver
            standard_personalities = [
                AgentPersonality(
                    name="Sam Rivera",
                    role="Collaboration Facilitator",
                    short_description="Expert in guiding productive discussions and synthesizing viewpoints",
                    detailed_background="I specialize in creating structured collaborative environments where diverse perspectives can be effectively integrated to solve complex problems.",
                    strengths=["diplomatic", "structured", "integrative"],
                    problem_solving_approach="collaborative"
                ),
                AgentPersonality(
                    name="Dr. Alex Morgan",
                    role="Research Specialist",
                    short_description="Data-driven analyst with expertise in methodical problem decomposition",
                    detailed_background="I approach problems by breaking them into analyzable components and applying evidence-based reasoning to develop comprehensive solutions.",
                    strengths=["analytical", "thorough", "evidence-based"],
                    problem_solving_approach="analytical"
                ),
                AgentPersonality(
                    name="Maya Chen",
                    role="Innovation Strategist",
                    short_description="Creative thinker focused on novel approaches and possibilities",
                    detailed_background="I bring fresh perspectives to problem-solving by identifying unconventional approaches and creative solutions that others might overlook.",
                    strengths=["creative", "visionary", "open-minded"],
                    problem_solving_approach="creative"
                ),
                AgentPersonality(
                    name="Jamie Washington",
                    role="Implementation Expert",
                    short_description="Practical problem-solver focused on actionable solutions",
                    detailed_background="I specialize in translating abstract ideas into concrete, implementable plans with clear steps and considerations for real-world constraints.",
                    strengths=["practical", "detail-oriented", "action-focused"],
                    problem_solving_approach="practical"
                )
            ]
            
            # Create agents for the problem solver
            agents = []
            for i, personality in enumerate(standard_personalities):
                agent_id = f"agent_{i+1}"
                agents.append(Agent(agent_id, personality))
            
            # Initialize the problem solver with the agents
            self.problem_solver = ProblemSolver(agents)
            
            # Use the problem solver to solve the problem
            solution_data = await self.problem_solver.solve(problem_statement)
            
            # Display the solution
            print("\n=== Solution ===")
            self.formatter.print_info(solution_data)
            
        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            self.formatter.print_error(f"Error during problem solving: {str(e)}")
            print("You can try again with a different problem statement or more specific context.")
    
    async def _show_help(self):
        """Show help information."""
        self.formatter.print_header("Available Commands")
        commands = [
            ("/help", "Show this help information"),
            ("/exit", "Exit the CLI"),
            ("/list", "List available agents"),
            ("/createagent <agent_id> [personality_type]", "Create a new agent"),
            ("/solve <problem_statement>", "Use a team of agents to solve a problem"),
            ("/startconvo [problem_statement]", "Start a multi-agent conversation"),
            ("/endconvo", "End the current conversation"),
            ("/personalities", "List available personality types"),
            ("/multiline", "Enter multi-line input mode (end with /end)")
        ]
        
        for cmd, desc in commands:
            self.formatter.print_command_help(cmd, desc)
    
    async def _list_personalities(self):
        """List available personality types."""
        self.formatter.print_header("Available Personality Types")
        personalities = [
            ("analytical", "Analytical Expert - Logical, systematic approach to problem-solving"),
            ("creative", "Creative Innovator - Novel ideas and unconventional solutions"),
            ("mediator", "Mediator/Collaborator - Facilitates discussion and finds common ground"),
            ("practical", "Practical Implementer - Focuses on efficient, realistic solutions"),
            ("critical", "Critical Evaluator - Identifies flaws and ensures robustness")
        ]
        
        for p_type, desc in personalities:
            self.formatter.print_info(f"{p_type}: {desc}")
    
    async def _list_agents(self):
        """List the available agents."""
        self.formatter.print_header("Available Agents")
        
        if not self.agents:
            self.formatter.print_info("No agents created yet. Use /createagent to create one.")
            return
        
        for agent_id, agent in self.agents.items():
            self.formatter.print_info(f"ID: {agent_id} | Name: {agent.name} | Role: {agent.role}")
    
    async def _create_agent(self, args: str):
        """
        Create a new agent.
        
        Args:
            args: Command arguments containing agent ID and optional personality type
        """
        parts = args.strip().split(maxsplit=1)
        
        if not parts:
            self.formatter.print_error("Please provide an agent ID")
            return
            
        agent_id = parts[0]
        
        if agent_id in self.agents:
            self.formatter.print_error(f"Agent with ID '{agent_id}' already exists")
            return
        
        # Default to analytical expert if no personality type specified
        personality_type = parts[1] if len(parts) > 1 else "analytical"
        
        # Get the appropriate personality function
        personality_func = {
            "analytical": get_analytical_expert,
            "creative": get_creative_innovator,
            "mediator": get_mediator_collaborator,
            "practical": get_practical_implementer,
            "critical": get_critical_evaluator
        }.get(personality_type.lower())
        
        if not personality_func:
            self.formatter.print_error(f"Unknown personality type '{personality_type}'")
            self.formatter.print_info("Available types: analytical, creative, mediator, practical, critical")
            return
            
        # Create the personality and agent
        personality = personality_func()
        
        # Override personality name with agent_id for easier identification
        personality.name = agent_id.capitalize()
        
        self.agents[agent_id] = Agent(agent_id, personality)
        self.formatter.print_success(f"Created agent '{agent_id}' with {personality_type} personality")
        
        # Set as current agent if we don't have one
        if not self.current_agent:
            self.current_agent = agent_id
            self.formatter.print_info(f"Set '{agent_id}' as the current agent")
    
    async def _start_conversation(self, problem_statement: str = None):
        """
        Start a conversation with agents.
        
        Args:
            problem_statement: Optional problem statement to focus the conversation
        """
        self.in_conversation = True
        
        if not self.agents:
            # Create a default agent if none exists
            await self._create_agent("assistant mediator")
        
        self.current_agent = list(self.agents.values())[0]
        
        self.formatter.print_header("Starting Conversation")
        if problem_statement:
            self.formatter.print_info(f"Problem: {problem_statement}")
            
            # Have the agent respond to the problem statement
            response = await self.current_agent.generate_response(
                f"The user wants to discuss the following problem: {problem_statement}. "
                f"Please acknowledge the problem and provide some initial thoughts."
            )
            self.formatter.print_agent_message(self.current_agent.name, response)
        
        self.formatter.print_info("Type your messages directly. Use /endconvo to end the conversation.")
    
    async def _end_conversation(self):
        """End the current conversation."""
        if not self.in_conversation:
            self.formatter.print_info("No active conversation to end.")
            return
        
        self.in_conversation = False
        self.formatter.print_info("Conversation ended.")
    
    async def _handle_conversation_message(self, message: str):
        """
        Handle a message in a conversation.
        
        Args:
            message: The message to process
        """
        if not self.current_agent:
            self.formatter.print_error("No agent selected for conversation.")
            return
        
        # Show that the agent is thinking
        self.formatter.print_thinking(self.current_agent.name)
        
        try:
            # Get the agent's response
            response = await self.current_agent.generate_response(message)
            
            # Display the response
            self.formatter.print_agent_message(self.current_agent.name, response)
            
        except Exception as e:
            if self.debug:
                traceback.print_exc()
            self.formatter.print_error(f"Error getting agent response: {str(e)}")


async def main() -> None:
    """Main entry point for the CLI application."""
    app = AgentCLI()
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())
