# OpenAI Agent Library Integration

## Overview

This document details how to integrate the OpenAI Agent Library with the Agent Orchestration Platform. The integration leverages OpenAI's Assistants API, function calling, knowledge retrieval, and thread management to create a powerful foundation for agent evolution and orchestration.

## Core Integration Components

### 1. OpenAI Agent Adapter

The adapter provides a consistent interface to the OpenAI Agent Library, abstracting the underlying API details:

```python
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

class OpenAIAgentAdapter:
    """Adapter for OpenAI Assistant API."""
    
    def __init__(self, openai_client: Any, event_bus: "EventBus"):
        """Initialize with OpenAI client."""
        self.client = openai_client
        self.event_bus = event_bus
        self.assistants = {}  # Cache of assistant details
    
    def create_assistant(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create an OpenAI Assistant."""
        tools = self._prepare_tools(spec.get("capabilities", []))
        
        # Create assistant
        assistant = self.client.beta.assistants.create(
            name=spec["name"],
            description=spec.get("description", ""),
            model=spec.get("model", "gpt-4-turbo"),
            instructions=spec.get("instructions", ""),
            tools=tools,
            metadata=spec.get("metadata", {})
        )
        
        # Cache assistant details
        self.assistants[assistant.id] = {
            "name": spec["name"],
            "capabilities": spec.get("capabilities", [])
        }
        
        # Publish event
        self.event_bus.publish("assistant_created", {
            "assistant_id": assistant.id,
            "name": spec["name"]
        })
        
        return {
            "assistant_id": assistant.id,
            "name": spec["name"],
            "model": spec.get("model", "gpt-4-turbo"),
            "capabilities": spec.get("capabilities", [])
        }
    
    def create_thread(self, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new conversation thread."""
        thread = self.client.beta.threads.create(
            metadata=metadata or {}
        )
        
        return {
            "thread_id": thread.id,
            "metadata": metadata or {}
        }
    
    def add_message(self, thread_id: str, content: str, role: str = "user") -> Dict[str, Any]:
        """Add a message to a thread."""
        message = self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
        
        return {
            "message_id": message.id,
            "thread_id": thread_id,
            "role": role
        }
    
    def run_assistant(
        self, 
        thread_id: str, 
        assistant_id: str,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run an assistant on a thread."""
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions=instructions
        )
        
        return {
            "run_id": run.id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": run.status
        }
    
    def get_run_status(self, thread_id: str, run_id: str) -> Dict[str, Any]:
        """Get the status of a run."""
        run = self.client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        result = {
            "run_id": run.id,
            "thread_id": thread_id,
            "status": run.status
        }
        
        if run.status == "requires_action":
            result["required_action"] = run.required_action
        
        return result
    
    def handle_required_action(
        self, 
        thread_id: str, 
        run_id: str, 
        function_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle a required action from a run."""
        run = self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=function_results
        )
        
        return {
            "run_id": run.id,
            "thread_id": thread_id,
            "status": run.status
        }
    
    def get_messages(
        self, 
        thread_id: str, 
        limit: int = 10, 
        order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """Get messages from a thread."""
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
            limit=limit,
            order=order
        )
        
        return [
            {
                "message_id": msg.id,
                "thread_id": thread_id,
                "role": msg.role,
                "content": msg.content[0].text.value if msg.content else "",
                "created_at": msg.created_at
            }
            for msg in messages.data
        ]
    
    def _prepare_tools(self, capabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare tools from capabilities."""
        tools = []
        
        for capability in capabilities:
            if "function" in capability:
                tools.append({
                    "type": "function",
                    "function": capability["function"]
                })
        
        # Always include code_interpreter if needed
        if any(cap.get("type") == "code_interpreter" for cap in capabilities):
            tools.append({"type": "code_interpreter"})
        
        # Always include retrieval if needed
        if any(cap.get("type") == "retrieval" for cap in capabilities):
            tools.append({"type": "retrieval"})
        
        return tools
```

### 2. Function Registry

The Function Registry maps OpenAI function calls to system capabilities:

```python
class FunctionRegistry:
    """Registry for functions available to OpenAI Assistants."""
    
    def __init__(self):
        """Initialize registry."""
        self.functions = {}
    
    def register_function(
        self, 
        name: str, 
        handler: callable, 
        schema: Dict[str, Any]
    ) -> None:
        """Register a function with the registry."""
        self.functions[name] = {
            "handler": handler,
            "schema": schema
        }
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all registered functions."""
        return [
            {
                "name": name,
                "description": fn["schema"].get("description", ""),
                "parameters": fn["schema"].get("parameters", {})
            }
            for name, fn in self.functions.items()
        ]
    
    def execute_function(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered function."""
        if name not in self.functions:
            raise ValueError(f"Function not found: {name}")
        
        try:
            result = self.functions[name]["handler"](**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
```

### 3. Thread Manager

The Thread Manager handles conversation state and context:

```python
class ThreadManager:
    """Manager for OpenAI thread lifecycle."""
    
    def __init__(
        self, 
        openai_adapter: OpenAIAgentAdapter,
        function_registry: FunctionRegistry,
        event_bus: "EventBus"
    ):
        """Initialize with dependencies."""
        self.openai_adapter = openai_adapter
        self.function_registry = function_registry
        self.event_bus = event_bus
        self.active_runs = {}
    
    def create_session(
        self, 
        agent_id: str, 
        user_id: str, 
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new session with thread."""
        # Create thread
        thread_metadata = {
            "agent_id": agent_id,
            "user_id": user_id,
            "session_id": str(uuid.uuid4()),
            **(metadata or {})
        }
        
        thread = self.openai_adapter.create_thread(thread_metadata)
        
        # Publish event
        self.event_bus.publish("session_created", {
            "session_id": thread_metadata["session_id"],
            "agent_id": agent_id,
            "user_id": user_id,
            "thread_id": thread["thread_id"]
        })
        
        return {
            "session_id": thread_metadata["session_id"],
            "thread_id": thread["thread_id"],
            "agent_id": agent_id,
            "user_id": user_id
        }
    
    async def process_message(
        self, 
        session_id: str, 
        thread_id: str, 
        assistant_id: str, 
        message: str
    ) -> Dict[str, Any]:
        """Process a user message and get assistant response."""
        # Add message to thread
        self.openai_adapter.add_message(thread_id, message)
        
        # Run assistant
        run = self.openai_adapter.run_assistant(thread_id, assistant_id)
        run_id = run["run_id"]
        
        # Track the run
        self.active_runs[run_id] = {
            "session_id": session_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "status": run["status"],
            "started_at": datetime.now().isoformat()
        }
        
        # Wait for completion or action
        await self._wait_for_run(thread_id, run_id)
        
        # Get latest messages
        messages = self.openai_adapter.get_messages(thread_id, limit=1)
        
        return {
            "session_id": session_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "response": messages[0]["content"] if messages else "",
            "status": self.active_runs[run_id]["status"]
        }
    
    async def _wait_for_run(self, thread_id: str, run_id: str) -> None:
        """Wait for a run to complete or require action."""
        max_attempts = 60  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            run_status = self.openai_adapter.get_run_status(thread_id, run_id)
            self.active_runs[run_id]["status"] = run_status["status"]
            
            if run_status["status"] == "completed":
                break
            
            if run_status["status"] == "requires_action":
                await self._handle_required_action(thread_id, run_id, run_status["required_action"])
                continue
            
            if run_status["status"] in ["failed", "cancelled", "expired"]:
                # Handle failure
                self.event_bus.publish("run_failed", {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "status": run_status["status"]
                })
                break
            
            # Wait before checking again
            import asyncio
            await asyncio.sleep(1)
    
    async def _handle_required_action(
        self, 
        thread_id: str, 
        run_id: str, 
        required_action: Dict[str, Any]
    ) -> None:
        """Handle a required action from a run."""
        if required_action["type"] != "submit_tool_outputs":
            return
        
        tool_calls = required_action["submit_tool_outputs"]["tool_calls"]
        tool_outputs = []
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = tool_call["function"]["arguments"]
            
            # Parse JSON arguments
            import json
            try:
                arguments = json.loads(function_args)
            except json.JSONDecodeError:
                arguments = {}
            
            # Execute function
            result = self.function_registry.execute_function(function_name, arguments)
            
            tool_outputs.append({
                "tool_call_id": tool_call["id"],
                "output": json.dumps(result)
            })
        
        # Submit tool outputs
        self.openai_adapter.handle_required_action(thread_id, run_id, tool_outputs)
```

## Integration Patterns

### 1. Assistant Creation

Creating specialized assistants based on domain requirements:

```python
def create_language_tutor_assistant(domain: str, target_language: str) -> Dict[str, Any]:
    """Create a specialized language tutor assistant."""
    
    # Define capabilities
    capabilities = [
        {
            "type": "function",
            "function": {
                "name": "assess_pronunciation",
                "description": "Assess the learner's pronunciation of a phrase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string"},
                        "recording_url": {"type": "string"}
                    },
                    "required": ["phrase", "recording_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_exercise",
                "description": "Generate a language exercise for the learner",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exercise_type": {"type": "string", "enum": ["vocabulary", "grammar", "conversation"]},
                        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                        "topic": {"type": "string"}
                    },
                    "required": ["exercise_type", "difficulty", "topic"]
                }
            }
        },
        {
            "type": "retrieval"  # Enable knowledge retrieval
        }
    ]
    
    # Define instructions
    instructions = f"""
        You are an expert {target_language} language tutor. Your goal is to help students learn
        {target_language} effectively through personalized instruction.
        
        Guidelines:
        1. Adapt to the student's proficiency level
        2. Provide clear explanations with examples
        3. Gently correct mistakes and explain the correct form
        4. Focus on practical conversation skills
        5. Use the assess_pronunciation function to evaluate pronunciation
        6. Use the generate_exercise function to create appropriate exercises
        7. Refer to learning materials when appropriate
    """
    
    # Create assistant specification
    assistant_spec = {
        "name": f"{target_language} Language Tutor",
        "description": f"Specialized tutor for {target_language} language learning",
        "model": "gpt-4-turbo",
        "instructions": instructions,
        "capabilities": capabilities,
        "metadata": {
            "domain": domain,
            "target_language": target_language
        }
    }
    
    # Create assistant
    return openai_adapter.create_assistant(assistant_spec)
```

### 2. Function Implementation

Implementing functions for OpenAI function calling:

```python
# Register pronunciation assessment function
function_registry.register_function(
    name="assess_pronunciation",
    handler=pronunciation_service.assess_pronunciation,
    schema={
        "name": "assess_pronunciation",
        "description": "Assess the learner's pronunciation of a phrase",
        "parameters": {
            "type": "object",
            "properties": {
                "phrase": {"type": "string"},
                "recording_url": {"type": "string"}
            },
            "required": ["phrase", "recording_url"]
        }
    }
)

# Register exercise generation function
function_registry.register_function(
    name="generate_exercise",
    handler=exercise_service.generate_exercise,
    schema={
        "name": "generate_exercise",
        "description": "Generate a language exercise for the learner",
        "parameters": {
            "type": "object",
            "properties": {
                "exercise_type": {"type": "string", "enum": ["vocabulary", "grammar", "conversation"]},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "topic": {"type": "string"}
            },
            "required": ["exercise_type", "difficulty", "topic"]
        }
    }
)
```

### 3. Knowledge Integration

Adding knowledge sources to assistants:

```python
async def add_knowledge_to_assistant(
    assistant_id: str,
    files: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Add knowledge files to an assistant."""
    
    # Upload files to OpenAI
    uploaded_files = []
    for file_data in files:
        file = await openai_client.files.create(
            file=file_data["content"],
            purpose="assistants"
        )
        uploaded_files.append(file.id)
    
    # Attach files to assistant
    updated_assistant = await openai_client.beta.assistants.update(
        assistant_id=assistant_id,
        file_ids=uploaded_files
    )
    
    return {
        "assistant_id": assistant_id,
        "file_ids": uploaded_files
    }
```

## Testing Strategy

### 1. Mock Client for Testing

```python
class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self):
        """Initialize mock client."""
        self.assistants = MockAssistants()
        self.beta = MockBeta()
        self.files = MockFiles()

class MockBeta:
    """Mock beta namespace."""
    
    def __init__(self):
        """Initialize mock beta."""
        self.assistants = MockAssistants()
        self.threads = MockThreads()

class MockAssistants:
    """Mock assistants API."""
    
    def __init__(self):
        """Initialize with empty state."""
        self.assistants = {}
        self.next_id = 1
    
    def create(self, **kwargs):
        """Create a mock assistant."""
        assistant_id = f"asst_{self.next_id}"
        self.next_id += 1
        
        assistant = MockAssistant(assistant_id, **kwargs)
        self.assistants[assistant_id] = assistant
        return assistant
    
    def update(self, assistant_id, **kwargs):
        """Update a mock assistant."""
        if assistant_id not in self.assistants:
            raise ValueError(f"Assistant not found: {assistant_id}")
        
        assistant = self.assistants[assistant_id]
        for key, value in kwargs.items():
            setattr(assistant, key, value)
        
        return assistant

# Additional mock classes for threads, messages, runs, etc.
```

### 2. Unit Test Example

```python
import unittest
from unittest.mock import MagicMock, patch

class TestOpenAIAgentAdapter(unittest.TestCase):
    """Test suite for OpenAIAgentAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_event_bus = MagicMock()
        
        # Configure mock returns
        mock_assistant = MagicMock()
        mock_assistant.id = "asst_test123"
        self.mock_client.beta.assistants.create.return_value = mock_assistant
        
        mock_thread = MagicMock()
        mock_thread.id = "thread_test123"
        self.mock_client.beta.threads.create.return_value = mock_thread
        
        # Initialize adapter
        self.adapter = OpenAIAgentAdapter(self.mock_client, self.mock_event_bus)
    
    def test_create_assistant(self):
        """Test creating an assistant."""
        # Define test spec
        spec = {
            "name": "Test Assistant",
            "description": "Test description",
            "model": "gpt-4-turbo",
            "capabilities": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "description": "Test function",
                        "parameters": {}
                    }
                }
            ]
        }
        
        # Call method
        result = self.adapter.create_assistant(spec)
        
        # Verify result
        self.assertEqual(result["assistant_id"], "asst_test123")
        self.assertEqual(result["name"], "Test Assistant")
        
        # Verify client call
        self.mock_client.beta.assistants.create.assert_called_once()
        
        # Verify event published
        self.mock_event_bus.publish.assert_called_once_with(
            "assistant_created",
            {
                "assistant_id": "asst_test123",
                "name": "Test Assistant"
            }
        )
```

## Integration with MCP

Exposing OpenAI capabilities through MCP:

```python
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp_server = FastMCP(name="Agent Orchestration Platform")

# Register assistant creation tool
@mcp_server.register_tool
def create_assistant(
    name: str,
    domain: str,
    assistant_type: str,
    configuration: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a specialized assistant."""
    
    # Determine assistant factory based on type
    if assistant_type == "language_tutor":
        assistant = create_language_tutor_assistant(
            domain=domain,
            target_language=configuration["target_language"]
        )
    elif assistant_type == "business_analyst":
        assistant = create_business_analyst_assistant(
            domain=domain,
            industry=configuration["industry"]
        )
    else:
        raise ValueError(f"Unsupported assistant type: {assistant_type}")
    
    # Store assistant mapping
    agent_id = str(uuid.uuid4())
    agent_repository.save_agent({
        "agent_id": agent_id,
        "name": name,
        "domain": domain,
        "assistant_id": assistant["assistant_id"],
        "assistant_type": assistant_type,
        "configuration": configuration
    })
    
    return {
        "agent_id": agent_id,
        "assistant_id": assistant["assistant_id"],
        "status": "created"
    }
```

## Best Practices

1. **Error Handling**
   - Implement comprehensive error handling for API failures
   - Use exponential backoff for rate limiting
   - Log detailed error information for debugging

2. **Cost Management**
   - Track API usage and costs
   - Implement caching where appropriate
   - Use lower-cost models for non-critical operations

3. **Testing**
   - Create comprehensive mock objects for testing
   - Test edge cases like token limits and timeouts
   - Use integration tests for critical flows

4. **Security**
   - Store API keys securely using environment variables or secrets manager
   - Validate inputs to prevent prompt injection
   - Sanitize outputs to prevent unsafe content

5. **Performance Optimization**
   - Batch operations where possible
   - Implement caching for frequently used data
   - Use concurrent processing for parallel operations
