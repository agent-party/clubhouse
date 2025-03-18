# Test Strategy

## Overview

This document outlines the test-driven development approach for the Agent Orchestration Platform, ensuring reliable and maintainable code. Following our quality-first development principles, we aim for comprehensive test coverage that validates both functionality and architectural integrity.

## Testing Principles

1. **Test First, Implement Later**
   - Write tests before implementing functionality
   - Use tests to clarify and document requirements
   - Ensure all features have corresponding test cases

2. **Comprehensive Coverage**
   - Target near 100% test coverage with strategic exclusions
   - Create tests at multiple levels (unit, integration, system)
   - Test both happy paths and error scenarios

3. **Independent and Repeatable**
   - Design tests to be independent of each other
   - Ensure tests can run in any order
   - Make tests deterministic and free of external dependencies

4. **Realistic and Representative**
   - Use realistic test data that represents production scenarios
   - Test edge cases and boundary conditions
   - Include performance and scale testing

## Test Levels

### Unit Tests

Focus on testing individual components in isolation.

```python
import unittest
from unittest.mock import MagicMock, patch
from agent_orchestration.evolution.engine import EvolutionEngine

class TestEvolutionEngine(unittest.TestCase):
    """Test suite for EvolutionEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent_factory = MagicMock()
        self.mock_knowledge_graph = MagicMock()
        self.mock_event_bus = MagicMock()
        
        # Create instance under test
        self.engine = EvolutionEngine(
            agent_factory=self.mock_agent_factory,
            knowledge_graph=self.mock_knowledge_graph,
            event_bus=self.mock_event_bus
        )
    
    def test_initialize_evolution(self):
        """Test initializing an evolution process."""
        # Setup
        spec = {
            "name": "Test Evolution",
            "domain": "education",
            "target_capabilities": ["language_teaching"],
            "population_size": 3,
            "max_generations": 5,
            "selection_criteria": [
                {
                    "metric_name": "user_satisfaction",
                    "weight": 0.7,
                    "optimization": "maximize"
                }
            ]
        }
        
        evolution_id = "evo_12345"
        self.mock_knowledge_graph.add_node.return_value = evolution_id
        
        # Execute
        result = self.engine.initialize_evolution(spec)
        
        # Assert
        self.assertEqual(result, evolution_id)
        self.mock_knowledge_graph.add_node.assert_called_once()
        self.mock_event_bus.publish.assert_called_once_with(
            "evolution_started",
            {
                "evolution_id": evolution_id,
                "domain": "education",
                "target_capabilities": ["language_teaching"]
            }
        )
```

### Integration Tests

Test interactions between components.

```python
import unittest
from unittest.mock import MagicMock, patch
import asyncio

class TestSessionManagerIntegration(unittest.TestCase):
    """Integration test for SessionManager with OpenAIAgentAdapter."""
    
    def setUp(self):
        """Set up test dependencies."""
        # Create real instances of some components
        self.openai_adapter = OpenAIAgentAdapter(MockOpenAIClient(), MockEventBus())
        self.function_registry = FunctionRegistry()
        self.event_bus = MockEventBus()
        
        # Register test functions
        self.function_registry.register_function(
            name="test_function",
            handler=self.test_function_handler,
            schema={
                "name": "test_function",
                "description": "Test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"}
                    },
                    "required": ["param1"]
                }
            }
        )
        
        # Create system under test
        self.thread_manager = ThreadManager(
            openai_adapter=self.openai_adapter,
            function_registry=self.function_registry,
            event_bus=self.event_bus
        )
    
    def test_function_handler(self, param1):
        """Test function handler."""
        return f"Processed: {param1}"
    
    def test_session_lifecycle(self):
        """Test the full lifecycle of a session."""
        # Create session
        session = self.thread_manager.create_session(
            agent_id="agent_123",
            user_id="user_123"
        )
        
        self.assertIsNotNone(session["session_id"])
        self.assertIsNotNone(session["thread_id"])
        
        # Process message
        async def run_test():
            response = await self.thread_manager.process_message(
                session_id=session["session_id"],
                thread_id=session["thread_id"],
                assistant_id="asst_123",
                message="Hello, can you help me?"
            )
            
            self.assertIsNotNone(response["response"])
            self.assertEqual(response["status"], "completed")
        
        asyncio.run(run_test())
        
        # Verify events
        self.assertTrue(any(e["type"] == "session_created" for e in self.event_bus.events))
        self.assertTrue(any(e["type"] == "message_sent" for e in self.event_bus.events))
        self.assertTrue(any(e["type"] == "message_received" for e in self.event_bus.events))
```

### System Tests

Test the entire system from an external perspective.

```python
import unittest
import requests
import time

class TestAgentEvolutionSystem(unittest.TestCase):
    """System test for agent evolution."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_url = "http://localhost:8000"
        self.api_key = "test_api_key"
        
        # Create test agent
        response = requests.post(
            f"{self.base_url}/agents",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "name": "Test Agent",
                "domain": "education",
                "capabilities": [
                    {
                        "name": "answer_question",
                        "description": "Answer educational questions",
                        "implementation": "qa_capability"
                    }
                ]
            }
        )
        
        self.agent_id = response.json()["agent_id"]
    
    def test_evolution_process(self):
        """Test a complete evolution process."""
        # Initialize evolution
        response = requests.post(
            f"{self.base_url}/evolutions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "name": "Improve QA Capability",
                "domain": "education",
                "target_capabilities": ["answer_question"],
                "population_size": 3,
                "max_generations": 2,
                "selection_criteria": [
                    {
                        "metric_name": "accuracy",
                        "weight": 0.7,
                        "optimization": "maximize"
                    },
                    {
                        "metric_name": "response_time",
                        "weight": 0.3,
                        "optimization": "minimize"
                    }
                ]
            }
        )
        
        evolution_id = response.json()["evolution_id"]
        
        # Create experiment
        response = requests.post(
            f"{self.base_url}/evolutions/{evolution_id}/experiments",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "name": "Test QA Variations",
                "baseline_agent_id": self.agent_id,
                "variation_strategies": [
                    {
                        "strategy_type": "prompt_variation",
                        "target_capability": "answer_question",
                        "parameters": {
                            "variation_count": 2
                        }
                    }
                ],
                "evaluation_metrics": ["accuracy", "response_time"]
            }
        )
        
        experiment_id = response.json()["experiment_id"]
        
        # Wait for experiment to complete
        max_wait_time = 60  # seconds
        poll_interval = 5  # seconds
        wait_time = 0
        
        while wait_time < max_wait_time:
            response = requests.get(
                f"{self.base_url}/evolutions/{evolution_id}/experiments/{experiment_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            status = response.json()["status"]
            if status in ["completed", "failed"]:
                break
            
            time.sleep(poll_interval)
            wait_time += poll_interval
        
        # Assert experiment completed
        self.assertEqual(status, "completed")
        
        # Verify metrics
        metrics = response.json()["metrics"]
        self.assertIsNotNone(metrics)
        
        # Verify selected variation
        selected_variation = response.json()["selected_variation"]
        self.assertIsNotNone(selected_variation)
```

## Test Categories

### Functional Tests

Test that components behave as expected.

```python
def test_feedback_processing():
    """Test that feedback is processed correctly."""
    # Arrange
    feedback_processor = FeedbackProcessor(mock_knowledge_graph, mock_event_bus)
    feedback = {
        "feedback_id": "fb_12345",
        "session_id": "sess_12345",
        "agent_id": "agent_12345",
        "user_id": "user_12345",
        "ratings": [
            {
                "metric": "helpfulness",
                "value": 4.5,
                "scale": "1-5"
            }
        ],
        "comments": "Very helpful but could be faster."
    }
    
    # Act
    feedback_processor.process_feedback(feedback)
    
    # Assert
    mock_knowledge_graph.add_node.assert_called_once()
    mock_event_bus.publish.assert_called_once_with(
        "feedback_processed",
        {
            "feedback_id": "fb_12345",
            "agent_id": "agent_12345"
        }
    )
```

### Performance Tests

Test system performance under various conditions.

```python
def test_session_throughput():
    """Test session handling throughput."""
    import time
    import concurrent.futures
    
    session_manager = SessionManager(openai_adapter, event_bus)
    
    # Create test data
    num_sessions = 50
    sessions = []
    
    for i in range(num_sessions):
        session = session_manager.start_session(
            agent_id=f"agent_{i}",
            user_id=f"user_{i}",
            context={"test": True}
        )
        sessions.append(session)
    
    # Measure throughput
    start_time = time.time()
    
    def send_message(session):
        return session_manager.send_message(
            session_id=session["session_id"],
            content="Test message",
            role="user"
        )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(send_message, sessions))
    
    end_time = time.time()
    
    # Calculate throughput
    duration = end_time - start_time
    throughput = num_sessions / duration
    
    # Assert minimum throughput
    self.assertGreaterEqual(throughput, 5)  # At least 5 messages per second
```

### Stress Tests

Test system behavior under high load.

```python
def test_concurrent_evolutions():
    """Test handling multiple concurrent evolution processes."""
    import time
    import concurrent.futures
    
    evolution_engine = EvolutionEngine(agent_factory, knowledge_graph, event_bus)
    
    # Create test data
    num_evolutions = 10
    specs = []
    
    for i in range(num_evolutions):
        spec = {
            "name": f"Evolution {i}",
            "domain": "test",
            "target_capabilities": ["test_capability"],
            "population_size": 3,
            "max_generations": 2,
            "selection_criteria": [
                {
                    "metric_name": "test_metric",
                    "weight": 1.0,
                    "optimization": "maximize"
                }
            ]
        }
        specs.append(spec)
    
    # Run concurrent evolutions
    def run_evolution(spec):
        evolution_id = evolution_engine.initialize_evolution(spec)
        return evolution_id
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_evolutions) as executor:
        evolution_ids = list(executor.map(run_evolution, specs))
    
    # Verify all evolutions started
    self.assertEqual(len(evolution_ids), num_evolutions)
    self.assertEqual(len(set(evolution_ids)), num_evolutions)  # All IDs should be unique
    
    # Verify system stability
    # Check CPU and memory usage, response time, etc.
```

## Mocking Strategy

### External Service Mocks

```python
class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self, simulate_errors=False):
        """Initialize with simulation settings."""
        self.assistants = MockAssistants(simulate_errors)
        self.beta = MockBeta(simulate_errors)
        self.files = MockFiles(simulate_errors)
        self.simulate_errors = simulate_errors

class MockNeo4jDriver:
    """Mock Neo4j driver for testing."""
    
    def __init__(self, data=None):
        """Initialize with optional test data."""
        self.data = data or {}
        self.executed_queries = []
    
    def session(self):
        """Return a mock session."""
        return MockNeo4jSession(self)

class MockKafkaProducer:
    """Mock Kafka producer for testing."""
    
    def __init__(self):
        """Initialize with empty state."""
        self.sent_messages = []
    
    def send(self, topic, value, key=None):
        """Mock sending a message."""
        self.sent_messages.append({
            "topic": topic,
            "value": value,
            "key": key
        })
```

### Event Bus Mock

```python
class MockEventBus:
    """Mock event bus for testing."""
    
    def __init__(self):
        """Initialize with empty state."""
        self.events = []
        self.subscribers = {}
    
    def publish(self, event_type, payload):
        """Publish an event."""
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": time.time()
        }
        self.events.append(event)
        
        # Notify subscribers
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                handler(payload)
    
    def subscribe(self, event_type, handler):
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
```

## Test Organization

### Directory Structure

```
tests/
├── unit/                   # Unit tests
│   ├── model/              # Tests for data models
│   ├── core/               # Tests for core services
│   ├── integration/        # Tests for integration components
│   └── data/               # Tests for data access layer
├── integration/            # Integration tests
│   ├── api/                # API integration tests
│   ├── database/           # Database integration tests
│   └── events/             # Event system integration tests
├── system/                 # System tests
│   ├── scenarios/          # Scenario-based tests
│   └── performance/        # Performance and stress tests
├── mocks/                  # Mock implementations
├── fixtures/               # Test fixtures and data
└── conftest.py             # pytest configuration
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run unit tests
python -m pytest tests/unit/

# Run specific test file
python -m pytest tests/unit/core/test_evolution_engine.py

# Run with coverage
python -m pytest --cov=agent_orchestration tests/

# Generate coverage report
python -m pytest --cov=agent_orchestration --cov-report=html tests/
```

## Continuous Integration

```yaml
# .github/workflows/test.yml
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        python -m pytest --cov=agent_orchestration tests/
    
    - name: Upload coverage report
      uses: actions/upload-artifact@v2
      with:
        name: coverage-report
        path: htmlcov/
```

## Test Data Management

### Fixture Pattern

```python
import pytest
from datetime import datetime

@pytest.fixture
def sample_agent():
    """Provide a sample agent for testing."""
    return {
        "agent_id": "agent_12345",
        "name": "Test Agent",
        "domain": "education",
        "description": "A test agent",
        "version": "1.0.0",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "capabilities": [
            {
                "name": "test_capability",
                "description": "A test capability",
                "implementation": "test_implementation",
                "effectiveness": 0.8
            }
        ],
        "knowledge_sources": [],
        "configuration": {}
    }

@pytest.fixture
def sample_evolution_spec():
    """Provide a sample evolution specification."""
    return {
        "name": "Test Evolution",
        "domain": "education",
        "target_capabilities": ["test_capability"],
        "population_size": 3,
        "max_generations": 5,
        "selection_criteria": [
            {
                "metric_name": "effectiveness",
                "weight": 1.0,
                "optimization": "maximize"
            }
        ]
    }
```

### Factory Pattern

```python
class AgentFactory:
    """Factory for creating test agents."""
    
    @staticmethod
    def create(
        agent_id=None,
        name=None,
        domain=None,
        capabilities=None,
        **kwargs
    ):
        """Create a test agent with specified or default properties."""
        agent = {
            "agent_id": agent_id or f"agent_{uuid.uuid4().hex[:8]}",
            "name": name or "Test Agent",
            "domain": domain or "test",
            "description": kwargs.get("description", "A test agent"),
            "version": kwargs.get("version", "1.0.0"),
            "created_at": kwargs.get("created_at", datetime.now()),
            "updated_at": kwargs.get("updated_at", datetime.now()),
            "capabilities": capabilities or [
                {
                    "name": "test_capability",
                    "description": "A test capability",
                    "implementation": "test_implementation",
                    "effectiveness": 0.8
                }
            ],
            "knowledge_sources": kwargs.get("knowledge_sources", []),
            "configuration": kwargs.get("configuration", {})
        }
        
        return agent
```

## Test Assertions

### Custom Assertions

```python
def assert_evolution_valid(evolution):
    """Assert that an evolution object is valid."""
    assert "evolution_id" in evolution
    assert "name" in evolution
    assert "domain" in evolution
    assert "target_capabilities" in evolution
    assert "population_size" in evolution
    assert "max_generations" in evolution
    assert "selection_criteria" in evolution
    assert len(evolution["selection_criteria"]) > 0
    
    for criterion in evolution["selection_criteria"]:
        assert "metric_name" in criterion
        assert "weight" in criterion
        assert "optimization" in criterion
        assert criterion["optimization"] in ["maximize", "minimize"]

def assert_event_published(event_bus, event_type, required_fields=None):
    """Assert that an event was published with required fields."""
    events = [e for e in event_bus.events if e["type"] == event_type]
    assert len(events) > 0, f"No events of type {event_type} were published"
    
    if required_fields:
        event = events[-1]  # Get the most recent event
        for field in required_fields:
            assert field in event["payload"], f"Field {field} missing from event payload"
```

## Test-Driven Development Workflow

1. **Write Test**
   - Define expected behavior
   - Create test case
   - Verify test fails (red)

2. **Implement Code**
   - Write minimal code to pass test
   - Verify test passes (green)

3. **Refactor**
   - Improve code quality
   - Maintain test coverage
   - Verify tests still pass

4. **Repeat**
   - Add additional tests
   - Implement more functionality
   - Continue until requirements are met
