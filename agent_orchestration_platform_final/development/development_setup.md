# Development Setup Guide

## Overview

This guide provides instructions for setting up a development environment for the Agent Orchestration Platform. Following our quality-first and test-driven development principles, this setup ensures a consistent environment for all developers.

## Prerequisites

- **Python**: 3.9+ (3.10 recommended)
- **Docker**: 20.10+ and Docker Compose
- **Git**: 2.30+
- **IDE**: VSCode (recommended) with Python and Neo4j extensions
- **OpenAI API Key**: Required for agent interactions

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/agent-orchestration-platform.git
cd agent-orchestration-platform
```

### 2. Environment Setup

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# OpenAI API
OPENAI_API_KEY=your_api_key_here

# Neo4j
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Development
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### 4. Start Development Infrastructure

Use Docker Compose to start the required services:

```bash
docker-compose -f docker-compose.dev.yml up -d
```

This will start:
- Neo4j database
- Kafka and Zookeeper
- Mock services (if needed)

### 5. Initialize Database

Run the database initialization script:

```bash
python -m agent_orchestration.scripts.init_db
```

## Development Workflow

### 1. Test-Driven Development Cycle

Following our TDD principles:

```bash
# 1. Write a failing test
nano tests/unit/core/test_evolution_engine.py

# 2. Run the test to verify it fails
python -m pytest tests/unit/core/test_evolution_engine.py -v

# 3. Implement code to make the test pass
nano agent_orchestration/core/evolution_engine.py

# 4. Run the test again to verify it passes
python -m pytest tests/unit/core/test_evolution_engine.py -v

# 5. Refactor while keeping tests passing
# 6. Run all tests to ensure no regressions
python -m pytest
```

### 2. Code Quality Checks

Run quality checks before committing:

```bash
# Format code with black
black agent_orchestration tests

# Sort imports with isort
isort agent_orchestration tests

# Run mypy for type checking
mypy agent_orchestration

# Run all tests with coverage
python -m pytest --cov=agent_orchestration --cov-report=html
```

### 3. Local Development Server

Start the development server:

```bash
python -m agent_orchestration.main
```

The server will be available at `http://localhost:8000`.

## Project Structure

```
agent_orchestration/
├── __init__.py
├── main.py                  # Application entry point
├── config.py                # Configuration loader
├── api/                     # API layer
│   ├── __init__.py
│   ├── mcp_server.py        # MCP server integration
│   └── resources/           # MCP resources
├── core/                    # Core business logic
│   ├── __init__.py
│   ├── agent_factory.py     # Agent creation
│   ├── evolution_engine.py  # Evolution processes
│   └── session_manager.py   # Session management
├── integration/             # External integrations
│   ├── __init__.py
│   ├── openai_adapter.py    # OpenAI integration
│   ├── kafka_event_bus.py   # Kafka integration
│   └── neo4j_graph.py       # Neo4j integration
├── data/                    # Data access layer
│   ├── __init__.py
│   ├── repositories/        # Repository implementations
│   └── models/              # Pydantic models
├── util/                    # Utility functions
│   ├── __init__.py
│   ├── logging.py           # Logging configuration
│   └── validation.py        # Validation helpers
└── scripts/                 # Utility scripts
    ├── __init__.py
    ├── init_db.py           # Database initialization
    └── generate_test_data.py # Test data generation
```

## IDE Setup

### VSCode Configuration

Create a `.vscode/settings.json` file:

```json
{
  "python.linting.enabled": true,
  "python.linting.mypyEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.nosetestsEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

Create a `.vscode/launch.json` file:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Main",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/agent_orchestration/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Python: Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}"
      ],
      "console": "integratedTerminal"
    }
  ]
}
```

## Dependency Management

### Core Dependencies

The `requirements.txt` file includes:

```
# API and Web
fastapi==0.95.1
uvicorn==0.22.0
pydantic==1.10.7
starlette==0.26.1

# OpenAI Integration
openai==1.3.0

# Database
neo4j==5.8.1

# Event Streaming
confluent-kafka==2.1.1

# Utilities
python-dotenv==1.0.0
loguru==0.7.0
uuid==1.30
```

### Development Dependencies

The `requirements-dev.txt` file includes:

```
# Testing
pytest==7.3.1
pytest-cov==4.1.0
pytest-asyncio==0.21.0
httpx==0.24.0

# Linting and Formatting
black==23.3.0
isort==5.12.0
mypy==1.3.0
flake8==6.0.0

# Type Stubs
types-requests==2.29.0.0
types-pyyaml==6.0.12.9

# Documentation
mkdocs==1.4.3
mkdocs-material==9.1.11
```

## Testing Setup

### Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── api/                 # API tests
│   ├── core/                # Core logic tests
│   ├── integration/         # Integration tests
│   └── data/                # Data layer tests
├── integration/             # Integration tests
│   ├── __init__.py
│   └── test_api.py          # API integration tests
└── system/                  # System tests
    ├── __init__.py
    └── test_evolution.py    # Full evolution process
```

### Common Test Fixtures

Create a `tests/conftest.py` file:

```python
import pytest
from unittest.mock import MagicMock
import asyncio
import uuid
from datetime import datetime

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    
    # Configure mock responses
    assistant = MagicMock()
    assistant.id = f"asst_{uuid.uuid4().hex[:10]}"
    client.beta.assistants.create.return_value = assistant
    
    thread = MagicMock()
    thread.id = f"thread_{uuid.uuid4().hex[:10]}"
    client.beta.threads.create.return_value = thread
    
    return client

@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    transaction = MagicMock()
    result = MagicMock()
    
    # Configure result mock
    result.single.return_value = {"id": "node_id"}
    
    # Link mocks
    transaction.run.return_value = result
    session.begin_transaction.return_value = transaction
    session.__enter__.return_value = session
    driver.session.return_value = session
    
    return driver

@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.events = []
    
    def publish(event_type, payload):
        """Record published events."""
        bus.events.append({
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        })
    
    bus.publish.side_effect = publish
    return bus

@pytest.fixture
def sample_agent_spec():
    """Provide a sample agent specification."""
    return {
        "name": "Test Agent",
        "domain": "education",
        "description": "A test agent for education domain",
        "capabilities": [
            {
                "name": "answer_question",
                "description": "Answer educational questions",
                "implementation": "qa_capability"
            }
        ],
        "configuration": {
            "model": "gpt-4-turbo"
        }
    }

@pytest.fixture
def sample_evolution_spec():
    """Provide a sample evolution specification."""
    return {
        "name": "Test Evolution",
        "domain": "education",
        "target_capabilities": ["answer_question"],
        "population_size": 3,
        "max_generations": 2,
        "selection_criteria": [
            {
                "metric_name": "accuracy",
                "weight": 0.7,
                "optimization": "maximize"
            }
        ]
    }
```

## Docker Setup

### Development Docker Compose

Create a `docker-compose.dev.yml` file:

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:4.4
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_dbms_memory_heap_max__size=2G
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
    networks:
      - agent-network

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"
    networks:
      - agent-network

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    networks:
      - agent-network

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    networks:
      - agent-network

networks:
  agent-network:
    driver: bridge

volumes:
  neo4j-data:
  neo4j-logs:
```

## Git Workflow

Follow this Git workflow:

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/new-feature-name
   ```

2. Make changes following TDD principles

3. Ensure all tests pass and run quality checks:
   ```bash
   # Format and lint
   black agent_orchestration tests
   isort agent_orchestration tests
   mypy agent_orchestration
   
   # Run tests
   python -m pytest
   ```

4. Commit your changes with descriptive messages:
   ```bash
   git add .
   git commit -m "feat: Add new capability for X"
   ```

5. Push your branch and create a pull request:
   ```bash
   git push -u origin feature/new-feature-name
   ```

## Common Development Tasks

### Adding a New Capability

1. Create a test file:
   ```bash
   touch tests/unit/core/capabilities/test_new_capability.py
   ```

2. Write tests for the new capability

3. Create the capability implementation:
   ```bash
   touch agent_orchestration/core/capabilities/new_capability.py
   ```

4. Implement the capability following the BaseCapability pattern

5. Register the capability in the capability registry

### Debugging Tips

1. Use VSCode debugging with breakpoints

2. Enable debug logging in `.env`:
   ```
   DEBUG=true
   LOG_LEVEL=DEBUG
   ```

3. Add contextual logging:
   ```python
   from agent_orchestration.util.logging import get_logger
   
   logger = get_logger(__name__)
   
   def my_function():
       logger.debug("Starting function with context", extra={"context": "value"})
       # Function logic
       logger.info("Operation completed", extra={"result": "success"})
   ```

4. Use the Neo4j Browser for database inspection:
   - Open http://localhost:7474 in your browser
   - Login with neo4j/password

5. Use Kafka UI for message inspection:
   - Open http://localhost:8080 in your browser

## Common Issues and Solutions

### Neo4j Connection Issues

If you encounter Neo4j connection issues:

```
Unable to connect to Neo4j at neo4j://localhost:7687
```

Check:
1. Neo4j container is running: `docker ps | grep neo4j`
2. Neo4j ports are exposed: `docker-compose -f docker-compose.dev.yml port neo4j 7687`
3. Correct credentials in `.env`
4. Reset the database if needed: `docker-compose -f docker-compose.dev.yml restart neo4j`

### Kafka Connection Issues

If you encounter Kafka issues:

```
No brokers available for Kafka connection
```

Check:
1. Kafka container is running: `docker ps | grep kafka`
2. Zookeeper is running: `docker ps | grep zookeeper`
3. Kafka is properly configured: `docker logs kafka`
4. Restart Kafka if needed: `docker-compose -f docker-compose.dev.yml restart kafka zookeeper`

## Continuous Integration

A GitHub Actions workflow has been set up in `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

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
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Check code formatting
      run: |
        black --check agent_orchestration tests
        isort --check agent_orchestration tests
    
    - name: Type checking
      run: |
        mypy agent_orchestration
    
    - name: Run tests
      run: |
        python -m pytest --cov=agent_orchestration tests/
        python -m pytest --cov=agent_orchestration --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

## Next Steps

1. Run the database initialization script:
   ```bash
   python -m agent_orchestration.scripts.init_db
   ```

2. Start the development server:
   ```bash
   python -m agent_orchestration.main
   ```

3. Begin development following the TDD workflow

4. Refer to the implementation guide documents for architecture and design details
