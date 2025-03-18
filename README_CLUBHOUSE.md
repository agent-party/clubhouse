# Clubhouse Kafka-Based Architecture

This document describes the architecture of the Clubhouse system, which uses Kafka for message-based communication between a command-line interface (CLI) and the core Clubhouse components.

## Overview

The Clubhouse architecture consists of several key components:

1. **Command-line Interface (CLI)**: A thin client that sends commands to and receives responses from the Clubhouse via Kafka.
2. **Clubhouse Application**: The core application that manages agents, conversations, and message processing.
3. **Kafka Messaging**: The message bus that enables communication between the CLI and the Clubhouse.

## Architecture Components

### Command-line Interface (CLI)

The CLI is a thin client that provides a user interface for interacting with agents. It has the following features:

- Sends commands to the Clubhouse via Kafka
- Receives responses and events from the Clubhouse
- Provides a simple REPL (Read-Eval-Print Loop) interface
- Supports multi-line input and colorized output

### Clubhouse Application

The Clubhouse is the core component that manages agents and processes messages. It consists of:

- **Message Router**: Routes incoming messages to the appropriate handlers based on message type
- **Message Handlers**:
  - `CreateAgentHandler`: Handles agent creation commands
  - `DeleteAgentHandler`: Handles agent deletion commands
  - `ProcessMessageHandler`: Handles message processing commands
- **Agent Manager**: Manages agent lifecycle (creation, retrieval, deletion)
- **Conversation Manager**: Manages conversation history and context
- **Event Publisher**: Publishes events to Kafka for CLI notification

### Kafka Messaging

Kafka serves as the message bus for the system, with the following topics:

- **Commands Topic**: Used by the CLI to send commands to the Clubhouse
- **Responses Topic**: Used by the Clubhouse to send responses back to the CLI
- **Events Topic**: Used by the Clubhouse to publish events (agent thinking, state changes, etc.)

## Message Flow

1. User enters a command in the CLI
2. CLI sends a command message to the Commands Topic
3. Clubhouse consumes the command message and routes it to the appropriate handler
4. Handler processes the command and generates a response
5. Clubhouse publishes the response to the Responses Topic
6. CLI consumes the response and displays it to the user
7. Clubhouse may also publish event messages to the Events Topic
8. CLI consumes events and provides real-time feedback to the user

## Message Types

The system uses standardized message types for communication:

### Commands
- `CreateAgentCommand`: Creates a new agent
- `DeleteAgentCommand`: Deletes an existing agent
- `ProcessMessageCommand`: Sends a message to an agent for processing

### Responses
- `AgentCreatedResponse`: Confirms successful agent creation
- `AgentDeletedResponse`: Confirms successful agent deletion
- `MessageProcessedResponse`: Contains the agent's response to a message

### Events
- `AgentThinkingEvent`: Indicates that an agent is processing a message
- `AgentErrorEvent`: Reports an error that occurred during processing
- `AgentStateChangedEvent`: Notifies of agent state changes (created, deleted, etc.)

## Running the System

### Prerequisites

- Python 3.8+
- Kafka 2.8.0+ with Zookeeper
- Optional: Schema Registry for message validation

### Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start Kafka and Zookeeper:
   ```
   # Using Docker Compose (if available)
   docker-compose up -d
   
   # Or manually start Kafka and Zookeeper services
   ```

### Running the Clubhouse

Start the Clubhouse with:

```
python scripts/run_clubhouse.py [options]
```

Options:
- `--bootstrap-servers`: Kafka bootstrap servers (default: localhost:9092)
- `--commands-topic`: Topic for receiving commands (default: clubhouse-commands)
- `--responses-topic`: Topic for sending responses (default: clubhouse-responses)
- `--events-topic`: Topic for sending/receiving events (default: clubhouse-events)
- `--group-id`: Consumer group ID (default: randomly generated)
- `--schema-registry-url`: Schema Registry URL (optional)
- `--debug`: Enable debug logging

### Running the CLI

Start the CLI with:

```
python scripts/kafka_cli/cli.py [options]
```

Options:
- `--bootstrap-servers`: Kafka bootstrap servers (default: localhost:9092)
- `--debug`: Enable debug logging

## CLI Commands

Once the CLI is running, you can use the following commands:

- `/help`: Show available commands
- `/create <agent_id> [personality_type]`: Create a new agent
- `/delete <agent_id>`: Delete an agent
- `/list`: List available agents
- `/start [agent_id]`: Start a conversation with an agent
- `/switch <agent_id>`: Switch to a different agent
- `/end`: End the current conversation
- `/exit` or `/quit`: Exit the CLI

## Development and Testing

### Running Tests

Run the test suite with:

```
pytest
```

For specific test modules:

```
pytest tests/clubhouse/messaging/test_message_router.py
pytest tests/clubhouse/services/test_agent_manager.py
# etc.
```

For integration tests:

```
pytest tests/clubhouse/integration/
```

### Development Guidelines

- Follow test-driven development practices
- Write tests before implementing features
- Ensure proper error handling and validation
- Follow clean code principles and SOLID design
- Add comprehensive type annotations

## Architecture Design Decisions

### Message Routing

The system uses a message router pattern to dispatch messages to the appropriate handlers. This approach:

- Keeps the codebase modular and maintainable
- Makes it easy to add new message types and handlers
- Follows the Single Responsibility Principle

### Service Registry

A service registry pattern is used for dependency management, which:

- Simplifies dependency injection
- Makes testing easier with mock services
- Reduces coupling between components

### Event-Driven Architecture

The event-driven architecture enables:

- Asynchronous communication between components
- Real-time feedback to users
- Loose coupling between the CLI and the Clubhouse

## Future Improvements

- Add authentication and authorization
- Implement message encryption for security
- Support for binary message formats for efficiency
- Expand test coverage with more integration tests
- Add support for WebSocket for web clients
- Implement persistent storage for conversations and agents
