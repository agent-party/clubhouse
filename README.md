# Clubhouse - Confluent Kafka Integration

This package provides a robust integration with Confluent Kafka, offering utilities for producing and consuming Kafka messages with support for JSON and Avro serialization using Schema Registry.

## Features

- Confluent Kafka integration with a clean, modular architecture
- Schema Registry integration with Avro serialization support
- Type-safe interfaces with Protocol definitions
- Comprehensive test suite following Test-Driven Development principles
- Docker-based local development environment

## Installation

### Development Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package in development mode with development dependencies
pip install -e ".[dev]"
```

## Usage

### Running the Kafka Demo

To run the Kafka demo with Docker:

```bash
# Start the Kafka environment
./run_kafka_demo.sh

# To stop the environment
./run_kafka_demo.sh stop
```

### Running Examples

```bash
# JSON serialization example
python -m mcp_demo.examples.kafka_example json_producer
python -m mcp_demo.examples.kafka_example json_consumer

# Avro serialization example with Schema Registry
python -m mcp_demo.examples.kafka_example avro_producer
python -m mcp_demo.examples.kafka_example avro_consumer
```

### Running the main application

```bash
python -m mcp_demo
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mcp_demo tests/
```

## Project Structure

```
mcp_demo/
├── core/            # Core components and utilities
├── services/        # Service implementations
├── schemas/         # Avro schema definitions
├── models/          # Data models and business logic
├── examples/        # Example scripts
├── __main__.py      # Application entry point
├── tests/           # Test suite
    ├── unit/        # Unit tests
    ├── integration/ # Integration tests
    ├── conftest.py  # Test fixtures and configuration
```

## Configuration

The Clubhouse uses a modular, type-safe configuration system based on Pydantic models. You can configure the application using environment variables:

| Environment Variable | Default Value | Description |
|----------------------|---------------|-------------|
| `MCP_HOST` | 127.0.0.1 | Host to bind the MCP server to |
| `MCP_PORT` | 8000 | Port to bind the MCP server to |
| `MCP_LOG_LEVEL` | info | Logging level (debug, info, warning, error, critical) |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Comma-separated list of Kafka broker addresses |
| `KAFKA_TOPIC_PREFIX` | "" | Prefix to add to all topics |
| `SCHEMA_REGISTRY_URL` | http://localhost:8081 | URL of the Schema Registry server |

For more details, see the [Configuration System Documentation](docs/configuration_system.md).

## License

MIT
