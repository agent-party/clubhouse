# Python Project with Kafka Integration

A Python project template with test-driven development setup, Docker integration, and Kafka in KRaft mode with Confluent Schema Registry and Avro serialization.

## Features

- Test-driven development with pytest and 100% test coverage
- Kafka integration using confluent-kafka with Schema Registry
- Avro serialization for Kafka messages
- Docker and Docker Compose setup with Kafka in KRaft mode
- Protocol-based service interfaces for better testability
- Service Registry pattern for dependency management
- Comprehensive type annotations and validation with Pydantic
- Code formatting with Black and isort
- Type checking with mypy

## Project Structure

```
project_name/
├── src/                 # Source code
│   └── project_name/    # Main package
│       ├── core/        # Core functionality
│       ├── services/    # Service implementations
│       ├── schemas/     # Avro schemas
│       └── models/      # Data models
├── tests/               # Tests
│   ├── conftest.py      # Pytest configuration
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── docker/              # Docker configuration
│   ├── Dockerfile       # Dockerfile for the application
│   └── entrypoint.sh    # Docker entrypoint script
├── docker-compose.yml   # Docker Compose configuration
└── pyproject.toml       # Project configuration
```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Make (optional)

### Local Development

1. Clone the repository:

```bash
git clone https://github.com/username/project_name.git
cd project_name
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

4. Run the tests:

```bash
pytest
```

### Docker Development

1. Build and start the containers:

```bash
docker-compose up -d
```

2. Access Kafdrop (Kafka UI) at http://localhost:9000

3. Access Schema Registry REST API at http://localhost:8081

4. Stop the containers:

```bash
docker-compose down
```

## Testing

This project follows test-driven development practices. Tests are located in the `tests` directory and are separated into unit and integration tests.

- Run all tests: `pytest`
- Run with coverage: `pytest --cov=project_name`
- Run only unit tests: `pytest tests/unit`
- Run only integration tests: `pytest tests/integration`

## Kafka Integration

The project includes Kafka in KRaft mode, which eliminates the need for ZooKeeper. The Docker Compose setup includes:

- Kafka broker in KRaft mode
- Schema Registry for Avro schema management
- Kafdrop for Kafka monitoring

### Using Confluent Kafka with Schema Registry

The project uses the Confluent Kafka Python client with Schema Registry integration, providing strong typing and validation for Kafka messages.

#### Configuration

Configure Kafka and Schema Registry using environment variables:

```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export SCHEMA_REGISTRY_URL=http://localhost:8081
```

When using Docker, these are automatically configured in the docker-compose.yml file.

#### Avro Schemas

Avro schemas are stored in the `src/project_name/schemas` directory. To create a new schema:

1. Create a new `.avsc` file in the schemas directory
2. Define your schema according to Avro schema specifications
3. Use the schema in your producer/consumer code

Example schema (`message.avsc`):

```json
{
  "type": "record",
  "name": "Message",
  "namespace": "project_name",
  "fields": [
    {"name": "id", "type": "string"},
    {"name": "content", "type": "string"},
    {"name": "timestamp", "type": "long"},
    {
      "name": "metadata",
      "type": {
        "type": "map",
        "values": "string"
      }
    }
  ]
}
```

#### Example Usage

See the examples directory for sample code showing how to use the Kafka integration:

```bash
# Produce JSON messages
python -m project_name.examples.kafka_example json_producer

# Consume JSON messages
python -m project_name.examples.kafka_example json_consumer

# Produce Avro messages with Schema Registry
python -m project_name.examples.kafka_example avro_producer

# Consume Avro messages with Schema Registry
python -m project_name.examples.kafka_example avro_consumer
```

## License

MIT
