"""
Example of using the Confluent Kafka service with Schema Registry.

This module demonstrates how to use the Confluent Kafka service with
Schema Registry and Avro serialization for producing and consuming messages.
"""

import os
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from project_name.services.confluent_kafka_service import (
    KafkaConfig,
    KafkaMessage,
    ConfluentKafkaService
)
from project_name.services.kafka_protocol import MessageHandlerProtocol


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class ExampleMessageHandler(MessageHandlerProtocol[Dict[str, Any], str]):
    """Example message handler implementation."""
    
    def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Handle a Kafka message.
        
        Args:
            value: Message value
            key: Optional message key
            headers: Optional message headers
        """
        logger.info(f"Received message with key: {key}")
        logger.info(f"Message value: {value}")
        if headers:
            logger.info(f"Message headers: {headers}")


def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load an Avro schema from a file.
    
    Args:
        schema_path: Path to the schema file
        
    Returns:
        Schema definition as a dictionary
    """
    with open(schema_path, "r") as f:
        return json.load(f)


def example_json_producer() -> None:
    """Example of producing JSON messages to Kafka."""
    # Get configuration from environment variables
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    # Configure Kafka
    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="example-producer"
    )
    
    # Create Kafka service
    kafka_service = ConfluentKafkaService(config)
    
    # Example topic
    topic = "example-json-topic"
    
    # Produce a few messages
    for i in range(5):
        # Create a message
        message = KafkaMessage(
            topic=topic,
            value={
                "id": str(uuid.uuid4()),
                "message": f"Hello, Kafka! Message {i}",
                "timestamp": int(time.time() * 1000)
            },
            key=f"key-{i}",
            headers={"source": "example-producer", "index": str(i)}
        )
        
        # Produce the message
        kafka_service.produce_message(message)
        logger.info(f"Produced message {i} to topic {topic}")
        
        # Wait a bit between messages
        time.sleep(1)


def example_json_consumer() -> None:
    """Example of consuming JSON messages from Kafka."""
    # Get configuration from environment variables
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    # Configure Kafka
    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="example-consumer",
        group_id="example-group",
        auto_offset_reset="earliest"
    )
    
    # Create Kafka service
    kafka_service = ConfluentKafkaService(config)
    
    # Example topic
    topic = "example-json-topic"
    
    # Create a message handler
    handler = ExampleMessageHandler()
    
    # Consume messages
    logger.info(f"Starting to consume from topic: {topic}")
    
    try:
        # Consume for 30 seconds then exit
        kafka_service.consume_messages([topic], handler)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        kafka_service.stop_consuming()


def example_avro_producer() -> None:
    """Example of producing Avro messages to Kafka with Schema Registry."""
    # Get configuration from environment variables
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    schema_registry_url = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    
    # Configure Kafka
    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="example-avro-producer",
        schema_registry_url=schema_registry_url
    )
    
    # Create Kafka service
    kafka_service = ConfluentKafkaService(config)
    
    # Load the schema
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "schemas",
        "message.avsc"
    )
    schema = load_schema(schema_path)
    
    # Set the Avro serializer
    kafka_service.set_avro_serializer(schema)
    
    # Example topic
    topic = "example-avro-topic"
    
    # Produce a few messages
    for i in range(5):
        # Create a message conforming to the schema
        message = KafkaMessage(
            topic=topic,
            value={
                "id": str(uuid.uuid4()),
                "content": f"Hello, Avro! Message {i}",
                "timestamp": int(time.time() * 1000),
                "metadata": {"index": str(i), "source": "example-avro-producer"}
            },
            key=f"key-{i}"
        )
        
        # Produce the message
        kafka_service.produce_message(message)
        logger.info(f"Produced Avro message {i} to topic {topic}")
        
        # Wait a bit between messages
        time.sleep(1)


def example_avro_consumer() -> None:
    """Example of consuming Avro messages from Kafka with Schema Registry."""
    # Get configuration from environment variables
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    schema_registry_url = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    
    # Configure Kafka
    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="example-avro-consumer",
        group_id="example-avro-group",
        auto_offset_reset="earliest",
        schema_registry_url=schema_registry_url
    )
    
    # Create Kafka service
    kafka_service = ConfluentKafkaService(config)
    
    # Load the schema
    schema_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "schemas",
        "message.avsc"
    )
    schema = load_schema(schema_path)
    
    # Set the Avro deserializer
    kafka_service.set_avro_deserializer(schema)
    
    # Example topic
    topic = "example-avro-topic"
    
    # Create a message handler
    handler = ExampleMessageHandler()
    
    # Consume messages
    logger.info(f"Starting to consume Avro messages from topic: {topic}")
    
    try:
        # Consume for 30 seconds then exit
        kafka_service.consume_messages([topic], handler)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Clean up
        kafka_service.stop_consuming()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m project_name.examples.kafka_example [json_producer|json_consumer|avro_producer|avro_consumer]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "json_producer":
        example_json_producer()
    elif command == "json_consumer":
        example_json_consumer()
    elif command == "avro_producer":
        example_avro_producer()
    elif command == "avro_consumer":
        example_avro_consumer()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m project_name.examples.kafka_example [json_producer|json_consumer|avro_producer|avro_consumer]")
        sys.exit(1)
