"""
Main entry point for the application.
"""

import os
import sys
import json
import uuid
import logging
import signal
import time
from typing import Dict, Any, Optional, List, Callable, cast

from project_name.core.service_registry import ServiceRegistry
from project_name.services.confluent_kafka_service import (
    KafkaConfig, 
    KafkaMessage, 
    ConfluentKafkaService
)
from project_name.services.kafka_protocol import MessageHandlerProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


class MessageHandler(MessageHandlerProtocol[Dict[str, Any], str]):
    """Message handler for Kafka messages."""
    
    def handle(self, value: Dict[str, Any], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Handle a message from Kafka.
        
        Args:
            value: The message value
            key: The message key
            headers: The message headers
        """
        logger.info(f"Received message with key: {key}")
        logger.info(f"Message value: {value}")
        if headers:
            logger.info(f"Message headers: {headers}")


def signal_handler(sig, frame):
    """Signal handler for graceful shutdown."""
    global running
    logger.info("Shutting down...")
    running = False


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


def main():
    """Main entry point for the application."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get configuration from environment variables
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    schema_registry_url = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    
    # Create service registry
    registry = ServiceRegistry()
    
    # Configure Kafka
    kafka_config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="project_name-client",
        group_id="project_name-group",
        auto_offset_reset="earliest",
        schema_registry_url=schema_registry_url
    )
    
    # Create and register Kafka service
    kafka_service = ConfluentKafkaService(kafka_config)
    registry.register("kafka_service", kafka_service)
    
    # Example topic
    topic = "example-topic"
    
    # Load the schema for Avro serialization
    try:
        schema_path = os.path.join(
            os.path.dirname(__file__),
            "schemas",
            "message.avsc"
        )
        schema = load_schema(schema_path)
        
        # Set the Avro serializer and deserializer
        kafka_service.set_avro_serializer(schema)
        kafka_service.set_avro_deserializer(schema)
        
        logger.info("Using Avro serialization with Schema Registry")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load Avro schema, falling back to JSON serialization: {e}")
    
    try:
        # Example: Produce a message
        logger.info(f"Producing a message to topic: {topic}")
        message = KafkaMessage(
            topic=topic,
            value={
                "id": str(uuid.uuid4()),
                "content": "Hello, Kafka!",
                "timestamp": int(time.time() * 1000),
                "metadata": {"source": "project_name", "version": "0.1.0"}
            },
            key="example-key"
        )
        
        kafka_service.produce_message(message)
        logger.info("Message produced successfully")
        
        # Example: Consume messages
        logger.info(f"Starting to consume from topic: {topic}")
        
        # Create a message handler
        handler = MessageHandler()
        
        # Start consuming in a separate thread (or process in a real application)
        import threading
        consumer_thread = threading.Thread(
            target=kafka_service.consume_messages,
            args=([topic], handler),
            daemon=True
        )
        consumer_thread.start()
        
        # Keep the main thread running until a signal is received
        while running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        # Clean up
        logger.info("Cleaning up resources...")
        if registry.has_service("kafka_service"):
            kafka_service = cast(ConfluentKafkaService, registry.get("kafka_service"))
            kafka_service.stop_consuming()
        
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
