# pragma: no cover
"""
Main entry point for the application.
"""

# pragma: no cover
import json

# pragma: no cover
import logging

# pragma: no cover
import os

# pragma: no cover
import signal

# pragma: no cover
import sys

# pragma: no cover
import time

# pragma: no cover
import uuid

# pragma: no cover
from typing import Any, Callable, Dict, List, Optional, cast

# pragma: no cover
from clubhouse.core.service_registry import ServiceRegistry

# pragma: no cover
from clubhouse.services.confluent_kafka_service import (
    ConfluentKafkaService,
    KafkaConfig,
    KafkaMessage,
)

# pragma: no cover
from clubhouse.services.kafka_protocol import MessageHandlerProtocol

# pragma: no cover
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# pragma: no cover - Main script execution code
logger = logging.getLogger(__name__)

# pragma: no cover
# Global flag for graceful shutdown
running = True


# pragma: no cover
class MessageHandler(MessageHandlerProtocol[Dict[str, Any], str]):
    """Message handler for Kafka messages."""

    def handle(
        self,
        value: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Handle a message.

        Args:
            value: Message value
            key: Optional message key
            headers: Optional message headers
        """
        logger.info(f"Received message: {value}")
        if key is not None:
            logger.info(f"Message key: {key}")
        if headers is not None:
            logger.info(f"Message headers: {headers}")


# pragma: no cover
def signal_handler(sig, frame):
    """Signal handler for graceful shutdown."""
    global running
    logger.info("Shutting down...")
    running = False


# pragma: no cover
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


# pragma: no cover
def main():
    """Main entry point for the application."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration from environment variables
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    schema_registry_url = os.environ.get("SCHEMA_REGISTRY_URL")
    consumer_group_id = os.environ.get("CONSUMER_GROUP_ID", f"consumer-{uuid.uuid4()}")

    # Create Kafka configuration
    config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="kafka-demo-client",
        group_id=consumer_group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        schema_registry_url=schema_registry_url,
    )

    # Create service registry and register services
    registry = ServiceRegistry()

    # Create Kafka service
    kafka_service = ConfluentKafkaService(config)
    registry.register("kafka", kafka_service)

    # Check command-line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python -m mcp_demo [producer|consumer]")
        sys.exit(1)

    mode = sys.argv[1]
    topic = os.environ.get("KAFKA_TOPIC", "demo-topic")

    if mode == "producer":
        # Producer mode
        logger.info(f"Starting in producer mode, sending to topic {topic}")

        try:
            # Produce messages
            for i in range(10):
                if not running:
                    break

                # Create a message
                message = KafkaMessage(
                    topic=topic,
                    value={
                        "id": str(uuid.uuid4()),
                        "message": f"Message {i}",
                        "timestamp": int(time.time()),
                    },
                    key=f"key-{i}",
                    headers={"source": "mcp_demo"},
                )

                # Produce the message
                logger.info(f"Producing message: {message.value}")
                kafka_service.produce_message(message)

                # Wait a bit
                time.sleep(1)

            logger.info("Producer completed")

        except Exception as e:
            logger.error(f"Error in producer: {e}")
            sys.exit(1)

    elif mode == "consumer":
        # Consumer mode
        logger.info(f"Starting in consumer mode, listening to topic {topic}")

        try:
            # Create a message handler
            handler = MessageHandler()

            # Consume messages
            logger.info(f"Consuming messages from {topic}")
            kafka_service.consume_messages([topic], handler)

        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            sys.exit(1)

        # Clean shutdown
        if not running:
            logger.info("Application shutdown complete")

    else:
        logger.error(f"Unknown mode: {mode}")
        logger.error("Usage: python -m mcp_demo [producer|consumer]")
        sys.exit(1)


# pragma: no cover
if __name__ == "__main__":
    main()
