"""
Main entry point for the Clubhouse application.

This module provides the main functionality for the Clubhouse application,
setting up services, configuring Kafka connections, and handling messages.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from dotenv import load_dotenv

from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.messaging.event_publisher import EventPublisher
from clubhouse.messaging.handlers import (
    CreateAgentHandler,
    DeleteAgentHandler,
    ProcessMessageHandler,
)
from clubhouse.messaging.message_router import MessageRouter
from clubhouse.messaging.schema_registrator import SchemaRegistrator
from clubhouse.services.agent_manager import AgentManager
from clubhouse.services.conversation_manager import ConversationManager
from clubhouse.services.confluent_kafka_service import (
    ConfluentKafkaService,
    KafkaConfig,
    KafkaMessage,
)
from clubhouse.services.kafka_protocol import MessageHandlerProtocol
from clubhouse.services.schema_registry import ConfluentSchemaRegistry
from scripts.kafka_cli.message_schemas import (
    AgentErrorEvent,
    MessageType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Signal handler for graceful shutdown."""
    global running
    logger.info("Shutting down clubhouse...")
    running = False


class ClubhouseMessageHandler(MessageHandlerProtocol[Dict[str, Any], str]):
    """Message handler for Kafka messages."""

    def __init__(self, message_router: MessageRouter, event_publisher: EventPublisher) -> None:
        """
        Initialize the message handler.

        Args:
            message_router: Router for dispatching messages to handlers
            event_publisher: Publisher for sending event messages
        """
        self._message_router = message_router
        self._event_publisher = event_publisher

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
        try:
            logger.debug(f"Received message: {value.get('message_type', 'unknown type')}")
            
            # Route the message to the appropriate handler
            response = self._message_router.route_message(value)
            
            # If a response was generated, publish it
            if response:
                logger.debug(f"Publishing response: {response.get('message_type', 'unknown type')}")
                self._event_publisher.publish_event(response, topic="clubhouse-responses")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            
            # Create and publish an error event
            error_event = AgentErrorEvent(
                message_id=str(uuid.uuid4()),
                agent_id=value.get("agent_id", "unknown"),
                error_message=str(e),
                error_type=type(e).__name__
            ).model_dump()
            
            self._event_publisher.publish_event(error_event)


def process_message(
    message: Dict[str, Any], 
    producer: Any, 
    responses_topic: str, 
    events_topic: str
) -> None:
    """
    Process a message and produce a response.
    
    Args:
        message: The message to process
        producer: Kafka producer to use for responses
        responses_topic: Topic to publish responses to
        events_topic: Topic to publish events to
    """
    try:
        logger.debug(f"Starting process_message with message: {message}")
        
        # Set up service registry and services
        service_registry = ServiceRegistry()
        logger.debug("Created service registry")
        
        # Configure services
        configure_services(service_registry)
        logger.debug("Configured services")
        
        # Get the message router and event publisher
        message_router = service_registry.get(MessageRouter)
        logger.debug(f"Retrieved message router: {message_router}")
        
        event_publisher = service_registry.get(EventPublisher)
        logger.debug(f"Retrieved event publisher: {event_publisher}")
        
        # Configure the event publisher with the producer
        event_publisher._producer = producer
        event_publisher._responses_topic = responses_topic
        event_publisher._events_topic = events_topic
        logger.debug(f"Configured event publisher with producer and topics")
        
        # Log all registered services for debugging
        logger.debug(f"Registered services by name: {list(service_registry._services.keys())}")
        logger.debug(f"Registered services by protocol: {[p.__name__ for p in service_registry._protocol_services.keys()]}")
        
        # Log the existence of the agent manager
        agent_manager = service_registry.get("agent_manager")
        logger.debug(f"Agent manager: {agent_manager}")
        
        # Route the message
        logger.debug(f"Processing message: {message.get('message_type', 'unknown type')}")
        response = message_router.route_message(message)
        logger.debug(f"Got response: {response}")
        
        # Publish response if available
        if response:
            logger.debug(f"Publishing response: {response.get('message_type', 'unknown type')}")
            event_publisher.publish_event(response, topic=responses_topic)
            logger.debug("Response published")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        
        # Create error response
        error_response = {
            "message_type": "ErrorResponse",
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "in_response_to": message.get("message_id"),
            "payload": {
                "error": str(e),
                "error_type": type(e).__name__
            }
        }
        
        # Publish error response
        producer.produce(
            topic=responses_topic,
            key=message.get("message_id"),
            value=json.dumps(error_response).encode("utf-8")
        )
        producer.flush()


def configure_services(service_registry: ServiceRegistry) -> None:
    """
    Configure and register all required services.
    
    Args:
        service_registry: Service registry to register services with
    """
    # Create and register agent factory first
    from clubhouse.agents.factory import AgentFactory
    agent_factory = AgentFactory(service_registry)
    service_registry.register(AgentFactory, agent_factory)
    service_registry.register("agent_factory", agent_factory)
    
    # Create and register core services
    agent_manager = AgentManager(service_registry, agent_factory=agent_factory)
    service_registry.register(AgentManager, agent_manager)
    service_registry.register("agent_manager", agent_manager)  # Register with string name for backward compatibility
    
    conversation_manager = ConversationManager()
    service_registry.register(ConversationManager, conversation_manager)
    service_registry.register("conversation_manager", conversation_manager)  # Register with string name for backward compatibility
    
    # Create and register messaging services
    event_publisher = EventPublisher(service_registry)
    service_registry.register(EventPublisher, event_publisher)
    service_registry.register("event_publisher", event_publisher)  # Register with string name for backward compatibility
    
    # Create message router and register handlers
    message_router = MessageRouter(service_registry)
    service_registry.register(MessageRouter, message_router)
    service_registry.register("message_router", message_router)  # Register with string name for backward compatibility
    
    # Register handlers
    message_router.register_handler(CreateAgentHandler(service_registry))
    message_router.register_handler(DeleteAgentHandler(service_registry))
    message_router.register_handler(ProcessMessageHandler(service_registry))


def main(
    bootstrap_servers: Optional[str] = None,
    commands_topic: Optional[str] = None,
    responses_topic: Optional[str] = None,
    events_topic: Optional[str] = None,
    group_id: Optional[str] = None,
    schema_registry_url: Optional[str] = None,
    register_schemas_only: bool = False
) -> None:
    """
    Main entry point for the Clubhouse application.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        commands_topic: Topic for receiving commands
        responses_topic: Topic for sending responses
        events_topic: Topic for sending/receiving events
        group_id: Consumer group ID
        schema_registry_url: Schema Registry URL
        register_schemas_only: If True, only register schemas and exit
    """
    # Ensure clean shutdown on SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set default values from environment if not provided
    bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    commands_topic = commands_topic or os.environ.get("COMMANDS_TOPIC", "clubhouse-commands")
    responses_topic = responses_topic or os.environ.get("RESPONSES_TOPIC", "clubhouse-responses")
    events_topic = events_topic or os.environ.get("EVENTS_TOPIC", "clubhouse-events")
    group_id = group_id or os.environ.get("GROUP_ID", "clubhouse")
    schema_registry_url = schema_registry_url or os.environ.get("SCHEMA_REGISTRY_URL")
    
    # Log configuration
    logger.info("Starting Clubhouse with configuration:")
    logger.info(f"  Bootstrap servers: {bootstrap_servers or 'Not configured'}")
    logger.info(f"  Commands topic: {commands_topic}")
    logger.info(f"  Responses topic: {responses_topic}")
    logger.info(f"  Events topic: {events_topic}")
    logger.info(f"  Consumer group ID: {group_id}")
    logger.info(f"  Schema registry URL: {schema_registry_url or 'Not configured'}")
    
    # Register schemas if schema registry URL is provided
    if schema_registry_url:
        schema_registry = ConfluentSchemaRegistry(schema_registry_url)
        registrator = SchemaRegistrator(schema_registry, topic_prefix=commands_topic.split("-")[0])
        
        logger.info("Registering schemas with Schema Registry...")
        num_registered = registrator.register_all_schemas()
        logger.info(f"Registered {num_registered} schemas")
        
        # Exit if only registering schemas
        if register_schemas_only:
            logger.info("Schema registration complete. Exiting.")
            return
    
    # Load environment variables
    load_dotenv()
    
    # Create Kafka configuration
    kafka_config = KafkaConfig(
        bootstrap_servers=bootstrap_servers,
        client_id="clubhouse-client",
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        schema_registry_url=schema_registry_url,
    )

    # Create service registry
    service_registry = ServiceRegistry()

    # Configure services
    configure_services(service_registry)

    # Create and register Kafka service
    kafka_service = ConfluentKafkaService(kafka_config)
    service_registry.register("kafka", kafka_service)

    # Get event publisher and set responses topic
    event_publisher = service_registry.get(EventPublisher)
    event_publisher._responses_topic = responses_topic
    event_publisher._events_topic = events_topic

    # Create message handler
    message_router = service_registry.get(MessageRouter)
    message_handler = ClubhouseMessageHandler(message_router, event_publisher)

    try:
        # Start consuming messages
        logger.info(f"Starting to consume messages from topic: {commands_topic}")
        kafka_service.consume_messages([commands_topic], message_handler)

        # Keep the application running until signaled to stop
        while running:
            time.sleep(1)

    except Exception as e:
        logger.error(f"Error in clubhouse: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("Clubhouse shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clubhouse Application")
    parser.add_argument("--bootstrap-servers", help="Kafka bootstrap servers")
    parser.add_argument("--commands-topic", help="Topic for receiving commands")
    parser.add_argument("--responses-topic", help="Topic for sending responses")
    parser.add_argument("--events-topic", help="Topic for sending/receiving events")
    parser.add_argument("--group-id", help="Consumer group ID")
    parser.add_argument("--schema-registry-url", help="Schema Registry URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--register-schemas-only", action="store_true", 
                        help="Only register schemas and exit")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run main function
    main(
        bootstrap_servers=args.bootstrap_servers,
        commands_topic=args.commands_topic,
        responses_topic=args.responses_topic,
        events_topic=args.events_topic,
        group_id=args.group_id,
        schema_registry_url=args.schema_registry_url,
        register_schemas_only=args.register_schemas_only
    )
