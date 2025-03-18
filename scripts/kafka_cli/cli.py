"""
Minimal command-line interface for interacting with agents through Kafka.

This module provides a thin command-line interface for sending commands to and
receiving responses from the Clubhouse components via Kafka messaging with Avro serialization.
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_origin, get_args

# Import third-party modules
try:
    from aioconsole import ainput
    AIOCONSOLE_AVAILABLE = True
except ImportError:
    AIOCONSOLE_AVAILABLE = False
    
try:
    from confluent_kafka import KafkaException
    from confluent_kafka.schema_registry import SchemaRegistryClient
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Import local modules
from scripts.kafka_cli.kafka_client import KafkaClient
from scripts.kafka_cli.formatter import CLIFormatter

# Import Pydantic for model definitions
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('agent_cli')

# Base message classes
class Message(BaseModel):
    """Base class for all messages."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: str
    
class Command(Message):
    """Base class for command messages."""
    type: str = "command"
    command_type: str
    
class Response(Message):
    """Base class for response messages."""
    type: str = "response"
    response_type: str
    command_id: str
    success: bool = True
    error_message: Optional[str] = None
    
class Event(Message):
    """Base class for event messages."""
    type: str = "event"
    event_type: str

# Command messages
class CreateAgentCommand(Command):
    """Command to create a new agent."""
    command_type: str = "create_agent"
    agent_id: str
    agent_type: str
    agent_name: str
    agent_description: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    
class DeleteAgentCommand(Command):
    """Command to delete an agent."""
    command_type: str = "delete_agent"
    agent_id: str
    
class ProcessMessageCommand(Command):
    """Command to process a message by an agent."""
    command_type: str = "process_message"
    agent_id: str
    message_content: str
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Response messages
class AgentCreatedResponse(Response):
    """Response to a create agent command."""
    response_type: str = "agent_created"
    agent_id: str
    
class AgentDeletedResponse(Response):
    """Response to a delete agent command."""
    response_type: str = "agent_deleted"
    agent_id: str
    
class MessageProcessedResponse(Response):
    """Response to a process message command."""
    response_type: str = "message_processed"
    agent_id: str
    response_content: str
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Event messages
class AgentThinkingEvent(Event):
    """Event emitted when an agent is thinking."""
    event_type: str = "agent_thinking"
    agent_id: str
    thinking_content: str
    conversation_id: Optional[str] = None
    
class AgentErrorEvent(Event):
    """Event emitted when an agent encounters an error."""
    event_type: str = "agent_error"
    agent_id: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    
class AgentStateChangedEvent(Event):
    """Event emitted when an agent's state changes."""
    event_type: str = "agent_state_changed"
    agent_id: str
    previous_state: str
    new_state: str
    
class AgentCLI:
    """CLI for interacting with agents via Kafka."""
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        schema_registry_url: str = None,
        topic_prefix: str = "clubhouse",
        use_avro: bool = True,
        client_id: str = None
    ):
        """Initialize the CLI.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            schema_registry_url: Schema Registry URL
            topic_prefix: Topic prefix for all messages
            use_avro: Whether to use Avro serialization
            client_id: Client ID for Kafka
        """
        # Set default environment variables if not provided
        bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        schema_registry_url = schema_registry_url or os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
        use_avro_str = os.environ.get("USE_AVRO", "true").lower()
        use_avro = use_avro and (use_avro_str == "true")
        
        # Store parameters
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        self.topic_prefix = topic_prefix
        self.use_avro = use_avro
        self.client_id = client_id or f"agent-cli-{os.getpid()}"
        
        # Set up formatter
        self.formatter = CLIFormatter()
        
        # Initialize Kafka client
        self.kafka_client = KafkaClient(
            bootstrap_servers=self.bootstrap_servers,
            schema_registry_url=self.schema_registry_url if use_avro else None,
            topic_prefix=self.topic_prefix,
            client_id=self.client_id,
            use_avro=self.use_avro
        )
        
        # Set up command handlers
        self.command_handlers = {
            "/help": self._handle_help,
            "/create": self._handle_create_agent,
            "/delete": self._handle_delete_agent,
            "/list": self._handle_list_agents,
            "/exit": self._handle_exit,
            "/quit": self._handle_exit
        }
        
        # State
        self.running = False
        self.current_agent_id = None
        self.command_history = []
        self.history_limit = 100
        
    async def start(self):
        """Start the CLI."""
        # Check dependencies
        if not AIOCONSOLE_AVAILABLE:
            self.formatter.print_error("aioconsole not available. Install it with `pip install aioconsole`")
            return
            
        if not KAFKA_AVAILABLE:
            self.formatter.print_error("confluent-kafka not available. Install it with `pip install confluent-kafka`")
            return
            
        try:
            # Connect to Kafka
            self.formatter.print_info(f"Connecting to Kafka at {self.bootstrap_servers}...")
            await self.kafka_client.connect()
            self.formatter.print_success("Connected to Kafka")
            
            # Register message handler
            self.kafka_client.set_message_handler(self._handle_message)
            
            # Make sure topics exist
            await self._ensure_topics_exist()
            
            # Enter main loop
            self.running = True
            
            # Print help
            self.formatter.print_info("Type /help for available commands")
            
            while self.running:
                try:
                    # Get user input
                    prompt = f"{self.current_agent_id}> " if self.current_agent_id else "> "
                    user_input = await ainput(prompt)
                    
                    # Skip empty lines
                    if not user_input.strip():
                        continue
                        
                    # Process commands
                    if user_input.startswith("/"):
                        await self._process_command(user_input)
                    else:
                        # Process as a message to the current agent
                        await self._process_message(user_input)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.formatter.print_error(f"Error: {str(e)}")
                    logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Error starting CLI: {str(e)}", exc_info=True)
            self.running = False
            
    async def _ensure_topics_exist(self) -> None:
        """Ensure that required Kafka topics exist, creating them if needed."""
        # Define topics
        command_topic = f"{self.topic_prefix}-commands"
        response_topic = f"{self.topic_prefix}-responses"
        event_topic = f"{self.topic_prefix}-events"
        
        # Check if topics exist
        topics_exist = await self.kafka_client.check_topics_exist([
            command_topic, response_topic, event_topic
        ])
        
        # Create missing topics if needed
        if not topics_exist:
            logger.info(f"Creating topics: {command_topic}, {response_topic}, {event_topic}")
            self.formatter.print_info("Creating required Kafka topics...")
            
            try:
                await self.kafka_client.create_topics([
                    (command_topic, 1, 3),
                    (response_topic, 1, 3),
                    (event_topic, 1, 3)
                ])
                self.formatter.print_success("Created required Kafka topics")
            except Exception as e:
                logger.error(f"Error creating topics: {str(e)}")
                self.formatter.print_error(f"Error creating topics: {str(e)}")
                self.formatter.print_warning("Continuing with existing topics")
        
        # Subscribe to topics
        try:
            await self.kafka_client.subscribe([response_topic, event_topic])
        except Exception as e:
            logger.error(f"Error subscribing to topics: {str(e)}")
            self.formatter.print_error(f"Error subscribing to topics: {str(e)}")
            
    def _handle_message(self, topic: str, key: str, value: Any) -> None:
        """Handle incoming Kafka messages.
        
        Args:
            topic: Kafka topic
            key: Message key
            value: Message value
        """
        try:
            # Log message receipt
            logger.debug(f"Received message on {topic}: {value}")
            
            # Parse message type
            message_type = value.get("type", "unknown") if isinstance(value, dict) else "unknown"
            
            # Handle based on topic and message type
            if topic == f"{self.topic_prefix}-responses":
                self._handle_response(value)
            elif topic == f"{self.topic_prefix}-events":
                self._handle_event(value)
            else:
                logger.warning(f"Received message on unexpected topic: {topic}")
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}", exc_info=True)
            self.formatter.print_error(f"Error handling message: {str(e)}")
            
    def _handle_response(self, response: Dict[str, Any]) -> None:
        """Handle a response message.
        
        Args:
            response: Response message
        """
        response_type = response.get("response_type", "unknown")
        
        # Print response based on type
        if response_type == "agent_created":
            agent_id = response.get("agent_id", "unknown")
            self.formatter.print_success(f"Agent {agent_id} created successfully")
            
            # Set current agent if not set
            if not self.current_agent_id:
                self.current_agent_id = agent_id
                self.formatter.print_info(f"Current agent set to {agent_id}")
                
        elif response_type == "agent_deleted":
            agent_id = response.get("agent_id", "unknown")
            self.formatter.print_success(f"Agent {agent_id} deleted successfully")
            
            # Clear current agent if it was deleted
            if self.current_agent_id == agent_id:
                self.current_agent_id = None
                self.formatter.print_info("Current agent cleared")
                
        elif response_type == "message_processed":
            agent_id = response.get("agent_id", "unknown")
            content = response.get("response_content", "")
            conversation_id = response.get("conversation_id", "unknown")
            
            # Print agent's response
            self.formatter.print_message(agent_id, content, is_response=True)
            
        else:
            # Unknown response type
            logger.warning(f"Received unknown response type: {response_type}")
            self.formatter.print_json(response)
            
    def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle an event message.
        
        Args:
            event: Event message
        """
        event_type = event.get("event_type", "unknown")
        
        # Print event based on type
        if event_type == "agent_thinking":
            agent_id = event.get("agent_id", "unknown")
            content = event.get("thinking_content", "")
            
            # Print agent's thinking
            self.formatter.print_thinking(agent_id, content)
            
        elif event_type == "agent_error":
            agent_id = event.get("agent_id", "unknown")
            error_message = event.get("error_message", "Unknown error")
            
            # Print error
            self.formatter.print_error(f"Agent {agent_id} error: {error_message}")
            
        elif event_type == "agent_state_changed":
            agent_id = event.get("agent_id", "unknown")
            previous_state = event.get("previous_state", "unknown")
            new_state = event.get("new_state", "unknown")
            
            # Print state change
            self.formatter.print_info(f"Agent {agent_id} state changed: {previous_state} -> {new_state}")
            
        else:
            # Unknown event type
            logger.warning(f"Received unknown event type: {event_type}")
            self.formatter.print_json(event)
            
    async def _process_command(self, command: str) -> None:
        """Process a CLI command.
        
        Args:
            command: Command string
        """
        # Split command into parts
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        # Get handler for command
        handler = self.command_handlers.get(cmd)
        
        if handler:
            try:
                # Call handler
                await handler(args)
            except Exception as e:
                logger.error(f"Error processing command {cmd}: {str(e)}", exc_info=True)
                self.formatter.print_error(f"Error processing command: {str(e)}")
        else:
            self.formatter.print_error(f"Unknown command: {cmd}")
            await self._handle_help([])
            
    async def _process_message(self, message: str) -> None:
        """Process a message to the current agent.
        
        Args:
            message: Message string
        """
        if not self.current_agent_id:
            self.formatter.print_error("No agent selected. Create or select an agent first.")
            return
            
        try:
            # Create message command
            command = ProcessMessageCommand(
                agent_id=self.current_agent_id,
                message_content=message,
            )
            
            # Send command
            self.formatter.print_message("You", message, is_response=False)
            await self.kafka_client.send_message(
                f"{self.topic_prefix}-commands",
                command.id,
                command.dict()
            )
            
            # Add to history
            self._add_to_history(command)
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}", exc_info=True)
            self.formatter.print_error(f"Error sending message: {str(e)}")
            
    def _add_to_history(self, command: Any) -> None:
        """Add a command to the history.
        
        Args:
            command: Command to add
        """
        self.command_history.append(command)
        
        # Trim history if too long
        if len(self.command_history) > self.history_limit:
            self.command_history = self.command_history[-self.history_limit:]
            
    async def _handle_help(self, args: List[str]) -> None:
        """Handle the /help command.
        
        Args:
            args: Command arguments
        """
        self.formatter.print_info("Available commands:")
        self.formatter.print_info("  /help                  - Show this help message")
        self.formatter.print_info("  /create <type> <name>  - Create a new agent")
        self.formatter.print_info("  /delete <agent_id>     - Delete an agent")
        self.formatter.print_info("  /list                  - List available agents")
        self.formatter.print_info("  /exit, /quit           - Exit the CLI")
        self.formatter.print_info("")
        self.formatter.print_info("To send a message to the current agent, simply type the message and press Enter.")
        
    async def _handle_create_agent(self, args: List[str]) -> None:
        """Handle the /create command.
        
        Args:
            args: Command arguments
        """
        if len(args) < 2:
            self.formatter.print_error("Usage: /create <type> <name>")
            return
            
        agent_type = args[0]
        agent_name = args[1]
        agent_id = f"{agent_type}-{agent_name.lower()}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        try:
            # Create agent command
            command = CreateAgentCommand(
                agent_id=agent_id,
                agent_type=agent_type,
                agent_name=agent_name,
            )
            
            # Send command
            self.formatter.print_info(f"Creating agent {agent_id}...")
            await self.kafka_client.send_message(
                f"{self.topic_prefix}-commands",
                command.id,
                command.dict()
            )
            
            # Add to history
            self._add_to_history(command)
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}", exc_info=True)
            self.formatter.print_error(f"Error creating agent: {str(e)}")
            
    async def _handle_delete_agent(self, args: List[str]) -> None:
        """Handle the /delete command.
        
        Args:
            args: Command arguments
        """
        if len(args) < 1:
            self.formatter.print_error("Usage: /delete <agent_id>")
            return
            
        agent_id = args[0]
        
        try:
            # Create delete command
            command = DeleteAgentCommand(
                agent_id=agent_id,
            )
            
            # Send command
            self.formatter.print_info(f"Deleting agent {agent_id}...")
            await self.kafka_client.send_message(
                f"{self.topic_prefix}-commands",
                command.id,
                command.dict()
            )
            
            # Add to history
            self._add_to_history(command)
            
        except Exception as e:
            logger.error(f"Error deleting agent: {str(e)}", exc_info=True)
            self.formatter.print_error(f"Error deleting agent: {str(e)}")
            
    async def _handle_list_agents(self, args: List[str]) -> None:
        """Handle the /list command.
        
        Args:
            args: Command arguments
        """
        self.formatter.print_info("Listing agents is not implemented yet.")
        
    async def _handle_exit(self, args: List[str]) -> None:
        """Handle the /exit command.
        
        Args:
            args: Command arguments
        """
        self.formatter.print_info("Exiting...")
        self.running = False
        
    async def stop(self):
        """Stop the CLI."""
        if self.kafka_client:
            await self.kafka_client.disconnect()

# Utility for schema registration (separated from CLI)
class SchemaManager:
    """Utility for managing schemas with the Schema Registry."""
    
    def __init__(
        self,
        bootstrap_servers: str = None,
        schema_registry_url: str = None,
        topic_prefix: str = "clubhouse"
    ):
        """Initialize the schema manager.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            schema_registry_url: Schema Registry URL
            topic_prefix: Topic prefix for all messages
        """
        # Set default environment variables if not provided
        self.bootstrap_servers = bootstrap_servers or os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.schema_registry_url = schema_registry_url or os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")
        self.topic_prefix = topic_prefix
        
        # Set up formatter
        self.formatter = CLIFormatter()
        
    async def register_schemas(self):
        """Register all message schemas with the Schema Registry."""
        # Import necessary modules
        from scripts.kafka_cli.schema_utils import register_all_schemas, SchemaConverter
        
        self.formatter.print_info(f"Registering schemas with Schema Registry at {self.schema_registry_url}...")
        
        if not self.schema_registry_url:
            logger.warning("Schema Registry URL not set, skipping schema registration")
            return False
            
        # Create Schema Registry client
        registry_client = SchemaRegistryClient({"url": self.schema_registry_url})
        
        # First, register base classes
        base_classes = [Command, Response, Event]
        registered_base = 0
        
        self.formatter.print_info("Registering base message classes...")
        try:
            for base_class in base_classes:
                subject = f"{self.topic_prefix}-{base_class.__name__}-value"
                avro_schema = SchemaConverter.pydantic_to_avro(base_class)
                avro_schema_str = json.dumps(avro_schema)
                try:
                    schema_id = SchemaConverter.register_schema(registry_client, subject, avro_schema_str)
                    if schema_id:
                        logger.debug(f"Registered base schema for {subject} with ID {schema_id}")
                        registered_base += 1
                        self.formatter.print_success(f"Registered {base_class.__name__} schema with ID {schema_id}")
                except Exception as e:
                    logger.warning(f"Could not register base schema for {subject}: {e}")
                    self.formatter.print_warning(f"Could not register {base_class.__name__} schema: {e}")
        except Exception as e:
            logger.warning(f"Error registering base schemas: {e}")
            self.formatter.print_error(f"Error registering base schemas: {e}")
            return False
            
        # All message classes to register
        message_classes = [
            CreateAgentCommand, DeleteAgentCommand, ProcessMessageCommand,
            AgentCreatedResponse, AgentDeletedResponse, MessageProcessedResponse,
            AgentThinkingEvent, AgentErrorEvent, AgentStateChangedEvent
        ]
        
        # Register schemas
        self.formatter.print_info("Registering specific message classes...")
        try:
            num_registered = register_all_schemas(
                registry_client, 
                message_classes, 
                topic_prefix=self.topic_prefix,
                include_null=True
            )
                
            self.formatter.print_success(f"Registered {num_registered} message schemas")
            return num_registered > 0 or registered_base > 0
                
        except Exception as e:
            logger.error(f"Error registering schemas: {str(e)}")
            self.formatter.print_error(f"Failed to register schemas: {str(e)}")
            return False
        
async def main():
    """Main entry point."""
    import argparse
    import uuid  # Import here to avoid circular import
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Agent CLI')
    parser.add_argument('--bootstrap-servers', help='Kafka bootstrap servers')
    parser.add_argument('--schema-registry-url', help='Schema Registry URL')
    parser.add_argument('--topic-prefix', help='Topic prefix', default='clubhouse')
    parser.add_argument('--client-id', help='Client ID')
    parser.add_argument('--no-avro', help='Disable Avro serialization', action='store_true')
    parser.add_argument('--register-schemas', help='Register schemas and exit', action='store_true')
    
    args = parser.parse_args()
    
    # Check if we should just register schemas
    if args.register_schemas:
        schema_manager = SchemaManager(
            bootstrap_servers=args.bootstrap_servers,
            schema_registry_url=args.schema_registry_url,
            topic_prefix=args.topic_prefix
        )
        success = await schema_manager.register_schemas()
        sys.exit(0 if success else 1)
    
    # Create and start CLI
    cli = AgentCLI(
        bootstrap_servers=args.bootstrap_servers,
        schema_registry_url=args.schema_registry_url,
        topic_prefix=args.topic_prefix,
        use_avro=not args.no_avro,
        client_id=args.client_id
    )
    
    try:
        await cli.start()
    except KeyboardInterrupt:
        pass
    finally:
        await cli.stop()
        
if __name__ == "__main__":
    asyncio.run(main())
