"""
Agent communication interfaces and implementations.

This module provides standardized message formats, routing, and handlers
for agent-to-agent communication using Kafka as the message bus.
"""

import json
import uuid
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Protocol, runtime_checkable, Union
from datetime import datetime, timezone, UTC
from uuid import UUID, uuid4

from clubhouse.agents.agent_protocol import AgentProtocol, AgentMessage, AgentResponse
from clubhouse.services.kafka_protocol import KafkaProducerProtocol, KafkaConsumerProtocol, MessageHandlerProtocol as KafkaMessageHandlerProtocol
from typing import cast, List, Dict, Any, Type

# Configure logger
logger = logging.getLogger(__name__)


class MessagePriority(str, Enum):
    """Priority levels for agent messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    COMMAND = "command"           # Direct command to an agent
    QUERY = "query"               # Information request
    NOTIFICATION = "notification" # Informational update, no response needed
    RESPONSE = "response"         # Response to a previous message
    ERROR = "error"               # Error notification


class RoutingStrategy(str, Enum):
    """Strategies for routing messages to agents."""
    DIRECT = "direct"             # Send directly to a specific agent
    BROADCAST = "broadcast"       # Send to all agents
    CAPABILITY = "capability"     # Send to agents with a specific capability
    GROUP = "group"               # Send to a defined group of agents


class MessageStatus(str, Enum):
    """Status of a message in its lifecycle."""
    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@runtime_checkable
class MessageHandlerProtocol(Protocol):
    """Protocol defining the interface for message handlers."""
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle an incoming message and optionally produce a response.
        
        Args:
            message: The message to handle
            
        Returns:
            Optional response message
        """
        ...


class EnhancedAgentMessage(dict):
    """
    Enhanced agent message format with routing information.
    
    This extends the basic AgentMessage dictionary with additional fields
    for message routing, tracking, and management.
    """
    
    @classmethod
    def create(
        cls,
        sender: str,
        content: Dict[str, Any],
        routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT,
        recipient: Optional[str] = None,
        message_type: MessageType = MessageType.COMMAND,
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        expires_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "EnhancedAgentMessage":
        """
        Create a new enhanced agent message.
        
        Args:
            sender: ID of the sending agent or system
            content: Message payload content
            routing_strategy: How the message should be routed
            recipient: Target recipient (required for DIRECT routing)
            message_type: Type of message being sent
            priority: Priority level of the message
            correlation_id: Optional ID to correlate related messages
            expires_at: Optional ISO timestamp when message expires
            metadata: Optional additional metadata for the message
            
        Returns:
            A new EnhancedAgentMessage instance
        """
        # Generate a unique message ID if not provided
        message_id = str(uuid4())
        
        # Validate routing based on strategy
        if routing_strategy == RoutingStrategy.DIRECT and recipient is None:
            raise ValueError("Direct routing requires a recipient")
            
        # Create message with standard fields
        message = cls({
            "message_id": message_id,
            "sender": sender,
            "timestamp": datetime.now(UTC).isoformat(),
            "routing": {
                "strategy": routing_strategy.value,
                "status": MessageStatus.CREATED.value
            },
            "type": message_type.value,
            "priority": priority.value,
            "content": content
        })
        
        # Add recipient for direct routing
        if recipient:
            message["routing"]["recipient"] = recipient
            
        # Add optional fields if provided
        if correlation_id:
            message["correlation_id"] = correlation_id
            
        if expires_at:
            message["expires_at"] = expires_at
            
        if metadata:
            message["metadata"] = metadata
            
        return message
    
    def create_response(
        self,
        sender: str,
        content: Dict[str, Any],
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "EnhancedAgentMessage":
        """
        Create a response to this message.
        
        Args:
            sender: ID of the responding agent or system
            content: Response content
            success: Whether the operation was successful
            metadata: Optional additional metadata for the response
            
        Returns:
            A new response message
        """
        # Determine message type based on success flag
        message_type = MessageType.RESPONSE if success else MessageType.ERROR
        
        # Get the original sender as the recipient for the response
        recipient = self.get("sender")
        
        # Create response message
        response = EnhancedAgentMessage.create(
            sender=sender,
            content=content,
            routing_strategy=RoutingStrategy.DIRECT,
            recipient=recipient,
            message_type=message_type,
            correlation_id=self.get("message_id"),
            metadata=metadata
        )
        
        # Add reference to original message
        response["in_response_to"] = self.get("message_id")
        
        # Add success flag
        response["success"] = success
        
        return response


class AgentCommunicationService:
    """
    Service for handling agent communication via Kafka.
    
    This service provides:
    1. Message publishing to Kafka topics
    2. Message subscription and routing to agents
    3. Message format standardization and validation
    """
    
    def __init__(
        self, 
        kafka_producer: KafkaProducerProtocol, 
        kafka_consumer: KafkaConsumerProtocol, 
        topic_prefix: str = "agent"
    ) -> None:
        """
        Initialize the agent communication service.
        
        Args:
            kafka_producer: Kafka producer for message publishing
            kafka_consumer: Kafka consumer for message subscription
            topic_prefix: Prefix for agent communication topics
        """
        self.kafka_producer = kafka_producer
        self.kafka_consumer = kafka_consumer
        self.topic_prefix = topic_prefix
        self.message_handlers: Dict[str, List[MessageHandlerProtocol]] = {}
        
        # Create standard topics if they don't exist
        self._ensure_standard_topics()
    
    def _ensure_standard_topics(self) -> None:
        """Ensure that the standard communication topics exist."""
        # Define standard topics for agent communication
        standard_topics = [
            self._get_topic_for_strategy(RoutingStrategy.DIRECT),
            self._get_topic_for_strategy(RoutingStrategy.BROADCAST),
            self._get_topic_for_strategy(RoutingStrategy.CAPABILITY),
            self._get_topic_for_strategy(RoutingStrategy.GROUP)
        ]
        
        # Create topics if they don't exist
        for topic in standard_topics:
            # Check if topic exists and create if needed
            try:
                if not hasattr(self.kafka_producer, 'topic_exists'):
                    logger.warning(f"Kafka producer does not support topic_exists check - assuming topic {topic} exists")
                    continue
                    
                if hasattr(self.kafka_producer, 'topic_exists') and not self.kafka_producer.topic_exists(topic):
                    if hasattr(self.kafka_producer, 'create_topic'):
                        self.kafka_producer.create_topic(topic)
                        logger.info(f"Created Kafka topic: {topic}")
                    else:
                        logger.warning(f"Kafka producer does not support creating topics - assuming topic {topic} exists")
            except Exception as e:
                logger.error(f"Failed to create Kafka topic {topic}: {e}")
    
    def _get_topic_for_strategy(self, strategy: RoutingStrategy) -> str:
        """
        Get the appropriate Kafka topic for a routing strategy.
        
        Args:
            strategy: Routing strategy to use
            
        Returns:
            Name of the Kafka topic to use
        """
        strategy_map = {
            RoutingStrategy.DIRECT: f"{self.topic_prefix}.direct",
            RoutingStrategy.BROADCAST: f"{self.topic_prefix}.broadcast",
            RoutingStrategy.CAPABILITY: f"{self.topic_prefix}.capability",
            RoutingStrategy.GROUP: f"{self.topic_prefix}.group"
        }
        
        return strategy_map.get(strategy, f"{self.topic_prefix}.direct")
    
    async def send_message(self, message: EnhancedAgentMessage) -> bool:
        """
        Send a message using the appropriate routing strategy.
        
        Args:
            message: Enhanced agent message to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        # Validate message format
        if not self._validate_message(message):
            logger.error(f"Invalid message format: {message}")
            return False
            
        # Determine topic based on routing strategy
        topic = self._get_topic_for_strategy(RoutingStrategy(message["routing"]["strategy"]))
        
        # Use message ID as key for routing
        key = message["message_id"]
        
        # Serialize message to JSON
        try:
            message_json = json.dumps(message)
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            return False
        
        try:
            # Send message to Kafka
            # The produce method returns None, don't use await here
            self.kafka_producer.produce(
                topic=topic, 
                value=message_json.encode('utf-8'), 
                key=key.encode('utf-8') if key else None
            )
            # Update message status to SENT
            message["routing"]["status"] = MessageStatus.SENT.value
            logger.debug(f"Sent message {message['message_id']} to topic {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            return False
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate message format and content.
        
        Args:
            message: Message to validate
            
        Returns:
            True if the message is valid, False otherwise
        """
        # Check required fields
        required_fields = ["message_id", "sender", "timestamp", "routing", "type", "content"]
        for field in required_fields:
            if field not in message:
                logger.error(f"Message missing required field: {field}")
                return False
        
        # Validate routing information
        routing = message.get("routing", {})
        strategy = routing.get("strategy")
        recipient = routing.get("recipient")
        
        # Direct routing requires a recipient
        if strategy == RoutingStrategy.DIRECT.value and not recipient:
            logger.error("Direct routing requires a recipient")
            return False
        
        return True
    
    def register_handler(self, agent_id: str, handler: MessageHandlerProtocol) -> None:
        """
        Register a message handler for an agent.
        
        Args:
            agent_id: ID of the agent
            handler: Message handler to register
        """
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = []
        
        self.message_handlers[agent_id].append(handler)
        logger.info(f"Registered message handler for agent {agent_id}")
    
    def unregister_handler(self, agent_id: str, handler: Optional[MessageHandlerProtocol] = None) -> None:
        """
        Unregister a message handler for an agent.
        
        Args:
            agent_id: ID of the agent
            handler: Specific handler to unregister, or None to unregister all handlers
        """
        if agent_id not in self.message_handlers:
            return
            
        if handler is None:
            # Remove all handlers for this agent
            self.message_handlers.pop(agent_id)
            logger.info(f"Unregistered all message handlers for agent {agent_id}")
        else:
            # Remove specific handler
            self.message_handlers[agent_id] = [
                h for h in self.message_handlers[agent_id] if h != handler
            ]
            logger.info(f"Unregistered specific message handler for agent {agent_id}")
    
    async def start_consumers(self) -> None:
        """Start Kafka consumers for agent message topics."""
        # Get all topics to subscribe to
        topics = [
            self._get_topic_for_strategy(RoutingStrategy.DIRECT),
            self._get_topic_for_strategy(RoutingStrategy.BROADCAST),
            self._get_topic_for_strategy(RoutingStrategy.CAPABILITY),
            self._get_topic_for_strategy(RoutingStrategy.GROUP)
        ]
        
        # Start consumers for each topic
        for topic in topics:
            logger.info(f"Starting consumer for topic: {topic}")
            
            # Use consistent consumer group for the service
            consumer_group = f"{self.topic_prefix}_consumer_group"
            
            try:
                # Start consumer with message handler
                self.kafka_consumer.subscribe([topic])
                logger.info(f"Subscribed to topic: {topic}")
                
                # Start polling in background
                # Note: This would typically be done in a separate task or thread
                # For simplicity, we'll assume the consumer is already configured to poll
            except Exception as e:
                logger.error(f"Failed to start consumer for topic {topic}: {e}")
    
    async def _message_consumer_handler(self, value: Union[Dict[str, Any], str, bytes], key: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> None:
        """
        Handle incoming messages from Kafka topics.
        
        Args:
            value: Message value (can be Dict, str, or bytes)
            key: Optional message key
            headers: Optional message headers
        """
        try:
            # Parse message JSON
            message = None
            if isinstance(value, bytes):
                message_str = value.decode('utf-8')
                message = json.loads(message_str)
            elif isinstance(value, str):
                message = json.loads(value)
            elif isinstance(value, dict):
                message = value
            
            # Check if we successfully parsed the message
            if message is None:
                logger.error(f"Failed to parse message of type: {type(value)}")
                return
            
            # Validate message format
            if not self._validate_message(message):
                logger.error(f"Received invalid message format: {message}")
                return
            
            # Extract routing information
            routing = message.get("routing", {})
            recipient_id = routing.get("recipient")
            strategy = RoutingStrategy(routing.get("strategy", RoutingStrategy.DIRECT.value))
            
            # Update message status
            message["routing"]["status"] = MessageStatus.DELIVERED.value
            
            # Process based on routing strategy
            if strategy == RoutingStrategy.DIRECT:
                # Direct message to specific agent
                if recipient_id:
                    await self._dispatch_message(recipient_id, message)
                else:
                    logger.error(f"Direct message missing recipient ID: {message}")
            
            elif strategy == RoutingStrategy.BROADCAST:
                # Send to all registered handlers
                for agent_id in self.message_handlers.keys():
                    await self._dispatch_message(agent_id, message)
            
            elif strategy == RoutingStrategy.CAPABILITY:
                # TODO: Implement capability-based routing
                # This would typically check which agents have registered for a specific capability
                # For now, just broadcast to all
                for agent_id in self.message_handlers.keys():
                    await self._dispatch_message(agent_id, message)
            
            elif strategy == RoutingStrategy.GROUP:
                # TODO: Implement group-based routing
                # This would check which agents are in a specific group
                # For now, just log it's not implemented
                logger.warning(f"Group-based routing not yet implemented")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message JSON: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _dispatch_message(self, agent_id: str, message: Dict[str, Any]) -> None:
        """
        Dispatch a message to all handlers for an agent.
        
        Args:
            agent_id: ID of the target agent
            message: Message to dispatch
        """
        if agent_id not in self.message_handlers:
            logger.warning(f"No message handlers registered for agent {agent_id}")
            return
            
        # Update message status to processing
        message["routing"]["status"] = MessageStatus.PROCESSING.value
        
        # Dispatch to all handlers for the agent
        handlers = self.message_handlers.get(agent_id, [])
        for handler in handlers:
            try:
                # Call handler with the message
                result = handler.handle_message(message)
                
                # If handler is async and returned a coroutine, await it
                if hasattr(result, '__await__'):
                    await result
                    
                logger.debug(f"Message {message['message_id']} processed by handler for agent {agent_id}")
            except Exception as e:
                logger.error(f"Error in message handler for agent {agent_id}: {e}")
                
                # Update message status to failed
                message["routing"]["status"] = MessageStatus.FAILED.value
