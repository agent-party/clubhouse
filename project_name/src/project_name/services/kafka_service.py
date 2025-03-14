from typing import Protocol, Dict, Any, Optional, List, Callable
import json
import logging
from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class KafkaProducerProtocol(Protocol):
    """Protocol defining the Kafka producer interface."""
    
    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None, 
               headers: Optional[Dict[str, str]] = None) -> None:
        """Produce a message to a topic."""
        ...
    
    def flush(self, timeout: Optional[float] = None) -> int:
        """Flush the producer."""
        ...


class KafkaConsumerProtocol(Protocol):
    """Protocol defining the Kafka consumer interface."""
    
    def subscribe(self, topics: List[str]) -> None:
        """Subscribe to a list of topics."""
        ...
    
    def poll(self, timeout: Optional[float] = None) -> Any:
        """Poll for new messages."""
        ...
    
    def close(self) -> None:
        """Close the consumer."""
        ...


class KafkaMessage(BaseModel):
    """Base model for Kafka messages with validation."""
    
    topic: str
    value: Dict[str, Any]
    key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None


class KafkaService:
    """
    Service for interacting with Kafka.
    
    Provides an abstraction over the Kafka producer and consumer clients.
    """
    
    def __init__(self, producer_config: Dict[str, Any], consumer_config: Dict[str, Any]) -> None:
        """
        Initialize the Kafka service.
        
        Args:
            producer_config: Configuration for the Kafka producer
            consumer_config: Configuration for the Kafka consumer
        """
        self._producer = Producer(producer_config)
        self._consumer = Consumer(consumer_config)
        self._running = False
    
    def produce_message(self, message: KafkaMessage) -> None:
        """
        Produce a message to a Kafka topic.
        
        Args:
            message: The message to produce
            
        Raises:
            ValidationError: If the message fails validation
            KafkaException: If there is an error producing the message
        """
        try:
            # Serialize the message value
            value = json.dumps(message.value).encode("utf-8")
            
            # Serialize the message key if provided
            key = message.key.encode("utf-8") if message.key else None
            
            # Delivery callback to log any errors
            def delivery_callback(err, msg):
                if err:
                    logger.error("Message delivery failed: %s", err)
                else:
                    logger.debug("Message delivered to %s [%d] at offset %d",
                                msg.topic(), msg.partition(), msg.offset())
            
            # Produce the message
            self._producer.produce(
                topic=message.topic,
                value=value,
                key=key,
                headers=message.headers,
                callback=delivery_callback
            )
            
            # Flush to ensure delivery
            self._producer.flush()
            
        except (ValidationError, KafkaException) as e:
            logger.error("Error producing message: %s", e)
            raise
    
    def consume_messages(self, topics: List[str], 
                        handler: Callable[[Dict[str, Any], Optional[str]], None],
                        timeout: float = 1.0) -> None:
        """
        Consume messages from Kafka topics.
        
        Args:
            topics: List of topics to consume from
            handler: Callback function to handle consumed messages
            timeout: Poll timeout in seconds
            
        Raises:
            KafkaException: If there is an error consuming messages
        """
        try:
            # Subscribe to the topics
            self._consumer.subscribe(topics)
            self._running = True
            
            logger.info("Started consuming from topics: %s", topics)
            
            # Consume messages
            while self._running:
                msg = self._consumer.poll(timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event - not an error
                        logger.debug("Reached end of partition")
                    else:
                        # Error
                        logger.error("Error while consuming: %s", msg.error())
                else:
                    # Process the message
                    try:
                        value = json.loads(msg.value().decode("utf-8"))
                        key = msg.key().decode("utf-8") if msg.key() else None
                        
                        # Call the handler with the message value and key
                        handler(value, key)
                        
                    except json.JSONDecodeError as e:
                        logger.error("Error decoding message: %s", e)
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
        
        except KafkaException as e:
            logger.error("Kafka error: %s", e)
            raise
        finally:
            # Close the consumer
            self.stop_consuming()
    
    def stop_consuming(self) -> None:
        """Stop consuming messages."""
        self._running = False
        self._consumer.close()
        logger.info("Stopped consuming messages")
