"""
Kafka Test Utilities.

This module provides utilities for Kafka-related integration tests including
fixtures for test isolation, cleanup, and common test patterns.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple

import httpx
from confluent_kafka.admin import AdminClient, NewTopic

from clubhouse.services.confluent_kafka_service import (
    ConfluentKafkaService,
    KafkaConfig,
    KafkaMessage
)
from clubhouse.services.kafka_protocol import MessageHandlerProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageCollector:
    """Utility class to collect and verify Kafka messages in tests."""
    
    def __init__(
        self, 
        max_messages: int = 1, 
        match_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        timeout_seconds: int = 10
    ):
        """
        Initialize the message collector.
        
        Args:
            max_messages: Maximum number of messages to collect before stopping
            match_condition: Optional function to determine if a message should be collected
            timeout_seconds: Maximum time to wait for all messages in seconds
        """
        self.messages: List[Tuple[Dict[str, Any], Optional[str], Optional[Dict[str, str]]]] = []
        self.max_messages = max_messages
        self.match_condition = match_condition
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.service: Optional[ConfluentKafkaService] = None
    
    def handle(
        self, 
        value: Dict[str, Any], 
        key: Optional[str] = None, 
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Handle a message by collecting it if it matches the condition.
        
        Args:
            value: Message value
            key: Optional message key
            headers: Optional message headers
        """
        logger.info(f"Received message: {value}")
        
        # Check for timeout
        if time.time() - self.start_time > self.timeout_seconds:
            logger.warning(f"MessageCollector timed out after {self.timeout_seconds} seconds")
            if self.service:
                self.service.stop_consuming()
            return
            
        # Check match condition if provided
        if self.match_condition and not self.match_condition(value):
            logger.info(f"Message did not match condition, skipping: {value}")
            return
            
        # Add message to collection
        self.messages.append((value, key, headers))
        logger.info(f"Collected message {len(self.messages)}/{self.max_messages}")
        
        # Stop consuming if we've reached the max
        if len(self.messages) >= self.max_messages and self.service:
            logger.info(f"Reached max messages ({self.max_messages}), stopping consumer")
            self.service.stop_consuming()
    
    def set_service(self, service: ConfluentKafkaService) -> None:
        """Set the Kafka service reference for this collector."""
        self.service = service
    
    def reset(self) -> None:
        """Reset the collector."""
        self.messages = []
        self.start_time = time.time()
    
    def wait_for_messages(self) -> bool:
        """
        Wait for messages to be collected.
        
        Returns:
            True if all expected messages were collected, False if timed out
        """
        end_time = self.start_time + self.timeout_seconds
        while time.time() < end_time:
            if len(self.messages) >= self.max_messages:
                return True
            time.sleep(0.1)
        return False


def create_test_topics(bootstrap_servers: str, topics: List[str], num_partitions: int = 1) -> None:
    """
    Create test topics in Kafka.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        topics: List of topic names to create
        num_partitions: Number of partitions for each topic
    """
    admin_client = AdminClient({"bootstrap.servers": bootstrap_servers})
    
    # Create new topics
    new_topics = [
        NewTopic(topic, num_partitions, 1) for topic in topics
    ]
    
    # Create topics
    logger.info(f"Creating test topics: {topics}")
    results = admin_client.create_topics(new_topics)
    
    # Wait for topic creation to complete
    for topic, future in results.items():
        try:
            future.result()
            logger.info(f"Topic {topic} created")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Topic {topic} already exists")
            else:
                logger.error(f"Failed to create topic {topic}: {e}")


def delete_test_topics(bootstrap_servers: str, topics: List[str]) -> None:
    """
    Delete test topics from Kafka.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        topics: List of topic names to delete
    """
    admin_client = AdminClient({"bootstrap.servers": bootstrap_servers})
    
    # Delete topics
    logger.info(f"Deleting test topics: {topics}")
    results = admin_client.delete_topics(topics)
    
    # Wait for topic deletion to complete
    for topic, future in results.items():
        try:
            future.result()
            logger.info(f"Topic {topic} deleted")
        except Exception as e:
            logger.error(f"Failed to delete topic {topic}: {e}")


def check_kafka_connection(bootstrap_servers: str, timeout: int = 5) -> bool:
    """
    Check if Kafka broker is accessible.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        # Create admin client with short timeout
        admin_config = {
            "bootstrap.servers": bootstrap_servers,
            "socket.timeout.ms": timeout * 1000,
            "request.timeout.ms": timeout * 1000
        }
        admin_client = AdminClient(admin_config)
        
        # Try to list topics as a connection test
        topics = admin_client.list_topics(timeout=timeout)
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Kafka at {bootstrap_servers}: {e}")
        return False


def check_schema_registry_connection(schema_registry_url: str, timeout: int = 5) -> bool:
    """
    Check if Schema Registry is accessible.
    
    Args:
        schema_registry_url: Schema Registry URL
        timeout: Connection timeout in seconds
        
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        # Make a simple HTTP request to the Schema Registry
        response = httpx.get(schema_registry_url, timeout=timeout)
        if response.status_code == 200:
            logger.info(f"Successfully connected to Schema Registry at {schema_registry_url}")
            return True
        else:
            logger.error(f"Schema Registry returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Failed to connect to Schema Registry at {schema_registry_url}: {e}")
        return False


def wait_for_kafka(bootstrap_servers: str, max_attempts: int = 30, delay: int = 1) -> bool:
    """
    Wait for Kafka to be accessible.
    
    Args:
        bootstrap_servers: Kafka bootstrap servers
        max_attempts: Maximum number of connection attempts
        delay: Delay between attempts in seconds
        
    Returns:
        True if Kafka becomes accessible, False otherwise
    """
    logger.info(f"Waiting for Kafka at {bootstrap_servers}")
    
    for attempt in range(max_attempts):
        if check_kafka_connection(bootstrap_servers):
            return True
            
        logger.info(f"Kafka not yet accessible (attempt {attempt+1}/{max_attempts}), waiting {delay}s...")
        time.sleep(delay)
        
    logger.error(f"Timed out waiting for Kafka after {max_attempts} attempts")
    return False


def wait_for_schema_registry(schema_registry_url: str, max_attempts: int = 30, delay: int = 1) -> bool:
    """
    Wait for Schema Registry to be accessible.
    
    Args:
        schema_registry_url: Schema Registry URL
        max_attempts: Maximum number of connection attempts
        delay: Delay between attempts in seconds
        
    Returns:
        True if Schema Registry becomes accessible, False otherwise
    """
    logger.info(f"Waiting for Schema Registry at {schema_registry_url}")
    
    for attempt in range(max_attempts):
        if check_schema_registry_connection(schema_registry_url):
            return True
            
        logger.info(f"Schema Registry not yet accessible (attempt {attempt+1}/{max_attempts}), waiting {delay}s...")
        time.sleep(delay)
        
    logger.error(f"Timed out waiting for Schema Registry after {max_attempts} attempts")
    return False
