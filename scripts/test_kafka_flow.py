#!/usr/bin/env python
"""
Test script for the Kafka message flow between CLI and Clubhouse.

This script simulates sending messages from the CLI to the Clubhouse
and receiving responses, allowing for quick testing of the message flow.
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from confluent_kafka import Producer, Consumer, KafkaError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def delivery_report(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def create_producer(bootstrap_servers: str) -> Producer:
    """Create a Kafka producer."""
    return Producer({
        'bootstrap.servers': bootstrap_servers,
        'client.id': f'test-producer-{uuid.uuid4()}'
    })


def create_consumer(bootstrap_servers: str, topics: List[str], group_id: Optional[str] = None) -> Consumer:
    """Create a Kafka consumer."""
    if group_id is None:
        group_id = f'test-consumer-{uuid.uuid4()}'
        
    consumer = Consumer({
        'bootstrap.servers': bootstrap_servers,
        'group.id': group_id,
        'auto.offset.reset': 'earliest'
    })
    
    consumer.subscribe(topics)
    return consumer


def send_command(producer: Producer, command: Dict[str, Any], topic: str) -> str:
    """
    Send a command to the specified Kafka topic.
    
    Args:
        producer: Kafka producer
        command: Command message to send
        topic: Destination topic
        
    Returns:
        Message ID of the sent command
    """
    # Ensure the message has an ID
    if 'message_id' not in command:
        command['message_id'] = str(uuid.uuid4())
    
    # Ensure the message has a timestamp
    if 'timestamp' not in command:
        from datetime import datetime, timezone
        command['timestamp'] = datetime.now(timezone.utc).isoformat()
    
    # Send the message
    producer.produce(
        topic=topic,
        key=command.get('message_id'),
        value=json.dumps(command).encode('utf-8'),
        callback=delivery_report
    )
    producer.flush()
    
    logger.info(f"Sent command: {command['message_type']} (ID: {command['message_id']})")
    return command['message_id']


def listen_for_responses(consumer: Consumer, expected_ids: List[str], timeout: int = 10) -> List[Dict[str, Any]]:
    """
    Listen for responses on the subscribed topics.
    
    Args:
        consumer: Kafka consumer
        expected_ids: List of message IDs we're expecting responses for
        timeout: Maximum time to wait for responses in seconds
        
    Returns:
        List of response messages received
    """
    responses = []
    remaining_ids = set(expected_ids)
    start_time = time.time()
    
    while remaining_ids and (time.time() - start_time) < timeout:
        msg = consumer.poll(1.0)
        
        if msg is None:
            continue
            
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                logger.debug(f"Reached end of partition for {msg.topic()} [{msg.partition()}]")
            else:
                logger.error(f"Consumer error: {msg.error()}")
            continue
            
        try:
            value = json.loads(msg.value().decode('utf-8'))
            logger.debug(f"Received message: {value.get('message_type')} on topic {msg.topic()}")
            
            # Check if this is a response to one of our commands
            if 'in_response_to' in value and value['in_response_to'] in remaining_ids:
                logger.info(f"Received response: {value.get('message_type')} for command {value['in_response_to']}")
                responses.append(value)
                remaining_ids.remove(value['in_response_to'])
                
            # Also collect event messages related to our commands
            if value.get('message_type', '').endswith('Event') and 'payload' in value:
                for expected_id in expected_ids:
                    if expected_id in str(value):
                        logger.info(f"Received event: {value.get('message_type')}")
                        responses.append(value)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    if remaining_ids:
        logger.warning(f"Timed out waiting for responses to: {remaining_ids}")
        
    return responses


def create_agent_test(producer: Producer, consumer: Consumer, commands_topic: str, agent_id: str, personality_type: str) -> Dict[str, Any]:
    """Run a test to create an agent."""
    command = {
        "message_type": "CreateAgentCommand",
        "payload": {
            "agent_id": agent_id,
            "personality_type": personality_type,
            "metadata": {
                "creator": "test_script"
            }
        }
    }
    
    command_id = send_command(producer, command, commands_topic)
    responses = listen_for_responses(consumer, [command_id])
    
    if not responses:
        logger.error("No response received for CreateAgentCommand")
        return {}
        
    return responses[0]


def delete_agent_test(producer: Producer, consumer: Consumer, commands_topic: str, agent_id: str) -> Dict[str, Any]:
    """Run a test to delete an agent."""
    command = {
        "message_type": "DeleteAgentCommand",
        "payload": {
            "agent_id": agent_id
        }
    }
    
    command_id = send_command(producer, command, commands_topic)
    responses = listen_for_responses(consumer, [command_id])
    
    if not responses:
        logger.error("No response received for DeleteAgentCommand")
        return {}
        
    return responses[0]


def process_message_test(producer: Producer, consumer: Consumer, commands_topic: str, 
                        agent_id: str, message_content: str) -> List[Dict[str, Any]]:
    """Run a test to process a message."""
    conversation_id = str(uuid.uuid4())
    
    command = {
        "message_type": "ProcessMessageCommand",
        "payload": {
            "agent_id": agent_id,
            "conversation_id": conversation_id,
            "content": message_content,
            "metadata": {
                "source": "test_script"
            }
        }
    }
    
    command_id = send_command(producer, command, commands_topic)
    responses = listen_for_responses(consumer, [command_id])
    
    if not responses:
        logger.error("No response received for ProcessMessageCommand")
        return []
        
    return responses


def run_tests(bootstrap_servers: str, commands_topic: str, responses_topic: str, events_topic: str):
    """Run a series of tests for the Kafka message flow."""
    # Create producer and consumer
    producer = create_producer(bootstrap_servers)
    consumer = create_consumer(bootstrap_servers, [responses_topic, events_topic])
    
    try:
        # Generate a unique agent ID for this test run
        test_agent_id = f"test-agent-{uuid.uuid4()}"
        
        logger.info("=== Running Create Agent Test ===")
        create_response = create_agent_test(producer, consumer, commands_topic, test_agent_id, "assistant")
        if not create_response:
            logger.error("Create agent test failed")
            return
            
        logger.info(f"Agent created successfully: {test_agent_id}")
        
        logger.info("=== Running Process Message Test ===")
        process_responses = process_message_test(
            producer, consumer, commands_topic, 
            test_agent_id, "Hello agent, how are you today?"
        )
        
        if not process_responses:
            logger.error("Process message test failed")
        else:
            # Find the actual response (not an event)
            for response in process_responses:
                if response.get("message_type") == "MessageProcessedResponse":
                    logger.info(f"Agent response: {response['payload'].get('content')}")
                    break
        
        logger.info("=== Running Delete Agent Test ===")
        delete_response = delete_agent_test(producer, consumer, commands_topic, test_agent_id)
        
        if not delete_response:
            logger.error("Delete agent test failed")
        else:
            logger.info(f"Agent deleted successfully: {test_agent_id}")
            
        logger.info("=== All Tests Completed ===")
        
    finally:
        # Clean up
        consumer.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the Kafka message flow")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--commands-topic", default="clubhouse-commands", help="Topic for sending commands")
    parser.add_argument("--responses-topic", default="clubhouse-responses", help="Topic for receiving responses")
    parser.add_argument("--events-topic", default="clubhouse-events", help="Topic for receiving events")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if specified
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Run the tests
    run_tests(
        args.bootstrap_servers,
        args.commands_topic,
        args.responses_topic,
        args.events_topic
    )
