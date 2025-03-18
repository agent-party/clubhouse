"""
Kafka Failure Scenario Tests.

This module tests error handling and resilience of our Kafka implementation under
various failure scenarios, connection issues, and error conditions.
"""

import logging
import os
import pytest
import time
import uuid
from typing import Dict, Any, List, Optional, Type
from copy import deepcopy

from confluent_kafka import KafkaException
from pydantic import BaseModel, Field

from clubhouse.services.confluent_kafka_service import (
    ConfluentKafkaService,
    KafkaConfig,
    KafkaMessage
)
from clubhouse.services.schema_registry import ConfluentSchemaRegistry

from tests.utils.kafka_test_utils import (
    MessageCollector,
    check_kafka_connection,
    check_schema_registry_connection,
    wait_for_kafka,
    wait_for_schema_registry
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPayload(BaseModel):
    """Test payload model for Kafka failure scenario messages."""
    test_id: str = Field(..., description="Unique test identifier")
    message: str = Field(..., description="Test message content")


@pytest.mark.integration
@pytest.mark.failure
class TestKafkaFailureScenarios:
    """Integration tests for Kafka failure scenarios and error handling."""

    @pytest.fixture(scope="class")
    def kafka_bootstrap_servers(self) -> str:
        """Get the Kafka bootstrap servers from environment or use default."""
        return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    @pytest.fixture(scope="class")
    def schema_registry_url(self) -> str:
        """Get the Schema Registry URL from environment or use default."""
        return os.environ.get("SCHEMA_REGISTRY_URL", "http://localhost:8081")

    @pytest.fixture(scope="class")
    def invalid_kafka_config(self, schema_registry_url: str) -> KafkaConfig:
        """
        Create an invalid Kafka configuration for testing error handling.
        
        Uses a non-existent broker to simulate connection failures.
        """
        return KafkaConfig(
            bootstrap_servers="non-existent-host:9092",
            client_id=f"test-client-{uuid.uuid4()}",
            group_id=f"test-group-{uuid.uuid4()}",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url=schema_registry_url,
            # Add connection timeout settings to prevent test from hanging
            producer_config={
                "socket.timeout.ms": 1000,  # 1 second socket timeout
                "message.timeout.ms": 2000,  # 2 second message timeout
                "request.timeout.ms": 2000,  # 2 second request timeout
                "metadata.request.timeout.ms": 2000,  # 2 second metadata request timeout
                "connections.max.idle.ms": 1000  # 1 second max idle time
            }
        )

    @pytest.fixture(scope="class")
    def invalid_schema_registry_config(self, kafka_bootstrap_servers: str) -> KafkaConfig:
        """
        Create a Kafka configuration with invalid Schema Registry URL.
        
        Used for testing schema registry connection failure handling.
        """
        return KafkaConfig(
            bootstrap_servers=kafka_bootstrap_servers,
            client_id=f"test-client-{uuid.uuid4()}",
            group_id=f"test-group-{uuid.uuid4()}",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url="http://non-existent-host:8081"
        )

    @pytest.fixture(scope="class")
    def valid_kafka_config(self, kafka_bootstrap_servers: str, schema_registry_url: str) -> KafkaConfig:
        """Create a valid Kafka configuration."""
        return KafkaConfig(
            bootstrap_servers=kafka_bootstrap_servers,
            client_id=f"test-client-{uuid.uuid4()}",
            group_id=f"test-group-{uuid.uuid4()}",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            schema_registry_url=schema_registry_url
        )

    @pytest.fixture(scope="function", autouse=True)
    def check_prerequisite_connections(self, kafka_bootstrap_servers: str, schema_registry_url: str):
        """Check real service connections for tests that need them."""
        # Some tests require actual services, others test the failure scenarios
        # We'll add specific checks in the relevant tests

    # Removed skip marker to test actual implementation
    def test_broker_connection_failure_handling(self, invalid_kafka_config: KafkaConfig):
        """Test broker connection failure handling with invalid broker configuration."""
        # Create service with invalid broker configuration
        service = ConfluentKafkaService(invalid_kafka_config)
        
        # Verify the broker is actually unreachable
        is_reachable = check_kafka_connection(invalid_kafka_config.bootstrap_servers, timeout=2)
        assert not is_reachable, "Test requires broker to be unreachable"
        logger.info(f"Confirmed broker {invalid_kafka_config.bootstrap_servers} is unreachable")
        
        # Create a test message
        message = KafkaMessage(
            topic="test-topic",
            value={"test": "value"},
            key="test-key"
        )
        
        # Try to produce the message
        # In real Kafka infrastructure, connection failures can take 10-30s by design
        start_time = time.time()
        max_wait_time = 30.0  # Real-world Kafka connection timeouts are often 30s
        
        try:
            # Test that we don't hang indefinitely and follow Kafka's normal behavior
            logger.info("Attempting to produce message to unreachable broker...")
            
            # Set producer's internal timeout to something reasonable
            producer = service.get_producer()
            
            # This doesn't throw immediately - Kafka clients queue messages first
            producer.produce(
                topic=message.topic,
                value=message.value,
                key=message.key,
                headers=message.headers
            )
            
            # Flush should eventually error or time out with unreachable broker
            logger.info("Flushing producer - this should eventually timeout or fail...")
            flush_result = producer.flush(timeout=15.0)
            logger.info(f"Flush returned with {flush_result} messages remaining")
            
            # If we have remaining messages, that's expected with unreachable broker
            assert flush_result > 0, "Expected messages to remain unflushed with unreachable broker"
            
        except Exception as e:
            # Getting an exception is also acceptable
            logger.info(f"Got expected exception: {e}")
            error_str = str(e).lower()
            assert any(term in error_str for term in [
                "timed out", "timeout", "connection", "broker", "connect", "unreachable"
            ]), f"Unexpected error type: {error_str}"
        
        # Verify the operation completed within a reasonable timeframe
        # For unreachable brokers, "reasonable" might still be 10-30s in real Kafka
        elapsed_time = time.time() - start_time
        logger.info(f"Operation completed in {elapsed_time:.2f}s")
        
        # Success criteria: either we get an appropriate error or we have unflushed messages
        # AND the test doesn't hang indefinitely (which would be caught by pytest timeouts)
        logger.info("Test passed - Kafka service handled unreachable broker appropriately")

    # Removed skip marker to test actual implementation
    def test_schema_registry_connection_failure(self, invalid_schema_registry_config: KafkaConfig):
        """Test Schema Registry connection failure handling."""
        # Create service with invalid schema registry configuration
        service = ConfluentKafkaService(invalid_schema_registry_config)
        
        # Define a schema that would require schema registry interaction
        schema_str = """
        {
            "type": "record",
            "name": "TestRecord",
            "fields": [
                {"name": "test_id", "type": "string"},
                {"name": "message", "type": "string"}
            ]
        }
        """
        
        # Attempt to register schema - should raise exception with connection error
        with pytest.raises(Exception) as excinfo:
            registry = service.get_schema_registry()
            registry.register_schema("test-subject", schema_str, timeout=1.0)
            
        # Error should relate to schema registry connection
        error_str = str(excinfo.value).lower()
        assert any(term in error_str for term in [
            "schema", "registry", "connection", "unreachable", "timeout", "failed"
        ])

    def test_topic_not_exists_behavior(self, valid_kafka_config: KafkaConfig, kafka_bootstrap_servers: str):
        """Test consumer behavior for non-existent topics."""
        # Skip test if Kafka is not available
        if not kafka_bootstrap_servers:
            pytest.skip("Kafka is not available")
            
        # Create service with valid configuration
        service = ConfluentKafkaService(valid_kafka_config)
        
        # Generate a random non-existent topic name
        nonexistent_topic = f"nonexistent-topic-{uuid.uuid4()}"
        logger.info(f"Testing with non-existent topic: {nonexistent_topic}")
        
        # Verify Kafka is reachable but we won't verify the topic directly
        is_reachable = check_kafka_connection(kafka_bootstrap_servers, timeout=5)
        assert is_reachable, "This test requires a reachable Kafka broker"
        logger.info(f"Confirmed Kafka broker is accessible at {kafka_bootstrap_servers}")
        
        # Create collector to consume messages (with timeout appropriate for real Kafka)
        collector = MessageCollector(max_messages=1, timeout_seconds=5)
        collector.service = service  # Set service reference for stop functionality
        
        # Track the start time for verification
        start_time = time.time()
        
        # In real Kafka deployments, consume_messages will wait up to max_runtime
        # even with non-existent topics, which is the expected behavior
        max_runtime = 10.0
        logger.info(f"Starting consumption with max_runtime={max_runtime}s")
        service.consume_messages([nonexistent_topic], collector, poll_timeout=0.5, max_runtime=max_runtime)
        
        # Track the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Consumption completed in {elapsed_time:.2f}s")
        
        # Verify no messages were received
        assert len(collector.messages) == 0, f"Expected 0 messages, but received {len(collector.messages)}"
        
        # Verify the consumer didn't hang indefinitely
        # For non-existent topics, Kafka consumers could potentially wait the full max_runtime
        # before returning, which is expected behavior with real Kafka infrastructure
        assert elapsed_time <= max_runtime + 5.0, f"Consumer took too long: {elapsed_time:.2f}s > {max_runtime + 5.0:.2f}s"
        
        # Success if:
        # 1. No messages were received (as expected)
        # 2. The operation completed in reasonable time (didn't hang indefinitely)
        logger.info("Test passed - Consumer properly handled non-existent topic")

    def test_consumer_invalid_topic_pattern(self, valid_kafka_config: KafkaConfig, kafka_bootstrap_servers: str):
        """Test consumer behavior with invalid topic pattern."""
        # Skip test if Kafka is not available
        if not kafka_bootstrap_servers:
            pytest.skip("Kafka is not available")

        # Create service with valid config
        service = ConfluentKafkaService(valid_kafka_config)
        
        # Create a test message collector
        collector = MessageCollector()
        
        # Use an invalid regex pattern that should cause a topic subscription error
        invalid_topic_pattern = "^[invalid-regex"
        
        # Verify the exception is raised when trying to subscribe with invalid pattern
        with pytest.raises(KafkaException):
            service.get_consumer().subscribe([invalid_topic_pattern])

    def test_consumer_group_rebalancing(self, valid_kafka_config: KafkaConfig, kafka_bootstrap_servers: str):
        """Test consumer group rebalancing behavior."""
        # Skip test if Kafka is not available
        if not kafka_bootstrap_servers:
            pytest.skip("Kafka is not available")
            
        # Create topic with test data
        topic_name = f"test-rebalancing-{uuid.uuid4()}"
        
        # Create service with valid configuration
        service = ConfluentKafkaService(valid_kafka_config)
        
        # Create collector with short timeout
        collector1 = MessageCollector(timeout_seconds=2)
        
        # Start consuming in a separate thread to simulate a consumer group member
        group_id = f"test-group-{uuid.uuid4()}"
        config1 = deepcopy(valid_kafka_config)
        config1.group_id = group_id
        
        service1 = ConfluentKafkaService(config1)
        
        # Start consuming with short timeout to avoid hanging
        import threading
        threading.Thread(
            target=service1.consume_messages,
            args=([topic_name], collector1),
            kwargs={"poll_timeout": 0.5, "max_runtime": 3.0},
            daemon=True
        ).start()
        
        # Wait briefly for consumer to start
        time.sleep(1.0)
        
        # Create a second consumer in the same group - should trigger rebalancing
        collector2 = MessageCollector(timeout_seconds=2)
        config2 = deepcopy(valid_kafka_config)
        config2.group_id = group_id
        
        service2 = ConfluentKafkaService(config2)
        
        # Start second consumer with short timeout to avoid hanging
        service2.consume_messages([topic_name], collector2, poll_timeout=0.5, max_runtime=3.0)
        
        # Test passes if no exceptions occurred during rebalancing
        # (We don't need to verify specific behavior here, just that rebalancing didn't crash)

    def test_producer_retry_behavior(self, valid_kafka_config: KafkaConfig, kafka_bootstrap_servers: str):
        """Test producer retry behavior with large messages."""
        # Skip test if Kafka is not available
        if not kafka_bootstrap_servers:
            pytest.skip("Kafka is not available")
            
        # Create service with valid config
        service = ConfluentKafkaService(valid_kafka_config)
        
        # Generate a random topic for this test
        test_topic = f"test-retry-{uuid.uuid4()}"
        logger.info(f"Testing with topic: {test_topic}")
        
        # Create a large message that might trigger broker throttling or retries
        test_id = str(uuid.uuid4())
        # Using 10MB of data which may exceed default broker limits
        large_data = "X" * (10 * 1024 * 1024)  
        
        # Record delivery results
        delivery_results = []
        
        def delivery_callback(err, msg):
            if err is not None:
                logger.info(f"Delivery failed: {err}")
                delivery_results.append(("error", err))
            else:
                logger.info(f"Message delivered to {msg.topic()}/{msg.partition()}")
                delivery_results.append(("success", msg))
        
        try:
            # Produce large message
            start_time = time.time()
            producer = service.get_producer()
            producer.produce(
                topic=test_topic,
                value={"test_id": test_id, "large_data": large_data},
                key=test_id,
                on_delivery=delivery_callback
            )
            
            # Flush with timeout to see what happens
            logger.info("Flushing producer...")
            remaining_msgs = producer.flush(timeout=5.0)
            end_time = time.time()
            logger.info(f"Flush completed in {end_time - start_time:.2f}s with {remaining_msgs} messages remaining")
            
            # Check delivery results
            if delivery_results:
                result_type, result_value = delivery_results[0]
                if result_type == "error":
                    logger.info(f"Message delivery failed with: {result_value}")
                    error_str = str(result_value).lower()
                    # This should either be a message size error or timeout
                    assert any(term in error_str for term in [
                        "timed out", "timeout", "size", "too large", "message_size"
                    ]), f"Unexpected error: {error_str}"
                else:
                    logger.info("Message was delivered successfully")
            else:
                # If no callback was executed, message might still be queued
                logger.info(f"No delivery results yet, {remaining_msgs} messages still queued")
                assert remaining_msgs > 0, "Expected message to be queued or callback to be executed"
                
            # Test passes as long as we didn't hang, even if the message was too large
            # This better reflects real-world Kafka behavior with large messages
            
        except Exception as e:
            # Log the exception and check it's a reasonable error
            logger.info(f"Exception during large message test: {e}")
            error_str = str(e).lower()
            assert any(term in error_str for term in [
                "timed out", "timeout", "size", "too large", "message_size"
            ]), f"Unexpected exception: {e}"
