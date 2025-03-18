"""
Kafka CLI package for agent interaction.

This package provides a command-line interface for interacting with agents
through Kafka-based messaging, implementing the standardized event-driven architecture
for agent communication.
"""

from . import kafka_client
from . import agent
from . import cli

__all__ = ["kafka_client", "agent", "cli"]
