#!/usr/bin/env python
"""
Agent Kafka CLI

A Kafka-based command-line interface for interacting with agents, enabling:
- Creation and management of agents with distinct personalities
- Conversational interactions with agents
- Multi-agent problem solving through Kafka events
- Parameter validation using Pydantic models
- Event-driven architecture for asynchronous operations

This implementation follows the refactoring checklist, particularly focusing on:
- Standardized parameter validation
- Consistent event handling
- Proper response structure
- Conversation capability integration
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to Python path to enable imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agent_kafka_cli")

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv not installed, skipping .env loading")

try:
    from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
    KAFKA_AVAILABLE = True
except ImportError:
    logger.warning("confluent-kafka package not installed, falling back to mock implementation")
    KAFKA_AVAILABLE = False

# Import clubhouse components
from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.capabilities.conversation_capability import (
    ConversationCapability,
    ConversationContext,
    ConversationMessage,
    ConversationParameters
)
from clubhouse.agents.capabilities.llm_capability import LLMCapability, LLMParameters
from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.personality import (
    get_personality,
    list_personalities,
    AgentPersonality
)

# Constants
DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_KAFKA_GROUP_ID = "agent_cli"
DEFAULT_KAFKA_TOPIC_PREFIX = "agents"

# Topics
TOPIC_AGENT_COMMANDS = f"{DEFAULT_KAFKA_TOPIC_PREFIX}.commands"
TOPIC_AGENT_EVENTS = f"{DEFAULT_KAFKA_TOPIC_PREFIX}.events"
TOPIC_AGENT_RESPONSES = f"{DEFAULT_KAFKA_TOPIC_PREFIX}.responses"
TOPIC_CONVERSATION = f"{DEFAULT_KAFKA_TOPIC_PREFIX}.conversation"
