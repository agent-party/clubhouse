#!/usr/bin/env python
"""
Run script for the Clubhouse application.

This script provides a simple way to start the clubhouse application,
configuring it to connect to Kafka and process messages. It supports
schema registration for message definitions with the Schema Registry.

The Clubhouse is the central component of the agent collaboration system,
responsible for:
1. Managing agents and their capabilities
2. Processing command messages from clients
3. Publishing response and event messages
4. Registering schemas for message serialization/deserialization

Usage:
    python run_clubhouse.py [options]

Example:
    python run_clubhouse.py --bootstrap-servers localhost:9092 --schema-registry-url http://localhost:8081
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from clubhouse.clubhouse_main import main


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Clubhouse Application Runner")
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
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Set debug environment variable if specified
    if args.debug:
        os.environ["DEBUG"] = "1"
        logging.getLogger().debug("Debug logging enabled")
    
    # Run the main function
    main(
        bootstrap_servers=args.bootstrap_servers,
        commands_topic=args.commands_topic,
        responses_topic=args.responses_topic, 
        events_topic=args.events_topic,
        group_id=args.group_id,
        schema_registry_url=args.schema_registry_url,
        register_schemas_only=args.register_schemas_only
    )
