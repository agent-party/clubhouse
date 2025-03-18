#!/usr/bin/env python3
"""
Agent Kafka CLI
--------------

Main entry point for the Kafka-based Agent CLI, which serves as a thin client
for sending commands to and receiving responses from the clubhouse system.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Configure Python path to include project root
PROJECT_ROOT = Path(__file__).parent.parent
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file if present
load_dotenv()

# Import the CLI class from our module
from scripts.kafka_cli.cli import AgentCLI


async def main() -> None:
    """
    Main entry point for the Kafka-based Agent CLI.
    
    This function initializes and starts the CLI interface, which connects
    to Kafka and provides a command-line interface for interacting with agents.
    """
    # Get configuration from environment variables
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    debug_mode = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
    
    # Initialize and start the CLI
    cli = AgentCLI(bootstrap_servers=bootstrap_servers, debug=debug_mode)
    await cli.start()


if __name__ == "__main__":
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        if os.environ.get("DEBUG", "").lower() in ("true", "1", "yes"):
            import traceback
            traceback.print_exc()
        sys.exit(1)
