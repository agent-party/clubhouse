#!/usr/bin/env python3
"""
Interactive CLI for testing the LLM capability with different providers.

This script provides a simple way to test the LLM capability with different providers,
models, and parameters without having to write test code.
"""
import asyncio
import os
import sys
import argparse
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to the path so we can import clubhouse modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
def read_env_file(env_path: Path) -> Dict[str, str]:
    """
    Directly read and parse a .env file without relying on libraries.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Handle key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                        
                    env_vars[key] = value
                    
        return env_vars
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        return {}

# Try multiple possible locations for .env file
env_paths = [
    Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) / '.env',  # Project root
    Path.cwd() / '.env',  # Current working directory
    Path.home() / '.env',  # User's home directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        env_vars = read_env_file(env_path)
        for key, value in env_vars.items():
            if not os.environ.get(key):
                os.environ[key] = value
        print(f"Found and processed .env file at {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print("Warning: No .env file found in standard locations.")

# Check if the API keys are now available
anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")
huggingface_key = os.environ.get("HUGGINGFACE_API_KEY")

print(f"Anthropic API key loaded: {'Yes' if anthropic_key else 'No'}")
print(f"OpenAI API key loaded: {'Yes' if openai_key else 'No'}")
print(f"HuggingFace API key loaded: {'Yes' if huggingface_key else 'No'}")

from clubhouse.agents.capabilities.llm_capability import LLMCapability, LLMProvider


async def set_api_key_if_needed(provider: LLMProvider) -> bool:
    """
    Check if the API key for the specified provider is set,
    and prompt the user to enter it if it's not.
    
    If the user provides a key, it will be saved to the .env file for future use.
    
    Args:
        provider: The LLM provider enum value
    
    Returns:
        bool: True if the key is set or was successfully set, False otherwise
    """
    # Get capability to use its API key management
    llm = LLMCapability()
    
    # Extract the provider name string from the enum
    provider_name = provider.value.upper()
    
    env_var_name = f"{provider_name}_API_KEY"
    
    if not os.environ.get(env_var_name):
        print(f"\nWarning: {env_var_name} not set in environment.")
        
        # Try to read from .env file if it exists at the project root
        env_file = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) / '.env'
        if env_file.exists():
            print(f"Checking .env file at {env_file}")
            env_vars = read_env_file(env_file)
            if env_var_name in env_vars:
                os.environ[env_var_name] = env_vars[env_var_name]
                print(f"Loaded {env_var_name} from .env file")
                return True
        
        # If still not set, ask the user
        key = input(f"Enter {provider_name} API key: ").strip()
        if key:
            os.environ[env_var_name] = key
            
            # Save the key to .env file for future use
            try:
                # Use the capability's method to save to .env
                llm._save_api_key_to_env_file(env_var_name, key)
                print(f"Saved {env_var_name} to .env file for future use")
            except Exception as e:
                print(f"Warning: Could not save API key to .env file: {e}")
                
            return True
        else:
            print(f"Cannot proceed without {provider_name} API key.")
            return False
    
    return True


async def get_user_input() -> Dict[str, Any]:
    """Get user input for the LLM capability parameters."""
    print("\n==== LLM Capability Interactive Test ====\n")
    
    # Get provider selection
    print("Select LLM provider:")
    print("1. Anthropic (Claude)")
    print("2. OpenAI (GPT)")
    print("3. HuggingFace")
    
    provider_choice = input("\nEnter choice (1-3) [1]: ").strip() or "1"
    
    providers = {
        "1": LLMProvider.ANTHROPIC,
        "2": LLMProvider.OPENAI,
        "3": LLMProvider.HUGGINGFACE
    }
    
    provider = providers.get(provider_choice, LLMProvider.ANTHROPIC)
    
    # Get model based on provider
    default_models = {
        LLMProvider.ANTHROPIC: "claude-3-haiku-20240307",
        LLMProvider.OPENAI: "gpt-4-turbo",
        LLMProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2"
    }
    
    default_model = default_models[provider]
    model = input(f"\nEnter model name [{default_model}]: ").strip() or default_model
    
    # Get prompt
    prompt = input("\nEnter your prompt: ").strip()
    while not prompt:
        print("Prompt cannot be empty.")
        prompt = input("Enter your prompt: ").strip()
    
    # Get temperature
    temp_str = input("\nEnter temperature (0.0-1.0) [0.7]: ").strip() or "0.7"
    try:
        temperature = float(temp_str)
        if temperature < 0 or temperature > 1:
            print("Invalid temperature. Using default 0.7.")
            temperature = 0.7
    except ValueError:
        print("Invalid temperature. Using default 0.7.")
        temperature = 0.7
    
    # Get max tokens
    max_tokens_str = input("\nEnter max tokens [1000]: ").strip() or "1000"
    try:
        max_tokens = int(max_tokens_str)
        if max_tokens < 1:
            print("Invalid max tokens. Using default 1000.")
            max_tokens = 1000
    except ValueError:
        print("Invalid max tokens. Using default 1000.")
        max_tokens = 1000
    
    # Get system prompt
    system_prompt = input("\nEnter system prompt (optional): ").strip() or None
    
    # Get streaming option
    stream_choice = input("\nStream response? (y/n) [n]: ").strip().lower() or "n"
    stream = stream_choice == "y"
    
    # Construct parameters
    params = {
        "prompt": prompt,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    if system_prompt:
        params["system_prompt"] = system_prompt
    
    return params


async def execute_capability(params: Dict[str, Any]) -> None:
    """Execute the LLM capability with the given parameters."""
    # Create the capability
    llm = LLMCapability()
    
    # Display information about the request
    print("\nSending request to LLM provider...")
    print(f"Provider: {params['provider']}")
    print(f"Model: {params['model']}")
    print(f"Streaming: {params['stream']}")
    
    try:
        # Execute the capability
        result = await llm.execute(**params)
        
        # Print the result
        print("\n----- Response -----")
        if result.result["status"] == "success":
            print(result.result["data"]["response"])
        else:
            print(f"Error: {result.result.get('error', 'Unknown error')}")
        
        print("\n----- Metadata -----")
        for key, value in result.metadata.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError executing LLM capability: {str(e)}")


async def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the LLM capability interactively")
    parser.add_argument("--list-env", action="store_true", help="List required environment variables")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional logging")
    args = parser.parse_args()
    
    if args.debug:
        logging_config()
    
    if args.list_env:
        print("Required environment variables:")
        print("- ANTHROPIC_API_KEY: For using Anthropic Claude models")
        print("- OPENAI_API_KEY: For using OpenAI GPT models")
        print("- HUGGINGFACE_API_KEY: For using HuggingFace models")
        return
    
    try:
        # Check for presence of API keys based on the user's choice
        params = await get_user_input()
        
        # Ensure the API key for the selected provider is available
        if not await set_api_key_if_needed(params["provider"]):
            return
        
        # Execute the capability
        await execute_capability(params)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {str(e)}")


def logging_config():
    """Configure logging for debug mode."""
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers to debug
    loggers = [
        'clubhouse',
        'anthropic',
        'openai'
    ]
    
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    print("Debug logging enabled")


if __name__ == "__main__":
    asyncio.run(main())
