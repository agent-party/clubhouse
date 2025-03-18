#!/usr/bin/env python3
"""
Script to fix the agent_demo.py file to resolve mypy errors.
This script performs targeted fixes for specific error patterns.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

TARGET_FILE = Path("clubhouse/agents/examples/agent_demo.py")


def fix_agent_demo() -> None:
    """Fix agent_demo.py mypy errors efficiently."""
    if not TARGET_FILE.exists():
        print(f"Error: {TARGET_FILE} does not exist!")
        sys.exit(1)

    # Read the file
    content = TARGET_FILE.read_text()

    # Apply all fixes
    content = fix_agent_input(content)
    content = fix_create_relationship_args(content)
    content = fix_create_constraint_args(content)
    content = fix_create_index_args(content)
    content = fix_agent_capability_usage(content)
    content = fix_method_signatures(content)

    # Write the fixed content back
    TARGET_FILE.write_text(content)
    print(f"Fixed {TARGET_FILE}")


def fix_agent_input(content: str) -> str:
    """Fix BaseAgentInput parameter issues."""
    # Replace BaseAgentInput with type parameter with proper content parameter
    pattern = r'BaseAgentInput\(\s*content="([^"]+)",\s*type="([^"]+)"\s*\)'
    replacement = r'BaseAgentInput(content="\1")'
    return re.sub(pattern, replacement, content)


def fix_create_relationship_args(content: str) -> str:
    """Fix create_relationship method calls."""
    # Fix the create_relationship calls with correct parameter names
    pattern = r'neo4j\.create_relationship\(\s*start_node_id=([^,]+),\s*end_node_id=([^,]+),\s*relationship_type="([^"]+)",\s*properties=({[^}]+})\s*\)'
    replacement = r'neo4j.create_relationship(from_node_id=\1, to_node_id=\2, relationship_type="\3", properties=\4)'
    return re.sub(pattern, replacement, content)


def fix_create_constraint_args(content: str) -> str:
    """Fix create_constraint method calls."""
    # Fix the create_constraint calls with correct parameters
    pattern = r'neo4j\.create_constraint\(\s*label="([^"]+)",\s*property_name="([^"]+)"\s*\)'
    replacement = r'neo4j.create_constraint(label="\1", property_key="\2")'
    return re.sub(pattern, replacement, content)


def fix_create_index_args(content: str) -> str:
    """Fix create_index method calls."""
    # Fix the create_index calls with correct parameters
    pattern = r'neo4j\.create_index\(\s*label="([^"]+)",\s*property_names=\[([^\]]+)\]\s*\)'
    replacement = r'neo4j.create_index(label="\1", property_keys=[\2])'
    return re.sub(pattern, replacement, content)


def fix_agent_capability_usage(content: str) -> str:
    """Fix AgentCapability.ECHO usage."""
    # Replace any remaining string capabilities with enum values
    patterns_replacements = [
        (r'"ECHO"', 'AgentCapability.TEXT_PROCESSING'),
        (r'"TEXT_TRANSFORM"', 'AgentCapability.TEXT_PROCESSING'),
        (r'"TEXT_ANALYSIS"', 'AgentCapability.REASONING'),
        (r'AgentCapability.ECHO', 'AgentCapability.TEXT_PROCESSING'),
    ]
    
    for pattern, replacement in patterns_replacements:
        content = content.replace(pattern, replacement)
    
    return content


def fix_method_signatures(content: str) -> str:
    """Fix method signatures and type annotations."""
    # Change SimpleAgent to BaseAgent in function signatures
    content = re.sub(
        r'def create_agent_relationships\(registry: ServiceRegistry, agents: List\[SimpleAgent\]\)',
        r'def create_agent_relationships(registry: ServiceRegistry, agents: List[BaseAgent])',
        content
    )
    
    # Fix type mismatch in assignment (Neo4jService vs MockNeo4jService)
    content = re.sub(
        r'(neo4j_service): MockNeo4jService = Neo4jService\(',
        r'\1: Neo4jServiceProtocol = Neo4jService(',
        content
    )
    
    return content


if __name__ == "__main__":
    fix_agent_demo()
