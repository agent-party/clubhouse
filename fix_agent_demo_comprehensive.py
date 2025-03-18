#!/usr/bin/env python3
"""
Comprehensive script to fix agent_demo.py mypy errors.

This script focuses on adhering to the project's quality-first approach by:
1. Following SOLID principles and clean code practices
2. Using Protocol interfaces for service contracts correctly
3. Adding comprehensive type annotations
4. Ensuring proper parameter names and types

The script systematically addresses all specific error categories identified
in the mypy output for agent_demo.py.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Match, Union, Callable

AGENT_DEMO_PATH = Path("clubhouse/agents/examples/agent_demo.py")


def fix_agent_demo() -> None:
    """Apply comprehensive fixes to agent_demo.py addressing all mypy errors."""
    if not AGENT_DEMO_PATH.exists():
        print(f"Error: {AGENT_DEMO_PATH} does not exist!")
        return

    # Read the file
    content = AGENT_DEMO_PATH.read_text()

    # Fix imports - ensure all necessary types are imported correctly
    content = fix_imports(content)
    
    # Fix Neo4j service instantiation - ensure proper types and abstract method implementation
    content = fix_service_instantiation(content)
    
    # Fix method parameter names and types
    content = fix_method_parameters(content)
    
    # Fix BaseAgentInput constructors
    content = fix_agent_input_usage(content)
    
    # Fix get_protocol calls
    content = fix_get_protocol_calls(content)
    
    # Write the fixed content back
    AGENT_DEMO_PATH.write_text(content)
    print(f"Fixed {AGENT_DEMO_PATH}")
    

def fix_imports(content: str) -> str:
    """Fix import statements to include all necessary types."""
    # Add AgentInputType to imports
    content = re.sub(
        r'from clubhouse\.agents\.protocols import AgentCapability, AgentState',
        r'from clubhouse.agents.protocols import AgentCapability, AgentState, AgentInputType',
        content
    )
    
    # Fix Neo4jConfig import to use Neo4jDatabaseConfig
    content = re.sub(
        r'from clubhouse\.core\.config\.models\.database import Neo4jConfig',
        r'from clubhouse.core.config.models.database import Neo4jDatabaseConfig',
        content
    )
    
    # Add missing imports for UUID and ConfigProtocol
    content = re.sub(
        r'from clubhouse\.core\.service_registry import ServiceRegistry',
        r'from clubhouse.core.service_registry import ServiceRegistry\nfrom clubhouse.core.config import ConfigProtocol\nfrom uuid import UUID, uuid4',
        content
    )
    
    return content


def fix_service_instantiation(content: str) -> str:
    """Fix Neo4j service instantiation and implement abstract methods."""
    # Fix Neo4jService instantiation with proper typing
    content = re.sub(
        r'neo4j_service: Neo4jServiceProtocol = Neo4jService\(Neo4jDatabaseConfig\(([^\)]+)\)\)',
        lambda m: f"""neo4j_service: Neo4jServiceProtocol = Neo4jService(config)

    # Create a proper config object
    config = ConfigProtocol[Neo4jDatabaseConfig]()
    config.set_value("uri", {m.group(1)})
    config.set_value("username", "neo4j")
    config.set_value("password", "password")""",
        content
    )
    
    # Fix MockNeo4jService instantiation and abstract methods
    content = re.sub(
        r'(def setup_test_environment.*?)\n(\s+)neo4j_service: Neo4jServiceProtocol = MockNeo4jService\(\)',
        r'\1\n\2# Create a mock Neo4j service and implement required abstract methods\n\2neo4j_service = MockNeo4jService()\n\2\n\2# Explicitly implement abstract methods\n\2def reset(self) -> None:\n\2    """Reset the database state."""\n\2    return None\n\2\n\2def health_check(self) -> bool:\n\2    """Check if the service is healthy."""\n\2    return True\n\2\n\2# Attach methods to the service instance\n\2neo4j_service.reset = reset.__get__(neo4j_service)\n\2neo4j_service.health_check = health_check.__get__(neo4j_service)',
        content,
        flags=re.DOTALL
    )
    
    return content


def fix_method_parameters(content: str) -> str:
    """Fix method parameter names and types to match protocol definitions."""
    # Fix create_constraint calls with correct parameter names
    content = re.sub(
        r'neo4j\.create_constraint\(\s*label="([^"]+)",\s*property_name="([^"]+)"\s*\)',
        r'neo4j.create_constraint(label="\1", property_key="\2")',
        content
    )
    
    # Fix create_index calls with correct parameter names
    content = re.sub(
        r'neo4j\.create_index\(\s*label="([^"]+)",\s*property_names=\[([^\]]+)\]\s*\)',
        r'neo4j.create_index(label="\1", property_keys=[\2])',
        content
    )
    
    # Fix create_relationship calls with correct parameter names
    content = re.sub(
        r'neo4j\.create_relationship\(\s*start_node_id=([^,]+),\s*end_node_id=([^,]+),\s*relationship_type="([^"]+)",\s*properties=({[^}]+})\s*\)',
        r'neo4j.create_relationship(from_node_id=\1, to_node_id=\2, relationship_type="\3", properties=\4)',
        content
    )
    
    # Ensure properties dict is properly typed
    content = re.sub(
        r'properties=({[^}]+})',
        r'properties=cast(Dict[str, Any], \1)',
        content
    )
    
    return content


def fix_agent_input_usage(content: str) -> str:
    """Fix BaseAgentInput constructor calls to use proper AgentInputType."""
    # Replace BaseAgentInput string type with enum
    content = re.sub(
        r'BaseAgentInput\(\s*content="([^"]+)",\s*input_type="text"\s*\)',
        r'BaseAgentInput(content="\1", input_type=AgentInputType.TEXT)',
        content
    )
    
    return content


def fix_get_protocol_calls(content: str) -> str:
    """Fix get_protocol calls to handle abstract protocol classes correctly."""
    # Properly handle abstract protocol classes in get_protocol calls
    content = re.sub(
        r'neo4j = registry\.get_protocol\(Neo4jServiceProtocol\)',
        r'neo4j = registry.get_protocol(Neo4jServiceProtocol)',
        content
    )
    
    # Add cast for abstract protocol
    content = re.sub(
        r'(neo4j) = registry\.get_protocol\(Neo4jServiceProtocol\)',
        r'\1 = cast(Neo4jServiceProtocol, registry.get_protocol(Neo4jServiceProtocol))',
        content
    )
    
    # Fix function signature for create_agent_relationships
    content = re.sub(
        r'def create_agent_relationships\(registry: ServiceRegistry, agents: List\[SimpleAgent\]\)',
        r'def create_agent_relationships(registry: ServiceRegistry, agents: List[BaseAgent])',
        content
    )
    
    # Fix the call site to use cast
    content = re.sub(
        r'create_agent_relationships\(registry, agents\)',
        r'create_agent_relationships(registry, cast(List[BaseAgent], agents))',
        content
    )
    
    return content


if __name__ == "__main__":
    fix_agent_demo()
