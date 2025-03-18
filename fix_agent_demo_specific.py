#!/usr/bin/env python3
"""
Script to fix specific mypy errors in agent_demo.py.
Follows the project's quality-first approach by ensuring proper type annotations
and adherence to protocol interfaces.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Match

AGENT_DEMO_PATH = Path("clubhouse/agents/examples/agent_demo.py")


def fix_agent_demo() -> None:
    """Apply targeted fixes to agent_demo.py addressing specific mypy errors."""
    if not AGENT_DEMO_PATH.exists():
        print(f"Error: {AGENT_DEMO_PATH} does not exist!")
        return

    # Read the file
    content = AGENT_DEMO_PATH.read_text()

    # 1. Fix get_service to get_protocol calls
    content = re.sub(
        r'registry\.get_service\(Neo4jServiceProtocol\)',
        r'registry.get_protocol(Neo4jServiceProtocol)',
        content
    )

    # 2. Fix BaseAgentInput with proper AgentInputType enum
    content = re.sub(
        r'import AgentCapability, AgentState',
        r'import AgentCapability, AgentState, AgentInputType',
        content
    )
    content = re.sub(
        r'BaseAgentInput\(content="([^"]+)", input_type="text"\)',
        r'BaseAgentInput(content="\1", input_type=AgentInputType.TEXT)',
        content
    )

    # 3. Fix Neo4jService instantiation - implement abstract methods
    neo4j_service_pattern = r'neo4j_service: Neo4jServiceProtocol = Neo4jService\(([^\)]+)\)'
    
    def neo4j_service_replacement(match: Match[str]) -> str:
        args = match.group(1)
        return f"""neo4j_service: Neo4jServiceProtocol = Neo4jService({args})
    
    # Implement required abstract methods
    def health_check(self) -> bool:
        return True
        
    def reset(self) -> None:
        return None"""
    
    content = re.sub(neo4j_service_pattern, neo4j_service_replacement, content)

    # 4. Fix MockNeo4jService instantiation - implement abstract methods
    mock_neo4j_pattern = r'neo4j_service: MockNeo4jService = MockNeo4jService\(\)'
    
    def mock_neo4j_replacement(match: Match[str]) -> str:
        return """neo4j_service: Neo4jServiceProtocol = MockNeo4jService()
        
    # Implement required abstract methods for MockNeo4jService
    def health_check(self) -> bool:
        return True
        
    def reset(self) -> None:
        return None"""
    
    content = re.sub(mock_neo4j_pattern, mock_neo4j_replacement, content)

    # 5. Fix list type compatibility with covariant type hint
    content = re.sub(
        r'def create_agent_relationships\(registry: ServiceRegistry, agents: List\[BaseAgent\]\)',
        r'def create_agent_relationships(registry: ServiceRegistry, agents: List[SimpleAgent])',
        content
    )
    
    # 6. Fix import for proper Neo4jConfig
    content = re.sub(
        r'from clubhouse\.core\.config\.models\.database import Neo4jConfig',
        r'from clubhouse.core.config.models.database import Neo4jDatabaseConfig as Neo4jConfig',
        content
    )

    # 7. Add import for AgentInputType
    content = re.sub(
        r'from clubhouse\.agents\.protocols import AgentCapability, AgentState',
        r'from clubhouse.agents.protocols import AgentCapability, AgentState, AgentInputType',
        content
    )

    # 8. Fix create_constraint parameter names
    content = re.sub(
        r'neo4j\.create_constraint\(\s*label="([^"]+)",\s*property_name="([^"]+)"\s*\)',
        r'neo4j.create_constraint(label="\1", property_key="\2")',
        content
    )
    
    # 9. Fix create_index parameter names
    content = re.sub(
        r'neo4j\.create_index\(\s*label="([^"]+)",\s*property_names=\[([^\]]+)\]\s*\)',
        r'neo4j.create_index(label="\1", property_keys=[\2])',
        content
    )
    
    # 10. Fix create_relationship parameter names
    content = re.sub(
        r'neo4j\.create_relationship\(\s*start_node_id=([^,]+),\s*end_node_id=([^,]+),\s*relationship_type="([^"]+)",\s*properties=({[^}]+})\s*\)',
        r'neo4j.create_relationship(from_node_id=\1, to_node_id=\2, relationship_type="\3", properties=\4)',
        content
    )

    # Write the fixed content back
    AGENT_DEMO_PATH.write_text(content)
    print(f"Fixed {AGENT_DEMO_PATH}")


if __name__ == "__main__":
    fix_agent_demo()
