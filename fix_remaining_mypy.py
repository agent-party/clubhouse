#!/usr/bin/env python3
"""
Script to fix remaining mypy errors in the codebase.
This script applies targeted fixes to specific files to address common error patterns.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set, Pattern


def fix_agent_demo() -> None:
    """Fix remaining issues in the agent_demo.py file."""
    file_path = Path("clubhouse/agents/examples/agent_demo.py")
    if not file_path.exists():
        print(f"Error: {file_path} does not exist!")
        return
    
    # Read the file
    content = file_path.read_text()
    
    # Fix Neo4jService assignment to MockNeo4jService
    content = re.sub(
        r'(neo4j_service: )MockNeo4jService( = Neo4jService\()',
        r'\1Neo4jServiceProtocol\2',
        content
    )
    
    # Fix create_constraint parameter names
    content = re.sub(
        r'neo4j\.create_constraint\(\s*label="([^"]+)",\s*property_name="([^"]+)"\s*\)',
        r'neo4j.create_constraint(label="\1", property_key="\2")',
        content
    )
    
    # Fix create_index parameter names
    content = re.sub(
        r'neo4j\.create_index\(\s*label="([^"]+)",\s*property_names=\[([^\]]+)\]\s*\)',
        r'neo4j.create_index(label="\1", property_keys=[\2])',
        content
    )
    
    # Fix create_relationship parameter names
    content = re.sub(
        r'neo4j\.create_relationship\(\s*start_node_id=([^,]+),\s*end_node_id=([^,]+),\s*relationship_type="([^"]+)",\s*properties=({[^}]+})\s*\)',
        r'neo4j.create_relationship(from_node_id=\1, to_node_id=\2, relationship_type="\3", properties=\4)',
        content
    )
    
    # Fix BaseAgentInput constructor calls
    content = re.sub(
        r'BaseAgentInput\(\s*content="([^"]+)"\s*\)',
        r'BaseAgentInput(content="\1", input_type="text")',
        content
    )
    
    # Fix get_protocol calls with correct type usage
    content = re.sub(
        r'registry\.get_protocol\(Neo4jServiceProtocol\)',
        r'registry.get_service(Neo4jServiceProtocol)',
        content
    )
    
    # Fix List[SimpleAgent] vs List[BaseAgent] in function call
    content = re.sub(
        r'(\s+)agents = create_agents\(registry\)',
        r'\1agents: List[BaseAgent] = create_agents(registry)',
        content
    )
    
    # Write the fixed content back
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def fix_mock_service() -> None:
    """Fix signature issues in mock_service.py file."""
    file_path = Path("clubhouse/services/neo4j/mock_service.py")
    if not file_path.exists():
        print(f"Error: {file_path} does not exist!")
        return
    
    # Read the file
    content = file_path.read_text()
    
    # Fix create_node method signature
    content = re.sub(
        r'def create_node\(self, labels: List\[str\], properties: Dict\[str, Any\], node_id: Optional\[str\] = None\) -> str:',
        r'def create_node(self, labels: Union[str, List[str]], properties: Dict[str, Any]) -> UUID:',
        content
    )
    
    # Fix implementation to match signature
    create_node_impl_pattern = r'def create_node.*?\n(\s+)""".*?"""\n.*?return node_id'
    create_node_impl_replacement = r'''def create_node(self, labels: Union[str, List[str]], properties: Dict[str, Any]) -> UUID:
        """
        Create a node in the graph with the given labels and properties.
        
        Args:
            labels: Labels for the node
            properties: Properties for the node
            
        Returns:
            UUID of the created node
        """
        # Convert string label to list for consistency
        if isinstance(labels, str):
            labels = [labels]
            
        # Generate a UUID if not provided in properties
        if "uuid" not in properties:
            properties["uuid"] = str(uuid4())
            
        # Create the node in our mock database
        node_id = str(uuid4())
        self._nodes[node_id] = {
            "labels": labels,
            "properties": properties
        }
        
        # Return the UUID (which may be different from the internal node_id)
        return UUID(properties["uuid"])'''
    
    # Use re.DOTALL to match across multiple lines
    content = re.sub(create_node_impl_pattern, create_node_impl_replacement, content, flags=re.DOTALL)
    
    # Fix update_node method signature
    content = re.sub(
        r'def update_node\(self, node_id: str, properties: Dict\[str, Any\]\) -> None:',
        r'def update_node(self, node_id: UUID, properties: Dict[str, Any]) -> None:',
        content
    )
    
    # Write the fixed content back
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def fix_redundant_cast() -> None:
    """Fix redundant casts in service_registry.py."""
    file_path = Path("clubhouse/core/service_registry.py")
    if not file_path.exists():
        print(f"Error: {file_path} does not exist!")
        return
    
    # Read the file
    content = file_path.read_text()
    
    # Fix redundant cast
    content = re.sub(
        r'return cast\(T, service\)',
        r'return service',
        content
    )
    
    # Fix no-any-return issue with correct type assertion
    content = re.sub(
        r'return registry  # type: ignore',
        r'return registry',  # Remove the type ignore comment
        content
    )
    
    # Write the fixed content back
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def fix_database_config() -> None:
    """Fix type issues in database config models."""
    file_path = Path("clubhouse/core/config/models/database.py")
    if not file_path.exists():
        print(f"Error: {file_path} does not exist!")
        return
    
    # Read the file
    content = file_path.read_text()
    
    # Fix field_validator to field_serializer where appropriate
    content = re.sub(
        r'@field_validator\(([^\)]+)\)\s*def ([^(]+)\([^,]+, [^,]+, data: dict\[Any, Any\]\)',
        r'@field_serializer(\1)\ndef \2(self, value: Any)',
        content
    )
    
    # Fix data.data to value
    content = re.sub(
        r'data\.data',
        r'value',
        content
    )
    
    # Write the fixed content back
    file_path.write_text(content)
    print(f"Fixed {file_path}")


def remove_unused_type_ignores() -> None:
    """Remove unused type: ignore comments from the codebase."""
    import subprocess
    
    # Get the list of files with unused type: ignore comments
    mypy_output = subprocess.check_output(
        ["mypy", "--config-file=pyproject.toml", "clubhouse/"], 
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Extract file paths with unused type: ignore comments
    unused_ignore_pattern = r'([^:]+):\d+: error: Unused "type: ignore" comment'
    file_paths = set(re.findall(unused_ignore_pattern, mypy_output))
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            continue
            
        # Read the file
        content = Path(file_path).read_text()
        
        # Remove unused type: ignore comments
        content = re.sub(r'\s*# type: ignore(\[[^\]]+\])?', '', content)
        
        # Write the fixed content back
        Path(file_path).write_text(content)
        print(f"Removed unused type: ignore comments from {file_path}")


def main() -> None:
    """Main function to run all fixes."""
    print("Fixing mypy errors...")
    
    # Fix specific files
    fix_agent_demo()
    fix_mock_service()
    fix_redundant_cast()
    fix_database_config()
    
    # Run mypy to check progress
    import subprocess
    subprocess.run(["mypy", "--config-file=pyproject.toml", "clubhouse/"])
    
    print("Done!")


if __name__ == "__main__":
    main()
