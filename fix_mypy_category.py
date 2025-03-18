#!/usr/bin/env python3
"""
Script to fix mypy errors by category.

This script allows targeting specific categories of mypy errors:
1. Protocol class instantiation issues
2. AgentCapability access patterns
3. get_protocol parameter fixes
4. Type annotation for function parameters
5. Dict-like attribute access vs subscript notation

Usage:
  python fix_mypy_category.py [category_number]
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

# Error categories
CATEGORIES = {
    1: "protocol_instantiation",
    2: "agent_capability_access",
    3: "get_protocol_calls",
    4: "type_annotations",
    5: "dict_attribute_access"
}

def fix_protocol_instantiation(files: List[Path]) -> int:
    """Fix protocol instantiations with concrete implementations."""
    fixed_count = 0
    for file_path in files:
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original = content
        
        # Replace protocol classes with base implementations
        replacements = [
            # Pattern, replacement
            (r"AgentInput\((.*?)\)", r"BaseAgentInput(\1)"),
            (r"AgentOutput\((.*?)\)", r"BaseAgentOutput(\1)"),
            (r"KafkaConsumer\((.*?)\)", r"BaseKafkaConsumer(\1)"),
            (r"KafkaProducer\((.*?)\)", r"BaseKafkaProducer(\1)")
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Add necessary imports
        needed_imports = []
        if "BaseAgentInput" in content and "from clubhouse.agents.base import BaseAgentInput" not in content:
            needed_imports.append("from clubhouse.agents.base import BaseAgentInput, BaseAgentOutput")
            
        if len(needed_imports) > 0:
            # Find import section
            import_section_end = 0
            for i, line in enumerate(content.split('\n')):
                if not (line.startswith('import ') or line.startswith('from ')):
                    if i > 1:  # Skip empty lines at the beginning
                        import_section_end = i - 1
                        break
            
            # Add imports
            lines = content.split('\n')
            for import_line in needed_imports:
                if import_section_end < len(lines):
                    lines.insert(import_section_end, import_line)
                    import_section_end += 1
                    
            content = '\n'.join(lines)
            
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"Fixed protocol instantiation in {file_path}")
            
    return fixed_count

def fix_agent_capability_access(files: List[Path]) -> int:
    """Fix accessing AgentCapability as enum."""
    fixed_count = 0
    for file_path in files:
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original = content
        
        # Replace AgentCapability.CAPABILITY with "CAPABILITY"
        # This assumes AgentCapabilities are registered by string names
        content = re.sub(r"AgentCapability\.([A-Z_]+)", r'"\1"', content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"Fixed capability access in {file_path}")
            
    return fixed_count

def fix_get_protocol_calls(files: List[Path]) -> int:
    """Fix get_protocol parameter types."""
    fixed_count = 0
    for file_path in files:
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original = content
        
        # Replace string-based get_protocol calls with Type-based ones
        content = re.sub(
            r'get_protocol\(["\']([^"\']+)["\']\)',
            r'get_protocol(Type[\1])',
            content
        )
        
        # Add Type to imports if needed
        if "Type[" in content and "from typing import Type" not in content:
            if "from typing import " in content:
                content = re.sub(
                    r'from typing import (.*?)',
                    r'from typing import \1, Type',
                    content
                )
            else:
                # Add import line
                import_section_end = 0
                for i, line in enumerate(content.split('\n')):
                    if not (line.startswith('import ') or line.startswith('from ')):
                        if i > 1:  # Skip empty lines at the beginning
                            import_section_end = i - 1
                            break
                
                lines = content.split('\n')
                if import_section_end < len(lines):
                    lines.insert(import_section_end, "from typing import Type")
                    content = '\n'.join(lines)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"Fixed get_protocol calls in {file_path}")
            
    return fixed_count

def fix_type_annotations(files: List[Path]) -> int:
    """Add missing type annotations."""
    fixed_count = 0
    for file_path in files:
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original = content
        
        # Find functions without type annotations
        # This is a simplified approach - a more sophisticated solution would use ast
        pattern = r'def ([^(]+)\(([^)]*)\):'
        matches = re.findall(pattern, content)
        
        # Process each match and add annotations
        for func_name, params in matches:
            # Skip if already has -> in the function signature
            if re.search(fr'def {re.escape(func_name)}\({re.escape(params)}\)\s*->', content):
                continue
                
            # Add basic Any return type
            new_signature = f'def {func_name}({params}) -> Any:'
            old_signature = f'def {func_name}({params}):'
            content = content.replace(old_signature, new_signature)
            
            # Ensure Any is imported
            if 'Any' not in content:
                if 'from typing import ' in content:
                    content = re.sub(
                        r'from typing import (.*?)',
                        r'from typing import \1, Any',
                        content
                    )
                else:
                    # Add import line
                    import_section_end = 0
                    for i, line in enumerate(content.split('\n')):
                        if not (line.startswith('import ') or line.startswith('from ')):
                            if i > 1:  # Skip empty lines at the beginning
                                import_section_end = i - 1
                                break
                    
                    lines = content.split('\n')
                    if import_section_end < len(lines):
                        lines.insert(import_section_end, "from typing import Any")
                        content = '\n'.join(lines)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"Added type annotations in {file_path}")
            
    return fixed_count

def fix_dict_attribute_access(files: List[Path]) -> int:
    """Fix dict-like attribute access issues."""
    fixed_count = 0
    for file_path in files:
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        original = content
        
        # This is a more complex fix that requires knowledge of the classes
        # For demonstration, we'll fix common patterns for AgentMessage and EnhancedAgentMessage
        dict_classes = ["AgentMessage", "EnhancedAgentMessage", "MessageContent"]
        
        for class_name in dict_classes:
            # Replace object.attribute with object["attribute"]
            # This is a simplified approach and might need manual review
            obj_pattern = fr'(\b{class_name}\b[^.]*)\.([a-zA-Z_][a-zA-Z0-9_]*)'
            content = re.sub(obj_pattern, r'\1["\2"]', content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"Fixed dict attribute access in {file_path}")
            
    return fixed_count

def find_error_files() -> List[Path]:
    """Run mypy and find files with errors."""
    import subprocess
    result = subprocess.run(
        ["mypy", "--config-file=pyproject.toml", "clubhouse/"],
        capture_output=True,
        text=True
    )
    
    error_files = set()
    for line in result.stdout.splitlines():
        if ": error:" in line:
            parts = line.split(": error:", 1)
            file_info = parts[0]
            
            # Extract filename
            file_parts = file_info.split(":", 1)
            filename = file_parts[0]
            error_files.add(Path(filename))
            
    return sorted(list(error_files))

def main():
    """Main entry point."""
    # Get category from command line
    category = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        category_num = int(sys.argv[1])
        if category_num in CATEGORIES:
            category = CATEGORIES[category_num]
    
    # Find files with errors
    print("Running mypy to find files with errors...")
    error_files = find_error_files()
    print(f"Found {len(error_files)} files with errors")
    
    # Apply fixes based on selected category
    if category == "protocol_instantiation":
        fixed = fix_protocol_instantiation(error_files)
    elif category == "agent_capability_access":
        fixed = fix_agent_capability_access(error_files)
    elif category == "get_protocol_calls":
        fixed = fix_get_protocol_calls(error_files)
    elif category == "type_annotations":
        fixed = fix_type_annotations(error_files)
    elif category == "dict_attribute_access":
        fixed = fix_dict_attribute_access(error_files)
    else:
        # If no category specified, show usage
        print("Please specify a category number:")
        for num, cat in CATEGORIES.items():
            print(f"  {num}: {cat}")
        return
    
    print(f"\nFixed {fixed} files for category '{category}'")
    
    # Run mypy again
    print("\nRunning mypy again to check progress...")
    import subprocess
    result = subprocess.run(
        ["mypy", "--config-file=pyproject.toml", "clubhouse/"],
        capture_output=True,
        text=True
    )
    
    # Count remaining errors
    remaining_errors = result.stdout.count(": error:")
    print(f"Remaining mypy errors: {remaining_errors}")

if __name__ == "__main__":
    main()
