#!/usr/bin/env python
"""
Script to automatically remove unused imports across the codebase.
This is a tool to help fix F401 flake8 errors.
"""

import os
import re
import sys
from pathlib import Path

def get_python_files(directory):
    """Find all Python files in directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def get_unused_imports(file_path):
    """Run flake8 on file and extract unused imports."""
    import subprocess
    
    result = subprocess.run(
        ["flake8", "--select=F401", file_path], 
        capture_output=True, 
        text=True
    )
    
    unused_imports = []
    for line in result.stdout.splitlines():
        if "F401" in line:
            parts = line.split("F401", 1)[1].strip()
            match = re.search(r"'([^']+)'", parts)
            if match:
                unused_imports.append(match.group(1))
    
    return unused_imports


def remove_unused_imports(file_path, unused_imports):
    """Remove unused imports from file."""
    if not unused_imports:
        return False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        # Skip line if marked for removal
        if skip_next:
            skip_next = False
            continue
            
        # Handle multi-line imports
        if i < len(lines) - 1 and line.strip().startswith("from ") and "import (" in line:
            # Extract the module part
            module = line.split("from ", 1)[1].split("import", 1)[0].strip()
            
            # Find the closing parenthesis
            j = i + 1
            import_block = [line]
            import_block_end = j
            while j < len(lines) and ")" not in lines[j]:
                import_block.append(lines[j])
                j += 1
            if j < len(lines):
                import_block.append(lines[j])
                import_block_end = j
            
            # Process each import in the block
            filtered_block = [import_block[0]]
            for j in range(1, len(import_block) - 1):
                line_content = import_block[j].strip().rstrip(',')
                full_import = f"{module}.{line_content}"
                if full_import not in unused_imports and line_content not in unused_imports:
                    filtered_block.append(import_block[j])
            
            filtered_block.append(import_block[-1])
            
            # Skip to the end of the block
            new_lines.extend(filtered_block)
            skip_next = False
            i = import_block_end
            modified = True
            continue
        
        # Handle single line imports
        skip_line = False
        for unused in unused_imports:
            # Handle 'from x import y' format
            if "from " in line and "import " in line:
                module = line.split("from ", 1)[1].split("import", 1)[0].strip()
                imports = line.split("import", 1)[1].strip()
                
                # Multiple imports on one line
                if "," in imports:
                    imports_list = [imp.strip() for imp in imports.split(",")]
                    filtered_imports = []
                    
                    for imp in imports_list:
                        if f"{module}.{imp}" not in unused_imports and imp not in unused_imports:
                            filtered_imports.append(imp)
                    
                    if not filtered_imports:
                        skip_line = True
                    elif len(filtered_imports) < len(imports_list):
                        new_line = f"from {module} import {', '.join(filtered_imports)}\n"
                        new_lines.append(new_line)
                        skip_line = True
                        modified = True
                # Single import
                elif f"{module}.{imports}" == unused or imports == unused:
                    skip_line = True
                    modified = True
            
            # Handle 'import x' format
            elif line.strip().startswith("import "):
                imports = line.strip()[7:].strip()
                if imports == unused:
                    skip_line = True
                    modified = True
                elif "," in imports:
                    imports_list = [imp.strip() for imp in imports.split(",")]
                    filtered_imports = [imp for imp in imports_list if imp != unused]
                    
                    if not filtered_imports:
                        skip_line = True
                    elif len(filtered_imports) < len(imports_list):
                        new_line = f"import {', '.join(filtered_imports)}\n"
                        new_lines.append(new_line)
                        skip_line = True
                        modified = True
        
        if not skip_line:
            new_lines.append(line)
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
    
    return modified


def fix_line_length(file_path, max_length=100):
    """Fix lines that exceed max length by breaking them up."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for line in lines:
        if len(line) > max_length + 1:  # +1 for newline
            # Handle import statements
            if "import " in line:
                if "from " in line and ", " in line:
                    module = line.split("from ", 1)[1].split("import", 1)[0].strip()
                    imports = line.split("import", 1)[1].strip()
                    imports_list = [imp.strip() for imp in imports.split(",")]
                    
                    new_line = f"from {module} import (\n"
                    for imp in imports_list:
                        new_line += f"    {imp},\n"
                    new_line += ")\n"
                    
                    new_lines.append(new_line)
                    modified = True
                else:
                    # Can't easily break other imports, just keep it
                    new_lines.append(line)
            else:
                # For now, just keep long lines
                # This function could be extended with more logic for breaking up other types of long lines
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    if modified:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
    
    return modified


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_imports.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    files = get_python_files(directory)
    
    for file_path in files:
        print(f"Processing {file_path}")
        unused_imports = get_unused_imports(file_path)
        if unused_imports:
            print(f"Removing unused imports: {', '.join(unused_imports)}")
            removed = remove_unused_imports(file_path, unused_imports)
            if removed:
                print(f"Fixed imports in {file_path}")
        
        # Uncomment if you want to also fix line length issues
        # fixed_length = fix_line_length(file_path)
        # if fixed_length:
        #     print(f"Fixed line length issues in {file_path}")


if __name__ == "__main__":
    main()
