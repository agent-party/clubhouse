#!/usr/bin/env python
"""
Script to fix the remaining E501 (line too long) issues in the codebase.
"""

import os
import sys
import re
import subprocess
from pathlib import Path

def get_files_with_long_lines(directory):
    """Find Python files with lines exceeding max length."""
    result = subprocess.run(
        ["flake8", "--select=E501", directory],
        capture_output=True,
        text=True
    )
    
    files_with_issues = {}
    for line in result.stdout.splitlines():
        file_path, line_info = line.split(":", 1)
        line_num = int(line_info.split(":", 1)[0])
        
        if file_path not in files_with_issues:
            files_with_issues[file_path] = []
        
        files_with_issues[file_path].append(line_num)
    
    return files_with_issues


def fix_long_line(file_path, line_num):
    """Fix a specific long line in a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if line_num >= len(lines):
        return False
    
    line = lines[line_num]
    
    # If line isn't actually too long, skip it
    if len(line.rstrip('\n')) <= 100:
        return False
    
    # Strategies to fix long lines
    fixed_line = None
    
    # Strategy 1: Fix long strings with variables
    if '"' in line or "'" in line:
        # Look for patterns like: f"... {var} ..." or "... " + var + " ..."
        # Split into multiline f-strings or string concatenation
        match = re.search(r'(f?"[^"]*")', line)
        if match:
            indent = len(line) - len(line.lstrip())
            continuation_indent = indent + 4
            
            # Try to break at spaces in the string
            parts = re.split(r'(\s+)', line)
            new_line = ""
            current_line = " " * indent
            
            for part in parts:
                if len(current_line + part) > 99:  # Leave room for quotes and continuation
                    new_line += current_line + "\\\n"
                    current_line = " " * continuation_indent + part
                else:
                    current_line += part
            
            if current_line:
                new_line += current_line
            
            fixed_line = new_line
    
    # Strategy 2: Break long function calls into multiple lines
    if fixed_line is None and "(" in line and ")" in line and "," in line:
        # Check if this is a function call with multiple arguments
        match = re.match(r'(\s*)(.+?\()(.+)(\).*)', line)
        if match:
            indent, func_part, args_part, closing_part = match.groups()
            args = []
            
            # Very simplistic splitting - won't handle nested function calls correctly
            current_arg = ""
            paren_level = 0
            
            for char in args_part:
                if char == '(' or char == '[' or char == '{':
                    paren_level += 1
                    current_arg += char
                elif char == ')' or char == ']' or char == '}':
                    paren_level -= 1
                    current_arg += char
                elif char == ',' and paren_level == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char
            
            if current_arg:
                args.append(current_arg.strip())
            
            if len(args) > 1:
                fixed_line = indent + func_part + "\n"
                for arg in args[:-1]:
                    fixed_line += indent + "    " + arg + ",\n"
                fixed_line += indent + "    " + args[-1] + "\n"
                fixed_line += indent + closing_part
    
    # If we successfully fixed the line, update the file
    if fixed_line:
        lines[line_num] = fixed_line
        with open(file_path, 'w') as f:
            f.writelines(lines)
        return True
    
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_long_lines.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    files_with_issues = get_files_with_long_lines(directory)
    
    for file_path, line_nums in files_with_issues.items():
        print(f"Processing {file_path}")
        for line_num in sorted(line_nums, reverse=True):  # Process from bottom to top to avoid line number changes
            # Line numbers in flake8 are 1-indexed, but Python lists are 0-indexed
            if fix_long_line(file_path, line_num - 1):
                print(f"  Fixed line {line_num}")


if __name__ == "__main__":
    main()
