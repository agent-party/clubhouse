#!/bin/bash
# Script to update test files to use clubhouse instead of mcp_demo

set -e  # Exit on error

# Function to display colored text
function echo_color() {
  local color="$1"
  local text="$2"
  case "$color" in
    "red") echo -e "\033[31m$text\033[0m" ;;
    "green") echo -e "\033[32m$text\033[0m" ;;
    "yellow") echo -e "\033[33m$text\033[0m" ;;
    "blue") echo -e "\033[34m$text\033[0m" ;;
    *) echo "$text" ;;
  esac
}

echo_color "blue" "Updating test files to use clubhouse instead of mcp_demo"

# Update all Python files in the tests directory
find tests -name "*.py" | while read file; do
    echo_color "yellow" "Processing $file"
    
    # Create a backup
    cp "$file" "${file}.bak"
    
    # Replace all occurrences of mcp_demo with clubhouse
    sed -i 's/from mcp_demo/from clubhouse/g; s/import mcp_demo/import clubhouse/g' "$file"
    
    # Replace any instances in test class names or docstrings
    sed -i 's/MCP Demo/Clubhouse/g; s/mcp_demo/clubhouse/g' "$file"
    
    echo_color "green" "Updated $file"
done

echo_color "blue" "Test file updates completed"
echo_color "yellow" "Run tests to verify changes"
