#!/bin/bash
# Script to rename the project from mcp_demo to clubhouse

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

echo_color "blue" "Starting project rename from mcp_demo to clubhouse"

# Copy Python files and update imports
echo_color "blue" "Copying and updating Python files"

find mcp_demo -name "*.py" | while read file; do
    # Get the corresponding path in the clubhouse directory
    target_file="clubhouse/${file#mcp_demo/}"
    target_dir=$(dirname "$target_file")
    
    # Create the directory if it doesn't exist
    mkdir -p "$target_dir"
    
    echo_color "yellow" "Processing $file → $target_file"
    
    # Copy and replace references
    sed 's/from mcp_demo/from clubhouse/g; s/import mcp_demo/import clubhouse/g' "$file" > "$target_file"
    
    echo_color "green" "Updated $target_file"
done

# Copy schema files
echo_color "blue" "Copying schema files"
if [ -d "mcp_demo/schemas" ]; then
    cp -r mcp_demo/schemas/* clubhouse/schemas/
    echo_color "green" "Copied schema files"
fi

# Create setup.py if it doesn't exist yet
if [ ! -f "setup.py" ]; then
    echo_color "yellow" "Creating setup.py"
    cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="clubhouse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "confluent-kafka>=2.0.0",
        "avro>=1.10.0",
        "requests>=2.25.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
    python_requires=">=3.8",
    description="Clubhouse - An intelligent agent system with Kafka integration",
    author="Clubhouse Team",
    url="https://github.com/username/clubhouse",
)
EOF
    echo_color "green" "Created setup.py"
fi

echo_color "blue" "Project rename completed successfully!"
echo_color "yellow" "Note: You should still review the codebase for any missed references"
echo_color "yellow" "Run tests to verify everything is working correctly"
