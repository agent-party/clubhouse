#!/bin/bash
# Script to run the assistant agent tests

# Change to the project directory
cd "$(dirname "$0")"

# Run all tests with coverage report
python -m pytest tests/unit/agents/test_assistant_agent.py -xvs --cov=clubhouse --cov-report=term

# Exit with the pytest exit code
exit $?
