#!/bin/bash
# Script to run tests with coverage for the mock Neo4j service

# Change to the project directory
cd "$(dirname "$0")"

# Run the tests with coverage
python -m pytest tests/clubhouse/services/neo4j/test_mock_service.py -v --cov=clubhouse.services.neo4j.mock_service --cov-report=term-missing

# Exit with the pytest exit code
exit $?
