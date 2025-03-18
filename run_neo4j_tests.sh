#!/bin/bash

echo "Starting Neo4j Docker container..."
docker-compose -f docker-compose-neo4j.yml up -d

# Give the container time to start
echo "Waiting for Neo4j to start..."
sleep 10

echo "Running Neo4j integration tests..."
python -m pytest tests/integration/services/test_neo4j_integration.py -v

echo "Tests completed. Cleaning up..."
docker-compose -f docker-compose-neo4j.yml down

echo "Done!"
