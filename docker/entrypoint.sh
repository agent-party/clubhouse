#!/bin/bash
set -e

# Wait for Kafka to be available
echo "Waiting for Kafka to be available..."
until nc -z kafka 9092; do
  echo "Kafka is unavailable - sleeping"
  sleep 1
done
echo "Kafka is up - executing command"

# Execute the command passed to the container
exec "$@"
