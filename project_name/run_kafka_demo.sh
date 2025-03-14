#!/bin/bash
# Script to run the Kafka demo with Confluent Kafka and Schema Registry

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo_color "red" "Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

# Start Docker Compose environment
echo_color "blue" "Starting Kafka, Schema Registry, and Kafdrop..."
docker-compose up -d

# Wait for services to be ready
echo_color "yellow" "Waiting for services to start up..."
sleep 10

# Check if Kafka is running properly
echo_color "blue" "Checking if Kafka is ready..."
KAFKA_READY=0
for i in {1..10}; do
  if docker-compose exec kafka bash -c "kafka-topics --bootstrap-server kafka:9092 --list" > /dev/null 2>&1; then
    KAFKA_READY=1
    break
  fi
  echo_color "yellow" "Kafka not ready yet, waiting..."
  sleep 5
done

if [ $KAFKA_READY -eq 0 ]; then
  echo_color "red" "Error: Kafka is not ready after waiting. Please check docker-compose logs."
  exit 1
fi

# Check if Schema Registry is running properly
echo_color "blue" "Checking if Schema Registry is ready..."
SCHEMA_REGISTRY_READY=0
for i in {1..10}; do
  if curl -s http://localhost:8081/subjects > /dev/null 2>&1; then
    SCHEMA_REGISTRY_READY=1
    break
  fi
  echo_color "yellow" "Schema Registry not ready yet, waiting..."
  sleep 2
done

if [ $SCHEMA_REGISTRY_READY -eq 0 ]; then
  echo_color "red" "Error: Schema Registry is not ready after waiting. Please check docker-compose logs."
  exit 1
fi

echo_color "green" "All services are up and running!"

# Create topics
echo_color "blue" "Creating Kafka topics..."
docker-compose exec kafka kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists --topic example-json-topic --partitions 3 --replication-factor 1
docker-compose exec kafka kafka-topics --bootstrap-server kafka:9092 --create --if-not-exists --topic example-avro-topic --partitions 3 --replication-factor 1

echo_color "green" "Topics created successfully!"
echo_color "blue" "Available topics:"
docker-compose exec kafka kafka-topics --bootstrap-server kafka:9092 --list

# Display service info
echo_color "blue" "Services information:"
echo "Kafka: localhost:9092"
echo "Schema Registry: http://localhost:8081"
echo "Kafdrop: http://localhost:9000"

echo_color "green" "Environment is ready! You can now run the example scripts."
echo 
echo_color "yellow" "Example Commands:"
echo "# Run the Avro producer example:"
echo "python -m project_name.examples.kafka_example avro_producer"
echo
echo "# In another terminal, run the Avro consumer example:"
echo "python -m project_name.examples.kafka_example avro_consumer"
echo
echo "# For JSON examples:"
echo "python -m project_name.examples.kafka_example json_producer"
echo "python -m project_name.examples.kafka_example json_consumer"
echo
echo_color "yellow" "To run the main application:"
echo "python -m project_name"
echo
echo_color "yellow" "To shut down the environment:"
echo "./run_kafka_demo.sh stop"
echo

# Check if the first argument is "stop" to shut down the environment
if [ "$1" = "stop" ]; then
  echo_color "blue" "Stopping Kafka environment..."
  docker-compose down
  echo_color "green" "Kafka environment stopped."
  exit 0
fi

# Offer to open Kafdrop in the browser
read -p "Would you like to open Kafdrop in your browser? (y/n): " open_kafdrop
if [ "$open_kafdrop" = "y" ] || [ "$open_kafdrop" = "Y" ]; then
  if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:9000
  elif command -v open > /dev/null; then
    open http://localhost:9000
  else
    echo_color "yellow" "Couldn't open browser automatically. Please open http://localhost:9000 manually."
  fi
fi

echo_color "green" "Demo setup complete!"
