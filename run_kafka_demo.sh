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

# Function to check if a service is ready
function check_service() {
  local service_name="$1"
  local url="$2"
  local max_attempts=10
  local attempt=0

  echo_color "yellow" "Checking if $service_name is ready..."
  
  while [ $attempt -lt $max_attempts ]; do
    if curl -s "$url" > /dev/null; then
      echo_color "green" "$service_name is ready!"
      return 0
    fi
    
    attempt=$((attempt + 1))
    echo_color "yellow" "Attempt $attempt/$max_attempts: $service_name not ready yet, waiting..."
    sleep 2
  done
  
  echo_color "red" "Error: $service_name did not become ready in time."
  return 1
}

# Check if Schema Registry is ready
check_service "Schema Registry" "http://localhost:8081/subjects" || exit 1

# Check if Kafdrop is ready
check_service "Kafdrop" "http://localhost:9000/api/cluster" || exit 1

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo_color "blue" "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo_color "blue" "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo_color "blue" "Installing requirements..."
pip install -e .

# Run the Kafka example
echo_color "blue" "Running Kafka example producer..."
echo_color "yellow" "This will create topics and produce sample messages to Kafka."
python -m clubhouse.examples.kafka_example produce

echo_color "green" "Producer completed successfully!"
echo_color "blue" "Running Kafka example consumer..."

python -m clubhouse.examples.kafka_example consume

echo_color "green" "Demo completed successfully!"
echo_color "blue" "Services are still running. You can:"
echo_color "yellow" "- View Kafka topics at http://localhost:9000"
echo_color "yellow" "- View Schema Registry subjects at http://localhost:8081/subjects"
echo_color "yellow" "- Run the example again with: python -m clubhouse.examples.kafka_example produce|consume"
echo_color "yellow" "- Stop the services with: docker-compose down"

# Deactivate virtual environment
deactivate

echo_color "blue" "Demo environment is ready for use!"

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
