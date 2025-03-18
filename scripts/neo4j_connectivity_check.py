"""
Simple script to test Neo4j connectivity directly.
This bypasses the service layer to verify basic connectivity to the Neo4j container.
"""

import logging
import time
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection parameters
uri = "bolt://localhost:7687"  # Standard Neo4j Bolt URI
username = "neo4j"
password = "testpassword"

def test_connection():
    """Test direct connection to Neo4j."""
    logger.info(f"Attempting to connect to Neo4j at {uri}")
    
    # Try to connect
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info("Driver created, testing connection...")
        
        # Use a simple query to test the connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                logger.info("✅ Successfully connected to Neo4j!")
                return True
            else:
                logger.error("❌ Query executed but returned unexpected result")
                return False
                
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {str(e)}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()
            logger.info("Driver closed")

if __name__ == "__main__":
    # Try to connect multiple times with a delay
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        logger.info(f"Connection attempt {attempt}/{max_attempts}")
        if test_connection():
            break
        else:
            if attempt < max_attempts:
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect after {max_attempts} attempts")
