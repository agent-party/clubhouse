"""
Agent demo script.

This script demonstrates how to use the agent infrastructure
and Neo4j integration to create, persist, and use agents.
"""

import logging
import os
import sys
import time
from typing import List, Optional, Type, Dict, Any, cast
from uuid import UUID

# Add parent directory to path to allow imports when running as script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from clubhouse.agents.examples.simple_agent import SimpleAgent
from clubhouse.agents.factory import AgentFactory
from clubhouse.agents.base import BaseAgent, BaseAgentInput, BaseAgentOutput
from clubhouse.agents.protocols import AgentCapability, AgentState, AgentInputType, AgentInputType, AgentInputType
from clubhouse.core.config.models.database import Neo4jDatabaseConfig, DatabaseConfig as Neo4jConfig
from clubhouse.core.service_registry import ServiceRegistry
from clubhouse.core.config import ConfigProtocol
from uuid import UUID, uuid4
from clubhouse.services.neo4j.service import Neo4jService
from clubhouse.services.neo4j.mock_service import MockNeo4jService
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_docker_compose() -> str:
    """
    Generate a docker-compose.yml file for Neo4j.
    
    Returns:
        Path to the generated docker-compose file
    """
    docker_compose_content = """
version: '3'

services:
  neo4j:
    image: neo4j:5.13.0
    container_name: clubhouse-neo4j
    environment:
      - NEO4J_AUTH=neo4j/password  # Default credentials used in the demo
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
"""
    # Create the docker-compose file
    compose_file_path = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
    with open(compose_file_path, "w") as f:
        f.write(docker_compose_content)
    
    logger.info(f"Created docker-compose file at: {compose_file_path}")
    return compose_file_path


def wait_for_neo4j(neo4j_config: Neo4jConfig, max_attempts: int = 5, retry_interval: int = 3) -> bool:
    """
    Wait for Neo4j to be available.
    
    Args:
        neo4j_config: Neo4j configuration
        max_attempts: Maximum number of connection attempts
        retry_interval: Interval between attempts in seconds
        
    Returns:
        True if Neo4j is available, False otherwise
    """
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_attempts} to connect to Neo4j at {neo4j_config.uri}...")
            
            # Try to connect to Neo4j
            driver = GraphDatabase.driver(
                neo4j_config.uri, 
                auth=(neo4j_config.username, neo4j_config.password)
            )
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            driver.close()
            logger.info("Successfully connected to Neo4j")
            return True
            
        except AuthError as e:
            logger.error(f"Authentication error: {str(e)}. Check your Neo4j credentials.")
            return False
            
        except ServiceUnavailable:
            logger.warning(f"Neo4j is not available. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)
    
    logger.error(f"Failed to connect to Neo4j after {max_attempts} attempts.")
    return False


def setup_services(use_mock: bool = True) -> Optional[ServiceRegistry]:
    """
    Set up and initialize the service registry with all required services.
    
    Args:
        use_mock: Whether to use the mock Neo4j service instead of a real one
        
    Returns:
        Initialized service registry or None if setup failed
    """
    # Create the service registry
    registry = ServiceRegistry()
    
    # Create Neo4j configuration (using defaults for demo purposes)
    neo4j_config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j"
    )
    
    try:
        if use_mock:
            # Use the mock Neo4j service
            logger.info("Using mock Neo4j service for demonstration")
            neo4j_service = MockNeo4jService(neo4j_config)
        else:
            # Check if Neo4j is available
            if not wait_for_neo4j(neo4j_config):
                logger.error("""
                Neo4j is not available. Please make sure Neo4j is running.
                
                You can start Neo4j using Docker:
                
                1. Navigate to the directory containing the docker-compose.yml file:
                   cd <path-to-docker-compose-directory>
                
                2. Start Neo4j:
                   docker-compose up -d
                   
                3. Wait a moment for Neo4j to initialize
                
                4. Try running this script again
                """)
                
                # Generate docker-compose file for convenience
                compose_file = generate_docker_compose()
                logger.info(f"A docker-compose.yml file has been generated at: {compose_file}")
                
                return None
            
            # Create and register the real Neo4j service  # type: ignore[type_assignment]
            neo4j_service = Neo4jService(neo4j_config)
        
        # Register the Neo4j service
        registry.register("neo4j", neo4j_service)
        
        # Initialize all services
        registry.initialize_all()
        
        return registry
    
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}", exc_info=True)
        return None


def setup_graph_schema(registry: ServiceRegistry) -> bool:
    """
    Set up the initial Neo4j graph schema with constraints and indexes.
    
    Args:
        registry: Service registry
    
    Returns:
        True if schema setup was successful, False otherwise
    """
    try:
        neo4j = cast(Neo4jServiceProtocol, registry.get_protocol(Neo4jServiceProtocol))
        
        # Create constraint for Agent UUIDs
        try:
            neo4j.create_constraint(
                label="Agent",
                property_name="uuid",
                constraint_type="uniqueness",
                constraint_name="unique_agent_uuid"
            )
            logger.info("Created constraint on Agent.uuid")
        except Exception as e:
            # Constraint might already exist
            logger.warning(f"Couldn't create constraint: {str(e)}")
        
        # Create indexes for performance
        try:
            neo4j.create_index(
                label="Agent",
                property_names=["name"],
                index_name="agent_name_index"
            )
            logger.info("Created index on Agent.name")
        except Exception as e:
            # Index might already exist
            logger.warning(f"Couldn't create index: {str(e)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to set up graph schema: {str(e)}", exc_info=True)
        return False


def create_example_agents(factory: AgentFactory) -> List[SimpleAgent]:
    """
    Create a set of example agents with different capabilities.
    
    Args:
        factory: Agent factory to use for creation
        
    Returns:
        List of created agents
    """
    # Create a simple echo agent
    echo_agent = factory.create_agent(
        agent_class=SimpleAgent,
        name="Echo Agent",
        description="A simple agent that echoes back input",  
        capabilities=[AgentCapability.TEXT_PROCESSING],
        tags=["simple", "echo", "demo"]
    )
    
    # Create a text processing agent
    text_processor = factory.create_agent(
        agent_class=SimpleAgent,
        name="Text Processor",
        description="An agent that can transform text to uppercase",  
        capabilities=[AgentCapability.TEXT_PROCESSING],
        tags=["text", "processor", "demo"]
    )
    
    # Create a text analyzer agent
    text_analyzer = factory.create_agent(
        agent_class=SimpleAgent,
        name="Text Analyzer",
        description="An agent that can analyze text properties",  
        capabilities=[AgentCapability.TEXT_PROCESSING, AgentCapability.REASONING],
        tags=["text", "analyzer", "demo"]
    )
    
    # Create a multi-purpose agent
    multi_agent = factory.create_agent(
        agent_class=SimpleAgent,
        name="Multi-Purpose Agent",
        description="An agent with all capabilities",
        capabilities=[AgentCapability.TEXT_PROCESSING, AgentCapability.REASONING, AgentCapability.PLANNING],
        tags=["multi", "all", "demo"],
        custom_properties={k: v for k, v in {"is_premium": True, "max_text_length": 10000}.items() if isinstance(k, str)}
    )
    
    return [echo_agent, text_processor, text_analyzer, multi_agent]


def persist_agents(factory: AgentFactory, agents: List[SimpleAgent]) -> List[UUID]:
    """
    Persist a list of agents to the database.
    
    Args:
        factory: Agent factory
        agents: List of agents to persist
        
    Returns:
        List of agent UUIDs
    """
    agent_ids = []
    
    for agent in agents:
        try:
            agent_id = factory.persist_agent(agent)
            agent_ids.append(agent_id)
        except Exception as e:
            logger.error(f"Failed to persist agent {agent.metadata.name}: {str(e)}")
    
    return agent_ids


def create_agent_relationships(registry: ServiceRegistry, agents: List[BaseAgent]) -> None:
    """
    Create relationships between agents in the graph database.
    
    Args:
        registry: Service registry
        agents: List of agents
    """
    try:
        neo4j = cast(Neo4jServiceProtocol, registry.get_protocol(Neo4jServiceProtocol))
        
        # Define some relationship types between agents
        relationships = [
            # Echo agent "CAN_COLLABORATE_WITH" Text Processor
            (str(agents[0].metadata.id), str(agents[1].metadata.id), "CAN_COLLABORATE_WITH", 
             {"reason": "Text processor enhances echo agent capabilities"}),
            
            # Text Processor "DEPENDS_ON" Echo Agent
            (str(agents[1].metadata.id), str(agents[0].metadata.id), "DEPENDS_ON", 
             {"priority": "high"}),
            
            # Text Analyzer "CAN_COLLABORATE_WITH" Text Processor
            (str(agents[2].metadata.id), str(agents[1].metadata.id), "CAN_COLLABORATE_WITH", 
             {"reason": "Processing before analysis improves results"}),
            
            # Multi-Purpose Agent "REPLACES" Echo Agent, Text Processor, and Text Analyzer
            (str(agents[3].metadata.id), str(agents[0].metadata.id), "REPLACES", 
             {"completely": True}),
            (str(agents[3].metadata.id), str(agents[1].metadata.id), "REPLACES", 
             {"completely": True}),
            (str(agents[3].metadata.id), str(agents[2].metadata.id), "REPLACES", 
             {"completely": True})
        ]
        
        # Create the relationships
        for start_id, end_id, rel_type, properties in relationships:
            try:
                neo4j.create_relationship(
                    start_node_id=start_id,
                    end_node_id=end_id,
                    relationship_type=rel_type,
                    properties=properties
                )
                logger.info(f"Created {rel_type} relationship from {start_id} to {end_id}")
            except Exception as e:
                logger.warning(f"Failed to create relationship {rel_type}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Failed to create agent relationships: {str(e)}")


def demonstrate_agent_usage(agents: List[SimpleAgent]) -> None:
    """
    Demonstrate using agents to process various inputs.
    
    Args:
        agents: List of agents to use
    """
    try:
        echo_agent, text_processor, text_analyzer, multi_agent = agents
        
        # Test the echo agent
        echo_input = BaseAgentInput(content="Hello, Agent System!", input_type=AgentInputType.TEXT)
        echo_result = echo_agent.process(echo_input)
        logger.info(f"Echo Agent result: {echo_result.content}")
        
        # Test the text processor
        uppercase_input = BaseAgentInput(content="Convert this text to uppercase please.", input_type=AgentInputType.TEXT)
        uppercase_result = text_processor.process(uppercase_input)
        logger.info(f"Text Processor result: {uppercase_result.content}")
        
        # Test the text analyzer
        count_input = BaseAgentInput(content="Count the number of words in this sample sentence.", input_type=AgentInputType.TEXT)
        count_result = text_analyzer.process(count_input)
        logger.info(f"Text Analyzer result: {count_result.content} words")
        
        # Test the multi-purpose agent with all capabilities
        multi_echo = multi_agent.process(echo_input)
        multi_upper = multi_agent.process(uppercase_input)
        multi_count = multi_agent.process(count_input)
        
        logger.info(f"Multi-Agent echo result: {multi_echo.content}")
        logger.info(f"Multi-Agent uppercase result: {multi_upper.content}")
        logger.info(f"Multi-Agent word count result: {multi_count.content} words")
    
    except Exception as e:
        logger.error(f"Error demonstrating agent usage: {str(e)}")


def run_neo4j_queries(registry: ServiceRegistry) -> None:
    """
    Run example Neo4j queries to demonstrate graph capabilities.
    
    Args:
        registry: Service registry
    """
    try:
        neo4j = cast(Neo4jServiceProtocol, registry.get_protocol(Neo4jServiceProtocol))
        
        # Query 1: Find all agents that can transform text
        logger.info("Finding agents that can transform text...")
        query1 = """
        MATCH (a:Agent)
        WHERE 'TEXT_PROCESSING' IN a.capabilities
        RETURN a.name, a.uuid, a.capabilities
        """
        result1 = neo4j.run_query(query1)
        for record in result1:
            logger.info(f"Text Transformer: {record['a.name']} ({record['a.uuid']})")
        
        # Query 2: Find all relationships between agents
        logger.info("\nFinding relationships between agents...")
        query2 = """
        MATCH (a1:Agent)-[r]->(a2:Agent)
        RETURN a1.name, type(r), r, a2.name
        """
        result2 = neo4j.run_query(query2)
        for record in result2:
            rel_properties = ", ".join([f"{k}: {v}" for k, v in record['r'].items() if k != "uuid"])
            logger.info(f"Relationship: {record['a1.name']} --[{record['type(r)']} {{{rel_properties}}}]--> {record['a2.name']}")
        
        # Query 3: Find paths between agents
        logger.info("\nFinding paths in the agent network...")
        query3 = """
        MATCH path = (a1:Agent)-[*1..3]->(a2:Agent)
        WHERE a1.name = 'Echo Agent' AND a2.name = 'Multi-Purpose Agent'
        RETURN [node in nodes(path) | node.name] AS node_names,
               [rel in relationships(path) | type(rel)] AS relationship_types
        """
        result3 = neo4j.run_query(query3)
        for record in result3:
            node_path = " -> ".join(record['node_names'])
            rel_path = " -> ".join(record['relationship_types'])
            logger.info(f"Path: {node_path}")
            logger.info(f"Relationships: {rel_path}")
    
    except Exception as e:
        logger.error(f"Error running Neo4j queries: {str(e)}")


def main() -> None:
    """Main entry point for the agent demo."""
    try:
        # Setup service registry with mock Neo4j service
        registry = setup_services(use_mock=True)
        if not registry:
            logger.error("Failed to set up services. Exiting.")
            sys.exit(1)
        
        logger.info("Services initialized successfully")
        
        # Set up graph schema
        if not setup_graph_schema(registry):
            logger.error("Failed to set up graph schema. Exiting.")
            sys.exit(1)
        
        # Create agent factory
        factory = AgentFactory(registry)
        
        # Create example agents
        agents = create_example_agents(factory)
        logger.info(f"Created {len(agents)} example agents")
        
        # Persist agents to Neo4j
        agent_ids = persist_agents(factory, agents)
        logger.info(f"Persisted agents with IDs: {agent_ids}")
        
        # Create relationships between agents
        create_agent_relationships(registry, cast(List[BaseAgent], agents))
        
        # Demonstrate agent functionality
        demonstrate_agent_usage(agents)
        
        # Run example Neo4j queries
        run_neo4j_queries(registry)
        
        # Clean up
        registry.shutdown_all()
        logger.info("Demo completed successfully")
        
        print("\n" + "="*80)
        print("Agent Demo completed successfully!")
        print("This demo used a mock Neo4j database to demonstrate the agent functionality.")
        print("In a real deployment, you would connect to a running Neo4j instance.")
        print("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"Error in agent demo: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()