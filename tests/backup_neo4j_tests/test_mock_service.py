"""
Test suite for the Mock Neo4j Service.

This module provides comprehensive tests for the MockNeo4jService class,
ensuring it correctly implements the Neo4jServiceProtocol interface and
behaves correctly during various operations.
"""

import pytest
import uuid
from typing import Dict, Any, List

from clubhouse.core.config.models.database import Neo4jDatabaseConfig
from clubhouse.services.neo4j.mock_service import MockNeo4jService
from clubhouse.services.neo4j.protocol import Neo4jServiceProtocol


@pytest.fixture
def neo4j_config() -> Neo4jDatabaseConfig:
    """Fixture providing a Neo4j configuration object."""
    return Neo4jDatabaseConfig(
        name="test-neo4j",
        hosts=["bolt://localhost:7687"],
        username="neo4j",
        password="password",
        database="neo4j"
    )


@pytest.fixture
def mock_service(neo4j_config: Neo4jDatabaseConfig) -> MockNeo4jService:
    """Fixture providing an initialized MockNeo4jService."""
    service = MockNeo4jService(neo4j_config)
    service.initialize()
    return service


class TestMockNeo4jService:
    """Test suite for the MockNeo4jService class."""

    def test_initialization(self, neo4j_config: Neo4jDatabaseConfig) -> None:
        """Test that the service initializes correctly."""
        service = MockNeo4jService(neo4j_config)
        assert isinstance(service, Neo4jServiceProtocol)
        assert service._config == neo4j_config
        assert service._nodes == {}
        assert service._relationships == {}
        assert service._labels == {}
        assert service._node_relationships == {}
        assert service._constraints == {}
        assert service._indexes == {}

    def test_service_lifecycle(self, mock_service: MockNeo4jService) -> None:
        """Test the initialization and shutdown methods."""
        mock_service.initialize()  # Should not raise
        mock_service.shutdown()  # Should not raise

    def test_create_node(self, mock_service: MockNeo4jService) -> None:
        """Test creating a node."""
        # Create a node with auto-generated ID
        labels = ["Person", "User"]
        properties = {"name": "John Doe", "age": 30}
        node_id = mock_service.create_node(labels, properties)
        
        assert node_id is not None
        assert mock_service._nodes[node_id]["labels"] == labels
        assert mock_service._nodes[node_id]["properties"] == properties
        
        # Create a node with specified ID
        specified_id = str(uuid.uuid4())
        labels2 = ["Organization"]
        properties2 = {"name": "Acme Inc."}
        node_id2 = mock_service.create_node(labels2, properties2, specified_id)
        
        assert node_id2 == specified_id
        assert mock_service._nodes[node_id2]["labels"] == labels2
        assert mock_service._nodes[node_id2]["properties"] == properties2
        
        # Verify label indices - using string representation for lookup
        assert "Person" in mock_service._labels
        assert "User" in mock_service._labels
        assert "Organization" in mock_service._labels
        assert str(node_id) in mock_service._labels["Person"]
        assert str(node_id) in mock_service._labels["User"]
        assert str(node_id2) in mock_service._labels["Organization"]

    def test_get_node(self, mock_service: MockNeo4jService) -> None:
        """Test retrieving a node."""
        # Create a node
        labels = ["TestNode"]
        properties = {"name": "Test Node", "value": 42}
        node_id = mock_service.create_node(labels, properties)
        
        # Retrieve the node
        node = mock_service.get_node(node_id)
        assert node is not None
        assert node == properties
        
        # Try to retrieve a non-existent node
        non_existent_id = str(uuid.uuid4())
        assert mock_service.get_node(non_existent_id) is None
        
        # Test with UUID object - ensure node_id is in string format if it's a UUID
        node_id_str = str(node_id) if isinstance(node_id, uuid.UUID) else node_id
        uuid_obj = uuid.UUID(node_id_str)
        node2 = mock_service.get_node(uuid_obj)
        assert node2 is not None
        assert node2 == properties

    def test_update_node(self, mock_service: MockNeo4jService) -> None:
        """Test updating a node."""
        # Create a node
        labels = ["TestNode"]
        properties = {"name": "Original Name", "value": 42}
        node_id = mock_service.create_node(labels, properties)
        
        # Update the node
        update_properties = {"name": "Updated Name", "new_property": "New Value"}
        success = mock_service.update_node(node_id, update_properties)
        
        assert success is True
        updated_node = mock_service.get_node(node_id)
        assert updated_node["name"] == "Updated Name"
        assert updated_node["value"] == 42  # Original property still exists
        assert updated_node["new_property"] == "New Value"  # New property added
        
        # Try to update a non-existent node
        non_existent_id = str(uuid.uuid4())
        assert mock_service.update_node(non_existent_id, {"test": "value"}) is False

    def test_delete_node(self, mock_service: MockNeo4jService) -> None:
        """Test deleting a node."""
        # Create a node
        labels = ["TestNode"]
        properties = {"name": "Node to Delete"}
        node_id = mock_service.create_node(labels, properties)
        
        # Delete the node
        success = mock_service.delete_node(node_id)
        assert success is True
        assert mock_service.get_node(node_id) is None
        assert node_id not in mock_service._nodes
        
        # Make sure the node is removed from label indices
        assert str(node_id) not in mock_service._labels["TestNode"]
        
        # Try to delete a non-existent node
        non_existent_id = str(uuid.uuid4())
        assert mock_service.delete_node(non_existent_id) is False

    def test_delete_node_with_relationships(self, mock_service: MockNeo4jService) -> None:
        """Test that deleting a node with relationships raises an error."""
        # Create two nodes
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        
        # Create a relationship between them
        mock_service.create_relationship(node1_id, node2_id, "CONNECTED_TO")
        
        # Try to delete node1 - should raise an error
        with pytest.raises(ValueError, match="Cannot delete node with existing relationships"):
            mock_service.delete_node(node1_id)
        
        # Verify nodes still exist
        assert mock_service.get_node(node1_id) is not None
        assert mock_service.get_node(node2_id) is not None

    def test_create_relationship(self, mock_service: MockNeo4jService) -> None:
        """Test creating a relationship between nodes."""
        # Create two nodes
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        
        # Create a relationship
        rel_type = "CONNECTED_TO"
        rel_properties = {"since": "2023-01-01", "strength": 0.8}
        rel_id = mock_service.create_relationship(node1_id, node2_id, rel_type, rel_properties)
        
        # Verify relationship exists
        assert rel_id in mock_service._relationships
        rel = mock_service._relationships[rel_id]
        assert rel["start_node_id"] == node1_id
        assert rel["end_node_id"] == node2_id
        assert rel["type"] == rel_type
        assert rel["properties"] == rel_properties
        
        # Verify relationship is in node_relationships
        assert rel_id in mock_service._node_relationships[node1_id]
        assert rel_id in mock_service._node_relationships[node2_id]
        
        # Test creating relationship with non-existent start node
        non_existent_id = str(uuid.uuid4())
        with pytest.raises(ValueError, match=f"Start node with ID {non_existent_id} not found"):
            mock_service.create_relationship(non_existent_id, node2_id, rel_type)
        
        # Test creating relationship with non-existent end node
        with pytest.raises(ValueError, match=f"End node with ID {non_existent_id} not found"):
            mock_service.create_relationship(node1_id, non_existent_id, rel_type)

    def test_get_relationship(self, mock_service: MockNeo4jService) -> None:
        """Test retrieving a relationship."""
        # Create two nodes and a relationship
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        rel_type = "CONNECTED_TO"
        rel_properties = {"since": "2023-01-01"}
        rel_id = mock_service.create_relationship(node1_id, node2_id, rel_type, rel_properties)
        
        # Retrieve the relationship
        rel = mock_service.get_relationship(rel_id)
        assert rel is not None
        assert rel["id"] == rel_id
        assert rel["start_node_id"] == node1_id
        assert rel["end_node_id"] == node2_id
        assert rel["type"] == rel_type
        assert rel["since"] == "2023-01-01"
        
        # Try to retrieve a non-existent relationship
        non_existent_id = str(uuid.uuid4())
        assert mock_service.get_relationship(non_existent_id) is None

    def test_update_relationship(self, mock_service: MockNeo4jService) -> None:
        """Test updating a relationship."""
        # Create two nodes and a relationship
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        rel_type = "CONNECTED_TO"
        rel_properties = {"since": "2023-01-01"}
        rel_id = mock_service.create_relationship(node1_id, node2_id, rel_type, rel_properties)
        
        # Update the relationship
        update_properties = {"since": "2023-02-01", "strength": 0.9}
        success = mock_service.update_relationship(rel_id, update_properties)
        
        assert success is True
        rel = mock_service.get_relationship(rel_id)
        assert rel["since"] == "2023-02-01"
        assert rel["strength"] == 0.9
        
        # Try to update a non-existent relationship
        non_existent_id = str(uuid.uuid4())
        assert mock_service.update_relationship(non_existent_id, {"test": "value"}) is False

    def test_delete_relationship(self, mock_service: MockNeo4jService) -> None:
        """Test deleting a relationship."""
        # Create two nodes and a relationship
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        rel_id = mock_service.create_relationship(node1_id, node2_id, "CONNECTED_TO")
        
        # Delete the relationship
        success = mock_service.delete_relationship(rel_id)
        assert success is True
        assert rel_id not in mock_service._relationships
        
        # Verify relationship is removed from node_relationships
        assert rel_id not in mock_service._node_relationships[node1_id]
        assert rel_id not in mock_service._node_relationships[node2_id]
        
        # Try to delete a non-existent relationship
        non_existent_id = str(uuid.uuid4())
        assert mock_service.delete_relationship(non_existent_id) is False

    def test_run_query_for_agents_with_capability(self, mock_service: MockNeo4jService) -> None:
        """Test running a query to find agents with a specific capability."""
        # Create some agent nodes with capabilities
        agent1_id = mock_service.create_node(
            ["Agent"], 
            {
                "name": "Agent 1", 
                "capabilities": ["ECHO", "TEXT_TRANSFORM"]
            }
        )
        agent2_id = mock_service.create_node(
            ["Agent"], 
            {
                "name": "Agent 2", 
                "capabilities": ["ECHO"]
            }
        )
        agent3_id = mock_service.create_node(
            ["Agent"], 
            {
                "name": "Agent 3", 
                "capabilities": ["TEXT_TRANSFORM", "TEXT_ANALYSIS"]
            }
        )
        
        # Run a query to find agents with TEXT_TRANSFORM capability
        query = """
        MATCH (a:Agent)
        WHERE 'TEXT_TRANSFORM' IN a.capabilities
        RETURN a.name, a.uuid, a.capabilities
        """
        results = mock_service.run_query(query)
        
        # Verify results
        assert len(results) == 2
        agent_names = [record["a.name"] for record in results]
        assert "Agent 1" in agent_names
        assert "Agent 3" in agent_names
        assert "Agent 2" not in agent_names

    def test_run_query_for_agent_relationships(self, mock_service: MockNeo4jService) -> None:
        """Test running a query to find relationships between agents."""
        # Create agent nodes
        agent1_id = mock_service.create_node(["Agent"], {"name": "Agent 1"})
        agent2_id = mock_service.create_node(["Agent"], {"name": "Agent 2"})
        agent3_id = mock_service.create_node(["Agent"], {"name": "Agent 3"})
        
        # Create relationships between agents
        mock_service.create_relationship(
            agent1_id, agent2_id, "DEPENDS_ON", {"priority": "high"}
        )
        mock_service.create_relationship(
            agent2_id, agent3_id, "CAN_COLLABORATE_WITH", {"reason": "Complementary skills"}
        )
        
        # Run a query to find all relationships between agents
        query = """
        MATCH (a1:Agent)-[r]->(a2:Agent)
        RETURN a1.name, type(r), r, a2.name
        """
        results = mock_service.run_query(query)
        
        # Verify results
        assert len(results) == 2
        relationships = [(r["a1.name"], r["type(r)"], r["a2.name"]) for r in results]
        assert ("Agent 1", "DEPENDS_ON", "Agent 2") in relationships
        assert ("Agent 2", "CAN_COLLABORATE_WITH", "Agent 3") in relationships

    def test_get_node_relationships(self, mock_service: MockNeo4jService) -> None:
        """Test getting relationships for a node."""
        # Create nodes
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        node3_id = mock_service.create_node(["TestNode"], {"name": "Node 3"})
        
        # Create relationships
        rel1_id = mock_service.create_relationship(node1_id, node2_id, "TYPE_A")
        rel2_id = mock_service.create_relationship(node1_id, node3_id, "TYPE_B")
        rel3_id = mock_service.create_relationship(node3_id, node1_id, "TYPE_C")
        
        # Get all relationships for node1
        all_rels = mock_service.get_node_relationships(node1_id)
        assert len(all_rels) == 3
        
        # Get outgoing relationships from node1
        outgoing_rels = mock_service.get_node_relationships(node1_id, direction="outgoing")
        assert len(outgoing_rels) == 2
        rel_types = [rel["type"] for rel in outgoing_rels]
        assert "TYPE_A" in rel_types
        assert "TYPE_B" in rel_types
        
        # Get incoming relationships to node1
        incoming_rels = mock_service.get_node_relationships(node1_id, direction="incoming")
        assert len(incoming_rels) == 1
        assert incoming_rels[0]["type"] == "TYPE_C"
        
        # Get relationships of specific types
        filtered_rels = mock_service.get_node_relationships(node1_id, relationship_types=["TYPE_A"])
        assert len(filtered_rels) == 1
        assert filtered_rels[0]["type"] == "TYPE_A"
        
        # Get relationships for a non-existent node
        non_existent_id = str(uuid.uuid4())
        assert mock_service.get_node_relationships(non_existent_id) == []
        
        # Get relationships for a node with no relationships
        node4_id = mock_service.create_node(["TestNode"], {"name": "Node 4"})
        assert mock_service.get_node_relationships(node4_id) == []

    def test_get_connected_nodes(self, mock_service: MockNeo4jService) -> None:
        """Test getting nodes connected to a specific node."""
        # Create nodes with different labels
        node1_id = mock_service.create_node(["Person"], {"name": "Person 1"})
        node2_id = mock_service.create_node(["Person"], {"name": "Person 2"})
        node3_id = mock_service.create_node(["Organization"], {"name": "Org 1"})
        
        # Create relationships
        mock_service.create_relationship(node1_id, node2_id, "KNOWS")
        mock_service.create_relationship(node1_id, node3_id, "WORKS_FOR")
        
        # Get all connected nodes
        all_connected = mock_service.get_connected_nodes(node1_id)
        assert len(all_connected) == 2
        
        # Get connected nodes with specific label
        orgs = mock_service.get_connected_nodes(node1_id, node_labels=["Organization"])
        assert len(orgs) == 1
        assert orgs[0]["name"] == "Org 1"
        
        # Get connected nodes with specific relationship type
        knows_nodes = mock_service.get_connected_nodes(node1_id, relationship_types=["KNOWS"])
        assert len(knows_nodes) == 1
        assert knows_nodes[0]["name"] == "Person 2"

    def test_find_nodes(self, mock_service: MockNeo4jService) -> None:
        """Test finding nodes based on labels and properties."""
        # Create various nodes
        mock_service.create_node(["Person"], {"name": "John", "age": 30})
        mock_service.create_node(["Person"], {"name": "Jane", "age": 28})
        mock_service.create_node(["Person", "Employee"], {"name": "Bob", "age": 35})
        mock_service.create_node(["Organization"], {"name": "Acme Inc."})
        
        # Find all Person nodes
        people = mock_service.find_nodes(labels=["Person"])
        assert len(people) == 3
        
        # Find Person nodes with specific property
        young_people = mock_service.find_nodes(labels=["Person"], properties={"age": 28})
        assert len(young_people) == 1
        assert young_people[0]["name"] == "Jane"
        
        # Find nodes with multiple labels
        employees = mock_service.find_nodes(labels=["Person", "Employee"])
        assert len(employees) == 1
        assert employees[0]["name"] == "Bob"
        
        # Find with a limit
        limited_results = mock_service.find_nodes(labels=["Person"], limit=2)
        assert len(limited_results) <= 2

    def test_find_paths(self, mock_service: MockNeo4jService) -> None:
        """Test finding paths between nodes."""
        # Create nodes
        node1_id = mock_service.create_node(["TestNode"], {"name": "Node 1"})
        node2_id = mock_service.create_node(["TestNode"], {"name": "Node 2"})
        node3_id = mock_service.create_node(["TestNode"], {"name": "Node 3"})
        
        # Create a direct relationship
        mock_service.create_relationship(node1_id, node2_id, "CONNECTED_TO")
        
        # Find direct path
        paths = mock_service.find_paths(node1_id, node2_id)
        assert len(paths) == 1
        path = paths[0]
        assert len(path) == 3  # start node, relationship, end node
        assert path[0]["node"]["name"] == "Node 1"
        assert path[2]["node"]["name"] == "Node 2"
        
        # Test with relationship type filter that matches
        paths = mock_service.find_paths(node1_id, node2_id, relationship_types=["CONNECTED_TO"])
        assert len(paths) == 1
        
        # Test with relationship type filter that doesn't match
        paths = mock_service.find_paths(node1_id, node2_id, relationship_types=["WRONG_TYPE"])
        assert len(paths) == 0
        
        # Test with non-existent nodes
        non_existent_id = str(uuid.uuid4())
        assert mock_service.find_paths(non_existent_id, node2_id) == []
        assert mock_service.find_paths(node1_id, non_existent_id) == []
        
        # Test with no path between nodes
        assert mock_service.find_paths(node1_id, node3_id) == []

    def test_create_index(self, mock_service: MockNeo4jService) -> None:
        """Test creating an index."""
        # Create an index with auto-generated name
        property_names = ["name", "age"]
        index_name = mock_service.create_index("Person", property_names)
        
        assert index_name in mock_service._indexes
        assert mock_service._indexes[index_name]["label"] == "Person"
        assert mock_service._indexes[index_name]["properties"] == property_names
        
        # Create an index with specified name
        specific_name = "custom_index"
        index_name2 = mock_service.create_index("Organization", ["name"], specific_name)
        
        assert index_name2 == specific_name
        assert specific_name in mock_service._indexes
        
        # Try to create an index with existing name
        with pytest.raises(ValueError, match=f"Index with name {specific_name} already exists"):
            mock_service.create_index("AnotherLabel", ["prop"], specific_name)

    def test_create_constraint(self, mock_service: MockNeo4jService) -> None:
        """Test creating a constraint."""
        # Create a constraint with auto-generated name
        constraint_name = mock_service.create_constraint("Person", "id")
        
        assert constraint_name in mock_service._constraints
        assert mock_service._constraints[constraint_name]["label"] == "Person"
        assert mock_service._constraints[constraint_name]["property"] == "id"
        assert mock_service._constraints[constraint_name]["type"] == "uniqueness"
        
        # Create a constraint with specified name and different type
        specific_name = "custom_constraint"
        constraint_name2 = mock_service.create_constraint(
            "Organization", 
            "name", 
            constraint_type="existence", 
            constraint_name=specific_name
        )
        
        assert constraint_name2 == specific_name
        assert specific_name in mock_service._constraints
        assert mock_service._constraints[specific_name]["type"] == "existence"
        
        # Try to create a constraint with existing name
        with pytest.raises(ValueError, match=f"Constraint with name {specific_name} already exists"):
            mock_service.create_constraint("AnotherLabel", "prop", constraint_name=specific_name)
        
        # Test uniqueness constraint violation
        # Create a node with a property
        mock_service.create_node(["TestLabel"], {"unique_prop": "value1"})
        
        # Create uniqueness constraint on that property
        mock_service.create_constraint("TestLabel", "unique_prop")
        
        # Try to create another node with the same property value
        with pytest.raises(ValueError, match="Uniqueness constraint violation"):
            mock_service.create_node(["TestLabel"], {"unique_prop": "value1"})
        
        # Creating a node with a different value should work
        mock_service.create_node(["TestLabel"], {"unique_prop": "value2"})

    def test_execute_batch(self, mock_service: MockNeo4jService) -> None:
        """Test executing a batch of operations."""
        # Prepare batch operations
        node_creation_operations = [
            # Create nodes
            ("create_node", {"labels": ["BatchTest"], "properties": {"name": "Node 1"}}),
            ("create_node", {"labels": ["BatchTest"], "properties": {"name": "Node 2"}}),
        ]
        
        # Execute the node creation operations to get node IDs
        partial_results = mock_service.execute_batch(node_creation_operations)
        node1_id, node2_id = partial_results
        
        # Create the remaining operations with the node IDs
        remaining_operations = [
            # Update a node
            ("update_node", {"node_id": node1_id, "properties": {"updated": True}}),
            
            # Create a relationship
            ("create_relationship", {
                "start_node_id": node1_id,
                "end_node_id": node2_id,
                "relationship_type": "BATCH_REL",
                "properties": {"batch": True}
            }),
            
            # Run a query
            ("run_query", {"query": "MATCH (a:Agent) WHERE 'TEXT_TRANSFORM' IN a.capabilities RETURN a.name"})
        ]
        
        # Execute the remaining operations
        results = mock_service.execute_batch(remaining_operations)
        
        # Verify results
        assert len(results) == 3
        assert results[0] is True  # update_node success
        
        # Check that we got a valid relationship ID back
        relationship_id = results[1]
        assert isinstance(relationship_id, str)
        assert relationship_id in mock_service._relationships
        
        assert isinstance(results[2], list)  # query results
        
        # Verify the operations were actually executed
        node1 = mock_service.get_node(node1_id)
        assert node1["name"] == "Node 1"
        assert node1["updated"] is True
        
        # Check that the relationship exists
        rel_id = results[1]
        rel = mock_service.get_relationship(rel_id)
        assert rel["start_node_id"] == node1_id
        assert rel["end_node_id"] == node2_id
        assert rel["type"] == "BATCH_REL"
        assert rel["batch"] is True

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
