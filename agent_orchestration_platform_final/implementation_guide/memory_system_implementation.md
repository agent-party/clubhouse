# Memory System Implementation Guide

## Overview

This document provides guidance for developers implementing the Memory System for the Agent Orchestration Platform. It covers development approaches, integration points, and best practices following our test-driven development methodology.

## Prerequisites

Before implementing the Memory System, ensure you have:

1. A solid understanding of the platform's event-driven architecture
2. Familiarity with Kafka for event publishing and consumption
3. Knowledge of vector embeddings and semantic search
4. Experience with graph databases (preferably Neo4j)
5. Access to appropriate LLM APIs for entity extraction

## Implementation Phases

### Phase 1: Core Memory Services

#### Step 1: Set Up Data Models

1. Implement the `MemoryItem`, `MemoryFilter`, and other Pydantic models
2. Create repositories interfaces following Protocol pattern:

```python
from typing import Dict, List, Optional, Protocol, Any
import uuid
from datetime import datetime

class VectorRepositoryProtocol(Protocol):
    """Protocol for vector storage operations."""
    
    def add(self, content: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None) -> str:
        """Add content to vector storage."""
        ...
    
    def search(self, query: str, filters: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content."""
        ...
    
    def update(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Update existing content."""
        ...
    
    def delete(self, memory_id: str) -> bool:
        """Delete content by ID."""
        ...
    
    def get(self, memory_id: str) -> Dict[str, Any]:
        """Get content by ID."""
        ...
```

#### Step 2: Implement Memory Service

Create the main service class with appropriate dependency injection:

```python
class MemoryService:
    """Service for managing agent and system memories."""
    
    def __init__(
        self,
        vector_repository: VectorRepositoryProtocol,
        graph_repository: GraphRepositoryProtocol,
        history_repository: HistoryRepositoryProtocol,
        event_bus: EventBusProtocol,
        embedding_model: EmbeddingModelProtocol
    ):
        self.vector_repository = vector_repository
        self.graph_repository = graph_repository
        self.history_repository = history_repository
        self.event_bus = event_bus
        self.embedding_model = embedding_model
        
    def add_memory(self, content: str, filters: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a new memory to the system."""
        # Validate filters
        self._validate_filters(filters)
        
        # Generate hash
        content_hash = self._generate_hash(content)
        
        # Create metadata dictionary
        full_metadata = {
            **filters,
            "created_at": datetime.now().isoformat(),
            "hash": content_hash
        }
        
        if metadata:
            full_metadata.update(metadata)
            
        # Generate embedding
        embedding = self.embedding_model.embed(content)
        
        # Store in vector repository
        memory_id = self.vector_repository.add(content, full_metadata, embedding)
        
        # Store in history repository
        self.history_repository.add_history(
            memory_id=memory_id,
            previous_content=None,
            new_content=content,
            event_type="CREATE",
            metadata=full_metadata
        )
        
        # Publish event
        self.event_bus.publish(
            topic="memory.created",
            payload={
                "memory_id": memory_id,
                "filters": filters,
                "metadata": metadata
            }
        )
        
        return {
            "memory_id": memory_id,
            "content": content,
            "created_at": full_metadata["created_at"],
            "metadata": full_metadata
        }
```

#### Step 3: Implement Repository Classes

For each repository (Vector, Graph, History), implement concrete classes:

```python
class SQLiteHistoryRepository:
    """SQLite-based implementation of history repository."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_history_table()
        
    def _create_history_table(self):
        """Create the history table if it doesn't exist."""
        with self._lock:
            with self.connection:
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_history (
                        entry_id TEXT PRIMARY KEY,
                        memory_id TEXT,
                        previous_content TEXT,
                        new_content TEXT,
                        event_type TEXT,
                        timestamp DATETIME,
                        metadata TEXT
                    )
                    """
                )
```

#### Step 4: Implement Event Integration

Create Kafka producers and consumers:

```python
class KafkaEventBus:
    """Kafka implementation of event bus."""
    
    def __init__(self, bootstrap_servers: str, client_id: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def publish(self, topic: str, payload: Dict[str, Any]):
        """Publish an event to a Kafka topic."""
        future = self.producer.send(topic, payload)
        # Optional: Wait for the result
        try:
            record_metadata = future.get(timeout=10)
        except KafkaError as e:
            logger.error(f"Failed to publish event to {topic}: {e}")
            raise
```

### Phase 2: Entity Extraction and Knowledge Graph

#### Step 1: Implement Entity Extractor

Create the entity extraction service:

```python
class EntityExtractor:
    """Extracts entities and relationships from memory content."""
    
    def __init__(
        self,
        llm_client: LLMClientProtocol,
        graph_repository: GraphRepositoryProtocol,
        event_bus: EventBusProtocol
    ):
        self.llm_client = llm_client
        self.graph_repository = graph_repository
        self.event_bus = event_bus
        
    async def process_memory(self, memory_id: str, content: str, metadata: Dict[str, Any]):
        """Process a memory for entity extraction."""
        # Extract entities
        entities = await self.extract_entities(content)
        
        # Extract relationships if entities were found
        relationships = []
        if entities:
            relationships = await self.extract_relationships(content, entities)
            
        # Update knowledge graph if entities or relationships were found
        if entities or relationships:
            await self.update_knowledge_graph(memory_id, entities, relationships, metadata)
            
        # Publish event
        self.event_bus.publish(
            topic="memory.entity.extracted",
            payload={
                "memory_id": memory_id,
                "entities": entities,
                "relationships": relationships
            }
        )
        
        return {
            "entities": entities,
            "relationships": relationships
        }
```

#### Step 2: Implement Graph Integration

Create the Neo4j graph repository:

```python
class Neo4jGraphRepository:
    """Neo4j implementation of graph repository."""
    
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def add_entity(self, entity_id: str, name: str, entity_type: str, attributes: Dict[str, Any], source_memory_id: str):
        """Add an entity to the graph."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.attributes = $attributes,
                    e.source_memory_id = $source_memory_id
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                attributes=attributes,
                source_memory_id=source_memory_id
            )
```

### Phase 3: Memory Deduplication and Summarization

Implement the Memory Deduper and Summarizer classes following similar patterns.

## Testing Strategy

### Unit Tests

For each component, create comprehensive unit tests:

```python
def test_add_memory():
    """Test adding a memory."""
    # Arrange
    vector_repo_mock = Mock(spec=VectorRepositoryProtocol)
    vector_repo_mock.add.return_value = "mem_123"
    
    graph_repo_mock = Mock(spec=GraphRepositoryProtocol)
    history_repo_mock = Mock(spec=HistoryRepositoryProtocol)
    event_bus_mock = Mock(spec=EventBusProtocol)
    embedding_model_mock = Mock(spec=EmbeddingModelProtocol)
    embedding_model_mock.embed.return_value = [0.1, 0.2, 0.3]
    
    memory_service = MemoryService(
        vector_repository=vector_repo_mock,
        graph_repository=graph_repo_mock,
        history_repository=history_repo_mock,
        event_bus=event_bus_mock,
        embedding_model=embedding_model_mock
    )
    
    # Act
    result = memory_service.add_memory(
        content="Test memory",
        filters={"user_id": "user123"},
        metadata={"source": "test"}
    )
    
    # Assert
    assert result["memory_id"] == "mem_123"
    assert result["content"] == "Test memory"
    vector_repo_mock.add.assert_called_once()
    history_repo_mock.add_history.assert_called_once()
    event_bus_mock.publish.assert_called_once_with(
        topic="memory.created",
        payload=ANY  # Use hamcrest matchers for detailed verification
    )
```

### Integration Tests

Create integration tests that verify the system works across components:

```python
def test_memory_entity_extraction_integration():
    """Test the integration between memory creation and entity extraction."""
    # Arrange - set up actual repositories with test databases
    
    # Act
    memory = memory_service.add_memory(
        content="John enjoys hiking in the mountains every weekend.",
        filters={"user_id": "user123"}
    )
    
    # Wait for entity extraction to complete
    time.sleep(1)  # Use a better waiting strategy in real tests
    
    # Assert
    entities = graph_repository.get_entities_for_memory(memory["memory_id"])
    assert len(entities) >= 2  # Should extract at least "John" and "mountains"
    
    # Verify relationships
    relationships = graph_repository.get_relationships_for_memory(memory["memory_id"])
    assert len(relationships) >= 1  # Should extract "John enjoys hiking"
```

## Performance Considerations

1. **Vector Storage**: Use dimensionality reduction techniques if needed
2. **Caching**: Implement caching for frequently accessed memories
3. **Batch Processing**: Process entity extraction in batches
4. **Background Processing**: Use asynchronous processing for non-critical operations
5. **Kafka Configuration**: Tune Kafka for optimal throughput

## Security Best Practices

1. **Access Control**: Implement strict access controls based on memory filters
2. **Data Validation**: Validate all inputs to prevent injection attacks
3. **Sensitive Data**: Never store sensitive data in memory content
4. **Encryption**: Consider encrypting sensitive memories at rest
5. **Audit Logging**: Maintain comprehensive logs of memory access

## Kafka Topics Configuration

Ensure Kafka topics are properly configured:

```bash
# Create memory topics with appropriate retention and partitioning
kafka-topics.sh --create --topic memory.created --partitions 3 --replication-factor 2
kafka-topics.sh --create --topic memory.updated --partitions 3 --replication-factor 2
kafka-topics.sh --create --topic memory.deleted --partitions 3 --replication-factor 2
kafka-topics.sh --create --topic memory.searched --partitions 3 --replication-factor 2
kafka-topics.sh --create --topic memory.entity.extracted --partitions 3 --replication-factor 2
kafka-topics.sh --create --topic memory.human.reviewed --partitions 3 --replication-factor 2
```

## Common Implementation Challenges

1. **Entity Extraction Quality**: LLM prompt engineering is critical
2. **Graph Performance**: Optimize Cypher queries for complex relationship searches
3. **Vector Search Tuning**: Balance precision vs. recall in embedding similarity
4. **Duplicate Detection**: Set appropriate similarity thresholds for deduplication
5. **Event Ordering**: Ensure proper ordering of events in Kafka consumers
