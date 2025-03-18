# Memory System Testing Guide

## Overview

This document provides guidance for thoroughly testing the Memory System components following our test-driven development approach. It ensures 100% test coverage with strategic exclusions, proper mocking of dependencies, and integration with the broader Agent Orchestration Platform.

## Testing Principles

1. **Write Tests Before Implementation**: Follow the TDD approach by writing tests before implementing functionality
2. **Target 100% Coverage**: Aim for comprehensive test coverage of all business logic
3. **Use Proper Mocking**: Mock dependencies appropriately to isolate components
4. **Test Error Handling**: Verify all error conditions and edge cases
5. **Integration Tests**: Ensure components work together as expected

## Test Structure

### Directory Structure

```
tests/
  unit/
    memory/
      test_memory_service.py
      test_entity_extractor.py
      test_memory_deduper.py
      test_memory_summarizer.py
      repositories/
        test_vector_repository.py
        test_graph_repository.py
        test_history_repository.py
  integration/
    memory/
      test_memory_entity_extraction.py
      test_memory_kafka_integration.py
      test_memory_knowledge_graph.py
  performance/
    memory/
      test_memory_service_performance.py
      test_vector_search_performance.py
  e2e/
    memory/
      test_memory_system_e2e.py
```

## Unit Testing

### Memory Service Tests

#### Test Case: Memory Service Initialization

```python
def test_memory_service_initialization():
    """Test that the memory service initializes correctly with all dependencies."""
    # Arrange
    vector_repo_mock = Mock(spec=VectorRepositoryProtocol)
    graph_repo_mock = Mock(spec=GraphRepositoryProtocol)
    history_repo_mock = Mock(spec=HistoryRepositoryProtocol)
    event_bus_mock = Mock(spec=EventBusProtocol)
    embedding_model_mock = Mock(spec=EmbeddingModelProtocol)
    
    # Act
    memory_service = MemoryService(
        vector_repository=vector_repo_mock,
        graph_repository=graph_repo_mock,
        history_repository=history_repo_mock,
        event_bus=event_bus_mock,
        embedding_model=embedding_model_mock
    )
    
    # Assert
    assert memory_service.vector_repository == vector_repo_mock
    assert memory_service.graph_repository == graph_repo_mock
    assert memory_service.history_repository == history_repo_mock
    assert memory_service.event_bus == event_bus_mock
    assert memory_service.embedding_model == embedding_model_mock
```

#### Test Case: Add Memory

```python
def test_add_memory_success():
    """Test successful memory addition."""
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
    
    content = "Test memory content"
    filters = {"user_id": "user123", "agent_id": "agent456"}
    metadata = {"source": "test", "importance": "high"}
    
    # Act
    result = memory_service.add_memory(content, filters, metadata)
    
    # Assert
    assert result["memory_id"] == "mem_123"
    assert result["content"] == content
    
    vector_repo_mock.add.assert_called_once()
    history_repo_mock.add_history.assert_called_once()
    event_bus_mock.publish.assert_called_once_with(
        topic="memory.created",
        payload=ANY  # Use a matcher to verify payload structure
    )
```

#### Test Case: Add Memory Validation Error

```python
def test_add_memory_validation_error():
    """Test memory addition with invalid filters."""
    # Arrange
    memory_service = create_mock_memory_service()
    
    content = "Test memory content"
    filters = {}  # Empty filters should cause validation error
    
    # Act & Assert
    with pytest.raises(MemoryValidationError) as excinfo:
        memory_service.add_memory(content, filters)
    
    assert "At least one filter (user_id, agent_id, or session_id) must be provided" in str(excinfo.value)
    
    # Verify no interactions with dependencies
    memory_service.vector_repository.add.assert_not_called()
    memory_service.history_repository.add_history.assert_not_called()
    memory_service.event_bus.publish.assert_not_called()
```

#### Test Case: Search Memories

```python
def test_search_memories_success():
    """Test successful memory search."""
    # Arrange
    vector_repo_mock = Mock(spec=VectorRepositoryProtocol)
    vector_repo_mock.search.return_value = [
        {
            "memory_id": "mem_123",
            "content": "Test memory 1",
            "score": 0.95,
            "metadata": {"user_id": "user123", "created_at": "2025-03-16T10:00:00Z"}
        },
        {
            "memory_id": "mem_456",
            "content": "Test memory 2",
            "score": 0.85,
            "metadata": {"user_id": "user123", "created_at": "2025-03-16T09:00:00Z"}
        }
    ]
    
    memory_service = create_memory_service_with_mocks(vector_repository=vector_repo_mock)
    
    query = "test query"
    filters = {"user_id": "user123"}
    
    # Act
    results = memory_service.search_memories(query, filters)
    
    # Assert
    assert len(results) == 2
    assert results[0]["memory_id"] == "mem_123"
    assert results[0]["score"] == 0.95
    
    vector_repo_mock.search.assert_called_once_with(query, filters, 10)
    memory_service.event_bus.publish.assert_called_once()
```

### Repository Tests

#### Test Case: Vector Repository Add

```python
def test_vector_repository_add():
    """Test adding an item to the vector repository."""
    # Arrange
    # Use a test database or in-memory store
    config = {"collection_name": "test_memories", "dimension": 3}
    vector_repo = VectorRepository(config)
    
    content = "Test content"
    metadata = {"user_id": "user123"}
    embedding = [0.1, 0.2, 0.3]
    
    # Act
    memory_id = vector_repo.add(content, metadata, embedding)
    
    # Assert
    assert memory_id is not None
    
    # Verify the item was added
    memory = vector_repo.get(memory_id)
    assert memory["content"] == content
    assert memory["metadata"]["user_id"] == "user123"
    
    # Clean up
    vector_repo.delete(memory_id)
```

## Integration Testing

### Memory and Entity Extraction Integration

```python
def test_memory_entity_extraction_integration():
    """Test that memory creation triggers entity extraction."""
    # Arrange
    # Set up actual services with test databases
    memory_service = create_test_memory_service()
    entity_extractor = create_test_entity_extractor()
    
    # Set up Kafka consumer to capture events
    consumer = create_test_kafka_consumer("memory.entity.extracted")
    
    # Act
    memory = memory_service.add_memory(
        content="John enjoys hiking in the mountains every weekend with his dog Max.",
        filters={"user_id": "test_user"}
    )
    
    # Wait for entity extraction to complete (use polling instead of sleep in real tests)
    wait_for_event(consumer, lambda event: event["memory_id"] == memory["memory_id"], timeout=5)
    
    # Assert
    # Get extracted entities
    entities = graph_repository.get_entities_for_memory(memory["memory_id"])
    
    # Verify entities were extracted
    assert any(e["name"] == "John" and e["type"] == "person" for e in entities)
    assert any(e["name"] == "mountains" and e["type"] == "location" for e in entities)
    assert any(e["name"] == "Max" and e["type"] == "animal" for e in entities)
    
    # Verify relationships were created
    relationships = graph_repository.get_relationships_for_memory(memory["memory_id"])
    assert any(r["source_name"] == "John" and r["target_name"] == "Max" and r["type"] == "owns" for r in relationships)
```

### Kafka Integration Tests

```python
def test_memory_kafka_event_flow():
    """Test the complete flow of memory events through Kafka."""
    # Arrange
    memory_service = create_test_memory_service()
    
    # Create consumers for all memory topics
    created_consumer = create_test_kafka_consumer("memory.created")
    entity_consumer = create_test_kafka_consumer("memory.entity.extracted")
    
    # Act
    memory = memory_service.add_memory(
        content="The user prefers dark mode in all applications.",
        filters={"user_id": "test_user", "agent_id": "test_agent"}
    )
    
    # Assert - Verify memory.created event
    created_event = wait_for_event(created_consumer, lambda e: e["memory_id"] == memory["memory_id"])
    assert created_event["filters"]["user_id"] == "test_user"
    
    # Assert - Verify memory.entity.extracted event
    entity_event = wait_for_event(entity_consumer, lambda e: e["memory_id"] == memory["memory_id"])
    assert any(e["name"] == "dark mode" for e in entity_event["entities"])
    
    # Update the memory
    updated_memory = memory_service.update_memory(
        memory_id=memory["memory_id"],
        content="The user prefers dark mode in all applications and uses it exclusively at night."
    )
    
    # Assert - Verify memory.updated event
    updated_consumer = create_test_kafka_consumer("memory.updated")
    updated_event = wait_for_event(updated_consumer, lambda e: e["memory_id"] == memory["memory_id"])
    assert updated_event["memory_id"] == memory["memory_id"]
```

## Performance Testing

```python
def test_memory_service_search_performance():
    """Test the performance of memory searches with increasing data volumes."""
    # Arrange
    memory_service = create_test_memory_service()
    
    # Create test data - vary the number of memories to test scaling
    for i in range(1000):
        memory_service.add_memory(
            content=f"Test memory content {i}",
            filters={"user_id": "perf_test_user"}
        )
    
    # Act - Measure search performance
    start_time = time.time()
    results = memory_service.search_memories(
        query="test memory content",
        filters={"user_id": "perf_test_user"},
        limit=10
    )
    end_time = time.time()
    
    # Assert
    assert len(results) == 10
    assert (end_time - start_time) < 0.5  # Search should complete in under 500ms
```

## Human-in-the-Loop Testing

```python
def test_human_review_workflow():
    """Test the human review workflow for critical memories."""
    # Arrange
    memory_service = create_test_memory_service()
    review_service = create_test_review_service()
    
    # Create a memory flagged for review
    memory = memory_service.add_memory(
        content="Critical information about system security.",
        filters={"user_id": "test_user"},
        metadata={"requires_review": True}
    )
    
    # Verify the memory is in the review queue
    review_queue = review_service.get_review_queue()
    assert any(item["memory_id"] == memory["memory_id"] for item in review_queue)
    
    # Act - Simulate human review
    review_service.approve_memory(
        memory_id=memory["memory_id"],
        reviewer_id="human_reviewer",
        comments="Verified and accurate."
    )
    
    # Assert
    # Check for human review event
    consumer = create_test_kafka_consumer("memory.human.reviewed")
    review_event = wait_for_event(consumer, lambda e: e["memory_id"] == memory["memory_id"])
    
    assert review_event["reviewer_id"] == "human_reviewer"
    assert review_event["action"] == "approved"
    
    # Verify memory metadata is updated
    updated_memory = memory_service.get_memory(memory["memory_id"])
    assert updated_memory["metadata"]["review_status"] == "approved"
    assert updated_memory["metadata"]["reviewer_id"] == "human_reviewer"
```

## Mocking Strategies

### Creating Mock Dependencies

```python
def create_mock_memory_service():
    """Create a MemoryService with all dependencies mocked."""
    vector_repo_mock = Mock(spec=VectorRepositoryProtocol)
    vector_repo_mock.add.return_value = str(uuid.uuid4())
    
    graph_repo_mock = Mock(spec=GraphRepositoryProtocol)
    history_repo_mock = Mock(spec=HistoryRepositoryProtocol)
    event_bus_mock = Mock(spec=EventBusProtocol)
    embedding_model_mock = Mock(spec=EmbeddingModelProtocol)
    embedding_model_mock.embed.return_value = [random.random() for _ in range(10)]
    
    return MemoryService(
        vector_repository=vector_repo_mock,
        graph_repository=graph_repo_mock,
        history_repository=history_repo_mock,
        event_bus=event_bus_mock,
        embedding_model=embedding_model_mock
    )
```

### Testing with LLM Dependencies

```python
def test_entity_extractor_with_llm_mock():
    """Test entity extraction with a mocked LLM."""
    # Arrange
    llm_client_mock = Mock(spec=LLMClientProtocol)
    llm_client_mock.extract_entities.return_value = [
        {"name": "John", "type": "person", "confidence": 0.95},
        {"name": "mountains", "type": "location", "confidence": 0.92}
    ]
    
    entity_extractor = EntityExtractor(
        llm_client=llm_client_mock,
        graph_repository=Mock(spec=GraphRepositoryProtocol),
        event_bus=Mock(spec=EventBusProtocol)
    )
    
    content = "John enjoys hiking in the mountains."
    
    # Act
    entities = entity_extractor.extract_entities(content)
    
    # Assert
    assert len(entities) == 2
    assert entities[0]["name"] == "John"
    assert entities[1]["name"] == "mountains"
    
    llm_client_mock.extract_entities.assert_called_once_with(content)
```

## Test Fixtures

```python
@pytest.fixture
def memory_service():
    """Fixture providing a memory service with test dependencies."""
    # Set up test databases
    vector_db = create_test_vector_db()
    graph_db = create_test_graph_db()
    history_db = create_test_history_db()
    
    # Create repositories
    vector_repo = VectorRepository(vector_db)
    graph_repo = GraphRepository(graph_db)
    history_repo = HistoryRepository(history_db)
    
    # Create event bus with test broker
    event_bus = EventBus(create_test_kafka_broker())
    
    # Create embedding model
    embedding_model = TestEmbeddingModel()
    
    # Create service
    service = MemoryService(
        vector_repository=vector_repo,
        graph_repository=graph_repo,
        history_repository=history_repo,
        event_bus=event_bus,
        embedding_model=embedding_model
    )
    
    yield service
    
    # Cleanup
    vector_db.clear()
    graph_db.clear()
    history_db.clear()
```

## Test Coverage Reporting

```python
# Generate coverage report
pytest --cov=agent_orchestration_platform.memory tests/
```

Coverage targets:
- 100% coverage for core business logic
- 90%+ coverage for repositories
- Exception handling must be fully tested

Strategic exclusions:
- Logging statements
- Debug-only code blocks
- External API integration wrappers (test through integration tests)

## Continuous Integration

Configure CI pipeline to:
1. Run unit tests on every commit
2. Run integration tests on feature branch merges
3. Run performance tests nightly
4. Generate coverage reports and fail if coverage decreases

Example GitHub Actions workflow:

```yaml
name: Memory System Tests

on:
  push:
    paths:
      - 'agent_orchestration_platform/memory/**'
      - 'tests/unit/memory/**'
      - 'tests/integration/memory/**'

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      kafka:
        image: wurstmeister/kafka:latest
        ports:
          - 9092:9092
      
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
          
      - name: Run unit tests
        run: pytest tests/unit/memory --cov=agent_orchestration_platform.memory
        
      - name: Run integration tests
        run: pytest tests/integration/memory
        
      - name: Upload coverage report
        uses: codecov/codecov-action@v1
```

## Best Practices for Memory System Testing

1. **Test Isolation**: Ensure each test can run independently
2. **Cleanup After Tests**: Reset test databases after each test
3. **Controlled Dependencies**: Use test doubles for external services
4. **Realistic Data**: Use realistic test data that mimics production scenarios
5. **Error Cases**: Test all error conditions and edge cases
6. **Event Verification**: Verify all expected events are published
7. **Performance Baselines**: Establish and test against performance baselines
8. **Security Testing**: Include tests for access control and data isolation
9. **Integration Testing**: Test integration with other platform components
10. **Human Feedback Flow**: Test the complete human-in-the-loop workflow
