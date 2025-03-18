# Memory System API Reference

## Overview

This document provides a comprehensive reference for the Memory System APIs available to developers building on the Agent Orchestration Platform. The Memory System provides capabilities for storing, retrieving, and managing memories for agents, supporting context-aware interactions and evolution.

## Core APIs

### Memory Service API

#### Add Memory

```python
def add_memory(
    content: str, 
    filters: Dict[str, Any], 
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Add a new memory to the system.
    
    Args:
        content: The memory content to store.
        filters: Dictionary containing at least one of user_id, agent_id, or session_id.
        metadata: Optional additional metadata to associate with the memory.
        
    Returns:
        Dictionary containing the created memory details including memory_id.
        
    Raises:
        ValidationError: If filters don't contain required fields.
        StorageError: If memory couldn't be stored.
    """
```

**Example:**

```python
memory = memory_service.add_memory(
    content="The user prefers responses in bullet points rather than paragraphs.",
    filters={"user_id": "user123", "agent_id": "agent456"},
    metadata={"source": "user_feedback", "confidence": 0.95}
)
```

#### Search Memories

```python
def search_memories(
    query: str, 
    filters: Dict[str, Any], 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for memories based on semantic relevance to query.
    
    Args:
        query: The search query.
        filters: Dictionary containing at least one of user_id, agent_id, or session_id.
        limit: Maximum number of results to return.
        
    Returns:
        List of matching memories ordered by relevance score.
        
    Raises:
        ValidationError: If filters don't contain required fields.
        SearchError: If search operation fails.
    """
```

**Example:**

```python
memories = memory_service.search_memories(
    query="What are the user's formatting preferences?",
    filters={"user_id": "user123"},
    limit=5
)
```

#### Update Memory

```python
def update_memory(
    memory_id: str, 
    content: str, 
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Update an existing memory.
    
    Args:
        memory_id: ID of the memory to update.
        content: New memory content.
        metadata: Optional metadata to update.
        
    Returns:
        Dictionary containing the updated memory details.
        
    Raises:
        NotFoundError: If memory_id doesn't exist.
        StorageError: If update operation fails.
    """
```

**Example:**

```python
updated_memory = memory_service.update_memory(
    memory_id="mem_abc123",
    content="The user prefers responses in bullet points with emoji prefixes.",
    metadata={"confidence": 0.98}
)
```

#### Delete Memory

```python
def delete_memory(
    memory_id: str
) -> bool:
    """
    Delete a memory.
    
    Args:
        memory_id: ID of the memory to delete.
        
    Returns:
        True if deletion was successful.
        
    Raises:
        NotFoundError: If memory_id doesn't exist.
        StorageError: If deletion operation fails.
    """
```

**Example:**

```python
success = memory_service.delete_memory("mem_abc123")
```

#### Get Memory History

```python
def get_memory_history(
    memory_id: str
) -> List[Dict[str, Any]]:
    """
    Get history of changes for a memory.
    
    Args:
        memory_id: ID of the memory to retrieve history for.
        
    Returns:
        List of history entries ordered by timestamp.
        
    Raises:
        NotFoundError: If memory_id doesn't exist.
    """
```

**Example:**

```python
history = memory_service.get_memory_history("mem_abc123")
```

### Entity Extractor API

#### Extract Entities

```python
async def extract_entities(
    content: str
) -> List[Dict[str, Any]]:
    """
    Extract entities from memory content.
    
    Args:
        content: The text content to extract entities from.
        
    Returns:
        List of extracted entities with type information.
    """
```

**Example:**

```python
entities = await entity_extractor.extract_entities(
    "John prefers to receive reports every Friday via email."
)
# Returns entities like: [{"name": "John", "type": "person"}, {"name": "Friday", "type": "day"}]
```

#### Extract Relationships

```python
async def extract_relationships(
    content: str, 
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities.
    
    Args:
        content: The text content to extract relationships from.
        entities: List of entities extracted from the content.
        
    Returns:
        List of relationships between entities.
    """
```

**Example:**

```python
relationships = await entity_extractor.extract_relationships(
    content="John prefers to receive reports every Friday via email.",
    entities=[{"name": "John", "type": "person"}, {"name": "reports", "type": "document"}]
)
# Returns relationships like: [{"source": "John", "relation": "prefers", "target": "reports"}]
```

### Memory Deduper API

#### Find Similar Memories

```python
async def find_similar_memories(
    content: str, 
    threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """
    Find memories similar to the given content.
    
    Args:
        content: The content to find similar memories for.
        threshold: Similarity threshold (0.0 to 1.0).
        
    Returns:
        List of similar memories with similarity scores.
    """
```

**Example:**

```python
similar = await memory_deduper.find_similar_memories(
    content="The user wants responses formatted with bullet points.",
    threshold=0.85
)
```

#### Merge Memories

```python
async def merge_memories(
    source_id: str, 
    target_id: str
) -> Dict[str, Any]:
    """
    Merge two memories, preserving history.
    
    Args:
        source_id: ID of the source memory.
        target_id: ID of the target memory.
        
    Returns:
        Dictionary containing the merged memory details.
        
    Raises:
        NotFoundError: If either memory_id doesn't exist.
    """
```

**Example:**

```python
merged = await memory_deduper.merge_memories(
    source_id="mem_abc123",
    target_id="mem_def456"
)
```

### Memory Summarizer API

#### Summarize Memory

```python
async def summarize_memory(
    memory_id: str
) -> str:
    """
    Generate a concise summary of a memory.
    
    Args:
        memory_id: ID of the memory to summarize.
        
    Returns:
        Summary text.
        
    Raises:
        NotFoundError: If memory_id doesn't exist.
    """
```

**Example:**

```python
summary = await memory_summarizer.summarize_memory("mem_abc123")
```

#### Summarize Memories

```python
async def summarize_memories(
    memory_ids: List[str]
) -> str:
    """
    Generate a summary of multiple memories.
    
    Args:
        memory_ids: List of memory IDs to summarize.
        
    Returns:
        Summary text.
        
    Raises:
        NotFoundError: If any memory_id doesn't exist.
    """
```

**Example:**

```python
summary = await memory_summarizer.summarize_memories([
    "mem_abc123", 
    "mem_def456", 
    "mem_ghi789"
])
```

## Event System Integration

### Memory Events

The Memory System publishes events to the following Kafka topics:

#### memory.created

Published when a new memory is created.

```json
{
  "event_type": "memory.created",
  "memory_id": "mem_abc123",
  "timestamp": "2025-03-16T09:30:00Z",
  "filters": {
    "user_id": "user123",
    "agent_id": "agent456"
  },
  "metadata": {
    "source": "user_feedback",
    "confidence": 0.95
  }
}
```

#### memory.updated

Published when an existing memory is modified.

```json
{
  "event_type": "memory.updated",
  "memory_id": "mem_abc123",
  "timestamp": "2025-03-16T09:35:00Z",
  "previous_hash": "a1b2c3d4e5f6",
  "new_hash": "f6e5d4c3b2a1",
  "metadata": {
    "confidence": 0.98
  }
}
```

#### memory.deleted

Published when a memory is deleted.

```json
{
  "event_type": "memory.deleted",
  "memory_id": "mem_abc123",
  "timestamp": "2025-03-16T09:40:00Z"
}
```

#### memory.searched

Published when a memory search is performed.

```json
{
  "event_type": "memory.searched",
  "query": "user preferences",
  "timestamp": "2025-03-16T09:45:00Z",
  "filters": {
    "user_id": "user123"
  },
  "result_count": 5
}
```

#### memory.entity.extracted

Published when entities are extracted from a memory.

```json
{
  "event_type": "memory.entity.extracted",
  "memory_id": "mem_abc123",
  "timestamp": "2025-03-16T09:50:00Z",
  "entities": [
    {
      "entity_id": "ent_123",
      "name": "John",
      "type": "person",
      "confidence": 0.97
    },
    {
      "entity_id": "ent_456",
      "name": "Friday",
      "type": "day",
      "confidence": 0.99
    }
  ]
}
```

#### memory.human.reviewed

Published when a human reviews or modifies a memory.

```json
{
  "event_type": "memory.human.reviewed",
  "memory_id": "mem_abc123",
  "timestamp": "2025-03-16T09:55:00Z",
  "reviewer_id": "user789",
  "action": "approved"
}
```

## Error Handling

All Memory System APIs follow the standard error handling patterns of the Agent Orchestration Platform:

1. All errors inherit from `MemoryError` base class
2. Each operation has specific error subtypes (e.g., `MemoryNotFoundError`)
3. Errors include detailed error codes and messages
4. Validation errors provide specific field information

Example error handling:

```python
try:
    memories = memory_service.search_memories(query, filters)
except MemoryValidationError as e:
    # Handle validation errors
    print(f"Validation error: {e.detail}")
except MemoryNotFoundError as e:
    # Handle not found errors
    print(f"Memory not found: {e.memory_id}")
except MemoryError as e:
    # Handle generic memory errors
    print(f"Memory error: {e}")
```

## Security Considerations

When using the Memory System APIs, follow these security guidelines:

1. Always validate user permissions before accessing memories
2. Use appropriate filters to prevent cross-contamination between users/agents
3. Never store sensitive information (passwords, keys) in memory content
4. Apply appropriate retention policies based on data sensitivity
5. Consider encryption for highly sensitive memories

## Performance Optimization

For optimal performance:

1. Use specific filters when searching to narrow results
2. Batch memory creations when possible
3. Implement caching for frequently accessed memories
4. Consider using summarization for large memory collections
5. Monitor memory.searched events to optimize query patterns
