# Memory System Integration Guide

## Overview

This document outlines how to integrate the Memory System with other components of the Agent Orchestration Platform. It provides guidance on connecting to the event-driven architecture, utilizing memory services within capabilities, and implementing human-in-the-loop feedback mechanisms.

## Event-Driven Integration

### Event Publishing and Consumption

The Memory System integrates with the platform's event-driven architecture through Kafka topics:

| Topic | Description | Producer | Consumers |
|-------|-------------|----------|-----------|
| `memory.created` | Published when a new memory is created | Memory Service | Entity Extractor, Knowledge Graph Service, Evolution Engine |
| `memory.updated` | Published when an existing memory is modified | Memory Service | Entity Extractor, Knowledge Graph Service |
| `memory.deleted` | Published when a memory is deleted | Memory Service | Knowledge Graph Service |
| `memory.searched` | Published when a memory search is performed | Memory Service | Usage Analytics |
| `memory.entity.extracted` | Published when entities are extracted from a memory | Entity Extractor | Knowledge Graph Service |
| `memory.human.reviewed` | Published when a memory is reviewed by a human | Human Review Service | Memory Service, Evolution Engine |

### Event Schema Example

```json
{
  "memory.created": {
    "memory_id": "string",
    "filters": {
      "user_id": "string",
      "agent_id": "string",
      "session_id": "string"
    },
    "metadata": {
      "additional_property": "string"
    },
    "content": "string",
    "created_at": "string (ISO timestamp)"
  },
  
  "memory.entity.extracted": {
    "memory_id": "string",
    "entities": [
      {
        "name": "string",
        "type": "string",
        "confidence": "number",
        "attributes": {}
      }
    ],
    "relationships": [
      {
        "source": "string",
        "target": "string",
        "type": "string",
        "confidence": "number",
        "attributes": {}
      }
    ]
  }
}
```

### Integration with Service Registry

Register memory-related services in the Service Registry:

```python
from agent_orchestration_platform.common.service_registry import ServiceRegistry
from agent_orchestration_platform.memory.services import MemoryService

# Register memory service
registry = ServiceRegistry()
registry.register("memory_service", MemoryService(...))
registry.register("entity_extractor", EntityExtractor(...))
registry.register("memory_deduper", MemoryDeduper(...))
registry.register("memory_summarizer", MemorySummarizer(...))
```

## Capabilities Integration

### Using Memory in Agent Capabilities

Each capability can access the Memory System through the Service Registry:

```python
from agent_orchestration_platform.capabilities.base import BaseCapability
from agent_orchestration_platform.memory.models import MemoryFilter
from pydantic import BaseModel, Field

class PreferenceSearchParams(BaseModel):
    """Parameters for searching user preferences."""
    query: str = Field(..., description="The search query for user preferences")
    user_id: str = Field(..., description="The user ID to search preferences for")

class SearchUserPreferencesCapability(BaseCapability):
    """Capability to search for user preferences in memory."""
    
    def __init__(self, service_registry):
        super().__init__(service_registry)
        self.memory_service = service_registry.get("memory_service")
    
    async def execute(self, params: PreferenceSearchParams):
        """Execute the capability to search for user preferences."""
        # Create memory filter
        memory_filter = MemoryFilter(
            user_id=params.user_id,
            tags=["preference", "setting"]
        )
        
        # Search memories
        memories = await self.memory_service.search_memories(
            query=params.query,
            filters=memory_filter,
            limit=5
        )
        
        # Process and return results
        preferences = [
            {
                "content": memory["content"],
                "created_at": memory["metadata"]["created_at"],
                "relevance": memory["score"]
            }
            for memory in memories
        ]
        
        return {
            "preferences": preferences,
            "count": len(preferences)
        }
```

### Adding Memories from Capabilities

```python
class UserPreferenceCapability(BaseCapability):
    """Capability to record user preferences."""
    
    def __init__(self, service_registry):
        super().__init__(service_registry)
        self.memory_service = service_registry.get("memory_service")
    
    async def execute(self, params):
        """Execute the capability to record a user preference."""
        # Create memory filter
        memory_filter = {
            "user_id": params.user_id,
            "tags": ["preference"]
        }
        
        # Add memory
        memory = await self.memory_service.add_memory(
            content=f"User preference: {params.preference_type} = {params.preference_value}",
            filters=memory_filter,
            metadata={
                "preference_type": params.preference_type,
                "preference_value": params.preference_value,
                "source": "user_input"
            }
        )
        
        return {
            "memory_id": memory["memory_id"],
            "status": "stored"
        }
```

## UI Integration for Human Review

### Memory Review Interface

The platform's UI should provide an interface for human review of memories:

```typescript
// Example React component for memory review
function MemoryReviewQueue() {
  const [memories, setMemories] = useState([]);
  
  useEffect(() => {
    // Fetch memories waiting for review
    fetchMemoriesForReview().then(setMemories);
  }, []);
  
  const handleApprove = async (memoryId, comments) => {
    await reviewMemory(memoryId, 'approve', comments);
    setMemories(memories.filter(m => m.id !== memoryId));
  };
  
  const handleReject = async (memoryId, comments) => {
    await reviewMemory(memoryId, 'reject', comments);
    setMemories(memories.filter(m => m.id !== memoryId));
  };
  
  return (
    <div className="memory-review-queue">
      <h2>Memories Awaiting Review</h2>
      {memories.map(memory => (
        <MemoryReviewCard
          key={memory.id}
          memory={memory}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      ))}
    </div>
  );
}
```

### Memory Review API Endpoints

```python
@router.get("/api/memories/review-queue")
async def get_review_queue(user_id: str, review_service = Depends(get_review_service)):
    """Get memories waiting for human review."""
    return await review_service.get_review_queue(reviewer_id=user_id)

@router.post("/api/memories/{memory_id}/review")
async def review_memory(
    memory_id: str,
    review: MemoryReview,
    review_service = Depends(get_review_service)
):
    """Submit a review for a memory."""
    result = await review_service.process_review(
        memory_id=memory_id,
        reviewer_id=review.reviewer_id,
        action=review.action,
        comments=review.comments
    )
    return result
```

## Knowledge Graph Integration

### Leveraging Memory Data in Knowledge Graph

The Entity Extractor feeds data into the Knowledge Graph:

```python
class KnowledgeGraphService:
    """Service for managing the knowledge graph."""
    
    def __init__(self, graph_repository, event_bus):
        self.graph_repository = graph_repository
        self.event_bus = event_bus
        
    async def process_entities(self, memory_id, entities, relationships, metadata):
        """Process entities and relationships extracted from memory."""
        # Add entities to graph
        for entity in entities:
            await self.graph_repository.add_entity(
                entity_id=f"{entity['name']}_{entity['type']}",
                name=entity['name'],
                entity_type=entity['type'],
                attributes=entity.get('attributes', {}),
                source_memory_id=memory_id
            )
        
        # Add relationships to graph
        for rel in relationships:
            await self.graph_repository.add_relationship(
                source_id=f"{rel['source']}_{rel['source_type']}",
                target_id=f"{rel['target']}_{rel['target_type']}",
                relationship_type=rel['type'],
                attributes=rel.get('attributes', {}),
                source_memory_id=memory_id
            )
        
        # Publish graph updated event
        await self.event_bus.publish(
            topic="knowledge_graph.updated",
            payload={
                "memory_id": memory_id,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "metadata": metadata
            }
        )
```

## Evolution Engine Integration

The Memory System integrates with the Evolution Engine to drive agent improvement:

```python
class MemoryBasedEvolutionStrategy:
    """Evolution strategy based on memory analysis."""
    
    def __init__(self, memory_service, evolution_engine):
        self.memory_service = memory_service
        self.evolution_engine = evolution_engine
        
    async def analyze_user_interactions(self, agent_id, time_period):
        """Analyze user interactions to suggest agent improvements."""
        # Get relevant memories
        memories = await self.memory_service.search_memories(
            query="",
            filters={
                "agent_id": agent_id,
                "created_after": time_period.start,
                "created_before": time_period.end
            },
            limit=100
        )
        
        # Analyze patterns in memories
        patterns = await self._identify_patterns(memories)
        
        # Generate evolution suggestions
        suggestions = await self._generate_suggestions(patterns)
        
        # Submit to evolution engine
        for suggestion in suggestions:
            await self.evolution_engine.submit_evolution_proposal(
                agent_id=agent_id,
                proposal_type="memory_derived",
                proposal_content=suggestion,
                evidence=self._format_evidence(memories, patterns)
            )
        
        return {
            "analyzed_memories": len(memories),
            "identified_patterns": len(patterns),
            "generated_suggestions": len(suggestions)
        }
```

## Security Integration

### Access Control Integration

```python
class MemoryAccessControlMiddleware:
    """Middleware to enforce access control for memories."""
    
    def __init__(self, auth_service):
        self.auth_service = auth_service
        
    async def process_request(self, request, memory_service):
        """Process a memory request and enforce access control."""
        # Extract user information from request
        user_id = request.user_id
        
        # Check user permissions
        user_permissions = await self.auth_service.get_user_permissions(user_id)
        
        # Modify memory filters based on permissions
        if "memory.read.all" not in user_permissions:
            # Restrict to only user's own memories
            if "filters" not in request.params:
                request.params["filters"] = {}
            
            # Force user_id filter to match authenticated user
            request.params["filters"]["user_id"] = user_id
            
            # Remove organization-wide filters if not authorized
            if ("organization_id" in request.params["filters"] and
                "memory.read.organization" not in user_permissions):
                del request.params["filters"]["organization_id"]
        
        return request
```

## Analytics Integration

Track memory usage for analytics:

```python
class MemoryAnalyticsService:
    """Service for tracking memory usage analytics."""
    
    def __init__(self, analytics_repository, event_bus):
        self.analytics_repository = analytics_repository
        self.event_bus = event_bus
        
    async def track_memory_search(self, user_id, agent_id, query, result_count):
        """Track a memory search event."""
        await self.analytics_repository.add_event(
            event_type="memory_search",
            user_id=user_id,
            agent_id=agent_id,
            properties={
                "query": query,
                "result_count": result_count,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def track_memory_creation(self, user_id, agent_id, memory_id):
        """Track a memory creation event."""
        await self.analytics_repository.add_event(
            event_type="memory_creation",
            user_id=user_id,
            agent_id=agent_id,
            properties={
                "memory_id": memory_id,
                "timestamp": datetime.now().isoformat()
            }
        )
```

## Deployment Integration

### Service Dependencies

The Memory System depends on these infrastructure components:

1. **Vector Database**: Qdrant, Pinecone, or Weaviate for vector storage
2. **Graph Database**: Neo4j for knowledge graph storage
3. **Relational Database**: PostgreSQL for history and metadata
4. **Message Broker**: Kafka for event publishing and consumption
5. **LLM Service**: OpenAI or equivalent for entity extraction

### Kubernetes Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: memory-service
  template:
    metadata:
      labels:
        app: memory-service
    spec:
      containers:
      - name: memory-service
        image: agent-platform/memory-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        - name: VECTOR_DB_HOST
          value: "qdrant-service"
        - name: GRAPH_DB_HOST
          value: "neo4j-service"
        - name: POSTGRES_HOST
          value: "postgres-service"
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "500m"
            memory: "512Mi"
```

## Monitoring Integration

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
MEMORY_CREATION_COUNTER = Counter(
    'memory_creation_total', 
    'Total number of memories created',
    ['user_id', 'agent_id']
)

MEMORY_SEARCH_COUNTER = Counter(
    'memory_search_total', 
    'Total number of memory searches',
    ['user_id', 'agent_id']
)

MEMORY_SEARCH_LATENCY = Histogram(
    'memory_search_latency_seconds',
    'Memory search latency in seconds',
    ['vector_db', 'query_complexity']
)

# Use within MemoryService
def add_memory(self, content, filters, metadata=None):
    # ... existing code ...
    
    # Record metric
    MEMORY_CREATION_COUNTER.labels(
        user_id=filters.get('user_id', 'unknown'),
        agent_id=filters.get('agent_id', 'unknown')
    ).inc()
    
    # ... rest of the function ...
```

## Troubleshooting Integration Issues

### Common Integration Problems

1. **Kafka Connection Issues**:
   - Check Kafka broker connectivity
   - Verify topic creation and permissions
   - Ensure proper serialization/deserialization

2. **Vector Database Performance**:
   - Optimize embedding dimensions
   - Check index configuration
   - Verify server resources

3. **Entity Extraction Quality**:
   - Review LLM prompt engineering
   - Adjust confidence thresholds
   - Implement feedback loop for extraction quality

4. **Access Control Failures**:
   - Verify middleware configuration
   - Check user permission assignments
   - Test with different user roles

### Debugging Tools

```python
# Debug logger for memory operations
import logging

logger = logging.getLogger("memory_system")
logger.setLevel(logging.DEBUG)

# Add handler to log to file
file_handler = logging.FileHandler("memory_system_debug.log")
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Add to memory service methods
def search_memories(self, query, filters, limit=10):
    logger.debug(f"Memory search request: query='{query}', filters={filters}, limit={limit}")
    
    # ... implementation ...
    
    logger.debug(f"Memory search results: found {len(results)} matches")
    return results
```

## Migration Strategies

When integrating with existing systems or upgrading components:

1. **Dual Write Strategy**: For transitioning to new storage
   - Write to both old and new systems
   - Read from old system, verify against new
   - Switch reads to new system when confident
   - Eventually remove old system writes

2. **Event Log Replay**: For rebuilding indices or views
   - Maintain event log of all memory operations
   - Replay events to build new storage or indices
   - Validate new storage against current state

## Conclusion

The Memory System is designed to integrate seamlessly with all components of the Agent Orchestration Platform. By following the patterns and examples in this guide, developers can ensure proper communication between the Memory System and other services while maintaining security, performance, and reliability.
