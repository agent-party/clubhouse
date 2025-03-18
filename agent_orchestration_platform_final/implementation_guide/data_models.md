# Data Models

## Overview

This document defines the core data models used throughout the Agent Orchestration Platform. Following the quality-first principles, these models use Pydantic for validation and schema definition, providing comprehensive type annotations and ensuring data integrity.

## Agent Models

### Agent Definition

```python
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator

class AgentCapability(BaseModel):
    """Represents a capability of an agent."""
    
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what the capability does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the capability")
    implementation: str = Field(..., description="Implementation identifier")
    effectiveness: float = Field(default=0.0, ge=0.0, le=1.0, description="Measured effectiveness (0-1)")
    
    class Config:
        frozen = True

class AgentKnowledge(BaseModel):
    """Represents a knowledge source for an agent."""
    
    source_id: str = Field(..., description="Identifier for the knowledge source")
    source_type: str = Field(..., description="Type of knowledge source (document, database, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        frozen = True

class AgentDefinition(BaseModel):
    """Defines an agent's core properties and capabilities."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Human-readable name")
    domain: str = Field(..., description="Domain the agent operates in")
    description: str = Field(..., description="Detailed description of the agent's purpose")
    version: str = Field(..., description="Semantic version of the agent")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    knowledge_sources: List[AgentKnowledge] = Field(default_factory=list, description="Knowledge sources")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    
    @validator('updated_at')
    def updated_at_must_be_after_created_at(cls, v, values):
        """Validate that updated_at is not before created_at."""
        if 'created_at' in values and v < values['created_at']:
            raise ValueError('updated_at must not be before created_at')
        return v
    
    class Config:
        validate_assignment = True
```

### Agent State

```python
class AgentState(BaseModel):
    """Represents the current state of an agent."""
    
    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Current status (active, inactive, evolving, etc.)")
    last_interaction: Optional[datetime] = Field(None, description="Timestamp of last interaction")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Current performance metrics")
    active_sessions: int = Field(default=0, ge=0, description="Number of active sessions")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage metrics")
    
    class Config:
        validate_assignment = True
```

### Agent Lineage

```python
class EvolutionEvent(BaseModel):
    """Represents an evolution event in an agent's history."""
    
    event_id: str = Field(..., description="Unique identifier for the event")
    event_type: str = Field(..., description="Type of evolution event")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the event occurred")
    changes: List[Dict[str, Any]] = Field(..., description="Changes made during this event")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics associated with the event")
    
    class Config:
        frozen = True

class AgentLineage(BaseModel):
    """Represents the evolutionary lineage of an agent."""
    
    agent_id: str = Field(..., description="Agent identifier")
    parent_id: Optional[str] = Field(None, description="Parent agent identifier, if any")
    root_id: str = Field(..., description="Root ancestor agent identifier")
    generation: int = Field(default=0, ge=0, description="Generation number in evolution")
    evolution_path: List[EvolutionEvent] = Field(default_factory=list, description="History of evolution events")
    
    class Config:
        validate_assignment = True
```

## Evolution Models

### Evolution Process

```python
class SelectionCriteria(BaseModel):
    """Criteria for selecting evolution candidates."""
    
    metric_name: str = Field(..., description="Name of the metric")
    weight: float = Field(..., gt=0.0, le=1.0, description="Weight of this metric in selection")
    optimization: str = Field(..., description="Whether to maximize or minimize this metric")
    threshold: Optional[float] = Field(None, description="Optional threshold value")
    
    @validator('optimization')
    def validate_optimization(cls, v):
        """Validate optimization value."""
        if v not in ['maximize', 'minimize']:
            raise ValueError('optimization must be either "maximize" or "minimize"')
        return v
    
    class Config:
        frozen = True

class EvolutionSpecification(BaseModel):
    """Specification for an evolution process."""
    
    evolution_id: str = Field(..., description="Unique identifier for the evolution process")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description of evolution goals")
    domain: str = Field(..., description="Domain for evolution")
    target_capabilities: List[str] = Field(..., min_items=1, description="Capabilities to evolve")
    population_size: int = Field(..., gt=0, description="Size of the agent population")
    max_generations: int = Field(..., gt=0, description="Maximum number of generations")
    selection_criteria: List[SelectionCriteria] = Field(..., min_items=1, description="Criteria for selection")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        validate_assignment = True
```

### Evolution Experiment

```python
class VariationStrategy(BaseModel):
    """Strategy for creating agent variations."""
    
    strategy_type: str = Field(..., description="Type of variation strategy")
    target_capability: str = Field(..., description="Capability to vary")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    
    class Config:
        frozen = True

class ExperimentSpecification(BaseModel):
    """Specification for an evolution experiment."""
    
    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    evolution_id: str = Field(..., description="Parent evolution process")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    baseline_agent_id: str = Field(..., description="Agent to use as baseline")
    variation_strategies: List[VariationStrategy] = Field(..., min_items=1, description="Strategies for creating variations")
    evaluation_metrics: List[str] = Field(..., min_items=1, description="Metrics to evaluate")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        validate_assignment = True

class ExperimentResult(BaseModel):
    """Results of an evolution experiment."""
    
    experiment_id: str = Field(..., description="Experiment identifier")
    status: str = Field(..., description="Status of the experiment")
    variations: List[str] = Field(default_factory=list, description="Agent variation IDs")
    metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Evaluation metrics by variation")
    selected_variation: Optional[str] = Field(None, description="Selected variation, if any")
    completion_time: Optional[datetime] = Field(None, description="When the experiment completed")
    
    class Config:
        validate_assignment = True
```

## Interaction Models

### Session

```python
class Message(BaseModel):
    """Message in an interaction session."""
    
    message_id: str = Field(..., description="Unique identifier for the message")
    session_id: str = Field(..., description="Session identifier")
    role: str = Field(..., description="Role of the message sender (user, agent, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was sent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate role value."""
        if v not in ['user', 'agent', 'system']:
            raise ValueError('role must be one of: user, agent, system')
        return v
    
    class Config:
        frozen = True

class Session(BaseModel):
    """Interaction session between a user and an agent."""
    
    session_id: str = Field(..., description="Unique identifier for the session")
    agent_id: str = Field(..., description="Agent identifier")
    user_id: str = Field(..., description="User identifier")
    status: str = Field(..., description="Session status (active, completed, etc.)")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    ended_at: Optional[datetime] = Field(None, description="End timestamp, if completed")
    messages: List[Message] = Field(default_factory=list, description="Session messages")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")
    
    @validator('ended_at')
    def ended_at_must_be_after_created_at(cls, v, values):
        """Validate that ended_at is not before created_at."""
        if v is not None and 'created_at' in values and v < values['created_at']:
            raise ValueError('ended_at must not be before created_at')
        return v
    
    class Config:
        validate_assignment = True
```

### Thread Management

```python
class Thread(BaseModel):
    """OpenAI thread with associated metadata."""
    
    thread_id: str = Field(..., description="OpenAI thread identifier")
    session_id: str = Field(..., description="Associated session identifier")
    agent_id: str = Field(..., description="Associated agent identifier")
    user_id: str = Field(..., description="Associated user identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        frozen = True

class RunState(BaseModel):
    """State of an OpenAI assistant run."""
    
    run_id: str = Field(..., description="OpenAI run identifier")
    thread_id: str = Field(..., description="OpenAI thread identifier")
    assistant_id: str = Field(..., description="OpenAI assistant identifier")
    status: str = Field(..., description="Current status of the run")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp, if completed")
    required_action: Optional[Dict[str, Any]] = Field(None, description="Required action, if any")
    last_error: Optional[Dict[str, Any]] = Field(None, description="Last error, if any")
    
    class Config:
        validate_assignment = True
```

## Memory Models

### Memory Item

```python
class MemoryItem(BaseModel):
    """Represents a single memory item in the system."""
    
    memory_id: str = Field(..., description="Unique identifier for the memory")
    content: str = Field(..., description="Memory content")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    hash: str = Field(..., description="Content hash for deduplication")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('updated_at')
    def updated_at_must_be_after_created_at(cls, v, values):
        """Validate that updated_at is not before created_at."""
        if v and 'created_at' in values and v < values['created_at']:
            raise ValueError('updated_at must not be before created_at')
        return v
    
    class Config:
        validate_assignment = True
```

### Memory Filter

```python
class MemoryFilter(BaseModel):
    """Filter criteria for memory retrieval."""
    
    user_id: Optional[str] = Field(None, description="User identifier filter")
    agent_id: Optional[str] = Field(None, description="Agent identifier filter")
    session_id: Optional[str] = Field(None, description="Session identifier filter")
    run_id: Optional[str] = Field(None, description="Run identifier filter")
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata filter")
    
    @validator('end_date')
    def end_date_must_be_after_start_date(cls, v, values):
        """Validate that end_date is not before start_date."""
        if v and 'start_date' in values and values['start_date'] and v < values['start_date']:
            raise ValueError('end_date must not be before start_date')
        return v
    
    class Config:
        validate_assignment = True
```

### Memory Search Result

```python
class MemorySearchResult(BaseModel):
    """Result of a memory search operation."""
    
    memory_id: str = Field(..., description="Memory identifier")
    content: str = Field(..., description="Memory content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Memory metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        frozen = True
```

### Entity and Relationship Models

```python
class Entity(BaseModel):
    """Entity extracted from memory content."""
    
    entity_id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    source_memory_id: str = Field(..., description="Memory ID where entity was extracted from")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for extraction")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        validate_assignment = True

class Relationship(BaseModel):
    """Relationship between entities extracted from memory content."""
    
    relationship_id: str = Field(..., description="Unique identifier for the relationship")
    source_entity_id: str = Field(..., description="Source entity identifier")
    target_entity_id: str = Field(..., description="Target entity identifier")
    type: str = Field(..., description="Relationship type")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Relationship attributes")
    source_memory_id: str = Field(..., description="Memory ID where relationship was extracted from")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for extraction")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        validate_assignment = True
```

### Memory History

```python
class MemoryHistoryEntry(BaseModel):
    """Entry in a memory's version history."""
    
    entry_id: str = Field(..., description="Unique identifier for the history entry")
    memory_id: str = Field(..., description="Memory identifier")
    previous_content: Optional[str] = Field(None, description="Previous memory content")
    new_content: Optional[str] = Field(None, description="New memory content")
    event_type: str = Field(..., description="Event type (CREATE, UPDATE, DELETE)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the event occurred")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('event_type')
    def validate_event_type(cls, v):
        """Validate event type value."""
        if v not in ['CREATE', 'UPDATE', 'DELETE']:
            raise ValueError('event_type must be one of "CREATE", "UPDATE", or "DELETE"')
        return v
    
    class Config:
        frozen = True
```

## Feedback Models

### Feedback

```python
class Rating(BaseModel):
    """Rating in a feedback submission."""
    
    metric: str = Field(..., description="Metric being rated")
    value: float = Field(..., description="Rating value")
    scale: str = Field(..., description="Scale used for the rating")
    
    class Config:
        frozen = True

class Feedback(BaseModel):
    """User feedback for an agent interaction."""
    
    feedback_id: str = Field(..., description="Unique identifier for the feedback")
    session_id: str = Field(..., description="Associated session identifier")
    agent_id: str = Field(..., description="Agent identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the feedback was submitted")
    ratings: List[Rating] = Field(..., min_items=1, description="Numeric ratings")
    comments: Optional[str] = Field(None, description="Textual comments")
    categories: List[str] = Field(default_factory=list, description="Feedback categories")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context information")
    
    class Config:
        frozen = True
```

### Feedback Analysis

```python
class FeedbackInsight(BaseModel):
    """Insight derived from feedback analysis."""
    
    insight_id: str = Field(..., description="Unique identifier for the insight")
    agent_id: str = Field(..., description="Agent identifier")
    metric: str = Field(..., description="Related metric")
    trend: str = Field(..., description="Trend direction (improving, declining, stable)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the insight (0-1)")
    description: str = Field(..., description="Human-readable description")
    supporting_data: Dict[str, Any] = Field(default_factory=dict, description="Supporting data")
    
    class Config:
        frozen = True

class AdaptationPlan(BaseModel):
    """Plan for adapting an agent based on feedback."""
    
    plan_id: str = Field(..., description="Unique identifier for the plan")
    agent_id: str = Field(..., description="Agent identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    insights: List[str] = Field(..., min_items=1, description="Insight identifiers")
    adaptations: List[Dict[str, Any]] = Field(..., min_items=1, description="Planned adaptations")
    expected_improvements: Dict[str, float] = Field(default_factory=dict, description="Expected metric improvements")
    priority: int = Field(default=1, ge=1, le=5, description="Priority level (1-5)")
    
    class Config:
        validate_assignment = True
```

## Event Models

### Event

```python
class Event(BaseModel):
    """Base event model for the system."""
    
    event_id: str = Field(..., description="Unique identifier for the event")
    event_type: str = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the event occurred")
    source: str = Field(..., description="Source of the event")
    payload: Dict[str, Any] = Field(..., description="Event payload")
    
    class Config:
        frozen = True
```

### Agent Events

```python
class AgentCreatedEvent(Event):
    """Event emitted when a new agent is created."""
    
    event_type: str = "agent_created"
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate required payload fields."""
        required_fields = ['agent_id', 'name', 'domain']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class AgentEvolvedEvent(Event):
    """Event emitted when an agent evolves."""
    
    event_type: str = "agent_evolved"
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate required payload fields."""
        required_fields = ['agent_id', 'parent_id', 'evolution_id', 'changes']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v
```

### Evolution Events

```python
class EvolutionStartedEvent(Event):
    """Event emitted when an evolution process starts."""
    
    event_type: str = "evolution_started"
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate required payload fields."""
        required_fields = ['evolution_id', 'domain', 'target_capabilities']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class ExperimentCompletedEvent(Event):
    """Event emitted when an experiment completes."""
    
    event_type: str = "experiment_completed"
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate required payload fields."""
        required_fields = ['experiment_id', 'evolution_id', 'status', 'metrics']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v
```

### Feedback Events

```python
class FeedbackSubmittedEvent(Event):
    """Event emitted when feedback is submitted."""
    
    event_type: str = "feedback_submitted"
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate required payload fields."""
        required_fields = ['feedback_id', 'agent_id', 'user_id', 'ratings']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v

class FeedbackAnalyzedEvent(Event):
    """Event emitted when feedback is analyzed."""
    
    event_type: str = "feedback_analyzed"
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate required payload fields."""
        required_fields = ['agent_id', 'insights', 'requires_adaptation']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required field: {field}")
        return v
```

## Neo4j Graph Schema

### Node Labels

- `Agent` - Represents an agent instance
- `Capability` - Represents an agent capability
- `User` - Represents a system user
- `Session` - Represents an interaction session
- `Feedback` - Represents user feedback
- `Evolution` - Represents an evolution process
- `Experiment` - Represents an evolution experiment
- `Insight` - Represents a feedback insight
- `KnowledgeItem` - Represents a piece of knowledge

### Relationship Types

- `HAS_CAPABILITY` - Links agent to capability
- `EVOLVED_FROM` - Links agent to its parent
- `PARTICIPATED_IN` - Links agent or user to session
- `PROVIDED_FEEDBACK` - Links user to feedback
- `RECEIVED_FEEDBACK` - Links agent to feedback
- `PART_OF` - Links experiment to evolution
- `GENERATED` - Links experiment to agent
- `DERIVED_FROM` - Links insight to feedback
- `KNOWS` - Links agent to knowledge

### Sample Cypher Schema

```cypher
// Agent node
CREATE CONSTRAINT ON (a:Agent) ASSERT a.agent_id IS UNIQUE;

// Capability node
CREATE CONSTRAINT ON (c:Capability) ASSERT (c.name, c.agent_id) IS NODE KEY;

// User node
CREATE CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE;

// Session node
CREATE CONSTRAINT ON (s:Session) ASSERT s.session_id IS UNIQUE;

// Feedback node
CREATE CONSTRAINT ON (f:Feedback) ASSERT f.feedback_id IS UNIQUE;

// Evolution node
CREATE CONSTRAINT ON (e:Evolution) ASSERT e.evolution_id IS UNIQUE;

// Experiment node
CREATE CONSTRAINT ON (e:Experiment) ASSERT e.experiment_id IS UNIQUE;

// Insight node
CREATE CONSTRAINT ON (i:Insight) ASSERT i.insight_id IS UNIQUE;

// Create indexes for frequent queries
CREATE INDEX agent_domain_idx FOR (a:Agent) ON (a.domain);
CREATE INDEX capability_name_idx FOR (c:Capability) ON (c.name);
CREATE INDEX feedback_timestamp_idx FOR (f:Feedback) ON (f.timestamp);
CREATE INDEX session_agent_idx FOR (s:Session) ON (s.agent_id);
