# Agent Implementation Guide

## Overview

This guide explains how to implement agents with personality traits in the Agent Orchestration Platform, focusing on:

1. Storing agent personalities in Neo4j
2. Connecting agents via Kafka for event-driven communication
3. Initializing agents with generated system prompts
4. Managing agent state and persistence

## Neo4j Agent Storage

### Agent Schema Design

The following Neo4j schema efficiently stores agent data and personality traits:

```cypher
// Agent node with basic properties
CREATE (a:Agent {
    agent_id: "unique-id",
    name: "Agent Name",
    role: "Generator|Critic|Refiner|Evaluator",
    created_at: datetime(),
    updated_at: datetime(),
    active: true
})

// Identity properties node
CREATE (i:AgentIdentity {
    specialization: "Domain Specialization",
    expertise_level: "Expert",
    backstory: "Agent backstory text..."
})

// Personality traits node
CREATE (p:AgentPersonality {
    openness: 85,
    conscientiousness: 60,
    extraversion: 70,
    agreeableness: 65,
    neuroticism: 30,
    creativity: 90,
    analytical: 60,
    decisiveness: 70,
    risk_tolerance: 75,
    adaptability: 80
})

// Communication style node
CREATE (c:CommunicationStyle {
    formality: 40,
    directness: 65,
    verbosity: 60,
    technical_level: 70,
    humor: 30
})

// Model configuration node
CREATE (m:ModelConfig {
    model_name: "gpt-4o",
    temperature: 0.9,
    top_p: 0.98,
    frequency_penalty: 0.2,
    max_tokens: 4000
})

// Establishing relationships
CREATE (a)-[:HAS_IDENTITY]->(i)
CREATE (a)-[:HAS_PERSONALITY]->(p)
CREATE (a)-[:HAS_COMMUNICATION_STYLE]->(c)
CREATE (a)-[:USES_MODEL_CONFIG]->(m)

// Knowledge domains
CREATE (kd1:KnowledgeDomain {
    name: "Creative Problem Solving",
    proficiency: 95,
    description: "Advanced techniques in lateral thinking"
})

CREATE (a)-[:HAS_PRIMARY_KNOWLEDGE]->(kd1)

// Capability relationships
CREATE (cap1:Capability {name: "GenerateIdeas"})
CREATE (cap2:Capability {name: "AnalyzeSolutions"})

CREATE (a)-[:HAS_CAPABILITY]->(cap1)
CREATE (a)-[:HAS_CAPABILITY]->(cap2)

// Historical interactions
CREATE (h:History {
    start_time: datetime(),
    end_time: datetime(),
    success_rate: 0.92,
    interactions_count: 57,
    total_cost: 0.34
})

CREATE (a)-[:HAS_HISTORY]->(h)
```

### Agent Data Access

The platform uses the `Neo4jService` to interact with agent data:

```python
class AgentRepository:
    """Repository for agent data in Neo4j."""
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol):
        """Initialize with Neo4j service."""
        self.neo4j = neo4j_service
        
    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get agent with all related data."""
        query = """
        MATCH (a:Agent {agent_id: $agent_id})
        OPTIONAL MATCH (a)-[:HAS_IDENTITY]->(i:AgentIdentity)
        OPTIONAL MATCH (a)-[:HAS_PERSONALITY]->(p:AgentPersonality)
        OPTIONAL MATCH (a)-[:HAS_COMMUNICATION_STYLE]->(c:CommunicationStyle)
        OPTIONAL MATCH (a)-[:USES_MODEL_CONFIG]->(m:ModelConfig)
        OPTIONAL MATCH (a)-[:HAS_PRIMARY_KNOWLEDGE]->(pk:KnowledgeDomain)
        OPTIONAL MATCH (a)-[:HAS_SECONDARY_KNOWLEDGE]->(sk:KnowledgeDomain)
        OPTIONAL MATCH (a)-[:HAS_CAPABILITY]->(cap:Capability)
        RETURN a, i, p, c, m,
               collect(DISTINCT pk) as primary_knowledge,
               collect(DISTINCT sk) as secondary_knowledge,
               collect(DISTINCT cap) as capabilities
        """
        
        result = await self.neo4j.execute_query(query, {"agent_id": agent_id})
        
        if not result or not result[0].get("a"):
            return None
            
        # Process results and convert to dictionary
        agent_data = node_to_dict(result[0]["a"])
        
        # Add related data
        if result[0].get("i"):
            agent_data["identity"] = node_to_dict(result[0]["i"])
            
        if result[0].get("p"):
            agent_data["personality"] = node_to_dict(result[0]["p"])
            
        if result[0].get("c"):
            agent_data["communication_style"] = node_to_dict(result[0]["c"])
            
        if result[0].get("m"):
            agent_data["model_config"] = node_to_dict(result[0]["m"])
            
        # Process knowledge domains and capabilities
        agent_data["primary_knowledge"] = [
            node_to_dict(k) for k in result[0]["primary_knowledge"] if k
        ]
        
        agent_data["secondary_knowledge"] = [
            node_to_dict(k) for k in result[0]["secondary_knowledge"] if k
        ]
        
        agent_data["capabilities"] = [
            node_to_dict(c) for c in result[0]["capabilities"] if c
        ]
        
        return agent_data
        
    async def create_agent(self, agent_data: Dict[str, Any]) -> str:
        """Create a new agent with all related data."""
        # Implementation details for creating an agent with all relationships
        # ...
        
    async def update_agent(self, agent_id: str, agent_data: Dict[str, Any]) -> bool:
        """Update an existing agent's data."""
        # Implementation details for updating an agent
        # ...
        
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent and all related nodes."""
        # Implementation details for deleting an agent
        # ...
        
    async def get_agents_by_role(self, role: str) -> List[Dict[str, Any]]:
        """Get all agents with a specific role."""
        query = """
        MATCH (a:Agent {role: $role, active: true})
        RETURN a
        ORDER BY a.name
        """
        
        result = await self.neo4j.execute_query(query, {"role": role})
        return [node_to_dict(record["a"]) for record in result]
        
    async def find_agents_by_traits(
        self,
        min_creativity: Optional[int] = None,
        min_analytical: Optional[int] = None,
        knowledge_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find agents matching specific trait criteria."""
        conditions = []
        params = {}
        
        if min_creativity is not None:
            conditions.append("p.creativity >= $min_creativity")
            params["min_creativity"] = min_creativity
            
        if min_analytical is not None:
            conditions.append("p.analytical >= $min_analytical")
            params["min_analytical"] = min_analytical
            
        where_clause = " AND ".join(conditions) if conditions else ""
        if where_clause:
            where_clause = f"WHERE {where_clause}"
            
        knowledge_match = ""
        if knowledge_domains:
            knowledge_match = """
            WITH a, p, c
            MATCH (a)-[:HAS_PRIMARY_KNOWLEDGE|HAS_SECONDARY_KNOWLEDGE]->(k:KnowledgeDomain)
            WHERE k.name IN $knowledge_domains
            """
            params["knowledge_domains"] = knowledge_domains
            
        query = f"""
        MATCH (a:Agent)-[:HAS_PERSONALITY]->(p:AgentPersonality)
        MATCH (a)-[:HAS_COMMUNICATION_STYLE]->(c:CommunicationStyle)
        {where_clause}
        {knowledge_match}
        RETURN DISTINCT a, p, c
        """
        
        result = await self.neo4j.execute_query(query, params)
        
        agents = []
        for record in result:
            agent = node_to_dict(record["a"])
            agent["personality"] = node_to_dict(record["p"])
            agent["communication_style"] = node_to_dict(record["c"])
            agents.append(agent)
            
        return agents
```

## Kafka Integration

### Kafka Message Structure

Agent communications use standardized Kafka message formats:

```python
class AgentMessage(BaseModel):
    """Standard message format for agent communication."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender_id: str
    recipient_id: Optional[str] = None
    message_type: str  # "command", "response", "event", "error"
    content: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
```

### Kafka Topics Structure

The platform uses the following Kafka topics:

```
agent.commands       # Commands for agents to execute capabilities
agent.responses      # Responses from capability executions
agent.events         # System events and notifications
agent.errors         # Error messages
agent.state.changes  # Agent state change notifications
evolution.process    # Evolutionary process tracking
human.approvals      # Human approval requests and responses
```

### Agent Kafka Service

Agents connect to Kafka using a standardized service:

```python
class AgentKafkaService:
    """Kafka service for agent communication."""
    
    def __init__(
        self,
        config: KafkaConfig,
        agent_id: str,
        message_handler: Callable[[AgentMessage], Awaitable[None]]
    ):
        """Initialize the Kafka service for an agent."""
        self.agent_id = agent_id
        self.message_handler = message_handler
        
        # Create Kafka producer/consumer
        self.kafka_service = ConfluentKafkaService(config)
        
        # Topics this agent subscribes to
        self.topics = [
            f"agent.commands.{agent_id}",  # Agent-specific commands
            "agent.commands.broadcast",    # Broadcast commands
            "agent.events",                # System events
            f"agent.responses.{agent_id}", # Responses to this agent
            "evolution.process"            # Evolution process updates
        ]
        
    async def start(self):
        """Start consuming messages."""
        # Set up consumer
        consumer = self.kafka_service.get_consumer()
        consumer.subscribe(self.topics)
        
        # Start consuming in a background task
        self.kafka_service.consume_messages(
            self.topics,
            self._handle_message,
            timeout=1.0
        )
        
    async def stop(self):
        """Stop consuming messages."""
        self.kafka_service.stop_consuming()
        
    async def send_message(self, message: AgentMessage):
        """Send a message to Kafka."""
        # Determine topic based on message type
        if message.message_type == "command":
            if message.recipient_id:
                topic = f"agent.commands.{message.recipient_id}"
            else:
                topic = "agent.commands.broadcast"
        elif message.message_type == "response":
            if message.recipient_id:
                topic = f"agent.responses.{message.recipient_id}"
            else:
                topic = "agent.responses"
        elif message.message_type == "event":
            topic = "agent.events"
        elif message.message_type == "error":
            topic = "agent.errors"
        else:
            topic = "agent.events"
            
        # Produce message to Kafka
        self.kafka_service.produce_message(
            KafkaMessage(
                topic=topic,
                value=message.dict(),
                key=message.message_id
            )
        )
        
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming Kafka message."""
        try:
            # Convert to AgentMessage
            agent_message = AgentMessage(**message)
            
            # Pass to agent's message handler
            await self.message_handler(agent_message)
            
        except Exception as e:
            logger.error(f"Error handling Kafka message: {str(e)}")
            # Log error but continue processing
```

## Agent Implementation

### Agent Factory

The `AgentFactory` creates agents with personality traits and system prompts:

```python
class AgentFactory:
    """Factory for creating agents with personality traits."""
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        system_prompt_generator: SystemPromptGenerator
    ):
        """Initialize the agent factory."""
        self.service_registry = service_registry
        self.system_prompt_generator = system_prompt_generator
        self.neo4j_service = service_registry.get_service(Neo4jServiceProtocol)
        self.agent_repository = AgentRepository(self.neo4j_service)
        
    async def create_agent(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        personality_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> AgentProtocol:
        """Create an agent of the specified type."""
        # Generate agent_id if not provided
        if not agent_id:
            agent_id = f"{agent_type.lower()}-{uuid.uuid4().hex[:8]}"
            
        # Use defaults if configs not provided
        personality_config = personality_config or self._get_default_personality(agent_type)
        model_config = model_config or self._get_default_model_config(agent_type)
        
        # Create agent specification
        agent_spec = {
            "agent_id": agent_id,
            "name": name or f"{agent_type} {agent_id[-5:]}",
            "role": role or agent_type,
            "identity": personality_config.get("identity", {}),
            "personality": personality_config.get("personality", {}),
            "communication_style": personality_config.get("communication_style", {}),
            "knowledge_domains": personality_config.get("knowledge_domains", {})
        }
        
        # Store in Neo4j
        await self.agent_repository.create_agent(agent_spec)
        
        # Generate system prompt
        system_prompt = self.system_prompt_generator.generate_prompt(
            identity=AgentIdentity(**agent_spec["identity"]),
            personality=AgentPersonality(**agent_spec["personality"]),
            communication=CommunicationStyle(**agent_spec["communication_style"]),
            knowledge=self._build_knowledge_domains(agent_spec.get("knowledge_domains", {})),
            capabilities=self._get_capabilities_for_agent_type(agent_type)
        )
        
        # Create the agent instance
        agent_class = self._get_agent_class(agent_type)
        agent = agent_class(
            agent_id=agent_id,
            name=agent_spec["name"],
            system_prompt=system_prompt,
            model_config=ModelConfiguration(**model_config),
            service_registry=self.service_registry
        )
        
        # Initialize the agent
        await agent.initialize()
        
        return agent
        
    def _get_agent_class(self, agent_type: str) -> Type[AgentProtocol]:
        """Get the agent class for the specified type."""
        agent_types = {
            "Generator": GeneratorAgent,
            "Critic": CriticAgent,
            "Refiner": RefinerAgent,
            "Evaluator": EvaluatorAgent,
            "Assistant": AssistantAgent
        }
        
        if agent_type not in agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return agent_types[agent_type]
        
    def _get_default_personality(self, agent_type: str) -> Dict[str, Any]:
        """Get default personality configuration for agent type."""
        # Implementation of default personalities by agent type
        # ...
        
    def _get_default_model_config(self, agent_type: str) -> Dict[str, Any]:
        """Get default model configuration for agent type."""
        # Implementation of default model configs by agent type
        # ...
        
    def _get_capabilities_for_agent_type(self, agent_type: str) -> List[str]:
        """Get list of capability names for agent type."""
        # Implementation of capability mapping by agent type
        # ...
        
    def _build_knowledge_domains(self, knowledge_data: Dict[str, Any]) -> KnowledgeDomains:
        """Build knowledge domains from dictionary data."""
        # Implementation of knowledge domain conversion
        # ...
```

### Agent Implementation

The core agent implementation combines personalities with capabilities:

```python
class GeneratorAgent(BaseAgent):
    """Generator agent implementation."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        model_config: ModelConfiguration,
        service_registry: ServiceRegistry
    ):
        """Initialize the generator agent."""
        self.system_prompt = system_prompt
        self.model_config = model_config
        self.service_registry = service_registry
        
        # Get services
        self.neo4j_service = service_registry.get_service(Neo4jServiceProtocol)
        self.llm_service = service_registry.get_service(LLMServiceProtocol)
        
        # Set up capabilities
        capabilities = [
            GenerateIdeasCapability(self),
            ExpandConceptsCapability(self),
            VisualizeOptionsCapability(self)
        ]
        
        # Initialize base agent
        super().__init__(agent_id, name, "Generator Agent", capabilities)
        
        # Set up Kafka service
        kafka_config = service_registry.get_service(KafkaConfigProvider).get_config()
        self.kafka_service = AgentKafkaService(
            config=kafka_config,
            agent_id=agent_id,
            message_handler=self._handle_message
        )
        
    async def initialize(self):
        """Initialize the agent."""
        await super().initialize()
        await self.kafka_service.start()
        
    async def shutdown(self):
        """Shutdown the agent."""
        await self.kafka_service.stop()
        await super().shutdown()
        
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages."""
        if message.message_type == "command":
            # Process command
            response = await self.process_command(message)
            
            # Send response
            await self.kafka_service.send_message(
                AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="response",
                    content=response,
                    parent_id=message.message_id,
                    trace_id=message.trace_id
                )
            )
        elif message.message_type == "event":
            # Handle event
            await self._handle_event(message)
            
    async def process_command(self, message: AgentMessage) -> Dict[str, Any]:
        """Process a command message."""
        try:
            # Extract command details
            command = message.content.get("command")
            parameters = message.content.get("parameters", {})
            
            if not command:
                return {
                    "status": "error",
                    "error": "No command specified"
                }
                
            # Execute capability
            result = await self.execute_capability(command, parameters)
            
            return {
                "status": "success",
                "data": result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def _handle_event(self, message: AgentMessage):
        """Handle event messages."""
        # Implementation of event handling
        # ...
        
    async def generate_content(self, prompt: str) -> str:
        """Generate content using the LLM."""
        return await self.llm_service.generate(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            model=self.model_config.model_name,
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            max_tokens=self.model_config.max_tokens,
            frequency_penalty=self.model_config.frequency_penalty,
            presence_penalty=self.model_config.presence_penalty
        )
```

## Evolutionary Integration

### Evolution Orchestrator

The `EvolutionOrchestrator` manages the evolutionary process using personalized agents:

```python
class EvolutionOrchestrator:
    """Orchestrates the evolutionary process with personalized agents."""
    
    def __init__(self, service_registry: ServiceRegistry):
        """Initialize the evolution orchestrator."""
        self.service_registry = service_registry
        self.neo4j_service = service_registry.get_service(Neo4jServiceProtocol)
        self.agent_repository = AgentRepository(self.neo4j_service)
        self.agent_factory = AgentFactory(
            service_registry,
            SystemPromptGenerator()
        )
        
        # Kafka for evolution process communication
        kafka_config = service_registry.get_service(KafkaConfigProvider).get_config()
        self.kafka_service = ConfluentKafkaService(kafka_config)
        
        # Active evolutionary processes
        self.active_processes: Dict[str, EvolutionProcess] = {}
        
    async def start_evolution(
        self,
        problem_statement: str,
        configuration: EvolutionConfiguration
    ) -> str:
        """Start a new evolutionary process."""
        # Generate process ID
        process_id = f"evo-{uuid.uuid4().hex[:8]}"
        
        # Create agents for the process
        generator = await self._get_or_create_agent("Generator", configuration.generator_config)
        critic = await self._get_or_create_agent("Critic", configuration.critic_config)
        refiner = await self._get_or_create_agent("Refiner", configuration.refiner_config)
        evaluator = await self._get_or_create_agent("Evaluator", configuration.evaluator_config)
        
        # Create evolution process
        process = EvolutionProcess(
            process_id=process_id,
            problem_statement=problem_statement,
            configuration=configuration,
            generator=generator,
            critic=critic,
            refiner=refiner,
            evaluator=evaluator,
            service_registry=self.service_registry
        )
        
        # Store in active processes
        self.active_processes[process_id] = process
        
        # Store process in Neo4j
        await self._store_process(process)
        
        # Start the process
        await process.start()
        
        return process_id
        
    async def _get_or_create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AgentProtocol:
        """Get an existing agent or create a new one."""
        if not config:
            # Use default configuration
            return await self.agent_factory.create_agent(agent_type)
            
        if "agent_id" in config:
            # Try to get existing agent
            agent_data = await self.agent_repository.get_agent(config["agent_id"])
            if agent_data:
                # Create agent from existing data
                # ...
                pass
                
        # Create new agent with configuration
        return await self.agent_factory.create_agent(
            agent_type,
            personality_config=config.get("personality"),
            model_config=config.get("model")
        )
        
    async def _store_process(self, process: EvolutionProcess):
        """Store evolution process in Neo4j."""
        # Implementation of process storage
        # ...
```

## Human-in-the-Loop Integration

Human approval is integrated into the agent workflow:

```python
class HumanApprovalService:
    """Service for human approval requests."""
    
    def __init__(self, service_registry: ServiceRegistry):
        """Initialize the human approval service."""
        self.service_registry = service_registry
        kafka_config = service_registry.get_service(KafkaConfigProvider).get_config()
        self.kafka_service = ConfluentKafkaService(kafka_config)
        
        # Neo4j for storing approval requests
        self.neo4j_service = service_registry.get_service(Neo4jServiceProtocol)
        
        # Pending approvals
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        
    async def request_approval(
        self,
        requester_id: str,
        request_type: str,
        content: Dict[str, Any],
        timeout_seconds: int = 300
    ) -> str:
        """Request human approval."""
        # Generate approval ID
        approval_id = f"approval-{uuid.uuid4().hex[:8]}"
        
        # Create approval request
        request = {
            "approval_id": approval_id,
            "requester_id": requester_id,
            "request_type": request_type,
            "content": content,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "timeout_at": (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat()
        }
        
        # Store in pending approvals
        self.pending_approvals[approval_id] = request
        
        # Store in Neo4j
        await self._store_approval_request(request)
        
        # Publish to Kafka
        self.kafka_service.produce_message(
            KafkaMessage(
                topic="human.approvals.requests",
                value=request,
                key=approval_id
            )
        )
        
        return approval_id
        
    async def get_approval_status(self, approval_id: str) -> Dict[str, Any]:
        """Get the status of an approval request."""
        if approval_id in self.pending_approvals:
            return self.pending_approvals[approval_id]
            
        # Retrieve from Neo4j if not in memory
        query = """
        MATCH (a:ApprovalRequest {approval_id: $approval_id})
        RETURN a
        """
        
        result = await self.neo4j_service.execute_query(query, {"approval_id": approval_id})
        
        if not result:
            raise ValueError(f"Approval request {approval_id} not found")
            
        return node_to_dict(result[0]["a"])
        
    async def submit_approval(
        self,
        approval_id: str,
        approved: bool,
        feedback: Optional[str] = None
    ) -> None:
        """Submit a human approval response."""
        # Check if approval exists
        if approval_id not in self.pending_approvals:
            # Try to retrieve from Neo4j
            request = await self.get_approval_status(approval_id)
            if not request:
                raise ValueError(f"Approval request {approval_id} not found")
                
            self.pending_approvals[approval_id] = request
            
        # Update approval status
        self.pending_approvals[approval_id].update({
            "status": "approved" if approved else "rejected",
            "feedback": feedback,
            "resolved_at": datetime.now().isoformat()
        })
        
        # Update in Neo4j
        await self._update_approval_request(self.pending_approvals[approval_id])
        
        # Publish to Kafka
        self.kafka_service.produce_message(
            KafkaMessage(
                topic="human.approvals.responses",
                value=self.pending_approvals[approval_id],
                key=approval_id
            )
        )
```

## Conclusion

The Agent Implementation Guide provides a comprehensive approach to building agents with personality traits and integration with Neo4j and Kafka:

1. **Personality-Driven Agents**: Agents with rich personalities stored in Neo4j
2. **Event-Driven Communication**: Kafka-based messaging for agent collaboration
3. **Dynamic System Prompts**: Generated prompts based on agent personalities
4. **Human Integration**: Approval workflows for human-in-the-loop operations
5. **Evolutionary Orchestration**: Coordinated evolutionary processes with specialized agents

This implementation approach ensures that agents can effectively collaborate in the evolutionary framework while maintaining their specialized roles and unique personality traits.
