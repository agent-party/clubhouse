# Collective Unconscious: Shared Agent Knowledge in Neo4j

## Overview

The Collective Unconscious is a shared knowledge structure in Neo4j that allows agents to maintain individual identity while accessing a common pool of knowledge, experiences, and contextual information. This approach enables agents to benefit from each other's learnings without sacrificing their unique personality traits and specializations.

## Core Principles

1. **Shared Knowledge, Individual Identity**: Agents maintain distinct personalities but share access to common knowledge structures
2. **Bidirectional Learning**: Agents both contribute to and learn from the collective knowledge
3. **Contextual Enrichment**: Shared structures provide richer context for agent decision-making
4. **Emergent Connections**: The system discovers and strengthens knowledge connections based on usage patterns
5. **Hierarchical Access**: Different knowledge levels from universally shared to agent-specific

## Neo4j Implementation

### Knowledge Structure Design

```cypher
// Global Knowledge Layer - Shared by all agents
CREATE (gk:GlobalKnowledge {
    knowledge_type: "universal",
    confidence: 0.95,
    last_validated: datetime()
})

// Domain Knowledge Layer - Shared by agents in the same domain
CREATE (dk:DomainKnowledge {
    domain: "creative_problem_solving",
    knowledge_type: "domain",
    confidence: 0.9,
    last_validated: datetime()
})

// Experience Layer - Shared problem-solving experiences
CREATE (ex:SharedExperience {
    experience_type: "problem_solution",
    outcome: "success",
    quality_score: 0.85,
    timestamp: datetime()
})

// Agent-Specific Knowledge Layer
CREATE (ak:AgentKnowledge {
    agent_id: "agent-123",
    knowledge_type: "personal",
    confidence: 0.8,
    created_at: datetime(),
    updated_at: datetime()
})

// Connect layers
CREATE (ak)-[:HAS_ACCESS_TO]->(dk)
CREATE (dk)-[:HAS_ACCESS_TO]->(gk)
CREATE (ak)-[:CONTRIBUTED_TO]->(ex)
CREATE (ex)-[:BELONGS_TO_DOMAIN]->(dk)

// Knowledge Nodes
CREATE (kn:KnowledgeNode {
    content: "Problem decomposition increases solution quality",
    type: "principle",
    confidence: 0.92
})

// Connect knowledge to layers
CREATE (dk)-[:CONTAINS]->(kn)

// Agent connection to knowledge
CREATE (a:Agent {agent_id: "agent-123"})-[:KNOWS]->(kn)
```

### Knowledge Sharing Patterns

The system implements several knowledge sharing patterns:

1. **Global Library**: Universal concepts available to all agents
2. **Domain Repositories**: Specialized knowledge for domain-specific agents
3. **Experience Records**: Shared problem-solving experiences and outcomes
4. **Agent Memory**: Private observations and learnings
5. **Concept Associations**: Connections between related knowledge elements

### Cross-Agent Knowledge Structures

```cypher
// Shared Problem-Solution Patterns
CREATE (ps:ProblemSolutionPattern {
    pattern_name: "Divide and Conquer",
    effectiveness: 0.88,
    applicability: "Complex problems with separable components",
    usage_count: 42
})

// Connect multiple agents to the pattern
MATCH (a:Agent) WHERE a.agent_id IN ["agent-123", "agent-456", "agent-789"]
MATCH (ps:ProblemSolutionPattern {pattern_name: "Divide and Conquer"})
CREATE (a)-[:HAS_USED {success_rate: 0.9, last_used: datetime()}]->(ps)

// Create cross-domain concept bridge
CREATE (c1:Concept {name: "Modularity", domain: "Software Engineering"})
CREATE (c2:Concept {name: "Compartmentalization", domain: "Biology"})
CREATE (c1)-[:ANALOGOUS_TO {strength: 0.75}]->(c2)

// Connect agents from different domains to the bridge
MATCH (a1:Agent {specialization: "Software Engineering"})
MATCH (a2:Agent {specialization: "Biology"})
MATCH (c1:Concept {name: "Modularity"})
MATCH (c2:Concept {name: "Compartmentalization"})
CREATE (a1)-[:UNDERSTANDS]->(c1)
CREATE (a2)-[:UNDERSTANDS]->(c2)
```

## Knowledge Sharing Mechanisms

### 1. Cross-Agent Learning

```python
class CrossAgentLearningManager:
    """Manages knowledge sharing between agents."""
    
    async def share_successful_solution(
        self,
        solution_id: str,
        source_agent_id: str,
        target_agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Share a successful solution with other agents."""
        # Get solution details
        solution = await self.solution_repository.get_solution(solution_id)
        
        # Get source agent details
        source_agent = await self.agent_repository.get_agent(source_agent_id)
        
        # If no target agents specified, find compatible agents
        if not target_agent_ids:
            target_agent_ids = await self._find_compatible_agents(
                source_agent_id,
                solution["domain"]
            )
            
        # Create knowledge sharing record
        knowledge_id = f"knowledge-{uuid.uuid4().hex[:8]}"
        
        # Extract key insights from solution
        insights = await self._extract_solution_insights(solution)
        
        # Create knowledge node in Neo4j
        query = """
        CREATE (k:KnowledgeNode {
            knowledge_id: $knowledge_id,
            content: $content,
            source_solution: $solution_id,
            source_agent: $source_agent_id,
            created_at: datetime(),
            domain: $domain,
            confidence: $confidence
        })
        RETURN k
        """
        
        params = {
            "knowledge_id": knowledge_id,
            "content": insights["pattern"],
            "solution_id": solution_id,
            "source_agent_id": source_agent_id,
            "domain": solution["domain"],
            "confidence": solution["quality_score"] if "quality_score" in solution else 0.8
        }
        
        await self.neo4j_service.execute_query(query, params)
        
        # Connect source agent to knowledge
        query = """
        MATCH (a:Agent {agent_id: $agent_id})
        MATCH (k:KnowledgeNode {knowledge_id: $knowledge_id})
        CREATE (a)-[:CONTRIBUTED {timestamp: datetime()}]->(k)
        CREATE (a)-[:KNOWS {confidence: 1.0}]->(k)
        """
        
        await self.neo4j_service.execute_query(query, {
            "agent_id": source_agent_id,
            "knowledge_id": knowledge_id
        })
        
        # Connect knowledge to appropriate layers
        await self._connect_to_knowledge_layers(knowledge_id, solution["domain"])
        
        # Share with target agents
        sharing_results = {}
        for agent_id in target_agent_ids:
            result = await self._share_with_agent(knowledge_id, agent_id)
            sharing_results[agent_id] = result
            
        return {
            "knowledge_id": knowledge_id,
            "insights": insights,
            "sharing_results": sharing_results
        }
        
    async def _share_with_agent(
        self,
        knowledge_id: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """Share knowledge with a specific agent."""
        # Check agent compatibility
        compatibility = await self._check_knowledge_compatibility(knowledge_id, agent_id)
        
        if compatibility < 0.5:
            return {
                "status": "rejected",
                "reason": "Low compatibility",
                "compatibility_score": compatibility
            }
            
        # Connect agent to knowledge with appropriate confidence
        query = """
        MATCH (a:Agent {agent_id: $agent_id})
        MATCH (k:KnowledgeNode {knowledge_id: $knowledge_id})
        CREATE (a)-[:KNOWS {
            confidence: $confidence,
            learned_at: datetime(),
            source: "shared"
        }]->(k)
        """
        
        await self.neo4j_service.execute_query(query, {
            "agent_id": agent_id,
            "knowledge_id": knowledge_id,
            "confidence": compatibility
        })
        
        # Update agent system prompt if appropriate
        if compatibility > 0.8:
            await self._update_agent_system_prompt(agent_id, knowledge_id)
            
        return {
            "status": "accepted",
            "compatibility_score": compatibility,
            "system_prompt_updated": compatibility > 0.8
        }
```

### 2. Emergent Knowledge Connections

```python
class EmergentConnectionDiscoverer:
    """Discovers emergent connections between knowledge entities."""
    
    async def discover_connections(
        self,
        min_similarity: float = 0.7,
        max_connections: int = 1000
    ) -> Dict[str, Any]:
        """Discover emergent connections between knowledge entities."""
        # Fetch knowledge nodes without too many connections
        query = """
        MATCH (k:KnowledgeNode)
        WHERE size((k)-[:SIMILAR_TO]->()) < 5
        RETURN k.knowledge_id AS id, k.content AS content, 
               k.domain AS domain, labels(k) AS types
        LIMIT 5000
        """
        
        knowledge_nodes = await self.neo4j_service.execute_query(query)
        
        # Prepare for embedding and comparison
        texts = [node["content"] for node in knowledge_nodes]
        
        # Get embeddings
        embeddings = await self.embedding_service.get_embeddings(texts)
        
        # Find similar pairs
        connections = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self._calculate_cosine_similarity(embeddings[i], embeddings[j])
                
                if similarity >= min_similarity:
                    connections.append({
                        "source_id": knowledge_nodes[i]["id"],
                        "target_id": knowledge_nodes[j]["id"],
                        "similarity": similarity,
                        "source_domain": knowledge_nodes[i]["domain"],
                        "target_domain": knowledge_nodes[j]["domain"]
                    })
                    
        # Sort by similarity
        connections.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to max_connections
        connections = connections[:max_connections]
        
        # Create connections in Neo4j
        created_count = 0
        for conn in connections:
            created = await self._create_connection(conn)
            if created:
                created_count += 1
                
        return {
            "discovered_connections": len(connections),
            "created_connections": created_count
        }
        
    async def _create_connection(self, connection: Dict[str, Any]) -> bool:
        """Create a connection between knowledge nodes in Neo4j."""
        query = """
        MATCH (k1:KnowledgeNode {knowledge_id: $source_id})
        MATCH (k2:KnowledgeNode {knowledge_id: $target_id})
        WHERE NOT (k1)-[:SIMILAR_TO]->(k2)
        CREATE (k1)-[:SIMILAR_TO {
            strength: $similarity,
            discovered_at: datetime(),
            cross_domain: $cross_domain
        }]->(k2)
        RETURN k1, k2
        """
        
        cross_domain = connection["source_domain"] != connection["target_domain"]
        
        result = await self.neo4j_service.execute_query(query, {
            "source_id": connection["source_id"],
            "target_id": connection["target_id"],
            "similarity": connection["similarity"],
            "cross_domain": cross_domain
        })
        
        return len(result) > 0
```

### 3. Collective Insight Generation

```python
class CollectiveInsightGenerator:
    """Generates insights from the collective knowledge pool."""
    
    async def generate_domain_insights(
        self,
        domain: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """Generate insights for a domain from collective knowledge."""
        # Get domain knowledge subgraph
        query = """
        MATCH (dk:DomainKnowledge {domain: $domain})-[:CONTAINS*1..2]->(k:KnowledgeNode)
        OPTIONAL MATCH (k)-[r:SIMILAR_TO|RELATES_TO]->(k2)
        WHERE r.strength > 0.7
        RETURN k, r, k2
        """
        
        subgraph = await self.neo4j_service.execute_query(query, {"domain": domain})
        
        # Extract knowledge elements
        knowledge_elements = self._extract_knowledge_elements(subgraph)
        
        # Generate insights
        insights = await self._generate_insights_from_elements(knowledge_elements, domain)
        
        # Store insights back in Neo4j
        for insight in insights:
            await self._store_insight(insight, domain)
            
        return {
            "domain": domain,
            "knowledge_elements_used": len(knowledge_elements),
            "insights_generated": len(insights),
            "insights": insights
        }
```

## Access Control Mechanisms

The collective unconscious implements several access control mechanisms:

### 1. Confidence-Based Access

```python
class ConfidenceAccessController:
    """Controls access to knowledge based on confidence levels."""
    
    async def get_accessible_knowledge(
        self,
        agent_id: str,
        domain: str,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get knowledge accessible to an agent with confidence threshold."""
        # Implementation details...
```

### 2. Personality-Based Knowledge Filtering

```python
class PersonalityKnowledgeFilter:
    """Filters knowledge based on agent personality traits."""
    
    async def filter_knowledge_for_agent(
        self,
        agent_id: str,
        available_knowledge: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter knowledge based on agent personality compatibility."""
        # Implementation details...
```

### 3. Dynamic Trust System

```python
class KnowledgeTrustSystem:
    """Manages trust levels for shared knowledge."""
    
    async def calculate_knowledge_trust(
        self,
        knowledge_id: str
    ) -> float:
        """Calculate trust level for a knowledge element."""
        # Implementation details...
```

## Benefits for Agent Context

1. **Enriched Context**: Agents access a broader context beyond their individual experiences
2. **Knowledge Transfer**: Successful strategies can propagate through the agent ecosystem
3. **Cross-Domain Insights**: Connections between domains enable novel problem-solving approaches
4. **Collective Learning**: The system becomes more capable as a whole while maintaining agent diversity
5. **Historical Memory**: Agents have access to past problem-solving experiences and outcomes

## Real-Time Knowledge Flow

The system implements real-time knowledge flow between agents:

```python
class KnowledgeFlowManager:
    """Manages real-time knowledge flow between agents."""
    
    async def process_knowledge_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Process a knowledge flow event."""
        # Implementation details for different event types:
        # - new_solution
        # - insight_generated
        # - pattern_discovered
        # - cross_domain_connection
        # - external_knowledge_added
```

## Conclusion

The Collective Unconscious framework provides a powerful shared knowledge structure that:

1. **Maintains Individual Identity**: Each agent retains its unique personality and specialization
2. **Enables Knowledge Sharing**: Agents contribute to and learn from the collective knowledge
3. **Creates Emergent Connections**: The system discovers connections across domains and knowledge elements
4. **Provides Rich Context**: Agents have access to contextualized knowledge beyond their individual experience
5. **Supports Continuous Learning**: The knowledge structure evolves and improves over time

This approach significantly enhances the capabilities of the Agent Orchestration Platform by enabling a form of collective intelligence while preserving the diversity of agent personalities and specializations.
