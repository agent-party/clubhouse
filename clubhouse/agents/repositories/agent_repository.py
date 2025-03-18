"""
Agent Repository

This module implements Neo4j storage and retrieval for agents, supporting:
1. Storage of agent configurations and performance metrics
2. Retrieval based on various search criteria
3. Performance tracking for agent evolution
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class AgentPerformanceMetrics(BaseModel):
    """Performance metrics for agent evaluation and evolution."""
    
    # Core metrics
    task_success_rate: float = Field(0.0, description="Rate of successful task completions", ge=0.0, le=1.0)
    context_preservation: float = Field(0.0, description="Accuracy in preserving critical context", ge=0.0, le=1.0)
    creativity_score: float = Field(0.0, description="Rating of creative solutions", ge=0.0, le=1.0)
    pragmatism_score: float = Field(0.0, description="Rating of practical, actionable outputs", ge=0.0, le=1.0)
    
    # Collaboration metrics
    reference_quality: float = Field(0.0, description="Quality of references to other agents", ge=0.0, le=1.0)
    contribution_relevance: float = Field(0.0, description="Relevance of contributions to discussion", ge=0.0, le=1.0)
    team_alignment: float = Field(0.0, description="Alignment with team objectives", ge=0.0, le=1.0)
    
    # Usage metrics
    total_invocations: int = Field(0, description="Total number of times agent was invoked")
    total_tokens_consumed: int = Field(0, description="Total tokens consumed by agent")
    
    # Timestamps
    last_evaluation: Optional[datetime] = Field(None, description="Timestamp of last evaluation")
    
    def calculate_overall_score(self) -> float:
        """Calculate an overall performance score based on all metrics."""
        if self.total_invocations == 0:
            return 0.0
            
        # Core performance weight (60%)
        core_score = (
            self.task_success_rate * 0.25 +
            self.context_preservation * 0.15 +
            self.creativity_score * 0.10 +
            self.pragmatism_score * 0.10
        ) * 0.60
        
        # Collaboration weight (40%)
        collab_score = (
            self.reference_quality * 0.15 +
            self.contribution_relevance * 0.15 +
            self.team_alignment * 0.10
        ) * 0.40
        
        return core_score + collab_score


class AgentRepository:
    """Repository for storing and retrieving agents using Neo4j."""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """
        Initialize the repository with Neo4j connection details.
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            username: Neo4j username (defaults to NEO4J_USERNAME env var)
            password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "testpassword")
        
        # Check if Neo4j driver is available
        if not NEO4J_AVAILABLE:
            print("WARNING: Neo4j Python driver not installed. Using in-memory storage instead.")
            self.driver = None
            self._in_memory_storage = {}
        else:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                self._setup_schema()
            except Exception as e:
                print(f"WARNING: Could not connect to Neo4j: {str(e)}. Using in-memory storage instead.")
                self.driver = None
                self._in_memory_storage = {}
    
    def _setup_schema(self) -> None:
        """Set up the Neo4j schema with constraints and indexes."""
        if not self.driver:
            return
            
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT agent_id_unique IF NOT EXISTS
                FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE
            """)
            
            # Create indexes
            session.run("""
                CREATE INDEX agent_role_idx IF NOT EXISTS
                FOR (a:Agent) ON (a.role)
            """)
            
            session.run("""
                CREATE INDEX agent_performance_idx IF NOT EXISTS
                FOR (a:Agent) ON (a.overall_score)
            """)
    
    def store_agent(self, agent_id: str, personality: Dict, metrics: Optional[AgentPerformanceMetrics] = None) -> str:
        """
        Store an agent in the repository.
        
        Args:
            agent_id: Unique identifier for the agent
            personality: Agent personality configuration
            metrics: Optional performance metrics
            
        Returns:
            The agent's ID
        """
        # Generate UUID if not provided
        if not agent_id:
            agent_id = str(uuid.uuid4())
            
        # Convert personality to storable format
        personality_data = self._prepare_personality_data(personality)
        
        # Use metrics or create default
        metrics_data = metrics.dict() if metrics else AgentPerformanceMetrics().dict()
        
        # Calculate overall score
        overall_score = metrics.calculate_overall_score() if metrics else 0.0
        
        if self.driver:
            with self.driver.session() as session:
                # Store in Neo4j
                result = session.run("""
                    MERGE (a:Agent {agent_id: $agent_id})
                    SET a.name = $name,
                        a.role = $role,
                        a.description = $description,
                        a.personality = $personality,
                        a.metrics = $metrics,
                        a.overall_score = $overall_score,
                        a.created_at = CASE WHEN a.created_at IS NULL THEN timestamp() ELSE a.created_at END,
                        a.updated_at = timestamp()
                    RETURN a.agent_id
                """, {
                    "agent_id": agent_id,
                    "name": personality_data.get("name", "Unknown"),
                    "role": personality_data.get("role", "Generic Agent"),
                    "description": personality_data.get("short_description", ""),
                    "personality": json.dumps(personality_data),
                    "metrics": json.dumps(metrics_data),
                    "overall_score": overall_score
                })
                return result.single()[0]
        else:
            # Store in memory
            self._in_memory_storage[agent_id] = {
                "agent_id": agent_id,
                "name": personality_data.get("name", "Unknown"),
                "role": personality_data.get("role", "Generic Agent"),
                "description": personality_data.get("short_description", ""),
                "personality": personality_data,
                "metrics": metrics_data,
                "overall_score": overall_score,
                "created_at": datetime.now().timestamp(),
                "updated_at": datetime.now().timestamp()
            }
            return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[Dict]:
        """
        Retrieve an agent by ID.
        
        Args:
            agent_id: The agent's unique ID
            
        Returns:
            The agent's data or None if not found
        """
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Agent {agent_id: $agent_id})
                    RETURN a
                """, {"agent_id": agent_id})
                
                record = result.single()
                if not record:
                    return None
                    
                agent_node = record[0]
                return self._normalize_agent_data(agent_node)
        else:
            # Retrieve from memory
            if agent_id not in self._in_memory_storage:
                return None
                
            return self._in_memory_storage[agent_id]
    
    def find_agents(self, criteria: Dict[str, Any], limit: int = 10) -> List[Dict]:
        """
        Find agents matching the given criteria.
        
        Args:
            criteria: Search criteria (role, min_score, etc.)
            limit: Maximum number of results
            
        Returns:
            List of matching agents
        """
        if self.driver:
            with self.driver.session() as session:
                # Build dynamic query based on criteria
                query_parts = ["MATCH (a:Agent)"]
                params = {}
                
                # Add WHERE clauses based on criteria
                where_clauses = []
                
                if "role" in criteria:
                    where_clauses.append("a.role = $role")
                    params["role"] = criteria["role"]
                
                if "min_score" in criteria:
                    where_clauses.append("a.overall_score >= $min_score")
                    params["min_score"] = criteria["min_score"]
                
                if "expertise" in criteria:
                    where_clauses.append("a.personality CONTAINS $expertise")
                    params["expertise"] = f'"{criteria["expertise"]}"'
                
                if where_clauses:
                    query_parts.append("WHERE " + " AND ".join(where_clauses))
                
                # Add order and limit
                query_parts.append("RETURN a ORDER BY a.overall_score DESC LIMIT $limit")
                params["limit"] = limit
                
                # Execute query
                result = session.run(" ".join(query_parts), params)
                
                return [self._normalize_agent_data(record[0]) for record in result]
        else:
            # Filter in-memory data
            filtered = self._in_memory_storage.values()
            
            if "role" in criteria:
                filtered = [a for a in filtered if a["role"] == criteria["role"]]
            
            if "min_score" in criteria:
                filtered = [a for a in filtered if a["overall_score"] >= criteria["min_score"]]
            
            if "expertise" in criteria:
                filtered = [a for a in filtered 
                            if criteria["expertise"] in json.dumps(a["personality"])]
            
            # Sort and limit
            sorted_agents = sorted(filtered, key=lambda a: a["overall_score"], reverse=True)
            return sorted_agents[:limit]
    
    def update_metrics(self, agent_id: str, metrics_update: Dict[str, float]) -> bool:
        """
        Update performance metrics for an agent.
        
        Args:
            agent_id: The agent's unique ID
            metrics_update: Dictionary of metrics to update
            
        Returns:
            Success status
        """
        # Get current agent data
        agent_data = self.get_agent(agent_id)
        if not agent_data:
            return False
        
        # Update metrics
        current_metrics = agent_data.get("metrics", {})
        for key, value in metrics_update.items():
            if key in current_metrics:
                current_metrics[key] = value
        
        # Calculate new overall score
        metrics_model = AgentPerformanceMetrics(**current_metrics)
        overall_score = metrics_model.calculate_overall_score()
        
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Agent {agent_id: $agent_id})
                    SET a.metrics = $metrics,
                        a.overall_score = $overall_score,
                        a.updated_at = timestamp()
                    RETURN a.agent_id
                """, {
                    "agent_id": agent_id,
                    "metrics": json.dumps(current_metrics),
                    "overall_score": overall_score
                })
                return result.single() is not None
        else:
            # Update in memory
            if agent_id not in self._in_memory_storage:
                return False
                
            self._in_memory_storage[agent_id]["metrics"] = current_metrics
            self._in_memory_storage[agent_id]["overall_score"] = overall_score
            self._in_memory_storage[agent_id]["updated_at"] = datetime.now().timestamp()
            return True
    
    def build_optimal_team(self, problem_description: str, team_size: int = 4) -> List[Dict]:
        """
        Build an optimal team of agents for solving a given problem.
        
        Args:
            problem_description: Description of the problem to solve
            team_size: Number of agents in the team
            
        Returns:
            List of selected agents
        """
        # In a real implementation, this would use semantic matching
        # For this demo, we'll use a simple approach
        
        # 1. First, get a high-performing facilitator
        facilitator = self.find_agents({
            "role": "Collaboration Facilitator", 
            "min_score": 0.6
        }, limit=1)
        
        if not facilitator:
            # Fallback to any high-performing agent
            facilitator = self.find_agents({"min_score": 0.7}, limit=1)
        
        selected_agents = facilitator if facilitator else []
        
        # 2. Fill the team with diverse, high-performing agents
        remaining_slots = team_size - len(selected_agents)
        if remaining_slots > 0:
            # Get top performing agents excluding already selected ones
            selected_ids = [a["agent_id"] for a in selected_agents]
            
            if self.driver:
                with self.driver.session() as session:
                    result = session.run("""
                        MATCH (a:Agent)
                        WHERE NOT a.agent_id IN $selected_ids
                        RETURN a
                        ORDER BY a.overall_score DESC
                        LIMIT $limit
                    """, {
                        "selected_ids": selected_ids,
                        "limit": remaining_slots
                    })
                    
                    for record in result:
                        selected_agents.append(self._normalize_agent_data(record[0]))
            else:
                # Filter in-memory
                remaining = [a for a in self._in_memory_storage.values() 
                            if a["agent_id"] not in selected_ids]
                sorted_remaining = sorted(remaining, key=lambda a: a["overall_score"], reverse=True)
                selected_agents.extend(sorted_remaining[:remaining_slots])
        
        return selected_agents
    
    def record_interaction(self, agent_id: str, performance_data: Dict[str, Any]) -> bool:
        """
        Record an agent interaction and update metrics.
        
        Args:
            agent_id: The agent's unique ID
            performance_data: Performance data from the interaction
            
        Returns:
            Success status
        """
        # Get current agent
        agent = self.get_agent(agent_id)
        if not agent:
            return False
        
        # Get current metrics
        current_metrics = agent.get("metrics", {})
        
        # Update invocation count
        current_metrics["total_invocations"] = current_metrics.get("total_invocations", 0) + 1
        
        # Update token count if provided
        if "tokens_consumed" in performance_data:
            current_metrics["total_tokens_consumed"] = current_metrics.get("total_tokens_consumed", 0) + performance_data["tokens_consumed"]
        
        # Update performance metrics if provided
        metric_fields = [
            "task_success_rate", "context_preservation", "creativity_score",
            "pragmatism_score", "reference_quality", "contribution_relevance",
            "team_alignment"
        ]
        
        for field in metric_fields:
            if field in performance_data:
                # Apply exponential moving average to smooth metrics
                current_value = current_metrics.get(field, 0.0)
                alpha = 0.3  # Weight for new observation
                new_value = (alpha * performance_data[field]) + ((1 - alpha) * current_value)
                current_metrics[field] = new_value
        
        # Update timestamp
        current_metrics["last_evaluation"] = datetime.now().isoformat()
        
        # Calculate new overall score
        metrics_model = AgentPerformanceMetrics(**current_metrics)
        overall_score = metrics_model.calculate_overall_score()
        
        # Update agent metrics
        if self.driver:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (a:Agent {agent_id: $agent_id})
                    SET a.metrics = $metrics,
                        a.overall_score = $overall_score,
                        a.updated_at = timestamp()
                    RETURN a.agent_id
                """, {
                    "agent_id": agent_id,
                    "metrics": json.dumps(current_metrics),
                    "overall_score": overall_score
                })
                return result.single() is not None
        else:
            # Update in memory
            self._in_memory_storage[agent_id]["metrics"] = current_metrics
            self._in_memory_storage[agent_id]["overall_score"] = overall_score
            self._in_memory_storage[agent_id]["updated_at"] = datetime.now().timestamp()
            return True
    
    def _prepare_personality_data(self, personality: Dict) -> Dict:
        """Prepare personality data for storage by normalizing structures."""
        # Handle pydantic models
        if hasattr(personality, "dict"):
            return personality.dict()
        return personality
    
    def _normalize_agent_data(self, agent_node: Any) -> Dict:
        """Normalize agent data from Neo4j node or in-memory storage."""
        if self.driver:
            # Neo4j node
            data = dict(agent_node)
            
            # Parse JSON fields
            if "personality" in data and isinstance(data["personality"], str):
                try:
                    data["personality"] = json.loads(data["personality"])
                except:
                    pass
                
            if "metrics" in data and isinstance(data["metrics"], str):
                try:
                    data["metrics"] = json.loads(data["metrics"])
                except:
                    pass
                
            return data
        else:
            # Already properly formatted in memory
            return agent_node
    
    def close(self) -> None:
        """Close the connection to Neo4j."""
        if self.driver:
            self.driver.close()
