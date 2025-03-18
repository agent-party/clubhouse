# Evolutionary Framework

## Introduction

The evolutionary framework integrates principles from evolutionary algorithms with multi-agent systems to create a robust approach for complex problem-solving. Building upon the foundational concepts described in the AI Agent Orchestration Framework paper, this implementation provides a structured method for agents to collaboratively evolve solutions through critical analysis and refinement.

## Core Principles

1. **Specialized Agent Roles**: Distinct agent types with focused responsibilities
2. **Iterative Improvement**: Systematic evolution of solutions through multiple generations
3. **Critical Analysis**: Socratic questioning and robust evaluation
4. **Diversity Preservation**: Maintaining a diverse solution population
5. **Human Oversight**: Strategic human intervention at configurable points

## Agent Roles

The evolutionary framework defines specialized agent roles that collaborate within the orchestration platform:

### Generator Agents

Generators create initial solution candidates or ideas. They implement the `GeneratorCapability` which extends the base capabilities with creative generation functions.

```python
@protocol
class GeneratorCapability(BaseCapability):
    """Capability for generating creative solutions to problems."""
    
    class Parameters(BaseModel):
        problem_statement: str
        constraints: Optional[List[str]] = None
        previous_attempts: Optional[List[Dict[str, Any]]] = None
        creativity_level: Optional[float] = 0.7
        
    class Response(BaseModel):
        solutions: List[Dict[str, Any]]
        rationale: Optional[str] = None
        
    async def generate(self, params: Parameters) -> Response:
        """Generate one or more candidate solutions to the given problem."""
        pass
```

### Critic Agents

Critics analyze solutions and identify flaws, inconsistencies, or areas for improvement. They implement the `CriticCapability` which extends base capabilities with analytical functions.

```python
@protocol
class CriticCapability(BaseCapability):
    """Capability for critically analyzing solutions."""
    
    class Parameters(BaseModel):
        solution: Dict[str, Any]
        evaluation_criteria: Optional[List[str]] = None
        previous_critiques: Optional[List[Dict[str, Any]]] = None
        critique_depth: Optional[str] = "standard"  # "light", "standard", "deep"
        
    class Response(BaseModel):
        critiques: List[Dict[str, str]]  # aspect: critique
        questions: Optional[List[str]] = None
        strengths: Optional[List[str]] = None
        improvement_suggestions: Optional[List[str]] = None
        
    async def critique(self, params: Parameters) -> Response:
        """Provide critical analysis of the given solution."""
        pass
```

### Refiner Agents

Refiners improve solutions based on feedback, implementing the `RefinerCapability` for solution enhancement.

```python
@protocol
class RefinerCapability(BaseCapability):
    """Capability for refining solutions based on feedback."""
    
    class Parameters(BaseModel):
        original_solution: Dict[str, Any]
        critiques: List[Dict[str, str]]
        improvement_suggestions: Optional[List[str]] = None
        constraints: Optional[List[str]] = None
        
    class Response(BaseModel):
        refined_solution: Dict[str, Any]
        changes_made: List[str]
        rationale: Optional[str] = None
        
    async def refine(self, params: Parameters) -> Response:
        """Refine a solution based on critical feedback."""
        pass
```

### Evaluator Agents

Evaluators score solutions against defined criteria, implementing the `EvaluatorCapability` for objective assessment.

```python
@protocol
class EvaluatorCapability(BaseCapability):
    """Capability for evaluating and ranking solutions."""
    
    class Parameters(BaseModel):
        solutions: List[Dict[str, Any]]
        evaluation_criteria: Dict[str, float]  # criterion: weight
        scoring_method: Optional[str] = "weighted_sum"  # "weighted_sum", "rank_based", "pareto"
        
    class Response(BaseModel):
        scores: List[Dict[str, float]]  # solution_id: score
        criterion_scores: Optional[Dict[str, Dict[str, float]]] = None  # solution_id: {criterion: score}
        ranking: List[str]  # solution_ids in rank order
        explanation: Optional[str] = None
        
    async def evaluate(self, params: Parameters) -> Response:
        """Evaluate and rank multiple solutions according to criteria."""
        pass
```

## Evolutionary Process

The evolutionary process is implemented as a workflow that coordinates agent interactions through multiple iterations:

![Evolutionary Process Flow](diagrams/evolutionary_flow.png)

### Process Stages

1. **Initialization**: Define problem and evaluation criteria
2. **Generation**: Create initial solution population
3. **Evaluation Cycle**:
   a. Critique solutions
   b. Refine based on critiques
   c. Evaluate refined solutions
   d. Select top candidates
4. **Diversity Management**: Ensure solution diversity
5. **Iteration**: Repeat the cycle for multiple generations
6. **Termination**: End when criteria are met or iterations complete
7. **Final Selection**: Choose the best solution(s)

### Implementation in Workflow Engine

The WorkflowEngine coordinates the evolutionary process:

```python
class EvolutionaryWorkflow:
    """Coordinates the evolutionary problem-solving process."""
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        problem_statement: str,
        evaluation_criteria: Dict[str, float],
        population_size: int = 5,
        max_generations: int = 3,
        diversity_threshold: float = 0.3,
        human_approval_checkpoints: Optional[List[str]] = None
    ):
        self.service_registry = service_registry
        self.problem_statement = problem_statement
        self.evaluation_criteria = evaluation_criteria
        self.population_size = population_size
        self.max_generations = max_generations
        self.diversity_threshold = diversity_threshold
        self.human_approval_checkpoints = human_approval_checkpoints or ["final_selection"]
        
        # Initialize agent access
        self.generator_agent = self.service_registry.get_service(GeneratorAgentProtocol)
        self.critic_agent = self.service_registry.get_service(CriticAgentProtocol)
        self.refiner_agent = self.service_registry.get_service(RefinerAgentProtocol)
        self.evaluator_agent = self.service_registry.get_service(EvaluatorAgentProtocol)
        
        # State tracking
        self.current_generation = 0
        self.solution_population = []
        self.evolution_history = []
        
    async def initialize(self) -> None:
        """Initialize the workflow and generate initial population."""
        # Generate initial population
        generator_params = GeneratorCapability.Parameters(
            problem_statement=self.problem_statement,
            constraints=None,  # Initial generation has no constraints
            creativity_level=0.8  # Higher creativity for diverse initial population
        )
        
        generation_result = await self.generator_agent.generate(generator_params)
        self.solution_population = generation_result.solutions
        self.evolution_history.append({
            "generation": 0,
            "event": "initialization",
            "solutions": self.solution_population,
            "timestamp": datetime.now().isoformat()
        })
        
    async def run_evolution_cycle(self) -> None:
        """Run a complete evolution cycle (generation)."""
        self.current_generation += 1
        
        # 1. Critique phase
        critiques = []
        for solution in self.solution_population:
            critic_params = CriticCapability.Parameters(
                solution=solution,
                evaluation_criteria=list(self.evaluation_criteria.keys())
            )
            critique_result = await self.critic_agent.critique(critic_params)
            critiques.append(critique_result)
            
        # 2. Refinement phase
        refined_solutions = []
        for solution, critique in zip(self.solution_population, critiques):
            refiner_params = RefinerCapability.Parameters(
                original_solution=solution,
                critiques=critique.critiques,
                improvement_suggestions=critique.improvement_suggestions
            )
            refinement_result = await self.refiner_agent.refine(refiner_params)
            refined_solutions.append(refinement_result.refined_solution)
            
        # 3. Evaluation phase
        evaluator_params = EvaluatorCapability.Parameters(
            solutions=refined_solutions,
            evaluation_criteria=self.evaluation_criteria
        )
        evaluation_result = await self.evaluator_agent.evaluate(evaluator_params)
        
        # 4. Selection phase
        selected_solutions = self._select_diverse_solutions(
            refined_solutions, 
            evaluation_result.scores, 
            evaluation_result.ranking
        )
        
        # 5. Update population
        self.solution_population = selected_solutions
        
        # 6. Record history
        self.evolution_history.append({
            "generation": self.current_generation,
            "event": "evolution_cycle",
            "critiques": critiques,
            "refined_solutions": refined_solutions,
            "evaluation": evaluation_result,
            "selected_solutions": selected_solutions,
            "timestamp": datetime.now().isoformat()
        })
        
        # 7. Human approval checkpoint if needed
        if "after_each_generation" in self.human_approval_checkpoints:
            await self._request_human_approval(f"Generation {self.current_generation} complete")
    
    def _select_diverse_solutions(
        self, 
        solutions: List[Dict[str, Any]], 
        scores: List[Dict[str, float]], 
        ranking: List[str]
    ) -> List[Dict[str, Any]]:
        """Select diverse high-quality solutions for the next generation."""
        # Implementation of diversity-aware selection algorithm
        # This combines selection pressure (favoring high scores)
        # with diversity preservation (maintaining different approaches)
        # ...
        
    async def _request_human_approval(self, checkpoint_name: str) -> None:
        """Request human approval at a defined checkpoint."""
        # Implementation of human approval workflow
        # ...
        
    async def run_full_evolution(self) -> Dict[str, Any]:
        """Run the complete evolutionary process until termination."""
        await self.initialize()
        
        for _ in range(self.max_generations):
            await self.run_evolution_cycle()
            
            # Check termination conditions
            if self._check_convergence():
                break
                
        # Final evaluation and selection
        final_result = await self._final_selection()
        
        # Human approval for final selection if needed
        if "final_selection" in self.human_approval_checkpoints:
            await self._request_human_approval("final_selection")
            
        return final_result
        
    def _check_convergence(self) -> bool:
        """Check if the evolution has converged to a stable solution."""
        # Implementation of convergence detection algorithm
        # ...
        
    async def _final_selection(self) -> Dict[str, Any]:
        """Perform final selection of the best solution(s)."""
        # Implementation of final selection logic
        # ...
```

## Diversity Management

Diversity management is a critical aspect of the evolutionary framework, preventing premature convergence and ensuring broad exploration of the solution space:

```python
class DiversityManager:
    """Manages diversity in the solution population."""
    
    def __init__(
        self, 
        embedding_service: EmbeddingServiceProtocol,
        diversity_threshold: float = 0.3,
        selection_pressure: float = 0.7
    ):
        self.embedding_service = embedding_service
        self.diversity_threshold = diversity_threshold
        self.selection_pressure = selection_pressure
        
    async def compute_solution_diversity(
        self, 
        solutions: List[Dict[str, Any]]
    ) -> float:
        """Compute the diversity score of a solution set."""
        if len(solutions) <= 1:
            return 0.0
            
        # Get embeddings for all solutions
        solution_texts = [json.dumps(solution) for solution in solutions]
        embeddings = await self.embedding_service.embed_texts(solution_texts)
        
        # Compute pairwise cosine similarities
        similarity_sum = 0.0
        pair_count = 0
        
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarity_sum += similarity
                pair_count += 1
                
        # Average similarity across all pairs
        avg_similarity = similarity_sum / pair_count if pair_count > 0 else 0.0
        
        # Convert similarity to diversity (1 - similarity)
        diversity = 1.0 - avg_similarity
        
        return diversity
        
    async def select_diverse_solutions(
        self,
        solutions: List[Dict[str, Any]],
        scores: Dict[str, float],
        num_to_select: int
    ) -> List[Dict[str, Any]]:
        """Select a diverse subset of solutions based on scores and diversity."""
        if len(solutions) <= num_to_select:
            return solutions
            
        # Get solution embeddings
        solution_texts = [json.dumps(solution) for solution in solutions]
        embeddings = await self.embedding_service.embed_texts(solution_texts)
        
        # Compute diversity contribution for each solution
        diversity_scores = self._compute_diversity_contributions(embeddings)
        
        # Combine quality scores with diversity scores
        combined_scores = {}
        for i, solution in enumerate(solutions):
            solution_id = str(i)
            quality_score = scores.get(solution_id, 0.0)
            diversity_score = diversity_scores[i]
            
            # Weighted combination
            combined_scores[solution_id] = (
                self.selection_pressure * quality_score +
                (1 - self.selection_pressure) * diversity_score
            )
            
        # Select top solutions by combined score
        selected_indices = sorted(
            range(len(solutions)), 
            key=lambda i: combined_scores[str(i)], 
            reverse=True
        )[:num_to_select]
        
        return [solutions[i] for i in selected_indices]
        
    def _compute_diversity_contributions(
        self, 
        embeddings: List[List[float]]
    ) -> List[float]:
        """Compute how much each solution contributes to overall diversity."""
        # Implementation of diversity contribution algorithm
        # ...
```

## Fitness Functions

The framework implements various fitness functions for evaluating solutions:

```python
class FitnessFunctions:
    """Provides fitness functions for solution evaluation."""
    
    @staticmethod
    def weighted_sum(
        solution: Dict[str, Any],
        criteria: Dict[str, float],
        criterion_scores: Dict[str, float]
    ) -> float:
        """Calculate weighted sum fitness score."""
        score = 0.0
        total_weight = sum(criteria.values())
        
        for criterion, weight in criteria.items():
            if criterion in criterion_scores:
                normalized_weight = weight / total_weight
                score += normalized_weight * criterion_scores[criterion]
                
        return score
        
    @staticmethod
    def rank_based(
        solution_id: str,
        criterion_rankings: Dict[str, List[str]]
    ) -> float:
        """Calculate rank-based fitness score."""
        total_rank = 0
        for criterion, rankings in criterion_rankings.items():
            if solution_id in rankings:
                # Convert to 0-indexed rank and invert (higher is better)
                rank = len(rankings) - rankings.index(solution_id) - 1
                total_rank += rank
                
        # Normalize to 0-1 range
        max_possible_rank = sum(len(rankings) - 1 for rankings in criterion_rankings.values())
        normalized_score = total_rank / max_possible_rank if max_possible_rank > 0 else 0.0
        
        return normalized_score
        
    @staticmethod
    def pareto_dominance(
        solution_id: str,
        all_solutions: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate Pareto dominance-based fitness."""
        # Count how many solutions this one dominates
        domination_count = 0
        solution_scores = all_solutions[solution_id]
        
        for other_id, other_scores in all_solutions.items():
            if other_id == solution_id:
                continue
                
            # Check if solution dominates other
            dominates = True
            at_least_one_better = False
            
            for criterion in solution_scores:
                if solution_scores[criterion] < other_scores.get(criterion, 0):
                    dominates = False
                    break
                if solution_scores[criterion] > other_scores.get(criterion, 0):
                    at_least_one_better = True
                    
            if dominates and at_least_one_better:
                domination_count += 1
                
        # Normalize by maximum possible domination count
        normalized_score = domination_count / (len(all_solutions) - 1) if len(all_solutions) > 1 else 0.0
        
        return normalized_score
```

## Neo4j Integration for Evolution Tracking

The evolutionary process is tracked in Neo4j for persistence and analysis:

```python
class EvolutionTracker:
    """Tracks the evolutionary process in Neo4j."""
    
    def __init__(self, neo4j_service: Neo4jServiceProtocol):
        self.neo4j_service = neo4j_service
        
    async def record_generation(
        self,
        workflow_id: str,
        generation: int,
        solutions: List[Dict[str, Any]],
        scores: Dict[str, float]
    ) -> None:
        """Record a generation in the evolution process."""
        # Create Generation node
        generation_query = """
        CREATE (g:Generation {
            workflow_id: $workflow_id,
            generation_number: $generation,
            timestamp: datetime()
        })
        RETURN g
        """
        
        generation_params = {
            "workflow_id": workflow_id,
            "generation": generation
        }
        
        generation_result = await self.neo4j_service.execute_query(
            generation_query, 
            generation_params
        )
        
        generation_id = generation_result[0]["g"]["id"]
        
        # Create Solution nodes and relationships
        for i, solution in enumerate(solutions):
            solution_id = f"{workflow_id}_{generation}_{i}"
            score = scores.get(str(i), 0.0)
            
            solution_query = """
            CREATE (s:Solution {
                solution_id: $solution_id,
                content: $content,
                score: $score
            })
            WITH s
            MATCH (g:Generation {id: $generation_id})
            CREATE (g)-[:CONTAINS]->(s)
            RETURN s
            """
            
            solution_params = {
                "solution_id": solution_id,
                "content": json.dumps(solution),
                "score": score,
                "generation_id": generation_id
            }
            
            await self.neo4j_service.execute_query(solution_query, solution_params)
            
    async def record_evolution(
        self,
        workflow_id: str,
        parent_solution_id: str,
        child_solution_id: str,
        critique_id: str,
        refinement_type: str
    ) -> None:
        """Record the evolution relationship between solutions."""
        evolution_query = """
        MATCH (parent:Solution {solution_id: $parent_id})
        MATCH (child:Solution {solution_id: $child_id})
        MATCH (critique:Critique {critique_id: $critique_id})
        CREATE (parent)-[:EVOLVED_TO {
            refinement_type: $refinement_type,
            timestamp: datetime()
        }]->(child)
        CREATE (critique)-[:INFLUENCED]->(child)
        """
        
        evolution_params = {
            "parent_id": parent_solution_id,
            "child_id": child_solution_id,
            "critique_id": critique_id,
            "refinement_type": refinement_type
        }
        
        await self.neo4j_service.execute_query(evolution_query, evolution_params)
        
    async def get_evolution_tree(self, workflow_id: str) -> Dict[str, Any]:
        """Retrieve the complete evolution tree for a workflow."""
        query = """
        MATCH (g:Generation {workflow_id: $workflow_id})-[:CONTAINS]->(s:Solution)
        OPTIONAL MATCH (s1:Solution)-[e:EVOLVED_TO]->(s2:Solution)
        WHERE s1.solution_id CONTAINS $workflow_id AND s2.solution_id CONTAINS $workflow_id
        RETURN g, s, e, s1, s2
        ORDER BY g.generation_number, s.solution_id
        """
        
        params = {"workflow_id": workflow_id}
        
        result = await self.neo4j_service.execute_query(query, params)
        
        # Transform the result into a structured tree
        # ...
        
        return tree
```

## Human-in-the-Loop Integration

The evolutionary framework integrates human oversight at configurable checkpoints:

```python
class HumanApprovalWorkflow:
    """Manages human approval workflows for the evolutionary process."""
    
    def __init__(
        self,
        notification_service: NotificationServiceProtocol,
        timeout_seconds: int = 300,
        default_action: str = "wait"  # "wait", "proceed", "abort"
    ):
        self.notification_service = notification_service
        self.timeout_seconds = timeout_seconds
        self.default_action = default_action
        self.pending_approvals = {}
        
    async def request_approval(
        self,
        workflow_id: str,
        checkpoint_name: str,
        context: Dict[str, Any],
        approvers: List[str]
    ) -> ApprovalResult:
        """Request human approval at a checkpoint."""
        # Generate approval request ID
        approval_id = str(uuid.uuid4())
        
        # Create approval request
        request = {
            "approval_id": approval_id,
            "workflow_id": workflow_id,
            "checkpoint_name": checkpoint_name,
            "context": context,
            "requested_at": datetime.now().isoformat(),
            "status": "pending",
            "approvers": approvers,
            "responses": {}
        }
        
        # Store in pending approvals
        self.pending_approvals[approval_id] = request
        
        # Send notification to approvers
        for approver in approvers:
            await self.notification_service.send_notification(
                recipient=approver,
                notification_type="approval_request",
                content={
                    "approval_id": approval_id,
                    "workflow_id": workflow_id,
                    "checkpoint_name": checkpoint_name,
                    "summary": self._generate_approval_summary(context),
                    "expiration": (datetime.now() + timedelta(seconds=self.timeout_seconds)).isoformat()
                }
            )
            
        # Wait for approval responses
        approval_result = await self._wait_for_approval(approval_id)
        
        return approval_result
        
    async def _wait_for_approval(self, approval_id: str) -> ApprovalResult:
        """Wait for approval responses, with timeout handling."""
        # Implementation of waiting logic with timeout
        # ...
        
    def _generate_approval_summary(self, context: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the approval context."""
        # Implementation of context summarization for humans
        # ...
        
    async def record_approval_response(
        self,
        approval_id: str,
        approver: str,
        decision: str,
        comments: Optional[str] = None
    ) -> None:
        """Record a human approver's response."""
        if approval_id not in self.pending_approvals:
            raise ValueError(f"Approval request {approval_id} not found")
            
        # Update the approval request with the response
        request = self.pending_approvals[approval_id]
        request["responses"][approver] = {
            "decision": decision,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if all approvers have responded
        if len(request["responses"]) == len(request["approvers"]):
            request["status"] = "completed"
```

## Practical Application Examples

The evolutionary framework can be applied to various domains:

### Strategic Planning

```python
# Example: Business Strategy Evolution
async def evolve_business_strategy(
    problem_statement: str,
    constraints: List[str],
    evaluation_criteria: Dict[str, float]
) -> Dict[str, Any]:
    """Evolve an optimal business strategy for a given problem."""
    # Initialize workflow with specialized agents for business strategy
    workflow = EvolutionaryWorkflow(
        service_registry=service_registry,
        problem_statement=problem_statement,
        evaluation_criteria=evaluation_criteria,
        population_size=5,
        max_generations=3,
        human_approval_checkpoints=["after_generation_1", "final_selection"]
    )
    
    # Run evolution process
    result = await workflow.run_full_evolution()
    
    return result
```

### Product Design

```python
# Example: Product Design Evolution
async def evolve_product_design(
    product_requirements: Dict[str, Any],
    constraints: List[str],
    evaluation_criteria: Dict[str, float]
) -> Dict[str, Any]:
    """Evolve an optimal product design based on requirements."""
    # Format problem statement from requirements
    problem_statement = f"Design a product that meets the following requirements: {json.dumps(product_requirements)}"
    
    # Initialize workflow with specialized agents for product design
    workflow = EvolutionaryWorkflow(
        service_registry=service_registry,
        problem_statement=problem_statement,
        evaluation_criteria=evaluation_criteria,
        population_size=7,  # Larger population for more design diversity
        max_generations=4,
        human_approval_checkpoints=["after_each_generation", "final_selection"]
    )
    
    # Run evolution process
    result = await workflow.run_full_evolution()
    
    return result
```

## Conclusion

The evolutionary framework provides a structured approach to complex problem-solving, combining the strengths of specialized AI agents with human oversight. By implementing well-defined interfaces, standardized communication patterns, and a robust persistence model, the framework enables the systematic evolution of solutions through critical analysis and refinement.

The integration with the Agent Orchestration Platform ensures that the evolutionary process benefits from the platform's core services like Neo4j persistence, event-driven communication, and human-in-the-loop approval workflows, creating a powerful environment for collaborative problem-solving between AI agents and humans.
