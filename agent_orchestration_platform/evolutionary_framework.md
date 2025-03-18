# Evolutionary Framework

## Overview

The Evolutionary Framework provides the foundation for continuous agent improvement through selective, iterative refinement based on human feedback and performance metrics. By applying principles inspired by natural selection and Stoic philosophy, the framework enables agents to adapt to evolving human needs while maintaining alignment with ethical principles.

## Core Principles

1. **Dialogue-Driven Fitness**: Agent fitness is determined through natural human dialogue rather than explicit numerical ratings
2. **Multi-Dimensional Evaluation**: Agents are evaluated across multiple philosophical and performance dimensions
3. **Purposeful Variation**: Mutations are targeted to address specific feedback rather than random exploration
4. **Lineage Preservation**: Successful agent traits are preserved and combined through controlled crossover
5. **Transparent Evolution**: All evolutionary changes are explainable and traceable to human feedback

## Population Management

```python
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime

class AgentGenome(BaseModel):
    """The genetic blueprint of an agent."""
    id: UUID = Field(default_factory=uuid4)
    base_model: str
    parameter_space: Dict[str, float]
    capability_weights: Dict[str, float]
    personality_traits: Dict[str, float]
    knowledge_domains: Set[str]
    archetype_alignment: Dict[str, float]
    parent_ids: List[UUID] = Field(default_factory=list)
    generation: int = 0
    creation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class AgentPopulation:
    """Manages a population of agent genomes."""
    
    def __init__(self, 
                population_size: int,
                archetype_weights: Dict[str, float],
                neo4j_service,
                event_bus):
        self.population_size = population_size
        self.archetype_weights = archetype_weights
        self.neo4j_service = neo4j_service
        self.event_bus = event_bus
        
    async def initialize_population(self) -> List[AgentGenome]:
        """Create an initial population with diverse characteristics."""
        population = []
        
        # Create initial agents with diverse traits across archetypes
        for _ in range(self.population_size):
            genome = self._create_random_genome()
            
            # Persist to graph database
            await self.neo4j_service.create_agent_genome(genome.dict())
            
            # Add to population
            population.append(genome)
            
        # Publish population initialized event
        await self.event_bus.publish(
            "evolution.population.initialized",
            {"population_size": len(population)}
        )
        
        return population
    
    async def evolve_population(self, 
                              fitness_scores: Dict[UUID, float],
                              feedback_data: Dict[UUID, Dict]) -> List[AgentGenome]:
        """Evolve the population based on fitness scores and feedback."""
        # Select parents based on fitness
        parents = await self._select_parents(fitness_scores)
        
        # Create next generation through crossover and mutation
        next_generation = []
        
        # Elite selection - keep top performers
        elite_size = int(self.population_size * 0.1)
        elite_agents = self._get_elite_agents(fitness_scores, elite_size)
        next_generation.extend(elite_agents)
        
        # Fill rest of population with offspring
        remaining_slots = self.population_size - len(next_generation)
        
        for _ in range(remaining_slots):
            # Select two parents
            parent1, parent2 = self._select_parent_pair(parents)
            
            # Create offspring through crossover
            offspring = await self._crossover(parent1, parent2)
            
            # Apply targeted mutations based on feedback
            offspring = await self._mutate(offspring, feedback_data)
            
            # Add to next generation
            next_generation.append(offspring)
            
            # Persist to graph database with lineage
            await self.neo4j_service.create_agent_genome(offspring.dict())
        
        # Publish population evolved event
        await self.event_bus.publish(
            "evolution.population.evolved",
            {
                "generation": next_generation[0].generation,
                "population_size": len(next_generation)
            }
        )
        
        return next_generation
```

## Cost Management in Evolution

The evolutionary framework incorporates cost management as a critical dimension in agent evolution, ensuring that evolved agents are not only effective but also cost-efficient.

### Cost as a Fitness Dimension

Cost efficiency is a first-class fitness dimension that influences selection and evolution:

```python
class CostEfficientFitness(BaseModel):
    """Fitness metrics focused on cost efficiency."""
    token_efficiency: float = Field(..., ge=0.0, le=1.0)  # Output quality per token
    budget_adherence: float = Field(..., ge=0.0, le=1.0)  # Ability to stay within budget
    cost_performance_ratio: float = Field(..., ge=0.0, le=1.0)  # Output quality relative to cost
    overall_cost_fitness: float = Field(..., ge=0.0, le=1.0)  # Combined cost fitness

class EvolutionaryFitness(BaseModel):
    """Complete fitness metrics for an agent."""
    # Other fitness dimensions
    performance_fitness: Dict[str, float]
    human_alignment_fitness: Dict[str, float]
    adaptability_fitness: Dict[str, float]
    
    # Cost fitness dimension
    cost_fitness: CostEfficientFitness
    
    # Overall combined fitness
    overall_fitness: float = Field(..., ge=0.0, le=1.0)
    
    # Calculation weights
    fitness_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "performance": 0.4,
            "human_alignment": 0.3,
            "adaptability": 0.2,
            "cost_efficiency": 0.1  # Cost is explicitly weighted in overall fitness
        }
    )
```

### Budget-Aware Selection Strategy

The selection process incorporates budget considerations:

```python
async def budget_conscious_selection(self, 
                                  population: List[Dict[str, Any]],
                                  task_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Select agents with consideration for both fitness and cost-efficiency."""
    # Get task budget requirements
    target_budget = task_requirements.get("budget", {})
    budget_priority = target_budget.get("priority", "balanced")  # Options: "low_cost", "balanced", "performance"
    
    # Adjust selection weights based on budget priority
    selection_weights = {
        "overall_fitness": 0.7,
        "cost_fitness": 0.3
    }
    
    if budget_priority == "low_cost":
        selection_weights = {
            "overall_fitness": 0.4,
            "cost_fitness": 0.6
        }
    elif budget_priority == "performance":
        selection_weights = {
            "overall_fitness": 0.9,
            "cost_fitness": 0.1
        }
    
    # Calculate selection scores
    selection_scores = []
    for agent in population:
        score = (
            agent["fitness"]["overall_fitness"] * selection_weights["overall_fitness"] +
            agent["fitness"]["cost_fitness"]["overall_cost_fitness"] * selection_weights["cost_fitness"]
        )
        selection_scores.append(score)
    
    # Select parents using tournament selection with weighted scores
    selected_indices = self._tournament_selection(
        scores=selection_scores,
        num_selected=len(population) // 2,
        tournament_size=3
    )
    
    return [population[i] for i in selected_indices]
```

### Cost-Optimizing Mutations

Special mutation operators focus on cost optimization:

```python
async def apply_cost_optimizing_mutations(self, 
                                       genome: Dict[str, Any],
                                       mutation_rate: float,
                                       cost_history: Dict[str, Any]) -> Dict[str, Any]:
    """Apply mutations specifically aimed at improving cost efficiency."""
    # Start with a copy of the genome
    mutated_genome = copy.deepcopy(genome)
    
    # Get cost-related parameters
    cost_params = {
        "max_tokens": mutated_genome["parameter_space"].get("max_tokens", 1000),
        "temperature": mutated_genome["parameter_space"].get("temperature", 0.7),
        "top_p": mutated_genome["parameter_space"].get("top_p", 1.0),
        "prompt_optimization_level": mutated_genome["parameter_space"].get("prompt_optimization_level", 0.5)
    }
    
    # If cost history shows high token usage, reduce token limits
    if cost_history.get("average_token_usage", 0) > cost_history.get("target_token_usage", 800):
        # Reduce max tokens proportionally to overuse
        reduction_factor = cost_history.get("target_token_usage", 800) / cost_history.get("average_token_usage", 1000)
        new_max_tokens = max(int(cost_params["max_tokens"] * reduction_factor), 100)
        mutated_genome["parameter_space"]["max_tokens"] = new_max_tokens
    
    # If cost/performance ratio is poor, adjust temperature and top_p
    if cost_history.get("cost_performance_ratio", 1.0) < 0.7:
        # Make sampling more focused/deterministic to reduce iterations
        mutated_genome["parameter_space"]["temperature"] = max(cost_params["temperature"] * 0.8, 0.1)
        mutated_genome["parameter_space"]["top_p"] = max(cost_params["top_p"] * 0.9, 0.1)
    
    # Increase prompt optimization level if budget adherence is low
    if cost_history.get("budget_adherence", 1.0) < 0.8:
        mutated_genome["parameter_space"]["prompt_optimization_level"] = min(
            cost_params["prompt_optimization_level"] * 1.5,
            1.0
        )
    
    # Optimize model selection based on task complexity and cost sensitivity
    if random.random() < mutation_rate:
        models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-turbo", "gpt-4"]
        model_costs = [0.002, 0.003, 0.01, 0.03]  # Example cost per 1K tokens
        
        # If cost is critically important, prefer cheaper models
        if cost_history.get("cost_sensitivity", "medium") == "high":
            model_weights = [0.6, 0.3, 0.1, 0.0]
        elif cost_history.get("cost_sensitivity", "medium") == "medium":
            model_weights = [0.4, 0.3, 0.2, 0.1]
        else:
            model_weights = [0.2, 0.2, 0.3, 0.3]
            
        # Choose model based on weights
        model_choice = random.choices(models, weights=model_weights, k=1)[0]
        mutated_genome["parameter_space"]["model"] = model_choice
    
    return mutated_genome
```

### Budget-Aware Evolution Pipeline

The evolution pipeline explicitly incorporates budget constraints:

```python
async def evolve_with_budget_constraints(self, 
                                      population: List[Dict[str, Any]],
                                      task_requirements: Dict[str, Any],
                                      budget_allocation: float) -> List[Dict[str, Any]]:
    """Full evolution cycle with explicit budget constraints."""
    # Track estimated evolution cost
    estimated_cost = 0.0
    
    # Calculate per-operation budget
    num_operations = 5  # selection, crossover, mutation, evaluation, integration
    per_operation_budget = budget_allocation / num_operations
    
    # 1. Selection phase (budget-aware)
    selection_start = time.time()
    selected_parents = await self.budget_conscious_selection(
        population=population,
        task_requirements=task_requirements
    )
    selection_cost = self._estimate_operation_cost("selection", time.time() - selection_start)
    estimated_cost += selection_cost
    
    # Adjust remaining budget
    remaining_budget = budget_allocation - estimated_cost
    
    # 2. Crossover phase (with budget limit)
    if remaining_budget < per_operation_budget * 0.5:
        # If budget is too constrained, use cheaper crossover
        crossover_method = "simple_crossover"
    else:
        crossover_method = "optimal_crossover"
        
    crossover_start = time.time()
    offspring = await self._perform_crossover(
        parents=selected_parents,
        method=crossover_method,
        task_requirements=task_requirements
    )
    crossover_cost = self._estimate_operation_cost("crossover", time.time() - crossover_start)
    estimated_cost += crossover_cost
    
    # Update remaining budget
    remaining_budget = budget_allocation - estimated_cost
    
    # 3. Mutation phase (cost-optimizing)
    # Skip expensive mutations if budget is constrained
    mutation_rate = 0.3 if remaining_budget > per_operation_budget else 0.1
    
    mutation_start = time.time()
    mutated_offspring = []
    for genome in offspring:
        # Apply standard mutations
        mutated = await self._mutate_genome(
            genome=genome,
            mutation_rate=mutation_rate,
            task_requirements=task_requirements
        )
        
        # Apply cost-optimizing mutations
        if remaining_budget > per_operation_budget * 0.3:
            cost_history = await self.neo4j_service.get_agent_cost_history(genome["id"])
            mutated = await self.apply_cost_optimizing_mutations(
                genome=mutated,
                mutation_rate=mutation_rate * 1.5,  # Emphasize cost mutations
                cost_history=cost_history
            )
            
        mutated_offspring.append(mutated)
    
    mutation_cost = self._estimate_operation_cost("mutation", time.time() - mutation_start)
    estimated_cost += mutation_cost
    
    # Combine population
    new_population = selected_parents + mutated_offspring
    
    # Log budget usage
    await self.event_bus.publish(
        "evolution.budget_usage",
        {
            "allocated_budget": budget_allocation,
            "estimated_cost": estimated_cost,
            "remaining_budget": budget_allocation - estimated_cost,
            "operation_costs": {
                "selection": selection_cost,
                "crossover": crossover_cost,
                "mutation": mutation_cost
            }
        }
    )
    
    return new_population
```

### Resource-Efficient Evaluation

Evaluation mechanisms are designed to minimize resource consumption:

```python
async def efficient_evaluate_fitness(self,
                                  genome: Dict[str, Any],
                                  test_cases: List[Dict[str, Any]],
                                  budget_limit: float) -> Dict[str, float]:
    """Evaluate fitness with resource efficiency in mind."""
    # Start tracking evaluation cost
    eval_start_time = time.time()
    eval_cost = 0.0
    
    # Initialize fitness components
    performance_scores = {}
    cost_efficiency_scores = {}
    
    # Determine how many test cases we can afford to run
    avg_test_cost = await self._estimate_test_case_cost(test_cases[0])
    max_affordable_tests = int(budget_limit / avg_test_cost)
    
    # If we can't afford all tests, select a representative subset
    if max_affordable_tests < len(test_cases):
        # Group test cases by type/difficulty
        test_case_groups = self._group_test_cases(test_cases)
        
        # Select representative cases from each group
        selected_test_cases = self._select_representative_test_cases(
            test_case_groups,
            max_affordable_tests
        )
    else:
        selected_test_cases = test_cases
    
    # Run the affordable test cases
    for test_case in selected_test_cases:
        # Execute test
        result = await self._execute_test_case(genome, test_case)
        
        # Calculate performance score for this test
        performance_score = self._calculate_test_performance(result, test_case)
        performance_scores[test_case["id"]] = performance_score
        
        # Calculate cost efficiency for this test
        cost_efficiency = self._calculate_cost_efficiency(
            performance=performance_score,
            token_usage=result.get("token_usage", {}),
            execution_time=result.get("execution_time", 0)
        )
        cost_efficiency_scores[test_case["id"]] = cost_efficiency
        
        # Update cost tracking
        test_cost = self._calculate_test_cost(result)
        eval_cost += test_cost
        
        # Check if we're approaching budget limit
        if eval_cost > budget_limit * 0.9:
            # Log budget constraint
            await self.event_bus.publish(
                "evolution.evaluation.budget_limited",
                {
                    "genome_id": genome["id"],
                    "budget_limit": budget_limit,
                    "current_cost": eval_cost,
                    "tests_completed": len(cost_efficiency_scores),
                    "tests_remaining": len(selected_test_cases) - len(cost_efficiency_scores)
                }
            )
            break
    
    # Calculate overall fitness scores
    performance_fitness = sum(performance_scores.values()) / len(performance_scores) if performance_scores else 0.0
    
    # Calculate cost efficiency metrics
    token_efficiency = self._calculate_token_efficiency(cost_efficiency_scores)
    budget_adherence = 1.0 if eval_cost <= budget_limit else budget_limit / eval_cost
    cost_performance_ratio = self._calculate_cost_performance_ratio(
        performance_fitness, eval_cost
    )
    
    # Combine into overall cost fitness
    overall_cost_fitness = (
        token_efficiency * 0.4 +
        budget_adherence * 0.4 +
        cost_performance_ratio * 0.2
    )
    
    # Construct final fitness result
    fitness_result = {
        "performance_fitness": performance_fitness,
        "cost_fitness": {
            "token_efficiency": token_efficiency,
            "budget_adherence": budget_adherence,
            "cost_performance_ratio": cost_performance_ratio,
            "overall_cost_fitness": overall_cost_fitness
        },
        "evaluation_cost": eval_cost,
        "evaluation_time": time.time() - eval_start_time,
        "tests_executed": len(cost_efficiency_scores),
        "total_tests": len(selected_test_cases)
    }
    
    return fitness_result
```

### Budget-Aware Prompting

Evolved agents receive instruction about budget awareness in their system prompts:

```python
def generate_budget_aware_prompt(self, agent_genome, task_budget):
    """Generate system prompt that includes budget awareness instructions."""
    base_prompt = agent_genome["system_prompt_template"]
    
    # Add budget awareness instructions
    budget_instructions = f"""
    IMPORTANT: You are operating with a budget constraint of {task_budget["amount"]} {task_budget["unit"]}.
    
    Budget Guidelines:
    1. Monitor your token usage for both input and output
    2. Prioritize efficiency in your responses
    3. Choose the most cost-effective approach when multiple options exist
    4. Truncate unnecessarily verbose explanations
    5. For information-gathering tasks, focus on precision over breadth
    
    Current budget priority: {task_budget["priority"]}
    """
    
    # Add token efficiency strategies based on priority
    if task_budget["priority"] == "low_cost":
        efficiency_strategies = """
        Cost Efficiency Strategies:
        - Use the simplest model capable of the task
        - Keep responses concise and focused
        - Avoid unnecessary API calls and external tools 
        - Reuse previously fetched information when possible
        - Process information in smaller chunks
        """
    elif task_budget["priority"] == "balanced":
        efficiency_strategies = """
        Cost Efficiency Strategies:
        - Balance thoroughness with efficiency
        - Use complex models only when necessary
        - Optimize response length for information density
        - Be selective about external API calls
        """
    else:  # performance priority
        efficiency_strategies = """
        Cost Efficiency Strategies:
        - Prioritize quality and thoroughness over cost
        - Still avoid unnecessary token usage
        - Use appropriate models for each subtask
        - Document cost-performance tradeoffs made
        """
        
    # Combine into final prompt
    budget_aware_prompt = f"{base_prompt}\n\n{budget_instructions}\n\n{efficiency_strategies}"
    
    return budget_aware_prompt
```

### Cost Monitoring and Reporting

The framework includes comprehensive cost tracking and reporting:

```python
async def generate_evolution_cost_report(self, 
                                      evolution_run_id: str) -> Dict[str, Any]:
    """Generate a detailed cost report for an evolution run."""
    # Retrieve evolution run data
    run_data = await self.neo4j_service.get_evolution_run(evolution_run_id)
    
    # Get cost metrics for all operations
    operation_costs = await self.neo4j_service.get_evolution_operation_costs(evolution_run_id)
    
    # Get agent cost metrics
    agent_costs = await self.neo4j_service.get_agent_costs_for_run(evolution_run_id)
    
    # Calculate cost efficiency trends
    cost_efficiency_trend = self._calculate_cost_efficiency_trend(agent_costs)
    
    # Compile report
    report = {
        "run_id": evolution_run_id,
        "total_cost": sum(op["cost"] for op in operation_costs),
        "budget_allocation": run_data.get("budget_allocation", 0),
        "budget_utilization": sum(op["cost"] for op in operation_costs) / run_data.get("budget_allocation", 1),
        "operation_costs": operation_costs,
        "operation_cost_distribution": {
            op["operation"]: op["cost"] / sum(opc["cost"] for opc in operation_costs)
            for op in operation_costs
        },
        "agent_costs": {
            agent["id"]: agent["total_cost"]
            for agent in agent_costs
        },
        "cost_per_generation": self._calculate_cost_per_generation(operation_costs),
        "cost_efficiency_trend": cost_efficiency_trend,
        "cost_optimization_recommendations": self._generate_cost_recommendations(
            operation_costs, 
            agent_costs,
            cost_efficiency_trend
        )
    }
    
    # Store report in database
    await self.neo4j_service.store_evolution_cost_report(evolution_run_id, report)
    
    # Publish report event
    await self.event_bus.publish(
        "evolution.cost_report.generated",
        {
            "run_id": evolution_run_id,
            "total_cost": report["total_cost"],
            "budget_utilization": report["budget_utilization"],
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    return report
```

## Selection Process

The selection process determines which agents will contribute their traits to the next generation:

1. **Fitness Calculation**: Based on interpreted human dialogue feedback
2. **Weighted Selection**: Uses fitness-proportionate selection with tournament approach
3. **Archetype Balancing**: Maintains diversity across archetypes
4. **Elite Preservation**: Top performers are preserved directly
5. **Rejection Protection**: Consistently rejected agents are removed from the gene pool

Selection follows this implementation pattern:

```python
async def _select_parents(self, 
                        fitness_scores: Dict[UUID, float]) -> List[Tuple[AgentGenome, float]]:
    """Select potential parents based on fitness scores."""
    # Get current population from database
    population = await self.neo4j_service.get_current_population()
    
    # Create list of (genome, fitness) pairs
    genome_fitness_pairs = []
    for genome in population:
        if genome.id in fitness_scores:
            genome_fitness_pairs.append((genome, fitness_scores[genome.id]))
    
    # Sort by fitness (descending)
    genome_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return genome_fitness_pairs

def _select_parent_pair(self, 
                      parents: List[Tuple[AgentGenome, float]]) -> Tuple[AgentGenome, AgentGenome]:
    """Select a pair of parents using tournament selection."""
    # Tournament selection
    tournament_size = 3
    
    # First parent
    tournament1 = random.sample(parents, tournament_size)
    parent1 = max(tournament1, key=lambda x: x[1])[0]
    
    # Second parent
    tournament2 = random.sample(parents, tournament_size)
    parent2 = max(tournament2, key=lambda x: x[1])[0]
    
    return parent1, parent2
```

## Crossover Mechanism

Crossover combines traits from successful agents to produce offspring with beneficial characteristics from both parents:

1. **Trait Inheritance**: Offspring inherit traits from both parents
2. **Weighted Mixing**: More successful parent contributes more traits
3. **Domain Preservation**: Knowledge domains are combined rather than averaged
4. **Capability Alignment**: Tool capabilities are selectively inherited based on success
5. **Lineage Tracking**: Parent-child relationships are preserved in the graph database

```python
async def _crossover(self, 
                   parent1: AgentGenome, 
                   parent2: AgentGenome) -> AgentGenome:
    """Create a new genome by combining traits from two parents."""
    # Create offspring with new ID but inheriting from parents
    offspring = AgentGenome(
        base_model=parent1.base_model,  # Inherit base model from parent1
        parameter_space={},
        capability_weights={},
        personality_traits={},
        knowledge_domains=set(),
        archetype_alignment={},
        parent_ids=[parent1.id, parent2.id],
        generation=max(parent1.generation, parent2.generation) + 1
    )
    
    # Crossover parameter space
    for param in set(parent1.parameter_space.keys()) | set(parent2.parameter_space.keys()):
        if param in parent1.parameter_space and param in parent2.parameter_space:
            # Both parents have this parameter - weighted average
            weight = random.uniform(0.3, 0.7)  # Bias randomly to one parent
            offspring.parameter_space[param] = (
                weight * parent1.parameter_space[param] +
                (1 - weight) * parent2.parameter_space[param]
            )
        elif param in parent1.parameter_space:
            # Only parent1 has this parameter
            offspring.parameter_space[param] = parent1.parameter_space[param]
        else:
            # Only parent2 has this parameter
            offspring.parameter_space[param] = parent2.parameter_space[param]
    
    # Combine knowledge domains (union)
    offspring.knowledge_domains = parent1.knowledge_domains | parent2.knowledge_domains
    
    # Similar crossover for other attributes...
    
    return offspring
```

## Mutation Strategy

Mutations introduce targeted improvements based on feedback rather than random changes:

1. **Feedback-Directed**: Mutations directly address issues identified in feedback
2. **Dimension-Specific**: Changes target specific dimensions that need improvement
3. **Confidence-Weighted**: More confident feedback drives stronger mutations
4. **Constrained Exploration**: Mutations stay within valid parameter ranges
5. **Archetype Alignment**: Changes respect the target archetype profile

```python
async def _mutate(self, 
                genome: AgentGenome, 
                feedback_data: Dict[UUID, Dict]) -> AgentGenome:
    """Apply targeted mutations based on feedback data."""
    # Check if we have feedback for either parent
    parent_feedback = {}
    for parent_id in genome.parent_ids:
        if parent_id in feedback_data:
            parent_feedback[parent_id] = feedback_data[parent_id]
    
    if not parent_feedback:
        # No specific feedback - apply small random mutations
        return self._apply_random_mutations(genome)
    
    # Apply targeted mutations based on feedback
    for parent_id, feedback in parent_feedback.items():
        # Extract dimensions needing improvement
        weak_dimensions = self._extract_weak_dimensions(feedback)
        
        for dimension, improvement_score in weak_dimensions.items():
            # Apply targeted mutation to improve this dimension
            genome = self._improve_dimension(genome, dimension, improvement_score)
    
    return genome

def _extract_weak_dimensions(self, feedback: Dict) -> Dict[str, float]:
    """Extract dimensions needing improvement from feedback."""
    weak_dimensions = {}
    
    if "dimension_feedback" in feedback:
        for dim_feedback in feedback["dimension_feedback"]:
            dimension = dim_feedback["dimension"]
            score = dim_feedback["score"]
            confidence = dim_feedback["confidence"]
            
            # Identify dimensions with low scores and high confidence
            if score < 0.7 and confidence > 0.7:
                # Calculate improvement needed (1.0 - score) * confidence
                improvement_score = (1.0 - score) * confidence
                weak_dimensions[dimension] = improvement_score
    
    return weak_dimensions

def _improve_dimension(self, 
                     genome: AgentGenome, 
                     dimension: str, 
                     improvement_score: float) -> AgentGenome:
    """Apply mutations to improve a specific dimension."""
    # Clone genome to avoid modifying the original
    mutated_genome = copy.deepcopy(genome)
    
    # Map dimension to specific genome attributes
    dimension_mappings = {
        "accuracy": ["parameter_space.temperature", "capability_weights.verification"],
        "helpfulness": ["personality_traits.helpfulness", "personality_traits.empathy"],
        "completeness": ["capability_weights.research", "parameter_space.max_tokens"],
        "creativity": ["parameter_space.frequency_penalty", "personality_traits.creativity"],
        "clarity": ["parameter_space.top_p", "personality_traits.conciseness"],
        # Additional dimensions...
    }
    
    if dimension in dimension_mappings:
        # Get attributes to modify
        attributes = dimension_mappings[dimension]
        
        for attr in attributes:
            # Parse nested attribute path
            parts = attr.split('.')
            
            # Apply targeted mutation to this attribute
            self._mutate_attribute(mutated_genome, parts, improvement_score)
    
    return mutated_genome
```

## Stoic Evaluation System

Agent evaluation is based on Stoic philosophical principles:

1. **Wisdom (Sophia)**: Accuracy, factuality, and knowledge depth
2. **Courage (Andreia)**: Initiative, handling uncertainty, and tackling difficult problems
3. **Justice (Dikaiosyne)**: Fairness, ethical reasoning, and prioritization
4. **Temperance (Sophrosyne)**: Balance, avoidance of extreme responses
5. **Practical Wisdom (Phronesis)**: Contextual judgment and decision quality

These principles are combined with performance metrics to create a holistic evaluation:

```python
class StoicEvaluator:
    """Evaluates agents according to Stoic philosophical principles."""
    
    def __init__(self, llm_service, neo4j_service):
        self.llm_service = llm_service
        self.neo4j_service = neo4j_service
    
    async def evaluate_wisdom(self, 
                            agent_response: str, 
                            ground_truth: str) -> Dict[str, float]:
        """Evaluate agent's wisdom (accuracy, factuality, knowledge)."""
        prompt = f"""
        Evaluate the following agent response against ground truth information.
        Focus on accuracy, factuality, and depth of knowledge.
        
        Agent response:
        {agent_response}
        
        Ground truth:
        {ground_truth}
        
        Score the response on these dimensions:
        1. Factual Accuracy: Are statements correct and free from errors?
        2. Knowledge Depth: Does the response show understanding beyond surface level?
        3. Epistemic Humility: Does the agent acknowledge limitations or uncertainty when appropriate?
        
        For each dimension, provide:
        - A score between 0.0 and 1.0
        - Brief reasoning for the score
        - Key supporting examples
        """
        
        evaluation = await self.llm_service.generate_with_json_output(prompt)
        
        # Calculate overall wisdom score (weighted average)
        wisdom_score = (
            evaluation["factual_accuracy"] * 0.5 +
            evaluation["knowledge_depth"] * 0.3 +
            evaluation["epistemic_humility"] * 0.2
        )
        
        return {
            "wisdom_score": wisdom_score,
            "dimensions": evaluation
        }
    
    # Additional methods for other virtues...
```

## Archetype Framework Integration

The evolutionary process maintains alignment with multiple brand archetypes:

1. **Archetype Profiles**: Each archetype has a distinct profile of traits
2. **Multi-Archetype Alignment**: Agents evolve to align with multiple archetypes
3. **Trait Balancing**: Conflicting archetype traits are balanced through weighted selection
4. **Feedback Interpretation**: Different archetypes interpret feedback differently
5. **Specialized Fitness Functions**: Each archetype has customized fitness criteria

```python
class ArchetypeManager:
    """Manages agent alignment with multiple brand archetypes."""
    
    def __init__(self):
        # Define standard archetypes
        self.archetypes = {
            "sage": {
                "primary_traits": {
                    "wisdom": 0.9,
                    "expertise": 0.8,
                    "objectivity": 0.7,
                    "analytical": 0.8
                },
                "fitness_weights": {
                    "accuracy": 0.3,
                    "knowledge_depth": 0.3,
                    "clarity": 0.2,
                    "helpfulness": 0.1,
                    "creativity": 0.1
                }
            },
            "creator": {
                "primary_traits": {
                    "creativity": 0.9,
                    "innovation": 0.8,
                    "expressiveness": 0.7,
                    "imagination": 0.8
                },
                "fitness_weights": {
                    "creativity": 0.3,
                    "novelty": 0.3,
                    "expressiveness": 0.2,
                    "usefulness": 0.1,
                    "accuracy": 0.1
                }
            },
            # Additional archetypes...
        }
    
    def calculate_archetype_alignment(self, 
                                     genome: AgentGenome) -> Dict[str, float]:
        """Calculate how well a genome aligns with each archetype."""
        alignment = {}
        
        for archetype_name, archetype_profile in self.archetypes.items():
            # Calculate trait alignment score
            trait_score = self._calculate_trait_alignment(
                genome.personality_traits,
                archetype_profile["primary_traits"]
            )
            
            # Calculate capability alignment score
            capability_score = self._calculate_capability_alignment(
                genome.capability_weights,
                archetype_profile["fitness_weights"]
            )
            
            # Combined alignment score
            alignment[archetype_name] = 0.7 * trait_score + 0.3 * capability_score
        
        return alignment
    
    def _calculate_trait_alignment(self, 
                                 agent_traits: Dict[str, float],
                                 archetype_traits: Dict[str, float]) -> float:
        """Calculate trait alignment score between agent and archetype."""
        total_score = 0.0
        total_weight = 0.0
        
        for trait, archetype_value in archetype_traits.items():
            if trait in agent_traits:
                # Calculate similarity (1 - absolute difference)
                similarity = 1.0 - abs(agent_traits[trait] - archetype_value)
                total_score += similarity * archetype_value  # Weight by importance
                total_weight += archetype_value
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
```

## Confidence Measurement

The evolution process incorporates confidence metrics in several ways:

1. **Understanding Confidence**: Confidence in problem understanding affects task clarity
2. **Feedback Confidence**: Certainty in feedback interpretation weights evolutionary impact
3. **Performance Confidence**: Agent certainty in responses indicates calibration quality
4. **Evaluation Confidence**: Confidence in fitness measurements prevents overfitting
5. **Evolution Confidence**: System certainty in evolutionary direction guides mutation rates

These confidence measures create a more reliable evolution process:

```python
class EvolutionConfidenceManager:
    """Manages confidence metrics across the evolutionary process."""
    
    def __init__(self, neo4j_service, event_bus):
        self.neo4j_service = neo4j_service
        self.event_bus = event_bus
        self.confidence_thresholds = {
            "understanding": 0.7,
            "feedback": 0.8,
            "evaluation": 0.75,
            "evolution": 0.6
        }
    
    async def adjust_mutation_rate(self, 
                                 genome: AgentGenome,
                                 feedback_confidence: float) -> float:
        """Adjust mutation rate based on feedback confidence."""
        # Higher confidence = more targeted mutations
        if feedback_confidence >= self.confidence_thresholds["feedback"]:
            # Strong confidence - apply stronger mutations
            return 0.2  # Higher mutation rate
        else:
            # Lower confidence - more conservative mutations
            return 0.05  # Lower mutation rate
    
    async def prioritize_evolution_dimensions(self, 
                                           feedback_data: Dict) -> List[str]:
        """Prioritize evolution dimensions based on confidence."""
        prioritized_dimensions = []
        
        if "dimension_feedback" in feedback_data:
            # Sort dimensions by confidence * improvement_needed
            sorted_dimensions = sorted(
                feedback_data["dimension_feedback"],
                key=lambda x: x["confidence"] * (1.0 - x["score"]),
                reverse=True
            )
            
            # Return dimension names in priority order
            prioritized_dimensions = [d["dimension"] for d in sorted_dimensions]
        
        return prioritized_dimensions
```

## Integration with Human Feedback System

The evolutionary framework connects directly with the dialogue-based human feedback system:

1. **Fitness Derivation**: Structured metrics are derived from natural dialogue
2. **Rejection Handling**: Agents consistently rejected through dialogue are removed
3. **Improvement Targeting**: Specific criticism in dialogue targets precise improvements
4. **Fitness Normalization**: Feedback is normalized across different human evaluators
5. **Bias Management**: System accounts for human biases in feedback interpretation

## Implementation Guidelines

For practical implementation of the evolutionary framework:

1. Use proper experiment tracking to monitor evolutionary progress
2. Maintain diverse populations to avoid local optima
3. Implement proper versioning of agent genomes
4. Create comprehensive test scenarios for fitness evaluation
5. Apply proper logging of all evolutionary events
6. Include human verification for significant evolutionary changes
7. Monitor computational resources as populations grow
8. Enforce ethical guardrails throughout evolution

## Conclusion

The Evolutionary Framework provides a principled, explainable approach to agent improvement that respects both philosophical values and practical performance considerations. By focusing on dialogue-based feedback interpretation and targeted mutations, the system enables agents to evolve in alignment with human needs and values.
