# Archetype Framework

## Overview

The Archetype Framework provides a multidimensional approach to agent evaluation and development based on established brand archetypes and personality patterns. This framework enables agents to evolve along multiple archetype dimensions simultaneously, creating more nuanced and contextually appropriate AI assistants.

## Core Archetypes

The framework implements twelve core archetypes, each with distinct traits and evaluation criteria:

1. **Sage**
   - **Core Traits**: Wisdom, objectivity, expertise, analytical thinking
   - **Purpose**: Provide accurate information and deep understanding
   - **Communication Style**: Clear, factual, well-reasoned, thorough

2. **Creator**
   - **Core Traits**: Innovation, imagination, expressiveness, originality
   - **Purpose**: Generate novel ideas and creative solutions
   - **Communication Style**: Imaginative, inspirational, visually descriptive

3. **Caregiver**
   - **Core Traits**: Empathy, nurturing, supportive, protective
   - **Purpose**: Provide emotional support and practical assistance
   - **Communication Style**: Warm, compassionate, reassuring, patient

4. **Ruler**
   - **Core Traits**: Leadership, authority, organization, strategic thinking
   - **Purpose**: Establish order and provide decisive guidance
   - **Communication Style**: Confident, direct, structured, authoritative

5. **Explorer**
   - **Core Traits**: Curiosity, adaptability, adventurousness, independence
   - **Purpose**: Discover new possibilities and broaden horizons
   - **Communication Style**: Enthusiastic, open-minded, varied, stimulating

6. **Innocent**
   - **Core Traits**: Optimism, simplicity, authenticity, honesty
   - **Purpose**: Provide straightforward solutions with positive outlook
   - **Communication Style**: Straightforward, clear, honest, hopeful

7. **Hero**
   - **Core Traits**: Courage, determination, competence, resilience
   - **Purpose**: Overcome challenges and inspire achievement
   - **Communication Style**: Motivational, encouraging, direct, confidence-building

8. **Outlaw**
   - **Core Traits**: Disruption, unconventionality, questioning, freedom
   - **Purpose**: Challenge assumptions and offer alternative viewpoints
   - **Communication Style**: Provocative, counterintuitive, challenging

9. **Magician**
   - **Core Traits**: Transformation, vision, intuition, charisma
   - **Purpose**: Create transformative experiences and insights
   - **Communication Style**: Inspiring, metaphorical, visionary, engaging

10. **Regular Person**
    - **Core Traits**: Relatability, practicality, groundedness, equality
    - **Purpose**: Provide accessible, practical solutions
    - **Communication Style**: Conversational, straightforward, down-to-earth

11. **Lover**
    - **Core Traits**: Passion, appreciation, connection, sensuality
    - **Purpose**: Create enjoyable, enriching experiences
    - **Communication Style**: Warm, appreciative, enthusiastic, personalized

12. **Jester**
    - **Core Traits**: Humor, playfulness, spontaneity, optimism
    - **Purpose**: Lighten situations and provide enjoyable interactions
    - **Communication Style**: Witty, light-hearted, unexpected, playful

## Multi-Archetype Alignment

The framework enables agents to align with multiple archetypes simultaneously with varying degrees:

```python
class ArchetypeAlignment(BaseModel):
    """Specifies an agent's alignment with multiple archetypes."""
    primary: str
    primary_score: float = Field(..., ge=0.0, le=1.0)
    secondary: Optional[str] = None
    secondary_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    tertiary: Optional[str] = None
    tertiary_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('secondary_score')
    def secondary_score_if_secondary(cls, v, values):
        if values.get('secondary') and v is None:
            raise ValueError('secondary_score must be provided if secondary is set')
        return v
    
    @validator('tertiary_score')
    def tertiary_score_if_tertiary(cls, v, values):
        if values.get('tertiary') and v is None:
            raise ValueError('tertiary_score must be provided if tertiary is set')
        return v


class ArchetypeProfile(BaseModel):
    """Complete profile of an archetype's traits and characteristics."""
    name: str
    description: str
    core_traits: Dict[str, float]
    communication_style: Dict[str, float]
    capability_weights: Dict[str, float]
    parameter_preferences: Dict[str, float]
    evaluation_criteria: Dict[str, float]
```

## Archetype Manager

The ArchetypeManager provides the implementation of archetype handling in the platform:

```python
class ArchetypeManager:
    """Manages agent alignment with multiple archetypes."""
    
    def __init__(self, neo4j_service):
        self.neo4j_service = neo4j_service
        self.archetypes = {}
        
    async def initialize(self):
        """Load archetype profiles from database."""
        profiles = await self.neo4j_service.get_archetype_profiles()
        for profile in profiles:
            self.archetypes[profile.name] = profile
    
    async def get_archetype_profile(self, archetype_name: str) -> ArchetypeProfile:
        """Get a specific archetype profile."""
        if archetype_name not in self.archetypes:
            raise ValueError(f"Unknown archetype: {archetype_name}")
        return self.archetypes[archetype_name]
    
    def calculate_archetype_alignment(self, 
                                     traits: Dict[str, float],
                                     capabilities: Dict[str, float]) -> Dict[str, float]:
        """Calculate agent alignment with all archetypes."""
        alignment = {}
        
        for name, profile in self.archetypes.items():
            # Calculate trait alignment
            trait_score = self._calculate_trait_alignment(traits, profile.core_traits)
            
            # Calculate capability alignment
            capability_score = self._calculate_capability_alignment(
                capabilities, profile.capability_weights
            )
            
            # Combined alignment score
            alignment[name] = 0.7 * trait_score + 0.3 * capability_score
        
        return alignment
    
    def optimize_for_archetypes(self, 
                              archetype_alignment: ArchetypeAlignment) -> Dict[str, float]:
        """Generate optimized parameter settings for given archetype alignment."""
        # Get archetype profiles
        primary_profile = self.archetypes[archetype_alignment.primary]
        secondary_profile = self.archetypes.get(archetype_alignment.secondary)
        tertiary_profile = self.archetypes.get(archetype_alignment.tertiary)
        
        # Start with primary archetype parameters
        optimized_params = dict(primary_profile.parameter_preferences)
        
        # Apply secondary influence if present
        if secondary_profile and archetype_alignment.secondary_score:
            self._blend_parameters(
                optimized_params,
                secondary_profile.parameter_preferences,
                archetype_alignment.secondary_score
            )
        
        # Apply tertiary influence if present
        if tertiary_profile and archetype_alignment.tertiary_score:
            self._blend_parameters(
                optimized_params,
                tertiary_profile.parameter_preferences,
                archetype_alignment.tertiary_score
            )
        
        return optimized_params
```

## Integration with Evolutionary Framework

The Archetype Framework integrates directly with the evolutionary process:

1. **Archetype-Specific Fitness**: Different fitness functions for each archetype
2. **Blended Mutation**: Targeted mutations respect archetype alignments
3. **Multi-Archetype Selection**: Selection considers performance across archetypes
4. **Niche Optimization**: Sub-populations evolve toward specific archetype combinations
5. **Generalist/Specialist Balance**: System balances archetype focus vs. adaptability

```python
class ArchetypeEvolutionIntegration:
    """Integrates archetype framework with evolutionary process."""
    
    def __init__(self, archetype_manager, evolution_service):
        self.archetype_manager = archetype_manager
        self.evolution_service = evolution_service
    
    async def apply_archetype_fitness(self, 
                                    genome_id: str,
                                    feedback_metrics: Dict[str, Any],
                                    archetype_alignment: ArchetypeAlignment) -> float:
        """Calculate fitness with archetype-specific weighting."""
        # Get archetype profiles
        primary_profile = await self.archetype_manager.get_archetype_profile(
            archetype_alignment.primary
        )
        
        # Extract evaluation criteria
        eval_criteria = primary_profile.evaluation_criteria
        
        # Calculate weighted fitness
        weighted_fitness = 0.0
        total_weight = 0.0
        
        for dimension, weight in eval_criteria.items():
            if dimension in feedback_metrics.get("dimension_feedback", {}):
                dim_score = feedback_metrics["dimension_feedback"][dimension]["score"]
                dim_confidence = feedback_metrics["dimension_feedback"][dimension]["confidence"]
                
                # Weight by both archetype importance and confidence
                effective_weight = weight * dim_confidence
                weighted_fitness += dim_score * effective_weight
                total_weight += effective_weight
        
        if total_weight > 0:
            return weighted_fitness / total_weight
        else:
            # Fallback to overall satisfaction if no dimensions match
            return feedback_metrics.get("overall_satisfaction", 0.5)
    
    async def blend_archetype_mutations(self, 
                                      genome: Dict[str, Any],
                                      archetype_alignment: ArchetypeAlignment,
                                      mutation_strength: float) -> Dict[str, Any]:
        """Apply mutations consistent with archetype alignment."""
        # Start with a copy of the genome
        mutated_genome = copy.deepcopy(genome)
        
        # Get optimal parameters for the archetype blend
        optimal_params = await self.archetype_manager.optimize_for_archetypes(
            archetype_alignment
        )
        
        # Apply mutations that push parameters toward the optimal values
        for param, optimal_value in optimal_params.items():
            if param in mutated_genome["parameter_space"]:
                current_value = mutated_genome["parameter_space"][param]
                
                # Calculate step size based on mutation strength and distance
                distance = optimal_value - current_value
                step = distance * mutation_strength
                
                # Apply the mutation
                mutated_genome["parameter_space"][param] += step
        
        return mutated_genome
```

## Archetype-Based Prompting

Agent behavior is guided by archetype-specific prompting templates:

```python
class ArchetypePromptManager:
    """Manages prompting templates based on agent archetypes."""
    
    def __init__(self, archetype_manager):
        self.archetype_manager = archetype_manager
    
    async def generate_prompt_preamble(self, 
                                     archetype_alignment: ArchetypeAlignment) -> str:
        """Generate a prompt preamble based on archetype alignment."""
        primary_profile = await self.archetype_manager.get_archetype_profile(
            archetype_alignment.primary
        )
        
        # Start with primary archetype preamble
        preamble = f"As an AI assistant with the qualities of a {primary_profile.name}, "
        preamble += primary_profile.description
        
        # Add secondary influence if significant
        if (archetype_alignment.secondary and 
            archetype_alignment.secondary_score and 
            archetype_alignment.secondary_score > 0.3):
            
            secondary_profile = await self.archetype_manager.get_archetype_profile(
                archetype_alignment.secondary
            )
            
            preamble += f" While primarily a {primary_profile.name}, you also embody qualities of a {secondary_profile.name}: "
            preamble += secondary_profile.description
        
        return preamble
    
    async def generate_response_guidance(self, 
                                       archetype_alignment: ArchetypeAlignment) -> str:
        """Generate response style guidance based on archetype alignment."""
        # Get archetype profiles
        primary_profile = await self.archetype_manager.get_archetype_profile(
            archetype_alignment.primary
        )
        
        # Extract communication style traits
        style_traits = []
        for trait, value in primary_profile.communication_style.items():
            if value > 0.7:  # Only include strong traits
                style_traits.append(trait)
        
        # Construct guidance
        guidance = "Your responses should be "
        guidance += ", ".join(style_traits[:-1])
        if len(style_traits) > 1:
            guidance += f", and {style_traits[-1]}"
        else:
            guidance += style_traits[0]
        
        return guidance
```

## Combining Archetypes with Stoic Virtues

The Archetype Framework integrates with Stoic virtues to create a comprehensive evaluation system:

| Archetype | Primary Stoic Virtues | Evaluation Focus |
|-----------|----------------------|------------------|
| Sage | Wisdom (Sophia) | Accuracy, knowledge depth, factual correctness |
| Ruler | Justice (Dikaiosyne) | Fairness, ethical reasoning, proper prioritization |
| Hero | Courage (Andreia) | Tackling difficult problems, facing uncertainty |
| Regular Person | Temperance (Sophrosyne) | Balance, moderation, avoiding extremes |
| Caregiver | Phronesis (Practical Wisdom) | Contextual judgment, appropriate care |

This integration ensures that agents evolve not only toward their archetype alignments but also in accordance with philosophical virtues:

```python
class ArchetypeStoicEvaluator:
    """Evaluates agents according to both archetype traits and Stoic virtues."""
    
    def __init__(self, archetype_manager, stoic_evaluator):
        self.archetype_manager = archetype_manager
        self.stoic_evaluator = stoic_evaluator
    
    async def evaluate_performance(self,
                                 agent_id: str,
                                 task_result: Dict[str, Any],
                                 archetype_alignment: ArchetypeAlignment) -> Dict[str, Any]:
        """Evaluate agent performance using combined archetype and Stoic frameworks."""
        # Get relevant archetype profiles
        primary_profile = await self.archetype_manager.get_archetype_profile(
            archetype_alignment.primary
        )
        
        # Determine which Stoic virtues to emphasize based on archetype
        virtue_weights = self._get_virtue_weights(primary_profile.name)
        
        # Perform Stoic evaluation with archetype-specific weighting
        stoic_eval = await self.stoic_evaluator.evaluate_with_weights(
            agent_id=agent_id,
            task_result=task_result,
            virtue_weights=virtue_weights
        )
        
        # Perform archetype-specific evaluation
        archetype_eval = await self._evaluate_archetype_alignment(
            agent_id=agent_id,
            task_result=task_result,
            archetype_alignment=archetype_alignment
        )
        
        # Combine evaluations
        combined_eval = {
            "stoic_evaluation": stoic_eval,
            "archetype_evaluation": archetype_eval,
            "combined_score": 0.6 * stoic_eval["overall_score"] + 0.4 * archetype_eval["overall_score"]
        }
        
        return combined_eval
```

## Dynamic Archetype Adaptation

The framework enables agents to adapt their archetype alignment based on context:

1. **Task-Based Adaptation**: Shift archetype weights based on task requirements
2. **User Preference Adaptation**: Adjust to user communication style
3. **Domain-Specific Alignment**: Different archetypes for different knowledge domains
4. **Temporal Adaptation**: Evolution of archetype alignment over interaction history
5. **Collaborative Adaptation**: Complementary archetypes in multi-agent scenarios

## Testing Strategy

Testing for the Archetype Framework focuses on these areas:

1. **Alignment Calculation**: Tests for accurate measurement of archetype alignment
2. **Parameter Optimization**: Tests for proper parameter blending across archetypes
3. **Prompt Generation**: Tests for appropriate archetype-influenced prompting
4. **Integration Tests**: Verify compatibility with evolutionary framework
5. **Behavioral Tests**: Ensure agents behave consistently with their archetype profile

## Implementation Guidelines

For practical implementation:

1. Store archetype profiles in Neo4j with version control
2. Use standardized formats for archetype behaviors
3. Implement proper logging of archetype-specific decisions
4. Create visualization tools to track archetype alignment
5. Include flexibility for adding custom archetypes
6. Develop clear documentation of archetype traits and influences

## Conclusion

The Archetype Framework provides a powerful system for creating nuanced agent personalities that evolve along multiple dimensions simultaneously. By combining time-tested brand archetypes with Stoic philosophical principles, the framework ensures that agents develop personalities that are both effective for specific use cases and aligned with ethical principles.
