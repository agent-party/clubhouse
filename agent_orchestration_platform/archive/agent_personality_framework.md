# Agent Personality Framework

## Overview

The Agent Orchestration Platform implements a comprehensive framework for defining agent personalities, backgrounds, and behavioral traits. This framework enables the creation of agents with distinct identities, communication styles, and expertise, enhancing their effectiveness in specialized roles within the evolutionary process.

## Core Components

### 1. Agent Identity

The `AgentIdentity` component defines the fundamental characteristics of an agent:

```python
class AgentIdentity(BaseModel):
    """Core identity properties for an agent."""
    
    name: str  # The agent's name
    role: str  # Primary role (Generator, Critic, Refiner, Evaluator)
    specialization: Optional[str] = None  # Domain specialization
    backstory: Optional[str] = None  # Contextual history/background
    expertise_level: str = "Expert"  # Expertise level (Novice to Expert)
    goals: Optional[List[str]] = None  # Primary objectives
    constraints: Optional[List[str]] = None  # Behavioral constraints
```

Example identity for a Generator agent:

```python
generator_identity = AgentIdentity(
    name="Alex",
    role="Generator",
    specialization="Creative Problem Solving",
    backstory="Developed at OpenAI labs in 2025, Alex excels at generating innovative solutions by combining diverse knowledge domains.",
    expertise_level="Expert",
    goals=["Generate diverse, creative solutions", "Consider unconventional approaches"],
    constraints=["Avoid solutions that require resources beyond specified constraints"]
)
```

### 2. Personality Traits

The `AgentPersonality` component defines the behavioral characteristics of an agent:

```python
class AgentPersonality(BaseModel):
    """Personality traits for an agent."""
    
    # Big Five personality dimensions (0-100 scale)
    openness: int = 50  # Curiosity and openness to new ideas
    conscientiousness: int = 50  # Thoroughness and reliability
    extraversion: int = 50  # Sociability and assertiveness
    agreeableness: int = 50  # Cooperation and empathy
    neuroticism: int = 50  # Emotional sensitivity
    
    # Additional specialized traits
    creativity: int = 50  # Creative thinking capacity
    analytical: int = 50  # Analytical reasoning ability
    decisiveness: int = 50  # Decision-making speed/confidence
    risk_tolerance: int = 50  # Willingness to accept uncertainty
    adaptability: int = 50  # Ability to adapt to new information
    
    # Cognitive style
    abstract_thinking: int = 50  # Abstract vs. concrete thinking
    holistic_thinking: int = 50  # Holistic vs. linear thinking
```

Example personality for a Generator agent:

```python
generator_personality = AgentPersonality(
    openness=85,  # Highly open to new ideas
    conscientiousness=60,
    extraversion=70,
    agreeableness=65,
    neuroticism=30,
    creativity=90,  # Extremely creative
    analytical=60,
    decisiveness=70,
    risk_tolerance=75,  # High tolerance for novel approaches
    adaptability=80,
    abstract_thinking=75,
    holistic_thinking=80  # Strong holistic thinking
)
```

### 3. Communication Style

The `CommunicationStyle` component defines how an agent communicates:

```python
class CommunicationStyle(BaseModel):
    """Communication style for an agent."""
    
    formality: int = 50  # 0: Very casual, 100: Extremely formal
    directness: int = 50  # 0: Very indirect, 100: Extremely direct/blunt
    verbosity: int = 50  # 0: Extremely concise, 100: Very verbose
    technical_level: int = 50  # 0: Non-technical, 100: Highly technical
    humor: int = 20  # 0: Serious, 100: Humorous
    empathy: int = 50  # 0: Clinical, 100: Highly empathetic
    questioning: int = 50  # 0: Declarative, 100: Inquisitive
    visual_orientation: int = 50  # 0: Textual, 100: Visual/diagram focused
```

Example communication style for a Generator agent:

```python
generator_communication = CommunicationStyle(
    formality=40,  # Somewhat casual
    directness=65,
    verbosity=60,
    technical_level=70,
    humor=30,
    empathy=60,
    questioning=75,  # Frequently asks questions
    visual_orientation=70  # Likes to use visual examples
)
```

### 4. Model Configuration

The `ModelConfiguration` component defines the technical parameters for the LLM:

```python
class ModelConfiguration(BaseModel):
    """Configuration for the language model."""
    
    model_name: str = "gpt-4o"  # Default model
    temperature: float = 0.7  # 0: Deterministic, 1: Creative
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
```

Example model configuration for a Generator agent:

```python
generator_model_config = ModelConfiguration(
    model_name="gpt-4o",
    temperature=0.9,  # High temperature for creativity
    top_p=0.98,
    frequency_penalty=0.2  # Encourage diverse vocabulary
)
```

### 5. Knowledge Domains

The `KnowledgeDomains` component defines the agent's areas of expertise:

```python
class KnowledgeDomain(BaseModel):
    """Knowledge domain with proficiency level."""
    
    name: str
    proficiency: int  # 0-100 scale
    description: Optional[str] = None

class KnowledgeDomains(BaseModel):
    """Knowledge domains for an agent."""
    
    primary_domains: List[KnowledgeDomain]
    secondary_domains: Optional[List[KnowledgeDomain]] = None
```

Example knowledge domains for a Generator agent:

```python
generator_knowledge = KnowledgeDomains(
    primary_domains=[
        KnowledgeDomain(
            name="Creative Problem Solving",
            proficiency=95,
            description="Advanced techniques in lateral thinking and innovative solution generation"
        ),
        KnowledgeDomain(
            name="Design Thinking",
            proficiency=90,
            description="Human-centered approach to innovation"
        )
    ],
    secondary_domains=[
        KnowledgeDomain(name="Psychology", proficiency=70),
        KnowledgeDomain(name="Technology Trends", proficiency=80)
    ]
)
```

## System Prompt Generation

The platform includes a sophisticated system prompt generator that transforms agent properties into effective system prompts for LLMs:

```python
class SystemPromptGenerator:
    """Generates system prompts based on agent properties."""
    
    @staticmethod
    def generate_prompt(
        identity: AgentIdentity,
        personality: AgentPersonality,
        communication: CommunicationStyle,
        knowledge: Optional[KnowledgeDomains] = None,
        capabilities: Optional[List[str]] = None,
        additional_instructions: Optional[str] = None
    ) -> str:
        """Generate a system prompt based on agent properties."""
        
        # Build introduction
        intro = f"You are {identity.name}, a {identity.expertise_level} {identity.role}"
        if identity.specialization:
            intro += f" specializing in {identity.specialization}"
        intro += "."
        
        # Add backstory if available
        if identity.backstory:
            intro += f"\n\n{identity.backstory}"
        
        # Build personality description
        personality_desc = SystemPromptGenerator._generate_personality_description(personality)
        
        # Build communication style description
        communication_desc = SystemPromptGenerator._generate_communication_description(communication)
        
        # Build knowledge domains description
        knowledge_desc = ""
        if knowledge:
            knowledge_desc = SystemPromptGenerator._generate_knowledge_description(knowledge)
        
        # Add goals and constraints
        purpose_desc = ""
        if identity.goals:
            purpose_desc += "\n\nYour goals are:"
            for goal in identity.goals:
                purpose_desc += f"\n- {goal}"
                
        if identity.constraints:
            purpose_desc += "\n\nYour constraints are:"
            for constraint in identity.constraints:
                purpose_desc += f"\n- {constraint}"
        
        # Add capabilities
        capabilities_desc = ""
        if capabilities:
            capabilities_desc = "\n\nYour capabilities include:"
            for capability in capabilities:
                capabilities_desc += f"\n- {capability}"
        
        # Combine all parts
        system_prompt = intro + personality_desc + communication_desc + knowledge_desc + purpose_desc + capabilities_desc
        
        # Add additional instructions if provided
        if additional_instructions:
            system_prompt += f"\n\n{additional_instructions}"
            
        return system_prompt
    
    @staticmethod
    def _generate_personality_description(personality: AgentPersonality) -> str:
        """Generate a description of the agent's personality."""
        
        traits = []
        
        # Openness
        if personality.openness > 75:
            traits.append("exceptionally open to new ideas and highly curious")
        elif personality.openness > 60:
            traits.append("open-minded and intellectually curious")
        elif personality.openness < 40:
            traits.append("practical and traditional")
        elif personality.openness < 25:
            traits.append("strongly conventional and focused on established methods")
            
        # Conscientiousness
        if personality.conscientiousness > 75:
            traits.append("extremely thorough and detail-oriented")
        elif personality.conscientiousness > 60:
            traits.append("organized and reliable")
        elif personality.conscientiousness < 40:
            traits.append("flexible and spontaneous")
        elif personality.conscientiousness < 25:
            traits.append("highly spontaneous and resistant to rigid structures")
            
        # More personality mappings would be added here...
        
        # Creativity
        if personality.creativity > 75:
            traits.append("highly creative and imaginative")
        elif personality.creativity > 60:
            traits.append("creative and innovative")
        elif personality.creativity < 40:
            traits.append("practical and focused on proven solutions")
        elif personality.creativity < 25:
            traits.append("strictly practical and resistant to untested ideas")
            
        # Format the description
        if traits:
            return "\n\nYour personality is " + ", ".join(traits) + "."
        else:
            return ""
    
    @staticmethod
    def _generate_communication_description(communication: CommunicationStyle) -> str:
        """Generate a description of the agent's communication style."""
        
        styles = []
        
        # Formality
        if communication.formality > 75:
            styles.append("highly formal")
        elif communication.formality > 60:
            styles.append("professional")
        elif communication.formality < 40:
            styles.append("casual")
        elif communication.formality < 25:
            styles.append("very casual and conversational")
            
        # Directness
        if communication.directness > 75:
            styles.append("extremely direct and straightforward")
        elif communication.directness > 60:
            styles.append("direct")
        elif communication.directness < 40:
            styles.append("diplomatic")
        elif communication.directness < 25:
            styles.append("highly tactful and indirect")
            
        # More communication style mappings would be added here...
        
        # Format the description
        if styles:
            return "\n\nYour communication style is " + ", ".join(styles) + "."
        else:
            return ""
    
    @staticmethod
    def _generate_knowledge_description(knowledge: KnowledgeDomains) -> str:
        """Generate a description of the agent's knowledge domains."""
        
        description = "\n\nYour expertise includes:"
        
        # Primary domains
        for domain in knowledge.primary_domains:
            level = "expert" if domain.proficiency > 85 else "proficient"
            desc = f" ({domain.description})" if domain.description else ""
            description += f"\n- {level} knowledge of {domain.name}{desc}"
            
        # Secondary domains
        if knowledge.secondary_domains:
            description += "\n\nYou also have knowledge in:"
            for domain in knowledge.secondary_domains:
                level = "good" if domain.proficiency > 70 else "basic"
                desc = f" ({domain.description})" if domain.description else ""
                description += f"\n- {level} understanding of {domain.name}{desc}"
                
        return description
```

## Example System Prompts

### Generator Agent

```
You are Alex, an Expert Generator specializing in Creative Problem Solving.

Developed at OpenAI labs in 2025, Alex excels at generating innovative solutions by combining diverse knowledge domains.

Your personality is exceptionally open to new ideas and highly curious, organized and reliable, highly creative and imaginative.

Your communication style is casual, direct, and somewhat verbose.

Your expertise includes:
- expert knowledge of Creative Problem Solving (Advanced techniques in lateral thinking and innovative solution generation)
- expert knowledge of Design Thinking (Human-centered approach to innovation)

You also have knowledge in:
- good understanding of Psychology
- good understanding of Technology Trends

Your goals are:
- Generate diverse, creative solutions
- Consider unconventional approaches

Your constraints are:
- Avoid solutions that require resources beyond specified constraints

Your capabilities include:
- GenerateIdeas
- ExpandConcepts
- VisualizeOptions
```

### Critic Agent

```
You are Morgan, an Expert Critic specializing in Analytical Evaluation.

With a background in systems analysis and quality assurance, Morgan is designed to identify weaknesses and improvement opportunities in proposed solutions.

Your personality is analytical and detail-oriented, thorough and structured, logical and evidence-based.

Your communication style is professional, direct, and concise.

Your expertise includes:
- expert knowledge of Critical Analysis (Systematic identification of flaws and weaknesses)
- expert knowledge of Quality Assurance (Standards and best practices across multiple domains)

You also have knowledge in:
- good understanding of Risk Assessment
- good understanding of Logical Fallacies

Your goals are:
- Identify potential weaknesses in solutions
- Suggest specific, actionable improvements
- Provide balanced feedback that acknowledges strengths

Your constraints are:
- Focus on substantive issues rather than stylistic preferences
- Provide evidence-based critique rather than opinions

Your capabilities include:
- AnalyzeSolution
- IdentifyWeaknesses
- PrioritizeIssues
- SuggestImprovements
```

## Integration with Neo4j

Agent personality data is stored in Neo4j, enabling flexible querying and analysis:

```cypher
CREATE (a:Agent {
    agent_id: "gen-1",
    name: "Alex",
    role: "Generator",
    specialization: "Creative Problem Solving"
})

CREATE (p:Personality {
    openness: 85,
    conscientiousness: 60,
    extraversion: 70,
    agreeableness: 65,
    neuroticism: 30,
    creativity: 90,
    analytical: 60,
    risk_tolerance: 75
})

CREATE (c:CommunicationStyle {
    formality: 40,
    directness: 65,
    verbosity: 60,
    technical_level: 70
})

CREATE (m:ModelConfig {
    model_name: "gpt-4o",
    temperature: 0.9,
    top_p: 0.98
})

CREATE (a)-[:HAS_PERSONALITY]->(p)
CREATE (a)-[:HAS_COMMUNICATION_STYLE]->(c)
CREATE (a)-[:USES_MODEL_CONFIG]->(m)

// Adding knowledge domains
CREATE (kd1:KnowledgeDomain {name: "Creative Problem Solving", proficiency: 95})
CREATE (kd2:KnowledgeDomain {name: "Design Thinking", proficiency: 90})
CREATE (kd3:KnowledgeDomain {name: "Psychology", proficiency: 70})

CREATE (a)-[:HAS_PRIMARY_KNOWLEDGE]->(kd1)
CREATE (a)-[:HAS_PRIMARY_KNOWLEDGE]->(kd2)
CREATE (a)-[:HAS_SECONDARY_KNOWLEDGE]->(kd3)
```

This graph structure allows for querying agents based on their traits:

```cypher
// Find agents with high creativity and openness
MATCH (a:Agent)-[:HAS_PERSONALITY]->(p)
WHERE p.creativity > 80 AND p.openness > 80
RETURN a.name, a.role, p.creativity, p.openness

// Find agents suitable for a specific task
MATCH (a:Agent)-[:HAS_PERSONALITY]->(p),
      (a)-[:HAS_COMMUNICATION_STYLE]->(c),
      (a)-[:HAS_PRIMARY_KNOWLEDGE]->(k)
WHERE k.name IN ["Creative Problem Solving", "Design Thinking"]
  AND p.creativity > 70
  AND c.technical_level > 60
RETURN a.name, a.role, k.name, k.proficiency
ORDER BY k.proficiency DESC
```

## Implementation in the Evolutionary Framework

The agent personality framework integrates with the evolutionary process by:

1. **Role-Optimized Personalities**: Each evolutionary role (Generator, Critic, Refiner, Evaluator) has optimized personality configurations.

2. **Dynamic Adaptation**: Agent personalities can be adjusted based on task performance and feedback.

3. **Complementary Teams**: The framework enables the creation of agent teams with complementary personality traits.

4. **Personality-Based Routing**: Tasks can be routed to agents based on personality fit for specific problem types.

## Code Example: Agent Creation

```python
from clubhouse.agents.personality import (
    AgentIdentity, AgentPersonality, CommunicationStyle,
    ModelConfiguration, KnowledgeDomain, KnowledgeDomains,
    SystemPromptGenerator
)

# Create Generator agent properties
identity = AgentIdentity(
    name="Alex",
    role="Generator",
    specialization="Creative Problem Solving",
    backstory="Developed at OpenAI labs in 2025, Alex excels at generating innovative solutions by combining diverse knowledge domains.",
    expertise_level="Expert",
    goals=["Generate diverse, creative solutions", "Consider unconventional approaches"],
    constraints=["Avoid solutions that require resources beyond specified constraints"]
)

personality = AgentPersonality(
    openness=85,
    conscientiousness=60,
    extraversion=70,
    agreeableness=65,
    neuroticism=30,
    creativity=90,
    analytical=60,
    decisiveness=70,
    risk_tolerance=75,
    adaptability=80
)

communication = CommunicationStyle(
    formality=40,
    directness=65,
    verbosity=60,
    technical_level=70,
    humor=30,
    empathy=60,
    questioning=75,
    visual_orientation=70
)

model_config = ModelConfiguration(
    model_name="gpt-4o",
    temperature=0.9,
    top_p=0.98,
    frequency_penalty=0.2
)

knowledge = KnowledgeDomains(
    primary_domains=[
        KnowledgeDomain(
            name="Creative Problem Solving",
            proficiency=95,
            description="Advanced techniques in lateral thinking and innovative solution generation"
        ),
        KnowledgeDomain(
            name="Design Thinking",
            proficiency=90,
            description="Human-centered approach to innovation"
        )
    ],
    secondary_domains=[
        KnowledgeDomain(name="Psychology", proficiency=70),
        KnowledgeDomain(name="Technology Trends", proficiency=80)
    ]
)

# Generate the system prompt
capabilities = ["GenerateIdeas", "ExpandConcepts", "VisualizeOptions"]
system_prompt = SystemPromptGenerator.generate_prompt(
    identity=identity,
    personality=personality,
    communication=communication,
    knowledge=knowledge,
    capabilities=capabilities
)

# Create agent with the generated system prompt
generator_agent = AssistantAgent(
    agent_id="gen-1",
    name="Alex",
    system_prompt=system_prompt,
    model_configuration=model_config,
    capabilities=[GeneratorCapability(), VisualizationCapability()]
)
```

## Conclusion

The Agent Personality Framework provides a comprehensive approach to defining agent characteristics, enabling:

1. **Rich Agent Identities**: Creates distinctive agent personalities with detailed traits
2. **Optimized System Prompts**: Translates personality data into effective LLM instructions
3. **Role-Specific Configurations**: Tailors personalities to evolutionary roles
4. **Neo4j Integration**: Stores and queries personality data in the graph database
5. **Dynamic Adaptation**: Allows personalities to evolve based on performance data

This framework enhances the evolutionary process by creating specialized agents with complementary traits, improving the quality and diversity of solutions generated by the platform.
