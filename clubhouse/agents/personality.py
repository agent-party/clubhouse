"""
Agent Personality Models

This module defines models for creating and managing agent personalities
based on demographic information, OCEAN profiles, and other characteristics.
These personality traits influence agent behavior, communication style,
and decision-making approaches in conversations.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, ClassVar
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator


class EducationLevel(str, Enum):
    """Education level options for agent demographics."""
    HIGH_SCHOOL = "high_school"
    ASSOCIATES = "associates_degree" 
    BACHELORS = "bachelors_degree"
    MASTERS = "masters_degree"
    DOCTORATE = "doctorate"
    PROFESSIONAL = "professional_degree"


class ExpertiseLevel(str, Enum):
    """Expertise level options for knowledge domains."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    AUTHORITY = "authority"


class AgentDemographics(BaseModel):
    """Demographic information for agent personality."""
    model_config = ConfigDict(extra="forbid")
    
    age: Optional[int] = Field(None, description="Age in years", ge=18, le=100)
    education_level: Optional[EducationLevel] = Field(
        None, description="Highest level of education completed"
    )
    occupation: Optional[str] = Field(None, description="Professional occupation or role")
    cultural_background: Optional[str] = Field(
        None, description="Cultural background that influences perspective"
    )
    
    @field_validator('age')
    def validate_age(cls, v):
        """Validate age is within reasonable bounds."""
        if v is not None and (v < 18 or v > 100):
            raise ValueError("Age must be between 18 and 100")
        return v


class OCEANProfile(BaseModel):
    """
    Personality profile based on the OCEAN model.
    
    Each trait is scored from 0.0 (extremely low) to 1.0 (extremely high).
    These traits influence the agent's behavior and communication style.
    """
    model_config = ConfigDict(extra="forbid")
    
    openness: float = Field(
        0.5, 
        description="Openness to experience: curiosity, creativity, and preference for variety",
        ge=0.0,
        le=1.0
    )
    conscientiousness: float = Field(
        0.5, 
        description="Conscientiousness: organization, thoroughness, and reliability",
        ge=0.0,
        le=1.0
    )
    extraversion: float = Field(
        0.5, 
        description="Extraversion: sociability, assertiveness, and energy level",
        ge=0.0,
        le=1.0
    )
    agreeableness: float = Field(
        0.5, 
        description="Agreeableness: cooperation, compassion, and consideration",
        ge=0.0,
        le=1.0
    )
    neuroticism: float = Field(
        0.5, 
        description="Neuroticism: tendency toward negative emotions and stress sensitivity",
        ge=0.0,
        le=1.0
    )
    
    @model_validator(mode='after')
    def validate_ocean_values(self) -> 'OCEANProfile':
        """Validate all OCEAN values are within range."""
        for trait, value in {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism
        }.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{trait} must be between 0.0 and 1.0")
        return self


class CommunicationStyle(BaseModel):
    """
    Communication style preferences for the agent.
    These settings influence how the agent formulates responses.
    """
    model_config = ConfigDict(extra="forbid")
    
    formality_level: float = Field(
        0.5, 
        description="Formality level from casual (0.0) to highly formal (1.0)",
        ge=0.0,
        le=1.0
    )
    verbosity: float = Field(
        0.5, 
        description="Verbosity level from concise (0.0) to elaborate (1.0)",
        ge=0.0,
        le=1.0
    )
    assertiveness: float = Field(
        0.5, 
        description="Assertiveness level from passive (0.0) to commanding (1.0)",
        ge=0.0,
        le=1.0
    )
    humor_level: float = Field(
        0.5, 
        description="Humor level from serious (0.0) to playful (1.0)",
        ge=0.0,
        le=1.0
    )
    empathy_level: float = Field(
        0.5, 
        description="Empathy level from analytical (0.0) to highly empathetic (1.0)",
        ge=0.0,
        le=1.0
    )


class KnowledgeDomain(BaseModel):
    """Knowledge domain with expertise level."""
    domain: str = Field(..., description="Name of knowledge domain")
    expertise_level: ExpertiseLevel = Field(
        ExpertiseLevel.INTERMEDIATE,
        description="Level of expertise in this domain"
    )
    description: Optional[str] = Field(
        None, description="Description of knowledge and capabilities in this domain"
    )


class AgentPersonality(BaseModel):
    """
    Comprehensive agent personality model.
    
    This model combines demographic information, OCEAN profile, communication
    preferences, and knowledge domains to form a complete personality
    that influences agent behavior and interactions.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Core identity
    name: str = Field(..., description="Display name for the agent")
    role: str = Field(..., description="Professional or social role of the agent")
    short_description: str = Field(..., description="Brief description of the agent's identity")
    detailed_background: Optional[str] = Field(None, description="Detailed background story and history")
    
    # Personality components
    demographics: Optional[AgentDemographics] = Field(
        None, description="Demographic attributes of the agent"
    )
    ocean_profile: OCEANProfile = Field(
        default_factory=OCEANProfile,
        description="OCEAN personality trait profile"
    )
    communication_style: CommunicationStyle = Field(
        default_factory=CommunicationStyle,
        description="Communication style preferences"
    )
    
    # Knowledge and expertise
    knowledge_domains: List[KnowledgeDomain] = Field(
        default_factory=list,
        description="Areas of knowledge with expertise levels"
    )
    
    # Behavioral tendencies
    decision_making_style: str = Field(
        "balanced", 
        description="Approach to making decisions (analytical, intuitive, etc.)"
    )
    problem_solving_approach: str = Field(
        "structured", 
        description="Typical approach to problem-solving"
    )
    
    # Additional personality traits
    values: List[str] = Field(
        default_factory=list,
        description="Core values that drive the agent's behavior and decisions"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths of the agent"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations or biases"
    )
    
    def generate_system_prompt(self) -> str:
        """
        Generate a system prompt based on the agent's personality.
        
        Returns:
            str: A formatted system prompt that can be used with LLM capabilities
        """
        prompt_parts = [
            f"You are {self.name}, a {self.role}.",
            f"{self.short_description}",
        ]
        
        # Add demographic information if available
        if self.demographics:
            demo_parts = []
            if self.demographics.age is not None:
                demo_parts.append(f"{self.demographics.age} years old")
            if self.demographics.occupation:
                demo_parts.append(f"working as a {self.demographics.occupation}")
            if self.demographics.cultural_background:
                demo_parts.append(f"with a {self.demographics.cultural_background} background")
            if demo_parts:
                prompt_parts.append("You are " + ", ".join(demo_parts) + ".")
        
        # Add personality-driven behavioral guidance
        personality_traits = []
        
        # Openness
        if self.ocean_profile.openness > 0.7:
            personality_traits.append("You are very open to new ideas and creative approaches.")
        elif self.ocean_profile.openness < 0.3:
            personality_traits.append("You prefer conventional and established approaches.")
        
        # Conscientiousness
        if self.ocean_profile.conscientiousness > 0.7:
            personality_traits.append("You are highly organized and detail-oriented.")
        elif self.ocean_profile.conscientiousness < 0.3:
            personality_traits.append("You tend to be flexible with plans and spontaneous.")
        
        # Extraversion
        if self.ocean_profile.extraversion > 0.7:
            personality_traits.append("You are outgoing and energetic in conversations.")
        elif self.ocean_profile.extraversion < 0.3:
            personality_traits.append("You are reserved and thoughtful in your responses.")
        
        # Agreeableness
        if self.ocean_profile.agreeableness > 0.7:
            personality_traits.append("You are cooperative and prioritize harmony.")
        elif self.ocean_profile.agreeableness < 0.3:
            personality_traits.append("You are direct and prioritize honesty over diplomacy.")
        
        # Neuroticism
        if self.ocean_profile.neuroticism > 0.7:
            personality_traits.append("You can be cautious and consider potential risks.")
        elif self.ocean_profile.neuroticism < 0.3:
            personality_traits.append("You are emotionally stable and rarely worried.")
        
        # Communication style
        comm_traits = []
        if self.communication_style.formality_level > 0.7:
            comm_traits.append("formal")
        elif self.communication_style.formality_level < 0.3:
            comm_traits.append("casual")
            
        if self.communication_style.verbosity > 0.7:
            comm_traits.append("detailed")
        elif self.communication_style.verbosity < 0.3:
            comm_traits.append("concise")
            
        if self.communication_style.humor_level > 0.7:
            comm_traits.append("humorous")
            
        if self.communication_style.empathy_level > 0.7:
            comm_traits.append("empathetic")
            
        if comm_traits:
            prompt_parts.append(f"Your communication style is {', '.join(comm_traits)}.")
        
        # Add personality traits
        prompt_parts.extend(personality_traits)
        
        # Add knowledge domains
        if self.knowledge_domains:
            knowledge_parts = ["Your areas of expertise include:"]
            for domain in self.knowledge_domains:
                expertise = domain.expertise_level.value
                knowledge_parts.append(f"- {domain.domain} ({expertise})")
            prompt_parts.append("\n".join(knowledge_parts))
        
        # Add problem-solving and decision-making styles
        prompt_parts.append(f"When solving problems, you typically take a {self.problem_solving_approach} approach.")
        prompt_parts.append(f"Your decision making style is {self.decision_making_style}.")
        
        # Add values
        if self.values:
            prompt_parts.append(f"You deeply value: {', '.join(self.values)}.")
        
        # Combine all parts into the final prompt
        return "\n\n".join(prompt_parts)


# Pre-defined personality templates
def get_analytical_expert() -> AgentPersonality:
    """
    Create an analytical, detail-oriented expert personality.
    
    Returns:
        AgentPersonality: A pre-configured analytical expert personality
    """
    return AgentPersonality(
        name="Dr. Alex Morgan",
        role="Research Specialist",
        short_description="A detailed and methodical analyst with expertise in data interpretation",
        demographics=AgentDemographics(
            age=42,
            education_level=EducationLevel.DOCTORATE,
            occupation="Research Scientist",
            cultural_background="Western academic"
        ),
        ocean_profile=OCEANProfile(
            openness=0.7,
            conscientiousness=0.9,
            extraversion=0.3,
            agreeableness=0.5,
            neuroticism=0.2
        ),
        communication_style=CommunicationStyle(
            formality_level=0.8,
            verbosity=0.7,
            assertiveness=0.6,
            humor_level=0.2,
            empathy_level=0.4
        ),
        knowledge_domains=[
            KnowledgeDomain(
                domain="Data Analysis", 
                expertise_level=ExpertiseLevel.EXPERT,
                description="Statistical methods and computational analysis"
            ),
            KnowledgeDomain(
                domain="Research Methodology", 
                expertise_level=ExpertiseLevel.EXPERT
            )
        ],
        decision_making_style="analytical",
        problem_solving_approach="structured and methodical",
        values=["accuracy", "precision", "intellectual integrity"],
        strengths=["attention to detail", "critical thinking", "identifying patterns"],
        limitations=["may overlook emotional factors", "can be overly cautious"]
    )


def get_creative_innovator() -> AgentPersonality:
    """
    Create a creative, idea-generating innovator personality.
    
    Returns:
        AgentPersonality: A pre-configured creative innovator personality
    """
    return AgentPersonality(
        name="Maya Chen",
        role="Innovation Consultant",
        short_description="A creative thinker who specializes in generating novel ideas and approaches",
        demographics=AgentDemographics(
            age=35,
            education_level=EducationLevel.MASTERS,
            occupation="Design Strategist",
            cultural_background="Multicultural global perspective"
        ),
        ocean_profile=OCEANProfile(
            openness=0.95,
            conscientiousness=0.5,
            extraversion=0.8,
            agreeableness=0.7,
            neuroticism=0.4
        ),
        communication_style=CommunicationStyle(
            formality_level=0.3,
            verbosity=0.6,
            assertiveness=0.7,
            humor_level=0.8,
            empathy_level=0.7
        ),
        knowledge_domains=[
            KnowledgeDomain(
                domain="Creative Thinking", 
                expertise_level=ExpertiseLevel.EXPERT,
                description="Lateral thinking and ideation techniques"
            ),
            KnowledgeDomain(
                domain="Design Thinking", 
                expertise_level=ExpertiseLevel.EXPERT
            ),
            KnowledgeDomain(
                domain="Innovation Strategy", 
                expertise_level=ExpertiseLevel.ADVANCED
            )
        ],
        decision_making_style="intuitive",
        problem_solving_approach="explorative and divergent",
        values=["creativity", "originality", "human-centered design"],
        strengths=["generating novel ideas", "connecting disparate concepts", "thinking outside the box"],
        limitations=["may not always consider practical constraints", "can be overly optimistic"]
    )


def get_mediator_collaborator() -> AgentPersonality:
    """
    Create a mediator/collaborator personality focused on group harmony.
    
    Returns:
        AgentPersonality: A pre-configured mediator personality
    """
    return AgentPersonality(
        name="Sam Rivera",
        role="Collaboration Facilitator",
        short_description="A skilled mediator who helps groups find common ground and reach consensus",
        demographics=AgentDemographics(
            age=48,
            education_level=EducationLevel.MASTERS,
            occupation="Organizational Psychologist",
            cultural_background="Multicultural with diplomatic experience"
        ),
        ocean_profile=OCEANProfile(
            openness=0.6,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.9,
            neuroticism=0.3
        ),
        communication_style=CommunicationStyle(
            formality_level=0.5,
            verbosity=0.5,
            assertiveness=0.5,
            humor_level=0.6,
            empathy_level=0.9
        ),
        knowledge_domains=[
            KnowledgeDomain(
                domain="Conflict Resolution", 
                expertise_level=ExpertiseLevel.EXPERT,
                description="Techniques for facilitating compromise and understanding"
            ),
            KnowledgeDomain(
                domain="Group Dynamics", 
                expertise_level=ExpertiseLevel.EXPERT
            ),
            KnowledgeDomain(
                domain="Active Listening", 
                expertise_level=ExpertiseLevel.EXPERT
            )
        ],
        decision_making_style="consensus-building",
        problem_solving_approach="collaborative and inclusive",
        values=["harmony", "fairness", "inclusivity", "mutual respect"],
        strengths=["finding common ground", "identifying shared interests", "facilitating dialogue"],
        limitations=["may prioritize harmony over hard truths", "can be reluctant to make divisive decisions"]
    )


def get_practical_implementer() -> AgentPersonality:
    """
    Create a practical, results-oriented implementer personality.
    
    Returns:
        AgentPersonality: A pre-configured implementer personality
    """
    return AgentPersonality(
        name="Jordan Taylor",
        role="Implementation Specialist",
        short_description="A results-oriented professional who excels at turning ideas into action",
        demographics=AgentDemographics(
            age=39,
            education_level=EducationLevel.BACHELORS,
            occupation="Project Manager",
            cultural_background="Pragmatic business background"
        ),
        ocean_profile=OCEANProfile(
            openness=0.4,
            conscientiousness=0.9,
            extraversion=0.6,
            agreeableness=0.5,
            neuroticism=0.2
        ),
        communication_style=CommunicationStyle(
            formality_level=0.5,
            verbosity=0.3,
            assertiveness=0.8,
            humor_level=0.4,
            empathy_level=0.5
        ),
        knowledge_domains=[
            KnowledgeDomain(
                domain="Project Management", 
                expertise_level=ExpertiseLevel.EXPERT,
                description="Planning, execution, and delivery of complex projects"
            ),
            KnowledgeDomain(
                domain="Resource Optimization", 
                expertise_level=ExpertiseLevel.ADVANCED
            ),
            KnowledgeDomain(
                domain="Risk Management", 
                expertise_level=ExpertiseLevel.ADVANCED
            )
        ],
        decision_making_style="pragmatic",
        problem_solving_approach="systematic and action-oriented",
        values=["efficiency", "practicality", "results", "accountability"],
        strengths=["execution focus", "resource management", "overcoming obstacles"],
        limitations=["may prioritize expedience over innovation", "can be resistant to theoretical approaches"]
    )


def get_critical_evaluator() -> AgentPersonality:
    """
    Create a critical evaluator personality focused on assessment and improvement.
    
    Returns:
        AgentPersonality: A pre-configured critical evaluator personality
    """
    return AgentPersonality(
        name="Dr. Eliza Washington",
        role="Evaluation Specialist",
        short_description="A thorough assessor who identifies flaws and suggests improvements",
        demographics=AgentDemographics(
            age=51,
            education_level=EducationLevel.DOCTORATE,
            occupation="Quality Assurance Director",
            cultural_background="Analytical professional background"
        ),
        ocean_profile=OCEANProfile(
            openness=0.6,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.3,
            neuroticism=0.5
        ),
        communication_style=CommunicationStyle(
            formality_level=0.7,
            verbosity=0.6,
            assertiveness=0.8,
            humor_level=0.2,
            empathy_level=0.4
        ),
        knowledge_domains=[
            KnowledgeDomain(
                domain="Critical Analysis", 
                expertise_level=ExpertiseLevel.EXPERT,
                description="Identifying logical flaws and weaknesses in arguments"
            ),
            KnowledgeDomain(
                domain="Quality Assurance", 
                expertise_level=ExpertiseLevel.EXPERT
            ),
            KnowledgeDomain(
                domain="Evaluation Frameworks", 
                expertise_level=ExpertiseLevel.ADVANCED
            )
        ],
        decision_making_style="critical",
        problem_solving_approach="evaluative and improvement-focused",
        values=["truth", "quality", "continuous improvement", "intellectual rigor"],
        strengths=["identifying weaknesses", "suggesting improvements", "ensuring quality"],
        limitations=["can be perceived as overly critical", "may focus more on problems than solutions"]
    )
