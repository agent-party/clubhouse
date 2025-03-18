# Human Interaction

## Overview

The human interaction system serves as the foundation for meaningful collaboration between humans and AI agents. It encompasses both initial problem definition and feedback collection through natural dialogue, creating a unified experience that respects human values while providing structured data for agent evolution.

## Principles

1. **Natural Dialogue**: Humans express requirements and feedback through natural conversation
2. **Implicit Understanding**: The system extracts meaning without requiring explicit ratings or structured input
3. **Contextual Continuity**: The same conversation flows naturally from problem definition to feedback
4. **Confidence Quantification**: The system measures certainty in its understanding and interpretation
5. **Stoic Alignment**: Interactions follow Stoic principles of wisdom, courage, justice, and temperance

## Human Liaison Agent

The Human Liaison Agent serves as the primary interface between humans and the AI ecosystem:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Literal, Any
from enum import Enum
from datetime import datetime
import uuid


class ConfidenceLevel(BaseModel):
    """Quantified confidence in understanding or interpretation."""
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    confidence_factors: Dict[str, float]


class RequirementElement(BaseModel):
    """An individual element of the requirements specification."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    importance: float = Field(default=1.0, ge=0.0, le=1.0)
    constraints: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    extracted_from: Optional[List[str]] = None  # References to dialogue exchanges


class TaskSpecification(BaseModel):
    """Structured task specification derived from conversation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    requirements: List[RequirementElement]
    constraints: Optional[List[str]] = None
    success_criteria: Optional[List[str]] = None
    required_capabilities: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None
    confidence: ConfidenceLevel


class HumanLiaisonAgent:
    """Agent responsible for human interaction throughout the task lifecycle."""
    
    def __init__(self, 
                llm_service,
                neo4j_service,
                event_bus,
                feedback_interpretation_service=None):
        self.llm_service = llm_service
        self.neo4j_service = neo4j_service
        self.event_bus = event_bus
        self.feedback_interpretation_service = feedback_interpretation_service
        self.conversations = {}  # Track active conversations
```

### Core Responsibilities

1. **Problem Definition & Clarification**
   - Elicit clear requirements through conversational dialogue
   - Calculate confidence in understanding
   - Translate natural language requirements into structured task specifications

2. **Task Transition & Monitoring**
   - Maintain conversation context across the task lifecycle
   - Provide status updates and intermediate results when appropriate
   - Facilitate mid-task clarifications if needed

3. **Feedback Collection & Interpretation**
   - Gather natural feedback through continuation of the same conversation
   - Interpret feedback dimensions without explicit numerical inputs
   - Calculate confidence-weighted feedback metrics for the evolution engine

## Feedback Interpretation Framework

The Feedback Interpretation Framework translates natural human dialogue into structured metrics for agent evolution:

```python
class FeedbackDimension(str, Enum):
    """Dimensions for evaluating agent performance."""
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    COMPLETENESS = "completeness"
    CREATIVITY = "creativity"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    ETHICS = "ethics"
    SAFETY = "safety"


class SentimentLevel(str, Enum):
    """Sentiment levels for feedback assessment."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class AcceptanceStatus(str, Enum):
    """Status of acceptance for agent output."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_IMPROVEMENT = "needs_improvement"


class DimensionFeedback(BaseModel):
    """Feedback on a specific performance dimension."""
    dimension: FeedbackDimension
    sentiment: SentimentLevel
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_quotes: List[str] = Field(default_factory=list)
    

class FeedbackMetrics(BaseModel):
    """Structured metrics extracted from human feedback dialogue."""
    overall_satisfaction: float = Field(..., ge=0.0, le=1.0)
    acceptance_status: AcceptanceStatus
    dimension_feedback: List[DimensionFeedback]
    improvement_suggestions: List[str] = Field(default_factory=list)
    interpretation_confidence: float = Field(..., ge=0.0, le=1.0)


class FeedbackInterpreter:
    """Service for interpreting human feedback from natural dialogue."""
    
    def __init__(self, llm_service, neo4j_service):
        self.llm_service = llm_service
        self.neo4j_service = neo4j_service
    
    async def interpret_dialogue(self,
                               dialogue_exchanges: List[Dict],
                               task_specification: Dict,
                               task_result: Dict) -> FeedbackMetrics:
        """Interpret feedback from dialogue exchanges."""
        # Format dialogue for interpretation
        formatted_dialogue = self._format_dialogue(dialogue_exchanges)
        
        # Create prompt for feedback interpretation
        prompt = f"""
        You are an expert at interpreting human feedback about AI-generated outputs.
        
        Review this conversation where a human is providing feedback about a completed task:
        
        {formatted_dialogue}
        
        Task specifications:
        {task_specification}
        
        Based on the conversation, determine:
        
        1. Overall satisfaction (0.0-1.0)
        2. Acceptance status: "accepted", "rejected", or "needs_improvement"
        3. Specific feedback dimensions:
           - Accuracy
           - Helpfulness
           - Completeness
           - Creativity
           - Clarity
           - Efficiency
           - Ethics
           - Safety
        4. For each dimension mentioned in the feedback, extract:
           - Sentiment (very_negative, negative, neutral, positive, very_positive)
           - Confidence in your interpretation (0.0-1.0)
           - Supporting quotes from the conversation
           - A normalized score (0.0-1.0)
        5. Specific improvement suggestions
        6. Your overall confidence in this interpretation
        
        Output your analysis as JSON matching the FeedbackMetrics schema.
        """
        
        # Get interpretation from LLM
        metrics_dict = await self.llm_service.generate_with_json_output(
            prompt=prompt,
            schema=FeedbackMetrics.schema()
        )
        
        # Create and validate FeedbackMetrics
        metrics = FeedbackMetrics(**metrics_dict)
        
        # Persist feedback metrics
        await self.neo4j_service.store_feedback_metrics(
            task_id=task_specification["id"],
            metrics=metrics.dict()
        )
        
        return metrics
```

### Interpretation Process

The feedback interpretation process follows these steps:

1. **Dialogue Extraction**: Isolate feedback-relevant parts of the conversation
2. **Sentiment Analysis**: Identify emotional tone toward different dimensions
3. **Issue Identification**: Extract specific problems or concerns
4. **Suggestion Extraction**: Identify constructive suggestions for improvement
5. **Acceptance Classification**: Determine if the output was accepted, rejected, or needs changes
6. **Confidence Calculation**: Assess certainty in the interpretation

## Integration with Evolution Framework

The human interaction system drives agent evolution through:

1. **Requirement-Based Initialization**: Initial agent configurations are based on task requirements
2. **Feedback-Directed Mutation**: Specific issues identified in feedback target precise mutations
3. **Confidence-Weighted Selection**: Higher confidence feedback has stronger evolutionary influence
4. **Archetype-Specific Interpretation**: Feedback is interpreted through the lens of relevant archetypes
5. **Rejection Protection**: Agents consistently rejected through feedback are removed from the gene pool

```python
class EvolutionIntegrationService:
    """Integrates human feedback with the evolution framework."""
    
    def __init__(self, evolution_service, neo4j_service, event_bus):
        self.evolution_service = evolution_service
        self.neo4j_service = neo4j_service
        self.event_bus = event_bus
    
    async def process_task_feedback(self, 
                                  task_id: str, 
                                  agent_id: str,
                                  feedback_metrics: FeedbackMetrics):
        """Process task feedback to influence agent evolution."""
        # Store feedback metrics
        await self.neo4j_service.store_agent_feedback(
            agent_id=agent_id,
            task_id=task_id,
            metrics=feedback_metrics.dict()
        )
        
        # Calculate fitness score from feedback
        fitness_score = self._calculate_fitness_score(feedback_metrics)
        
        # Update agent fitness in evolution system
        await self.evolution_service.update_agent_fitness(
            agent_id=agent_id,
            fitness_score=fitness_score,
            feedback_data=feedback_metrics.dict()
        )
        
        # Trigger evolutionary step if needed
        if await self._should_trigger_evolution():
            await self.evolution_service.evolve_population()
    
    def _calculate_fitness_score(self, 
                               feedback_metrics: FeedbackMetrics) -> float:
        """Calculate a fitness score from feedback metrics."""
        # Base score on overall satisfaction
        base_score = feedback_metrics.overall_satisfaction
        
        # Adjust based on acceptance status
        if feedback_metrics.acceptance_status == AcceptanceStatus.ACCEPTED:
            status_modifier = 1.2  # Boost for accepted outputs
        elif feedback_metrics.acceptance_status == AcceptanceStatus.NEEDS_IMPROVEMENT:
            status_modifier = 0.8  # Penalty for outputs needing improvement
        else:  # REJECTED
            status_modifier = 0.5  # Major penalty for rejected outputs
        
        # Calculate dimension-weighted score
        dimension_score = 0.0
        total_weight = 0.0
        
        for dim_feedback in feedback_metrics.dimension_feedback:
            # Weight by confidence in the interpretation
            weight = dim_feedback.confidence
            dimension_score += dim_feedback.score * weight
            total_weight += weight
        
        # Combine scores
        if total_weight > 0:
            combined_score = (
                0.5 * base_score + 
                0.5 * (dimension_score / total_weight)
            ) * status_modifier
        else:
            combined_score = base_score * status_modifier
        
        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, combined_score))
```

## Natural Dialogue Collection

The platform prioritizes natural conversation for several reasons:

1. **Reduced Cognitive Load**: Humans express feedback naturally without form-filling
2. **Richer Information**: Dialogue provides context, nuance, and emotional content
3. **Relationship Building**: Conversation builds rapport and trust
4. **Elicitation of Tacit Knowledge**: Natural dialogue reveals unstated requirements
5. **Adaptation to Human Style**: System adapts to different human communication styles

### Example Dialogue Flow

**Human**: "I need to create an interactive dashboard that shows our sales data from the last quarter. It should highlight our best-performing products and identify any concerning trends."

**Liaison Agent**: *[Conducts dialogue to clarify requirements]*

"I understand you need an interactive dashboard for last quarter's sales data. Just to make sure I've got everything:
1. What specific metrics should be included?
2. Who is the primary audience for this dashboard?
3. Are there any specific visualization types you prefer?"

**Human**: "We need to show revenue, units sold, and profit margin. The audience is our executive team who aren't very technical. Bar charts and line graphs would work best."

**Liaison Agent**: *[Calculates 85% confidence]*

"Thanks for the clarification! Here's what I understand about the dashboard you need:

**Dashboard Purpose**: Visualize Q4 sales data with focus on best-performing products and concerning trends
**Key Metrics**: Revenue, units sold, profit margin
**Target Audience**: Executive team (non-technical)
**Preferred Visuals**: Bar charts and line graphs

I'll arrange for this dashboard to be created. Is there anything else I should know before I get started?"

*[System handles the task execution with appropriate agents]*

**Liaison Agent**: "I've completed the sales dashboard based on your requirements. You can access it here: [link]. It shows Q4 sales data with visualizations for revenue, units sold, and profit margins. You can filter by region using the dropdown in the top right.

How well does this dashboard meet your needs? Is there anything that could be improved?"

**Human**: "This is mostly good, but the colors are hard to distinguish for someone with color blindness like me. Also, I was hoping to see a way to compare quarters."

**Liaison Agent**: *[Interprets feedback along multiple dimensions]*

"I understand the dashboard needs two improvements:
1. More accessible colors for color-blind users
2. Adding quarter-to-quarter comparison functionality

Would you like me to update the dashboard with these changes? And besides these two issues, does the dashboard effectively show your best-performing products and concerning trends as requested?"

## Testing Approach

The human interaction system follows test-driven development with these test categories:

1. **Dialogue Understanding Tests**: Verify accurate extraction of requirements
2. **Confidence Calculation Tests**: Ensure appropriate confidence scoring
3. **Feedback Interpretation Tests**: Validate accurate extraction of feedback dimensions
4. **Integration Tests**: Verify proper connection to evolution framework
5. **End-to-End Tests**: Test complete workflows from problem definition to feedback

```python
@pytest.mark.asyncio
async def test_feedback_interpretation():
    """Test interpretation of natural language feedback into metrics."""
    # Mock dependencies
    mock_llm = AsyncMock()
    mock_neo4j = AsyncMock()
    
    # Configure mock responses
    mock_llm.generate_with_json_output.return_value = {
        "overall_satisfaction": 0.75,
        "acceptance_status": "needs_improvement",
        "dimension_feedback": [
            {
                "dimension": "accuracy",
                "sentiment": "positive",
                "confidence": 0.9,
                "supporting_quotes": ["The product analysis is good"],
                "score": 0.85
            },
            {
                "dimension": "accessibility",
                "sentiment": "negative",
                "confidence": 0.95,
                "supporting_quotes": ["colors are hard to distinguish for someone with color blindness"],
                "score": 0.3
            }
        ],
        "improvement_suggestions": [
            "More accessible colors for color-blind users",
            "Add quarter-to-quarter comparison"
        ],
        "interpretation_confidence": 0.9
    }
    
    # Create interpreter
    interpreter = FeedbackInterpreter(mock_llm, mock_neo4j)
    
    # Sample dialogue
    dialogue = [
        {"role": "human", "content": "This is mostly good, but the colors are hard to distinguish for someone with color blindness like me. Also, I was hoping to see a way to compare quarters."},
        {"role": "assistant", "content": "I understand the dashboard needs two improvements..."}
    ]
    
    # Sample task specification
    task_spec = {
        "id": "task-123",
        "title": "Sales Dashboard",
        "description": "Create a sales dashboard for Q4 data"
    }
    
    # Sample task result
    task_result = {
        "url": "https://dashboard.example.com"
    }
    
    # Process feedback
    result = await interpreter.interpret_dialogue(dialogue, task_spec, task_result)
    
    # Verify interpretation
    assert result.overall_satisfaction == 0.75
    assert result.acceptance_status == AcceptanceStatus.NEEDS_IMPROVEMENT
    assert len(result.dimension_feedback) == 2
    assert result.dimension_feedback[1].dimension == FeedbackDimension.ACCESSIBILITY
    assert result.dimension_feedback[1].score == 0.3
    assert len(result.improvement_suggestions) == 2
    
    # Verify LLM was called correctly
    mock_llm.generate_with_json_output.assert_called_once()
    prompt = mock_llm.generate_with_json_output.call_args[1]["prompt"]
    assert "interpreting human feedback" in prompt.lower()
```

## Ethical Considerations

The human interaction system includes several ethical safeguards:

1. **Transparent Interpretation**: The system explains how it understood feedback
2. **Confidence Reporting**: Low confidence triggers clarification questions
3. **Bias Detection**: The system identifies and mitigates biases in feedback
4. **Privacy Protection**: Sensitive information is properly handled
5. **Cultural Adaptation**: The system respects cultural differences in communication

## Conclusion

The human interaction system creates a seamless experience from problem definition through feedback collection. By interpreting natural dialogue rather than requiring structured input, it reduces cognitive load on humans while providing rich data for agent evolution. The confidence-based approach ensures reliable evolution even with ambiguous feedback, leading to continuous improvement in alignment with human needs and values.
