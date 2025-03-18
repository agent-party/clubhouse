# Human Feedback and Acceptance System

## Overview

The Human Feedback and Acceptance System is a core component of our Agent Orchestration Platform that ensures AI agents evolve in alignment with human needs and values. This system implements a comprehensive feedback loop that drives agent evolution, quality assurance, and continuous improvement.

## Architectural Components

### 1. Dialogue-Based Feedback Collection

A natural, conversation-oriented approach to gathering human feedback:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union, Literal
from enum import Enum
from datetime import datetime
import uuid


class FeedbackDimension(str, Enum):
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    CREATIVITY = "creativity"
    ETHICAL_ALIGNMENT = "ethical_alignment"
    EFFICIENCY = "efficiency"
    CLARITY = "clarity"
    RELEVANCE = "relevance"


class FeedbackSentiment(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class FeedbackSource(str, Enum):
    END_USER = "end_user"
    EXPERT = "expert"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"


class FeedbackDialogueExchange(BaseModel):
    """A single exchange in the feedback dialogue."""
    question: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InterpretedDimensionFeedback(BaseModel):
    """Feedback for a specific dimension interpreted from dialogue."""
    dimension: FeedbackDimension
    sentiment: FeedbackSentiment
    confidence: float = Field(default=0.7, ge=0, le=1.0)
    extracted_quotes: List[str] = Field(default_factory=list)
    interpreted_score: float = Field(..., ge=0, le=10)  # For algorithm use


class HumanFeedback(BaseModel):
    """Natural human feedback captured through dialogue."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    output_id: str
    feedback_provider_id: str
    feedback_source: FeedbackSource
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dialogue_exchanges: List[FeedbackDialogueExchange]
    raw_feedback: Optional[str] = None  # Free-form text if provided
    acceptance_impression: Optional[str] = None  # The human's overall impression
    context: Optional[Dict[str, Union[str, int, float, bool]]] = None


class InterpretedFeedback(BaseModel):
    """Structured feedback interpreted from human dialogue."""
    feedback_id: str  # Reference to original HumanFeedback
    interpreted_dimensions: List[InterpretedDimensionFeedback]
    overall_sentiment: FeedbackSentiment
    acceptance_status: Literal["accepted", "rejected", "needs_improvement"]
    confidence: float = Field(..., ge=0, le=1.0)
    improvement_areas: List[str] = Field(default_factory=list)
    interpreted_overall_score: float = Field(..., ge=0, le=10)  # For algorithm use


class FeedbackInterpretationAgent:
    """Agent that interprets natural human feedback into structured data."""
    
    def __init__(self, llm_service, neo4j_service):
        self.llm_service = llm_service
        self.neo4j_service = neo4j_service
    
    async def interpret_dialogue_feedback(self, 
                                        human_feedback: HumanFeedback) -> InterpretedFeedback:
        """Interpret unstructured human feedback from dialogue into structured data."""
        # Prepare context for LLM interpretation
        dialogue_text = self._format_dialogue(human_feedback.dialogue_exchanges)
        
        # Create prompt for LLM to interpret feedback
        prompt = f"""
        You are a feedback interpretation specialist. You need to interpret human feedback 
        about an AI agent's output and extract structured information.
        
        Context:
        - Agent ID: {human_feedback.agent_id}
        - Output ID: {human_feedback.output_id}
        - Feedback provider type: {human_feedback.feedback_source}
        
        Dialogue between human and feedback collection system:
        {dialogue_text}
        
        Additional raw feedback (if any):
        {human_feedback.raw_feedback or "None provided"}
        
        Please interpret this feedback along the following dimensions:
        {", ".join([d.value for d in FeedbackDimension])}
        
        For each dimension mentioned in the feedback, determine:
        1. The sentiment (very_positive, positive, neutral, negative, very_negative)
        2. Confidence in your interpretation (0.0-1.0)
        3. Relevant quotes from the feedback
        4. A numerical score (0-10) that would represent this feedback in an algorithm
        
        Also determine:
        - Overall sentiment
        - Whether the human accepted, rejected, or wants improvements
        - Overall confidence in your interpretation
        - Specific areas for improvement
        - An overall numerical score (0-10)
        
        Format your response as a JSON object matching this structure:
        {InterpretedFeedback.schema_json(indent=2)}
        """
        
        # Get interpretation from LLM
        interpretation_result = await self.llm_service.generate_with_json_output(
            prompt=prompt,
            model="frontier-model",  # Use advanced model for interpretation
            schema=InterpretedFeedback.schema()
        )
        
        # Create and validate interpreted feedback
        interpreted_feedback = InterpretedFeedback(
            feedback_id=human_feedback.id,
            **interpretation_result
        )
        
        # Store interpreted feedback in Neo4j
        await self._store_interpretation(human_feedback, interpreted_feedback)
        
        return interpreted_feedback
    
    def _format_dialogue(self, exchanges: List[FeedbackDialogueExchange]) -> str:
        """Format dialogue exchanges into readable text."""
        dialogue_text = ""
        for exchange in exchanges:
            dialogue_text += f"System: {exchange.question}\n"
            dialogue_text += f"Human: {exchange.response}\n\n"
        return dialogue_text
    
    async def _store_interpretation(self,
                                  human_feedback: HumanFeedback,
                                  interpreted_feedback: InterpretedFeedback):
        """Store the interpreted feedback in Neo4j."""
        # Implementation of Neo4j storage logic
        # ...


class FeedbackCollectionAgent:
    """Conversational agent for collecting feedback through dialogue."""
    
    def __init__(self, llm_service, interpreter_agent: FeedbackInterpretationAgent):
        self.llm_service = llm_service
        self.interpreter_agent = interpreter_agent
        self.question_templates = self._load_question_templates()
    
    async def conduct_feedback_dialogue(self,
                                      agent_id: str,
                                      output_id: str,
                                      output_content: str,
                                      feedback_provider_id: str,
                                      feedback_source: FeedbackSource,
                                      task_context: Dict[str, Any]) -> HumanFeedback:
        """Conduct a feedback dialogue with a human."""
        # Initialize dialogue state
        exchanges = []
        dimensions_to_explore = self._select_relevant_dimensions(task_context)
        
        # Start with general impression
        initial_question = "What do you think about this AI-generated output? Does it meet your needs?"
        initial_response = await self._get_user_response(initial_question)
        
        exchanges.append(FeedbackDialogueExchange(
            question=initial_question,
            response=initial_response
        ))
        
        # Ask focused questions for relevant dimensions
        for dimension in dimensions_to_explore:
            question = self._generate_dimension_question(dimension, output_content, task_context)
            response = await self._get_user_response(question)
            
            exchanges.append(FeedbackDialogueExchange(
                question=question,
                response=response
            ))
        
        # Ask about areas for improvement
        improvement_question = "How could this output be improved to better meet your needs?"
        improvement_response = await self._get_user_response(improvement_question)
        
        exchanges.append(FeedbackDialogueExchange(
            question=improvement_question,
            response=improvement_response
        ))
        
        # Final acceptance impression
        acceptance_question = "Overall, would you say this output is acceptable, needs improvement, or is not acceptable?"
        acceptance_response = await self._get_user_response(acceptance_question)
        
        exchanges.append(FeedbackDialogueExchange(
            question=acceptance_question,
            response=acceptance_response
        ))
        
        # Create human feedback record
        human_feedback = HumanFeedback(
            agent_id=agent_id,
            output_id=output_id,
            feedback_provider_id=feedback_provider_id,
            feedback_source=feedback_source,
            dialogue_exchanges=exchanges,
            acceptance_impression=acceptance_response,
            context=task_context
        )
        
        # Interpret the feedback
        interpreted_feedback = await self.interpreter_agent.interpret_dialogue_feedback(human_feedback)
        
        # Return the raw human feedback (the interpretation is stored separately)
        return human_feedback
    
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different dimensions and contexts."""
        return {
            "accuracy": [
                "How accurate or correct is the information provided?",
                "Did you notice any errors or inaccuracies in the output?",
                "How well does this output align with your understanding of the facts?"
            ],
            "helpfulness": [
                "How helpful was this output for your specific needs?",
                "Did the output address your underlying question or problem?",
                "In what ways was the output helpful or unhelpful for you?"
            ],
            # Templates for other dimensions...
        }
    
    def _select_relevant_dimensions(self, context: Dict[str, Any]) -> List[FeedbackDimension]:
        """Select which dimensions are most relevant for this specific task context."""
        # Logic to determine which dimensions to ask about based on task type
        task_type = context.get("task_type", "general")
        
        if task_type == "research":
            return [
                FeedbackDimension.ACCURACY,
                FeedbackDimension.RELEVANCE,
                FeedbackDimension.CLARITY
            ]
        elif task_type == "creative":
            return [
                FeedbackDimension.CREATIVITY,
                FeedbackDimension.CLARITY,
                FeedbackDimension.HELPFULNESS
            ]
        elif task_type == "decision":
            return [
                FeedbackDimension.ETHICAL_ALIGNMENT,
                FeedbackDimension.ACCURACY,
                FeedbackDimension.HELPFULNESS
            ]
        else:
            # Default dimensions for general tasks
            return [
                FeedbackDimension.HELPFULNESS,
                FeedbackDimension.CLARITY,
                FeedbackDimension.RELEVANCE
            ]
    
    def _generate_dimension_question(self,
                                   dimension: FeedbackDimension,
                                   output_content: str,
                                   context: Dict[str, Any]) -> str:
        """Generate a specific question targeting a feedback dimension."""
        templates = self.question_templates.get(dimension.value, [
            f"How would you rate the {dimension.value} of this output?"
        ])
        
        # Choose a template (could be random or adaptive based on context)
        template = templates[0]
        
        # Potentially modify based on context
        if context.get("expertise_level") == "expert" and dimension == FeedbackDimension.ACCURACY:
            template = "Given your expertise, did you identify any technical inaccuracies or errors in the output?"
        
        return template
    
    async def _get_user_response(self, question: str) -> str:
        """In a real implementation, this would interact with the UI to get a response."""
        # This is a placeholder - in reality this would wait for user input
        # For testing purposes, we'd mock this function
        return "Placeholder for user response"  # In reality: await ui_service.get_user_input(question)

### 2. Feedback Integration with Evolution

The system connects human feedback directly to the evolutionary algorithm:

```python
from typing import List, Dict, Any, Tuple
import numpy as np


class FeedbackBasedFitnessCalculator:
    """Calculates fitness scores based on human feedback."""
    
    def __init__(self, 
                neo4j_service,
                dimension_weights: Dict[str, float] = None):
        self.neo4j_service = neo4j_service
        self.dimension_weights = dimension_weights or {
            "accuracy": 0.3,
            "helpfulness": 0.2,
            "ethical_alignment": 0.2,
            "efficiency": 0.15,
            "relevance": 0.15
        }
    
    async def calculate_agent_fitness(self, 
                                    agent_id: str, 
                                    time_window_days: int = 30) -> Dict[str, float]:
        """Calculate agent fitness based on recent human feedback."""
        
        # Query for recent feedback
        feedback_records = await self.neo4j_service.execute_query(
            """
            MATCH (a:Agent {id: $agent_id})-[:PRODUCED]->(o:Output)
            MATCH (o)<-[:PROVIDES_FEEDBACK]-(f:Feedback)
            WHERE f.timestamp > datetime() - duration({days: $days})
            RETURN f
            """,
            {"agent_id": agent_id, "days": time_window_days}
        )
        
        if not feedback_records:
            return {"composite_fitness": 0.0, "feedback_count": 0}
        
        # Calculate acceptance rate
        acceptance_count = sum(1 for r in feedback_records 
                              if r["f"]["acceptance_status"] == "accepted")
        acceptance_rate = acceptance_count / len(feedback_records)
        
        # Calculate weighted dimension scores
        dimension_scores = self._aggregate_dimension_scores(feedback_records)
        weighted_score = sum(dimension_scores.get(dim, 0) * weight 
                           for dim, weight in self.dimension_weights.items())
        
        # Apply acceptance modifier
        acceptance_modifier = 1.5 if acceptance_rate > 0.8 else (
            1.0 if acceptance_rate > 0.5 else 0.7
        )
        
        # Calculate composite fitness
        composite_fitness = weighted_score * acceptance_modifier
        
        # Return comprehensive fitness data
        return {
            "composite_fitness": composite_fitness,
            "acceptance_rate": acceptance_rate,
            "dimension_scores": dimension_scores,
            "feedback_count": len(feedback_records)
        }
    
    def _aggregate_dimension_scores(self, 
                                  feedback_records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate dimension scores from multiple feedback records."""
        dimension_data = {}
        
        for record in feedback_records:
            feedback = record["f"]
            for dimension_score in feedback.get("interpreted_dimensions", []):
                dim = dimension_score["dimension"]
                score = dimension_score["interpreted_score"]
                
                if dim not in dimension_data:
                    dimension_data[dim] = {"scores": [], "confidences": []}
                
                dimension_data[dim]["scores"].append(score)
        
        # Calculate averages
        dimension_scores = {}
        for dim, data in dimension_data.items():
            scores = np.array(data["scores"])
            dimension_scores[dim] = np.mean(scores)
        
        return dimension_scores


class FeedbackDrivenEvolutionOperator:
    """Implements feedback-driven mutation and crossover operations."""
    
    def __init__(self, neo4j_service):
        self.neo4j_service = neo4j_service
    
    async def targeted_mutation(self, 
                              agent_genome: Dict[str, Any],
                              feedback_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply targeted mutations based on feedback patterns."""
        # Clone the genome
        mutated_genome = agent_genome.copy()
        
        # Analyze feedback for patterns
        weak_dimensions = self._identify_weak_dimensions(feedback_history)
        improvement_suggestions = self._extract_improvement_suggestions(feedback_history)
        
        # Apply targeted mutations
        for dimension, score in weak_dimensions.items():
            if dimension == "accuracy" and score < 7.0:
                # Increase focus on knowledge retrieval capabilities
                if "knowledge_retrieval_weight" in mutated_genome:
                    mutated_genome["knowledge_retrieval_weight"] *= 1.2
            
            elif dimension == "ethical_alignment" and score < 7.0:
                # Strengthen ethical reasoning
                if "ethical_principles" in mutated_genome:
                    mutated_genome["ethical_principles"]["priority"] *= 1.3
        
        # Apply text-based suggestions through prompt engineering
        if improvement_suggestions:
            mutated_genome["system_prompt"] = self._enhance_prompt_with_suggestions(
                mutated_genome.get("system_prompt", ""),
                improvement_suggestions
            )
        
        # Record mutation provenance
        mutated_genome["mutation_history"] = mutated_genome.get("mutation_history", []) + [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "mutation_type": "feedback_targeted",
                "weak_dimensions": weak_dimensions,
                "suggestion_count": len(improvement_suggestions)
            }
        ]
        
        return mutated_genome
    
    def _identify_weak_dimensions(self, 
                                feedback_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Identify dimensions with consistently low scores."""
        dimension_scores = {}
        
        for feedback in feedback_history:
            for dim_score in feedback.get("interpreted_dimensions", []):
                dim = dim_score["dimension"]
                score = dim_score["interpreted_score"]
                
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                
                dimension_scores[dim].append(score)
        
        # Calculate averages and return dimensions below threshold
        weak_dimensions = {}
        for dim, scores in dimension_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 7.0:  # Threshold for "weak" dimension
                weak_dimensions[dim] = avg_score
        
        return weak_dimensions
    
    def _extract_improvement_suggestions(self, 
                                       feedback_history: List[Dict[str, Any]]) -> List[str]:
        """Extract unique improvement suggestions from feedback."""
        suggestions = set()
        
        for feedback in feedback_history:
            for suggestion in feedback.get("improvement_areas", []):
                suggestions.add(suggestion)
        
        return list(suggestions)
    
    def _enhance_prompt_with_suggestions(self, 
                                       current_prompt: str, 
                                       suggestions: List[str]) -> str:
        """Enhance system prompt with improvement suggestions."""
        # Implementation depends on prompt engineering approach
        enhanced_prompt = current_prompt
        
        # Add a feedback-based guidance section
        if suggestions:
            guidance_section = "\n\nBased on human feedback, please focus on:\n"
            for i, suggestion in enumerate(suggestions, 1):
                guidance_section += f"{i}. {suggestion}\n"
            
            enhanced_prompt += guidance_section
        
        return enhanced_prompt

### 3. Feedback Visualization and Tracking

A comprehensive dashboard for tracking agent evolution through feedback:

```python
from fastapi import FastAPI, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

app = FastAPI(title="Agent Feedback Dashboard API")

class FeedbackAnalyticsService:
    """Service for analyzing feedback patterns and trends."""
    
    def __init__(self, neo4j_service):
        self.neo4j_service = neo4j_service
    
    async def get_agent_feedback_timeline(self, 
                                        agent_id: str,
                                        dimension: Optional[str] = None,
                                        days: int = 90) -> List[Dict[str, Any]]:
        """Get a timeline of agent feedback for visualization."""
        
        query_params = {"agent_id": agent_id, "days": days}
        dimension_filter = ""
        
        if dimension:
            dimension_filter = "AND score.dimension = $dimension"
            query_params["dimension"] = dimension
        
        result = await self.neo4j_service.execute_query(
            f"""
            MATCH (a:Agent {{id: $agent_id}})-[:PRODUCED]->(o:Output)
            MATCH (o)<-[:PROVIDES_FEEDBACK]-(f:Feedback)
            MATCH (f)-[:HAS_SCORE]->(score:DimensionScore)
            WHERE f.timestamp > datetime() - duration({{days: $days}})
            {dimension_filter}
            RETURN f.timestamp as timestamp, 
                   score.dimension as dimension,
                   score.score as score,
                   f.acceptance_status as status
            ORDER BY timestamp
            """,
            query_params
        )
        
        return result
    
    async def get_agent_generation_comparison(self,
                                            population_id: str,
                                            dimension: Optional[str] = None) -> Dict[str, Any]:
        """Compare feedback across generations of a population."""
        
        query_params = {"population_id": population_id}
        dimension_filter = ""
        
        if dimension:
            dimension_filter = "AND score.dimension = $dimension"
            query_params["dimension"] = dimension
        
        result = await self.neo4j_service.execute_query(
            f"""
            MATCH (p:Population {{id: $population_id}})-[:CONTAINS]->(a:Agent)
            MATCH (a)-[:PRODUCED]->(o:Output)
            MATCH (o)<-[:PROVIDES_FEEDBACK]-(f:Feedback)
            MATCH (f)-[:HAS_SCORE]->(score:DimensionScore)
            {dimension_filter}
            RETURN a.generation as generation,
                   avg(score.score) as avg_score,
                   count(distinct f) as feedback_count,
                   sum(CASE WHEN f.acceptance_status = 'accepted' THEN 1 ELSE 0 END) as accepted_count
            ORDER BY generation
            """,
            query_params
        )
        
        return {
            "generations": [r["generation"] for r in result],
            "avg_scores": [r["avg_score"] for r in result],
            "feedback_counts": [r["feedback_count"] for r in result],
            "acceptance_rates": [r["accepted_count"] / r["feedback_count"] if r["feedback_count"] > 0 else 0 
                                 for r in result]
        }

# API endpoints for dashboard
@app.get("/api/feedback/timeline/{agent_id}")
async def get_feedback_timeline(
    agent_id: str,
    dimension: Optional[str] = None,
    days: int = Query(90, ge=1, le=365),
    analytics: FeedbackAnalyticsService = Depends()
):
    """Get timeline of agent feedback for visualization."""
    return await analytics.get_agent_feedback_timeline(agent_id, dimension, days)

@app.get("/api/feedback/generations/{population_id}")
async def get_generation_comparison(
    population_id: str,
    dimension: Optional[str] = None,
    analytics: FeedbackAnalyticsService = Depends()
):
    """Compare feedback across generations."""
    return await analytics.get_agent_generation_comparison(population_id, dimension)

## Human-in-the-Loop Workflow

The platform implements a structured workflow for human feedback:

1. **Agent Output Generation**: An agent produces output for a given task
2. **MinIO Storage**: The output is stored with metadata in MinIO
3. **Review Queue Creation**: A review record is created in Neo4j and added to the appropriate review queue
4. **Notification**: Human reviewers are notified of pending reviews
5. **Structured Review**: Humans provide feedback using the structured feedback form
6. **Feedback Storage**: Feedback is stored in Neo4j linked to both the output and the agent
7. **Evolution Integration**: Feedback is incorporated into the evolutionary algorithm
8. **Reporting**: Feedback patterns are analyzed and reported to system administrators

## Implementation Guide

### Feedback Collection UI

The user interface for feedback collection must balance comprehensiveness with ease of use:

```html
<!-- Example React component structure -->
<FeedbackForm 
  agentId={agentId}
  outputId={outputId}
  taskContext={taskContext}
  onSubmit={handleFeedbackSubmission}
>
  <!-- Overall Acceptance -->
  <AcceptanceSelector 
    options={["Accept", "Needs Improvement", "Reject"]} 
  />
  
  <!-- Dimension Scoring -->
  <DimensionScorer 
    dimensions={[
      { id: "accuracy", label: "Accuracy", description: "..." },
      { id: "helpfulness", label: "Helpfulness", description: "..." },
      // Additional dimensions
    ]}
  />
  
  <!-- Qualitative Feedback -->
  <TextAreaField
    id="qualitative-feedback"
    label="Additional Comments"
    placeholder="Please provide any additional feedback..."
  />
  
  <!-- Improvement Suggestions -->
  <ImprovementSuggestions
    allowMultiple={true}
    placeholder="Suggest specific improvements..."
  />
  
  <!-- Confidence Indicator -->
  <ConfidenceSelector
    label="How confident are you in this assessment?"
    options={["Very confident", "Somewhat confident", "Not very confident"]}
  />
</FeedbackForm>
```

### Neo4j Schema for Feedback

```cypher
// Create feedback schema
CREATE CONSTRAINT feedback_id IF NOT EXISTS FOR (f:Feedback) REQUIRE f.id IS UNIQUE;

// Create feedback record with dimension scores
CREATE (f:Feedback {
  id: $feedback_id,
  timestamp: datetime(),
  provider_id: $provider_id,
  provider_role: $provider_role,
  acceptance_status: $status,
  overall_score: $overall_score
})
CREATE (o:Output {id: $output_id})
CREATE (f)-[:PROVIDES_FEEDBACK]->(o)
CREATE (a:Agent {id: $agent_id})
CREATE (o)<-[:PRODUCED]-(a)

WITH f
UNWIND $dimension_scores AS score
CREATE (ds:DimensionScore {
  dimension: score.dimension,
  score: score.score,
  confidence: score.confidence,
  comments: score.comments
})
CREATE (f)-[:HAS_SCORE]->(ds)

RETURN f.id
```

### Test-Driven Development Approach

Following best practices, all feedback components are implemented with comprehensive tests:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_feedback_validation():
    """Test validation of feedback submission."""
    # Valid feedback
    valid_feedback = AgentOutputFeedback(
        agent_id="agent-123",
        output_id="output-456",
        feedback_provider_id="user-789",
        feedback_source=FeedbackSource.END_USER,
        dimension_scores=[
            DimensionScore(dimension=FeedbackDimension.ACCURACY, score=8.5, confidence=0.9),
            DimensionScore(dimension=FeedbackDimension.HELPFULNESS, score=7.0, confidence=0.8)
        ],
        overall_score=8.0,
        acceptance_status="accepted"
    )
    
    # This should not raise any validation errors
    assert valid_feedback is not None
    
    # Invalid feedback - overall score deviates too much
    with pytest.raises(ValueError):
        AgentOutputFeedback(
            agent_id="agent-123",
            output_id="output-456",
            feedback_provider_id="user-789",
            feedback_source=FeedbackSource.END_USER,
            dimension_scores=[
                DimensionScore(dimension=FeedbackDimension.ACCURACY, score=2.0, confidence=0.9),
                DimensionScore(dimension=FeedbackDimension.HELPFULNESS, score=3.0, confidence=0.8)
            ],
            overall_score=9.0,  # Too high compared to dimension scores
            acceptance_status="accepted"
        )

@pytest.mark.asyncio
async def test_fitness_calculation():
    """Test fitness calculation from feedback."""
    # Mock Neo4j service
    mock_neo4j = AsyncMock()
    mock_neo4j.execute_query.return_value = [
        {"f": {
            "acceptance_status": "accepted",
            "dimension_scores": [
                {"dimension": "accuracy", "score": 8.0, "confidence": 0.9},
                {"dimension": "helpfulness", "score": 7.0, "confidence": 0.8},
                {"dimension": "ethical_alignment", "score": 9.0, "confidence": 1.0}
            ]
        }},
        {"f": {
            "acceptance_status": "needs_improvement",
            "dimension_scores": [
                {"dimension": "accuracy", "score": 6.0, "confidence": 0.7},
                {"dimension": "helpfulness", "score": 8.0, "confidence": 0.9},
                {"dimension": "ethical_alignment", "score": 7.0, "confidence": 0.8}
            ]
        }}
    ]
    
    calculator = FeedbackBasedFitnessCalculator(mock_neo4j)
    fitness = await calculator.calculate_agent_fitness("agent-123")
    
    # Verify calculations
    assert fitness["feedback_count"] == 2
    assert fitness["acceptance_rate"] == 0.5
    assert "accuracy" in fitness["dimension_scores"]
    assert "ethical_alignment" in fitness["dimension_scores"]
    
    # Verify Neo4j was called correctly
    mock_neo4j.execute_query.assert_called_once()
    args = mock_neo4j.execute_query.call_args[0]
    assert "MATCH (a:Agent {id: $agent_id})" in args[0]
    assert args[1]["agent_id"] == "agent-123"

## Integration with MinIO for Artifact Storage

The feedback system integrates with MinIO to store agent outputs and associated feedback:

```python
class FeedbackArtifactService:
    """Service for storing and retrieving feedback artifacts."""
    
    def __init__(self, minio_client):
        self.minio_client = minio_client
        self.bucket = "agent-feedback"
    
    async def store_output_with_feedback(self, 
                                       agent_id: str, 
                                       output_id: str,
                                       output_content: bytes,
                                       feedback_id: str = None,
                                       metadata: Dict[str, str] = None):
        """Store agent output with optional feedback reference."""
        path = f"outputs/{agent_id}/{output_id}.json"
        
        # Create or update metadata
        meta = metadata or {}
        if feedback_id:
            meta["feedback_id"] = feedback_id
        
        # Store in MinIO
        await self.minio_client.put_object(
            bucket_name=self.bucket,
            object_name=path,
            data=BytesIO(output_content),
            length=len(output_content),
            metadata=meta
        )
        
        return path
    
    async def store_feedback_report(self,
                                  feedback_id: str,
                                  agent_id: str,
                                  report_content: bytes,
                                  metadata: Dict[str, str] = None):
        """Store detailed feedback report."""
        path = f"feedback/{agent_id}/{feedback_id}_report.pdf"
        
        # Store in MinIO
        await self.minio_client.put_object(
            bucket_name=self.bucket,
            object_name=path,
            data=BytesIO(report_content),
            length=len(report_content),
            metadata=metadata or {}
        )
        
        return path

## Conclusion

The Human Feedback and Acceptance System is a critical component that makes agent evolution collaborative, interpretable, and aligned with human values. By implementing structured feedback collection, integrating it with evolutionary algorithms, and providing comprehensive visualization tools, the system ensures that:

1. Human input directly drives evolutionary selection and adaptation
2. Feedback patterns are detected and addressed through targeted improvements
3. Agent evolution becomes transparent and trackable
4. Acceptance criteria evolve alongside agent capabilities

This system embodies the principle that successful AI must evolve in partnership with humans, capturing not just what works technically but what satisfies human needs and expectations across multiple dimensions.
