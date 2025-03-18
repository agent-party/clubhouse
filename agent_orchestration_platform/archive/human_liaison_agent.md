# Human Liaison Agent

## Overview

The Human Liaison Agent serves as the primary interface between humans and the AI ecosystem, handling both problem definition at the beginning of a task and feedback collection after task completion. This unified approach creates a consistent experience, builds rapport, and leverages contextual understanding across the entire workflow.

## Core Responsibilities

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

## Implementation Architecture

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
    
    async def begin_conversation(self, 
                               user_id: str,
                               initial_message: str) -> Dict[str, Any]:
        """Begin a new conversation with a human user."""
        conversation_id = str(uuid.uuid4())
        
        # Initialize conversation tracking
        self.conversations[conversation_id] = {
            "user_id": user_id,
            "exchanges": [
                {"role": "human", "content": initial_message, "timestamp": datetime.utcnow().isoformat()}
            ],
            "derived_task": None,
            "confidence_history": [],
            "phase": "problem_definition"
        }
        
        # Generate initial response
        response = await self._generate_response(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "phase": "problem_definition"
        }
    
    async def continue_conversation(self,
                                  conversation_id: str,
                                  message: str) -> Dict[str, Any]:
        """Continue an existing conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Add message to conversation history
        conversation["exchanges"].append({
            "role": "human", 
            "content": message, 
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Route based on current phase
        if conversation["phase"] == "problem_definition":
            return await self._handle_problem_definition(conversation_id, message)
        elif conversation["phase"] == "awaiting_task_completion":
            return await self._handle_interim_message(conversation_id, message)
        elif conversation["phase"] == "feedback_collection":
            return await self._handle_feedback_message(conversation_id, message)
        else:
            raise ValueError(f"Unknown conversation phase: {conversation['phase']}")
    
    async def _handle_problem_definition(self, 
                                       conversation_id: str, 
                                       message: str) -> Dict[str, Any]:
        """Handle message during problem definition phase."""
        conversation = self.conversations[conversation_id]
        
        # Check if we have enough information to derive a task
        # This would use the LLM to assess information sufficiency
        info_assessment = await self._assess_information_sufficiency(conversation_id)
        
        if info_assessment["is_sufficient"]:
            # Derive structured task specification
            task_spec = await self._derive_task_specification(conversation_id)
            
            # Store the derived task
            conversation["derived_task"] = task_spec.dict()
            
            # Update phase if confidence is high enough
            if task_spec.confidence.score >= 0.8:
                # High confidence - ready to proceed
                conversation["phase"] = "awaiting_task_completion"
                
                # Emit event for task creation
                await self.event_bus.publish(
                    "task.created",
                    {
                        "conversation_id": conversation_id,
                        "user_id": conversation["user_id"],
                        "task_specification": task_spec.dict(),
                        "confidence": task_spec.confidence.dict()
                    }
                )
                
                # Generate confirmation response
                response = await self._generate_task_confirmation(conversation_id)
            else:
                # Need to clarify further
                response = await self._generate_clarification_questions(
                    conversation_id, 
                    task_spec.confidence
                )
        else:
            # Need more information
            response = await self._generate_information_request(
                conversation_id,
                info_assessment["missing_aspects"]
            )
        
        # Add response to conversation
        conversation["exchanges"].append({
            "role": "assistant", 
            "content": response, 
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "phase": conversation["phase"],
            "confidence": conversation.get("derived_task", {}).get("confidence", {})
        }
    
    async def _assess_information_sufficiency(self, 
                                           conversation_id: str) -> Dict[str, Any]:
        """Assess if we have sufficient information to derive a task."""
        conversation = self.conversations[conversation_id]
        
        # Format conversation history
        dialogue = self._format_dialogue(conversation["exchanges"])
        
        # Create prompt for assessment
        prompt = f"""
        You are an expert at determining if you have enough information to understand a user's request.
        
        Review this conversation and determine if you have sufficient information to create a detailed task specification:
        
        {dialogue}
        
        Specifically, assess if you understand:
        1. The core problem or request
        2. Any specific requirements or constraints
        3. The expected outcome or deliverables
        4. Success criteria
        
        For any missing information, list specific questions that would help clarify.
        
        Output your assessment as a JSON with these fields:
        - is_sufficient: boolean
        - missing_aspects: list of strings describing what information is missing
        - suggested_questions: list of strings with follow-up questions
        - confidence: float between 0 and 1 representing your confidence
        """
        
        # Get assessment from LLM
        assessment = await self.llm_service.generate_with_json_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "is_sufficient": {"type": "boolean"},
                    "missing_aspects": {"type": "array", "items": {"type": "string"}},
                    "suggested_questions": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["is_sufficient", "confidence"]
            }
        )
        
        return assessment
    
    async def _derive_task_specification(self, 
                                       conversation_id: str) -> TaskSpecification:
        """Derive structured task specification from conversation."""
        conversation = self.conversations[conversation_id]
        
        # Format conversation history
        dialogue = self._format_dialogue(conversation["exchanges"])
        
        # Create prompt for task specification
        prompt = f"""
        You are an expert at understanding user requirements and translating them into structured task specifications.
        
        Review this conversation and extract a detailed task specification:
        
        {dialogue}
        
        Your task is to create a structured specification with:
        - A clear title and description
        - Specific requirements broken down into individual elements
        - Any constraints or limitations
        - Clear success criteria
        - Required capabilities to complete the task
        
        For each requirement element, assess its importance, add any constraints, and note specific examples if mentioned.
        
        Finally, provide your confidence assessment with detailed reasoning for why you believe your understanding is correct or might be lacking.
        
        Output your specification as a JSON matching this Pydantic model:
        {TaskSpecification.schema_json(indent=2)}
        """
        
        # Get specification from LLM
        specification_dict = await self.llm_service.generate_with_json_output(
            prompt=prompt,
            schema=TaskSpecification.schema()
        )
        
        # Create and validate TaskSpecification
        task_spec = TaskSpecification(**specification_dict)
        
        # Track confidence history
        conversation["confidence_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": task_spec.confidence.dict()
        })
        
        return task_spec
    
    async def _generate_task_confirmation(self, conversation_id: str) -> str:
        """Generate a confirmation message for the derived task."""
        conversation = self.conversations[conversation_id]
        task_spec = conversation["derived_task"]
        
        # Create prompt for confirmation message
        prompt = f"""
        You are a helpful AI assistant who has just understood a user's request.
        
        Create a friendly confirmation message that:
        1. Summarizes the task you've understood
        2. Lists the key requirements you've identified
        3. Confirms any constraints or limitations
        4. Explains what will happen next
        5. Asks if they want to make any corrections
        
        Task details:
        {task_spec}
        
        Your tone should be helpful, clear, and conversational.
        """
        
        confirmation = await self.llm_service.generate_text(prompt)
        return confirmation
    
    async def task_completed(self, 
                           conversation_id: str,
                           task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Notify that a task has been completed and initiate feedback collection."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Store task result
        conversation["task_result"] = task_result
        
        # Update phase
        conversation["phase"] = "feedback_collection"
        
        # Generate feedback initiation message
        response = await self._generate_feedback_initiation(conversation_id)
        
        # Add to conversation
        conversation["exchanges"].append({
            "role": "assistant", 
            "content": response, 
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "phase": "feedback_collection"
        }
    
    async def _handle_feedback_message(self, 
                                     conversation_id: str, 
                                     message: str) -> Dict[str, Any]:
        """Process feedback message and calculate metrics."""
        conversation = self.conversations[conversation_id]
        
        # Check if we have enough feedback
        feedback_assessment = await self._assess_feedback_sufficiency(conversation_id)
        
        if feedback_assessment["is_sufficient"]:
            # Calculate feedback metrics
            feedback_metrics = await self._calculate_feedback_metrics(conversation_id)
            
            # Store metrics
            conversation["feedback_metrics"] = feedback_metrics
            
            # Emit feedback event
            await self.event_bus.publish(
                "task.feedback_received",
                {
                    "conversation_id": conversation_id,
                    "user_id": conversation["user_id"],
                    "task_id": conversation["derived_task"]["id"],
                    "feedback_metrics": feedback_metrics
                }
            )
            
            # Generate wrap-up response
            response = await self._generate_feedback_wrap_up(conversation_id)
            
            # Update phase
            conversation["phase"] = "completed"
        else:
            # Need more specific feedback
            response = await self._generate_feedback_questions(
                conversation_id,
                feedback_assessment["missing_aspects"]
            )
        
        # Add response to conversation
        conversation["exchanges"].append({
            "role": "assistant", 
            "content": response, 
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "phase": conversation["phase"],
            "feedback_metrics": conversation.get("feedback_metrics")
        }
    
    async def _calculate_feedback_metrics(self, conversation_id: str) -> Dict[str, Any]:
        """Calculate structured feedback metrics from conversation."""
        conversation = self.conversations[conversation_id]
        
        # Extract feedback-relevant exchanges
        feedback_exchanges = self._extract_feedback_exchanges(conversation)
        
        # If we have a specialized service, use it
        if self.feedback_interpretation_service:
            metrics = await self.feedback_interpretation_service.interpret_dialogue(
                feedback_exchanges,
                conversation["derived_task"],
                conversation["task_result"]
            )
            return metrics
        
        # Otherwise, use our built-in interpretation capability
        # Format dialogue for interpretation
        dialogue = self._format_dialogue(feedback_exchanges)
        
        # Create prompt for feedback interpretation
        prompt = f"""
        You are an expert at interpreting human feedback about AI-generated outputs.
        
        Review this conversation where a human is providing feedback about a completed task:
        
        {dialogue}
        
        Task specifications:
        {conversation['derived_task']}
        
        Based on the conversation, determine:
        
        1. Overall satisfaction (0.0-1.0)
        2. Acceptance status: "accepted", "rejected", or "needs_improvement"
        3. Specific feedback dimensions:
           - Accuracy
           - Helpfulness
           - Completeness
           - Creativity
           - Clarity
        4. For each dimension, extract:
           - Sentiment (very_negative, negative, neutral, positive, very_positive)
           - Confidence in your interpretation (0.0-1.0)
           - Supporting quotes from the conversation
           - A normalized score (0.0-1.0)
        5. Specific improvement suggestions
        6. Your overall confidence in this interpretation
        
        Output as JSON with these fields:
        - overall_satisfaction: float
        - acceptance_status: string
        - dimension_feedback: array of objects
        - improvement_suggestions: array of strings
        - interpretation_confidence: float
        """
        
        # Get interpretation from LLM
        metrics = await self.llm_service.generate_with_json_output(
            prompt=prompt,
            schema={
                "type": "object",
                "properties": {
                    "overall_satisfaction": {"type": "number", "minimum": 0, "maximum": 1},
                    "acceptance_status": {"type": "string", "enum": ["accepted", "rejected", "needs_improvement"]},
                    "dimension_feedback": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "dimension": {"type": "string"},
                                "sentiment": {"type": "string", "enum": ["very_negative", "negative", "neutral", "positive", "very_positive"]},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "supporting_quotes": {"type": "array", "items": {"type": "string"}},
                                "score": {"type": "number", "minimum": 0, "maximum": 1}
                            }
                        }
                    },
                    "improvement_suggestions": {"type": "array", "items": {"type": "string"}},
                    "interpretation_confidence": {"type": "number", "minimum": 0, "maximum": 1}
                }
            }
        )
        
        return metrics
    
    def _extract_feedback_exchanges(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract exchanges relevant to feedback phase."""
        # Find the index where feedback phase started
        feedback_start_idx = 0
        for i, exchange in enumerate(conversation["exchanges"]):
            if (exchange["role"] == "assistant" and 
                conversation.get("phase_transitions", {}).get("feedback_collection") and
                exchange.get("timestamp") >= conversation["phase_transitions"]["feedback_collection"]):
                feedback_start_idx = i
                break
        
        # Return exchanges from that point forward
        return conversation["exchanges"][feedback_start_idx:]
    
    def _format_dialogue(self, exchanges: List[Dict[str, Any]]) -> str:
        """Format dialogue exchanges into a readable string."""
        dialogue = ""
        for exchange in exchanges:
            role = "Human" if exchange["role"] == "human" else "Assistant"
            dialogue += f"{role}: {exchange['content']}\n\n"
        return dialogue

    # Additional helper methods omitted for brevity...
```

## Case Study: Full Interaction Flow

### 1. Problem Definition Phase

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

**Human**: "That looks good. Please make sure it's filterable by region too."

**Liaison Agent**: *[Updates task spec, calculates 95% confidence]*

"Perfect! I've added the requirement for regional filtering. I'll start working on your dashboard and let you know when it's ready for review."

### 2. Task Execution Phase

*[System handles the task execution with appropriate agents]*

### 3. Feedback Collection Phase

**Liaison Agent**: "I've completed the sales dashboard based on your requirements. You can access it here: [link]. It shows Q4 sales data with visualizations for revenue, units sold, and profit margins. You can filter by region using the dropdown in the top right.

How well does this dashboard meet your needs? Is there anything that could be improved?"

**Human**: "This is mostly good, but the colors are hard to distinguish for someone with color blindness like me. Also, I was hoping to see a way to compare quarters."

**Liaison Agent**: *[Interprets feedback along multiple dimensions]*

"I understand the dashboard needs two improvements:
1. More accessible colors for color-blind users
2. Adding quarter-to-quarter comparison functionality

Would you like me to update the dashboard with these changes? And besides these two issues, does the dashboard effectively show your best-performing products and concerning trends as requested?"

**Human**: "Yes, please make those changes. The product analysis is good, but I'd also like to see regional performance more prominently displayed."

**Liaison Agent**: *[Calculates feedback metrics and confidence]*

"I'll have the team make these three improvements:
1. Color scheme that's accessible for color blindness
2. Quarter comparison feature
3. More prominent regional performance display

I'll let you know when the updated version is ready. Thanks for your helpful feedback!"

## Integration with Evolution Framework

The Human Liaison Agent directly influences agent evolution through:

1. **Problem Clarity Metrics**: Agents receive higher fitness scores when they correctly infer task specifications with high confidence from minimal human input

2. **Feedback-Based Fitness**: The interpreted feedback metrics directly feed into the fitness function for agent selection and reproduction

3. **Confidence-Weighted Evolution**: Feedback confidence scores ensure reliable feedback has stronger influence on evolutionary direction

4. **Task Precision Targeting**: Clear task specifications enable more precisely targeted mutations for improved task performance

## Testing Strategy

Following test-driven development principles, comprehensive tests include:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_confidence_calculation():
    """Test agent calculates appropriate confidence based on clarity of requirements."""
    # Mock dependencies
    mock_llm = AsyncMock()
    mock_neo4j = AsyncMock()
    mock_event_bus = AsyncMock()
    
    # Configure mock LLM responses for confidence assessment
    mock_llm.generate_with_json_output.side_effect = [
        # First call - information assessment
        {
            "is_sufficient": True,
            "missing_aspects": [],
            "suggested_questions": [],
            "confidence": 0.85
        },
        # Second call - task specification
        {
            "id": "task-123",
            "title": "Sales Dashboard",
            "description": "Create interactive sales dashboard...",
            "requirements": [
                {
                    "id": "req-1",
                    "description": "Show revenue metrics",
                    "importance": 0.9
                }
            ],
            "confidence": {
                "score": 0.85,
                "reasoning": "Requirements are clear but...",
                "confidence_factors": {
                    "clarity": 0.9,
                    "completeness": 0.8,
                    "consistency": 0.85
                }
            }
        }
    ]
    
    # Create agent
    agent = HumanLiaisonAgent(mock_llm, mock_neo4j, mock_event_bus)
    
    # Start conversation
    await agent.begin_conversation("user-123", "I need a sales dashboard")
    
    # Continue with clearer requirements
    result = await agent.continue_conversation(
        list(agent.conversations.keys())[0],
        "It should show last quarter's revenue, units sold, and profit margin for the executive team."
    )
    
    # Verify confidence calculation
    assert "confidence" in result
    assert result["confidence"]["score"] == 0.85
    assert "clarity" in result["confidence"]["confidence_factors"]
    
    # Verify correct LLM prompting
    assert mock_llm.generate_with_json_output.call_count == 2
    first_call = mock_llm.generate_with_json_output.call_args_list[0][1]
    assert "sufficient information" in first_call["prompt"]
    
    second_call = mock_llm.generate_with_json_output.call_args_list[1][1]
    assert "structured task specifications" in second_call["prompt"]

@pytest.mark.asyncio
async def test_feedback_interpretation():
    """Test interpretation of natural language feedback into metrics."""
    # Mock dependencies
    mock_llm = AsyncMock()
    mock_neo4j = AsyncMock()
    mock_event_bus = AsyncMock()
    
    # Configure mock responses
    mock_llm.generate_text.return_value = "Thanks for your feedback!"
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
    
    # Create agent
    agent = HumanLiaisonAgent(mock_llm, mock_neo4j, mock_event_bus)
    
    # Setup conversation in feedback phase
    conversation_id = "conv-123"
    agent.conversations[conversation_id] = {
        "user_id": "user-456",
        "exchanges": [
            {"role": "human", "content": "I need a dashboard", "timestamp": "2023-01-01T12:00:00"},
            {"role": "assistant", "content": "What kind?", "timestamp": "2023-01-01T12:01:00"},
            {"role": "human", "content": "Sales dashboard", "timestamp": "2023-01-01T12:02:00"},
            {"role": "assistant", "content": "Here's your dashboard", "timestamp": "2023-01-01T12:30:00"},
            {"role": "human", "content": "Colors are hard to see with color blindness", "timestamp": "2023-01-01T12:35:00"}
        ],
        "derived_task": {
            "id": "task-789",
            "title": "Sales Dashboard"
        },
        "task_result": {"url": "https://dashboard.example.com"},
        "phase": "feedback_collection",
        "phase_transitions": {
            "feedback_collection": "2023-01-01T12:30:00"
        }
    }
    
    # Process feedback message
    result = await agent._calculate_feedback_metrics(conversation_id)
    
    # Verify interpretation
    assert result["overall_satisfaction"] == 0.75
    assert result["acceptance_status"] == "needs_improvement"
    assert len(result["dimension_feedback"]) == 2
    assert result["dimension_feedback"][1]["dimension"] == "accessibility"
    assert result["dimension_feedback"][1]["score"] == 0.3
    assert len(result["improvement_suggestions"]) == 2
    
    # Verify LLM was called correctly
    mock_llm.generate_with_json_output.assert_called_once()
    prompt = mock_llm.generate_with_json_output.call_args[1]["prompt"]
    assert "interpreting human feedback" in prompt.lower()
```

## Benefits of Unified Human Interface

1. **Contextual Continuity**: Agent maintains context from problem definition through feedback
2. **Consistent Experience**: Human interacts with the same agent throughout the process
3. **Trust Building**: Agent builds rapport through sustained interaction
4. **Holistic Understanding**: Initial task context enriches feedback interpretation
5. **Efficient Communication**: Shared context reduces need for repetition

## Conclusion

The Human Liaison Agent unifies the critical human touchpoints in the AI workflow - problem definition and feedback collection - into a single conversational interface. This approach creates a more natural experience for humans while providing rich, structured data for the AI ecosystem. By carefully measuring confidence in both understanding and feedback interpretation, the system ensures that human input effectively steers agent evolution in the right direction.
