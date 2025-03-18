"""
Language Tutor Implementation Example

This module demonstrates how to implement the Language Tutor scenario (E1)
using the OpenAI Agent Library integrated with the Agent Orchestration Platform.

Following the test-driven development approach, we include tests first,
followed by the implementation.
"""

import unittest
import os
from typing import Dict, List, Optional, Union, Any, Protocol
from pydantic import BaseModel, Field
from datetime import datetime
from unittest.mock import patch, MagicMock

# ========================
# Schema Definitions
# ========================

class LearnerProfile(BaseModel):
    """Schema for learner profile data."""
    learner_id: str
    target_language: str
    native_language: str
    proficiency_level: str = Field(..., description="Beginner, Intermediate, Advanced, or Fluent")
    learning_goals: List[str]
    learning_preferences: Dict[str, Any]
    available_time_weekly: int = Field(..., description="Available time in minutes per week")


class LearningActivity(BaseModel):
    """Schema for a learning activity."""
    activity_id: str
    activity_type: str = Field(..., description="Vocabulary, Grammar, Conversation, Reading, Writing, Listening")
    difficulty_level: str
    expected_duration_minutes: int
    instructions: str
    content: Dict[str, Any]
    required_resources: List[str] = []


class LearningSession(BaseModel):
    """Schema for a learning session."""
    session_id: str
    learner_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    activities: List[str] = []  # List of activity_ids
    feedback: Optional[Dict[str, Any]] = None
    progress_metrics: Dict[str, Any] = {}


class FeedbackItem(BaseModel):
    """Schema for feedback on a learning activity."""
    activity_id: str
    comprehension_level: int = Field(..., description="1-5 scale where 5 is complete understanding")
    engagement_level: int = Field(..., description="1-5 scale where 5 is highly engaged")
    difficulty_perception: int = Field(..., description="1-5 scale where 5 is too difficult")
    comments: Optional[str] = None


# ========================
# Protocol Interfaces
# ========================

class LanguageModelService(Protocol):
    """Protocol for language model service."""
    
    def create_assistant(self, name: str, instructions: str, tools: List[Dict[str, Any]]) -> Any:
        """Create an assistant with specified configuration."""
        ...
    
    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Create a new conversation thread."""
        ...
    
    def add_message(self, thread_id: str, role: str, content: str) -> Any:
        """Add a message to a thread."""
        ...
    
    def run_assistant(self, thread_id: str, assistant_id: str, 
                     tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Run an assistant on a thread."""
        ...
    
    def get_response(self, thread_id: str, run_id: str) -> str:
        """Get the assistant's response from a run."""
        ...


class LearnerRepository(Protocol):
    """Protocol for learner data repository."""
    
    def get_learner_profile(self, learner_id: str) -> LearnerProfile:
        """Retrieve a learner's profile."""
        ...
    
    def update_learner_profile(self, profile: LearnerProfile) -> None:
        """Update a learner's profile."""
        ...
    
    def save_learning_session(self, session: LearningSession) -> None:
        """Save a learning session."""
        ...
    
    def get_learning_sessions(self, learner_id: str) -> List[LearningSession]:
        """Retrieve a learner's past sessions."""
        ...
    
    def save_feedback(self, learner_id: str, feedback: FeedbackItem) -> None:
        """Save feedback for a learning activity."""
        ...


class ActivityRepository(Protocol):
    """Protocol for activity data repository."""
    
    def create_activity(self, activity: LearningActivity) -> str:
        """Create a new learning activity and return its ID."""
        ...
    
    def get_activity(self, activity_id: str) -> LearningActivity:
        """Retrieve a learning activity."""
        ...
    
    def find_activities(self, criteria: Dict[str, Any]) -> List[LearningActivity]:
        """Find activities matching criteria."""
        ...


# ========================
# Event Definitions
# ========================

class Event(BaseModel):
    """Base event class."""
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str
    payload: Dict[str, Any]


class EventBus(Protocol):
    """Protocol for event bus."""
    
    def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        ...
    
    def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to an event type."""
        ...


# ========================
# Implementation Classes
# ========================

class OpenAILanguageModelService:
    """Implementation of LanguageModelService using OpenAI."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo"):
        """Initialize with API key and model name."""
        self.api_key = api_key
        self.model_name = model_name
        # In real implementation, this would use the actual OpenAI client
        self.client = MagicMock()
    
    def create_assistant(self, name: str, instructions: str, tools: List[Dict[str, Any]]) -> Any:
        """Create an assistant with specified configuration."""
        return self.client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model=self.model_name,
            tools=tools
        )
    
    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Create a new conversation thread."""
        return self.client.beta.threads.create(
            metadata=metadata or {}
        )
    
    def add_message(self, thread_id: str, role: str, content: str) -> Any:
        """Add a message to a thread."""
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role=role,
            content=content
        )
    
    def run_assistant(self, thread_id: str, assistant_id: str, 
                     tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Run an assistant on a thread."""
        return self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            tools=tools or []
        )
    
    def get_response(self, thread_id: str, run_id: str) -> str:
        """Get the assistant's response from a run."""
        # In a real implementation, this would wait for the run to complete and fetch messages
        run = self.client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        
        if run.status == "completed":
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id,
                order="desc",
                limit=1
            )
            return messages.data[0].content[0].text.value
        
        return ""


class Neo4jLearnerRepository:
    """Implementation of LearnerRepository using Neo4j."""
    
    def __init__(self, graph_uri: str, username: str, password: str):
        """Initialize with Neo4j connection details."""
        # In real implementation, this would connect to Neo4j
        self.db = MagicMock()
    
    def get_learner_profile(self, learner_id: str) -> LearnerProfile:
        """Retrieve a learner's profile."""
        # Simulate database query
        if learner_id == "test_learner":
            return LearnerProfile(
                learner_id="test_learner",
                target_language="Spanish",
                native_language="English",
                proficiency_level="Beginner",
                learning_goals=["Travel conversation", "Basic reading"],
                learning_preferences={"visual": True, "interactive": True},
                available_time_weekly=180
            )
        raise ValueError(f"Learner not found: {learner_id}")
    
    def update_learner_profile(self, profile: LearnerProfile) -> None:
        """Update a learner's profile."""
        # Simulate database update
        pass
    
    def save_learning_session(self, session: LearningSession) -> None:
        """Save a learning session."""
        # Simulate database insert
        pass
    
    def get_learning_sessions(self, learner_id: str) -> List[LearningSession]:
        """Retrieve a learner's past sessions."""
        # Simulate database query
        return []
    
    def save_feedback(self, learner_id: str, feedback: FeedbackItem) -> None:
        """Save feedback for a learning activity."""
        # Simulate database insert
        pass


class LanguageTutorService:
    """Main service for the Language Tutor scenario."""
    
    def __init__(
        self,
        language_model_service: LanguageModelService,
        learner_repository: LearnerRepository,
        activity_repository: ActivityRepository,
        event_bus: EventBus
    ):
        """Initialize with required dependencies."""
        self.language_model_service = language_model_service
        self.learner_repository = learner_repository
        self.activity_repository = activity_repository
        self.event_bus = event_bus
        self.assistants = {}
    
    def initialize_language_learning(self, learner_id: str) -> Dict[str, Any]:
        """Initialize a personalized language learning experience for a learner."""
        # Get learner profile
        profile = self.learner_repository.get_learner_profile(learner_id)
        
        # Create language tutor assistant
        tutor_assistant = self.create_language_tutor(profile)
        self.assistants[f"{learner_id}_tutor"] = tutor_assistant
        
        # Create conversation partner assistant
        conversation_assistant = self.create_conversation_partner(profile)
        self.assistants[f"{learner_id}_conversation"] = conversation_assistant
        
        # Generate initial learning plan
        learning_plan = self.generate_learning_plan(profile)
        
        # Publish initialization event
        self.event_bus.publish(Event(
            event_type="learning_initialized",
            source="language_tutor_service",
            payload={
                "learner_id": learner_id,
                "target_language": profile.target_language,
                "proficiency_level": profile.proficiency_level,
                "learning_plan": learning_plan
            }
        ))
        
        return {
            "learner_id": learner_id,
            "tutor_assistant_id": tutor_assistant.id,
            "conversation_assistant_id": conversation_assistant.id,
            "learning_plan": learning_plan
        }
    
    def create_language_tutor(self, profile: LearnerProfile) -> Any:
        """Create a language tutor assistant customized for the learner."""
        instructions = f"""
            You are a {profile.target_language} language tutor for a {profile.proficiency_level} level student 
            whose native language is {profile.native_language}. 
            
            Learning goals: {', '.join(profile.learning_goals)}
            
            Adapt your teaching style to match these preferences: {profile.learning_preferences}
            
            Focus on practical conversation skills first, then gradually introduce grammar concepts as needed.
            Always be encouraging and patient. Provide clear explanations with examples.
            When the student makes a mistake, gently correct them and explain the correction.
        """
        
        # Define tutor tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "assess_pronunciation",
                    "description": "Assess the learner's pronunciation of a phrase",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "phrase": {"type": "string"},
                            "recording_url": {"type": "string"}
                        },
                        "required": ["phrase", "recording_url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_exercise",
                    "description": "Generate a language exercise for the learner",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "exercise_type": {"type": "string", "enum": ["vocabulary", "grammar", "conversation"]},
                            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                            "topic": {"type": "string"}
                        },
                        "required": ["exercise_type", "difficulty", "topic"]
                    }
                }
            }
        ]
        
        return self.language_model_service.create_assistant(
            name=f"{profile.target_language} Tutor for {profile.learner_id}",
            instructions=instructions,
            tools=tools
        )
    
    def create_conversation_partner(self, profile: LearnerProfile) -> Any:
        """Create a conversation partner assistant customized for the learner."""
        instructions = f"""
            You are a conversation partner for a {profile.proficiency_level} level {profile.target_language} learner 
            whose native language is {profile.native_language}.
            
            Simulate natural conversations about: {', '.join(profile.learning_goals)}
            
            Use only vocabulary and grammar appropriate for their level. Speak clearly and naturally.
            If they don't understand, rephrase using simpler language rather than switching to their native language.
            Gently correct critical errors that would interfere with understanding, but don't interrupt the flow
            for minor mistakes.
        """
        
        return self.language_model_service.create_assistant(
            name=f"{profile.target_language} Conversation Partner for {profile.learner_id}",
            instructions=instructions,
            tools=[]
        )
    
    def generate_learning_plan(self, profile: LearnerProfile) -> Dict[str, Any]:
        """Generate a personalized learning plan based on the learner's profile."""
        # Calculate sessions per week based on available time
        session_length = 30  # minutes
        sessions_per_week = profile.available_time_weekly // session_length
        
        # Create a simple learning plan structure
        learning_plan = {
            "weekly_schedule": {
                "sessions_per_week": sessions_per_week,
                "session_length_minutes": session_length
            },
            "initial_focus_areas": [],
            "first_week_activities": []
        }
        
        # Determine focus areas based on goals
        if "Travel conversation" in profile.learning_goals:
            learning_plan["initial_focus_areas"].append({
                "area": "Travel phrases",
                "priority": "High"
            })
        
        if "Basic reading" in profile.learning_goals:
            learning_plan["initial_focus_areas"].append({
                "area": "Common vocabulary",
                "priority": "Medium"
            })
            learning_plan["initial_focus_areas"].append({
                "area": "Simple text reading",
                "priority": "Medium"
            })
        
        # Generate first week activities
        learning_plan["first_week_activities"] = [
            {
                "day": 1,
                "activity_type": "Vocabulary",
                "focus": "Greetings and introductions",
                "duration_minutes": session_length
            },
            {
                "day": 3,
                "activity_type": "Conversation",
                "focus": "Introducing yourself",
                "duration_minutes": session_length
            },
            {
                "day": 5,
                "activity_type": "Reading",
                "focus": "Simple signs and notices",
                "duration_minutes": session_length
            }
        ]
        
        return learning_plan
    
    def start_learning_session(self, learner_id: str, activity_type: str) -> Dict[str, Any]:
        """Start a new learning session for the specified learner."""
        profile = self.learner_repository.get_learner_profile(learner_id)
        
        # Create a new session
        session_id = f"session_{learner_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        session = LearningSession(
            session_id=session_id,
            learner_id=learner_id,
            start_time=datetime.now()
        )
        
        # Create a thread for the session
        thread = self.language_model_service.create_thread(
            metadata={
                "learner_id": learner_id,
                "session_id": session_id,
                "activity_type": activity_type
            }
        )
        
        # Determine which assistant to use
        assistant_id = None
        initial_message = ""
        
        if activity_type == "Conversation":
            assistant_id = self.assistants.get(f"{learner_id}_conversation").id
            initial_message = "Hello! I'm ready to practice conversation with you. What would you like to talk about today?"
        else:
            assistant_id = self.assistants.get(f"{learner_id}_tutor").id
            initial_message = f"Welcome to your {activity_type} session! Let's get started with today's lesson."
        
        # Add initial message from the assistant
        self.language_model_service.add_message(
            thread_id=thread.id,
            role="assistant",
            content=initial_message
        )
        
        # Publish session start event
        self.event_bus.publish(Event(
            event_type="learning_session_started",
            source="language_tutor_service",
            payload={
                "learner_id": learner_id,
                "session_id": session_id,
                "activity_type": activity_type
            }
        ))
        
        # Save the session
        self.learner_repository.save_learning_session(session)
        
        return {
            "session_id": session_id,
            "thread_id": thread.id,
            "assistant_id": assistant_id
        }
    
    def submit_learner_message(self, thread_id: str, assistant_id: str, message: str) -> Dict[str, Any]:
        """Submit a message from the learner and get a response."""
        # Add the learner's message to the thread
        self.language_model_service.add_message(
            thread_id=thread_id,
            role="user",
            content=message
        )
        
        # Run the assistant
        run = self.language_model_service.run_assistant(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        
        # Get the response (in a real implementation, this would handle async properly)
        response = self.language_model_service.get_response(thread_id, run.id)
        
        return {
            "thread_id": thread_id,
            "run_id": run.id,
            "response": response
        }
    
    def submit_session_feedback(self, session_id: str, feedback: Dict[str, Any]) -> None:
        """Submit feedback for a learning session."""
        # Retrieve the session
        sessions = [s for s in self.learner_repository.get_learning_sessions("test_learner") 
                   if s.session_id == session_id]
        
        if not sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = sessions[0]
        
        # Update the session with feedback
        session.feedback = feedback
        session.end_time = datetime.now()
        
        # Save the updated session
        self.learner_repository.save_learning_session(session)
        
        # Publish feedback event
        self.event_bus.publish(Event(
            event_type="learning_feedback_submitted",
            source="language_tutor_service",
            payload={
                "learner_id": session.learner_id,
                "session_id": session_id,
                "feedback": feedback
            }
        ))


# ========================
# Tests
# ========================

class TestLanguageTutorService(unittest.TestCase):
    """Test suite for LanguageTutorService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.language_model_service = MagicMock(spec=LanguageModelService)
        self.learner_repository = MagicMock(spec=LearnerRepository)
        self.activity_repository = MagicMock(spec=ActivityRepository)
        self.event_bus = MagicMock(spec=EventBus)
        
        # Set up mock returns
        self.learner_repository.get_learner_profile.return_value = LearnerProfile(
            learner_id="test_learner",
            target_language="Spanish",
            native_language="English",
            proficiency_level="Beginner",
            learning_goals=["Travel conversation", "Basic reading"],
            learning_preferences={"visual": True, "interactive": True},
            available_time_weekly=180
        )
        
        # Setup mock assistants
        mock_tutor = MagicMock()
        mock_tutor.id = "tutor_assistant_id"
        mock_conversation = MagicMock()
        mock_conversation.id = "conversation_assistant_id"
        
        self.language_model_service.create_assistant.side_effect = [mock_tutor, mock_conversation]
        
        # Setup mock thread
        mock_thread = MagicMock()
        mock_thread.id = "thread_id"
        self.language_model_service.create_thread.return_value = mock_thread
        
        # Setup mock run
        mock_run = MagicMock()
        mock_run.id = "run_id"
        self.language_model_service.run_assistant.return_value = mock_run
        
        # Initialize service
        self.service = LanguageTutorService(
            self.language_model_service,
            self.learner_repository,
            self.activity_repository,
            self.event_bus
        )
    
    def test_initialize_language_learning(self):
        """Test initialization of language learning for a learner."""
        result = self.service.initialize_language_learning("test_learner")
        
        # Verify learner profile was retrieved
        self.learner_repository.get_learner_profile.assert_called_once_with("test_learner")
        
        # Verify assistants were created
        self.assertEqual(self.language_model_service.create_assistant.call_count, 2)
        
        # Verify event was published
        self.event_bus.publish.assert_called_once()
        call_args = self.event_bus.publish.call_args[0][0]
        self.assertEqual(call_args.event_type, "learning_initialized")
        
        # Verify return value
        self.assertEqual(result["learner_id"], "test_learner")
        self.assertEqual(result["tutor_assistant_id"], "tutor_assistant_id")
        self.assertEqual(result["conversation_assistant_id"], "conversation_assistant_id")
        self.assertIn("learning_plan", result)
    
    def test_start_learning_session(self):
        """Test starting a learning session."""
        # First initialize learning to set up assistants
        self.service.initialize_language_learning("test_learner")
        
        # Now test starting a session
        result = self.service.start_learning_session("test_learner", "Conversation")
        
        # Verify thread was created
        self.language_model_service.create_thread.assert_called_once()
        
        # Verify initial message was added
        self.language_model_service.add_message.assert_called_once()
        
        # Verify session was saved
        self.learner_repository.save_learning_session.assert_called_once()
        
        # Verify event was published
        call_args = self.event_bus.publish.call_args[0][0]
        self.assertEqual(call_args.event_type, "learning_session_started")
        
        # Verify return value
        self.assertEqual(result["thread_id"], "thread_id")
        self.assertEqual(result["assistant_id"], "conversation_assistant_id")
        self.assertIn("session_id", result)
    
    def test_submit_learner_message(self):
        """Test submitting a learner message and getting a response."""
        # Setup mock response
        self.language_model_service.get_response.return_value = "¡Hola! ¿Cómo estás?"
        
        result = self.service.submit_learner_message("thread_id", "assistant_id", "Hello")
        
        # Verify message was added
        self.language_model_service.add_message.assert_called_once_with(
            thread_id="thread_id",
            role="user",
            content="Hello"
        )
        
        # Verify assistant was run
        self.language_model_service.run_assistant.assert_called_once_with(
            thread_id="thread_id",
            assistant_id="assistant_id"
        )
        
        # Verify response was retrieved
        self.language_model_service.get_response.assert_called_once()
        
        # Verify return value
        self.assertEqual(result["response"], "¡Hola! ¿Cómo estás?")
        self.assertEqual(result["thread_id"], "thread_id")
        self.assertEqual(result["run_id"], "run_id")


if __name__ == "__main__":
    unittest.main()
