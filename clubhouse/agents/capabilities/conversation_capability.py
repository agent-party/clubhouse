"""
Conversation capability for managing multi-turn interactions.

This module provides a capability for managing conversational context, history,
and state tracking for multi-turn interactions between agents and users.
It handles message history, context persistence, and reference resolution.
"""

import logging
import asyncio
import time
import traceback
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, cast
from datetime import datetime, timezone
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError

# Set up logging
logger = logging.getLogger(__name__)

class ConversationMessage(BaseModel):
    """Model for a message in a conversation."""
    message_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for this message")
    message: str = Field(..., description="The content of the message")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="When the message was created")
    sender: Optional[str] = Field(None, description="Identifier of the message sender (user or agent)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information for this message")

class ConversationContext(BaseModel):
    """Model for conversation context."""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    topic: Optional[str] = Field(None, description="The main topic of conversation")
    participants: List[str] = Field(default_factory=list, description="Participants in the conversation")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional context attributes")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="When the conversation was created")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="When the conversation was last updated")
    message_count: int = Field(0, description="Number of messages in the conversation")
    message_history: List[Dict[str, Any]] = Field(default_factory=list, description="Recent message history")
    reference_message_id: Optional[str] = Field(None, description="ID of a referenced message")
    
    def __getattr__(self, name: str) -> Any:
        """
        Provide dynamic access to attributes stored in the attributes dictionary.
        
        This allows accessing attributes like context.language instead of context.attributes['language']
        
        Args:
            name: The attribute name to access
            
        Returns:
            The attribute value if found in attributes
            
        Raises:
            AttributeError: If the attribute is not found in the model or attributes
        """
        # Check if the attribute exists in the attributes dictionary
        if name in self.attributes:
            return self.attributes[name]
            
        # If not found, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Support dynamic setting of attributes in the attributes dictionary.
        
        This allows setting attributes like context.language = 'en' instead of 
        context.attributes['language'] = 'en'
        
        Args:
            name: The attribute name to set
            value: The value to set
        """
        # First check if this is a model field
        if name in self.model_fields:
            super().__setattr__(name, value)
        else:
            # For non-model fields, store in attributes
            if hasattr(self, 'attributes'):
                self.attributes[name] = value
            else:
                super().__setattr__(name, value)

class ConversationParameters(BaseModel):
    """Parameters for the conversation capability."""
    message: str = Field(..., description="Message to add to the conversation")
    conversation_id: str = Field(..., description="Identifier for the conversation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the message")
    context: Optional[Dict[str, Any]] = Field(None, description="Contextual information for this message")
    resolve_references: bool = Field(False, description="Whether to resolve references to previous messages")

class ConversationCapability(BaseCapability):
    """
    Capability for managing conversation context and history.
    
    This capability maintains conversation state across multiple turns of dialogue,
    tracks message history, and manages context for natural conversations.
    """
    
    parameters_schema = ConversationParameters
    
    def __init__(self) -> None:
        """Initialize the ConversationCapability."""
        super().__init__()
        # Store conversations by ID
        self._conversations: Dict[str, ConversationContext] = {}
        # Store messages by conversation ID
        self._messages: Dict[str, List[ConversationMessage]] = {}
        # For testing error scenarios
        self._force_error = False
        
    @property
    def name(self) -> str:
        """
        Get the unique identifier for this capability.
        
        Returns:
            The capability name as a string
        """
        return "conversation"
    
    @property
    def description(self) -> str:
        """
        Get a human-readable description of what this capability does.
        
        Returns:
            Description string
        """
        return "Manage conversations with context tracking and history management for multi-turn interactions"
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters specification for this capability.
        
        Returns:
            Dictionary mapping parameter names to specifications
        """
        # Generate parameter descriptions from the Pydantic model
        schema = ConversationParameters.model_json_schema()
        parameters = {}
        
        for name, field_schema in schema.get("properties", {}).items():
            parameters[name] = {
                "description": field_schema.get("description", ""),
                "type": field_schema.get("type", "string"),
                "required": name in schema.get("required", [])
            }
            
        return parameters
        
    def validate_parameters(self, **parameters: Any) -> Dict[str, Any]:
        """
        Validate and convert raw parameters for the conversation capability.
        
        Args:
            **parameters: Raw parameters to validate
            
        Returns:
            Dict[str, Any]: Dictionary of validated parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # First, check for required parameters directly before Pydantic validation
        if "message" not in parameters:
            raise ValidationError("Missing required parameter: 'message'", self.name)
            
        if "conversation_id" not in parameters:
            raise ValidationError("Missing required parameter: 'conversation_id'", self.name)
            
        # Check types directly for critical parameters
        if not isinstance(parameters.get("conversation_id"), str):
            raise ValidationError("Parameter 'conversation_id' must be a string", self.name)
            
        if not isinstance(parameters.get("message"), str):
            raise ValidationError("Parameter 'message' must be a string", self.name)
        
        # Check optional parameters if provided
        if "metadata" in parameters and parameters["metadata"] is not None:
            if not isinstance(parameters["metadata"], dict):
                raise ValidationError("Parameter 'metadata' must be a dictionary", self.name)
                
        if "context" in parameters and parameters["context"] is not None:
            if not isinstance(parameters["context"], dict):
                raise ValidationError("Parameter 'context' must be a dictionary", self.name)
                
        try:
            # Use Pydantic's built-in validation for additional checks
            validated = ConversationParameters(**parameters)
            # Return as a dictionary
            return validated.model_dump()
        except PydanticValidationError as e:
            # Convert Pydantic validation error to our ValidationError format
            raise ValidationError(f"Invalid parameters for conversation capability: {str(e)}", self.name)
    
    def version(self) -> str:
        """
        Get the version of this capability.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        return "1.0.0"
    
    def _initialize_conversation(self, conversation_id: str, context: Optional[Dict[str, Any]] = None) -> ConversationContext:
        """
        Initialize a new conversation or get an existing one.
        
        Args:
            conversation_id: The unique identifier for the conversation
            context: Optional initial context for the conversation
            
        Returns:
            The conversation context object
        """
        if conversation_id in self._conversations:
            # Return existing conversation
            return self._conversations[conversation_id]
            
        # Create a new conversation context
        new_context = {
            "conversation_id": conversation_id,
            "message_history": [],
            **(context or {})
        }
        
        conversation = ConversationContext(**new_context)
        self._conversations[conversation_id] = conversation
        self._messages[conversation_id] = []
        
        return conversation
    
    def _add_message_to_conversation(
        self, 
        conversation_id: str, 
        message: str, 
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        resolve_references: bool = False
    ) -> ConversationMessage:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: The conversation identifier
            message: The message content
            metadata: Optional metadata for the message
            context: Optional context for the message
            resolve_references: Whether to resolve references to previous messages
            
        Returns:
            The created message object
        """
        # Get or create the conversation
        conversation = self._initialize_conversation(conversation_id, context)
        
        # Initialize metadata if None
        if metadata is None:
            metadata = {}
            
        # Resolve references if requested
        if resolve_references and len(self._messages.get(conversation_id, [])) > 0:
            resolved_data = self._resolve_references_in_message(conversation_id, message)
            if resolved_data:
                metadata.update(resolved_data)
                
                # Update query chain in conversation context
                if "query_chain" not in conversation.attributes:
                    conversation.attributes["query_chain"] = []
                
                # Add current query to the chain
                query_data = {
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "has_references": True,
                    "referenced_entities": resolved_data.get("referenced_entity", "")
                }
                conversation.attributes["query_chain"].append(query_data)
        
        # Create a new message
        message_data = {
            "message_id": str(uuid4()),
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "context": context or {}
        }
        
        message_obj = ConversationMessage(**message_data)
        
        # Add to messages store
        if conversation_id not in self._messages:
            self._messages[conversation_id] = []
            
        self._messages[conversation_id].append(message_obj)
        
        # Update conversation context
        conversation.message_count += 1
        
        # Update message history in context (keep last 5 messages)
        message_history = self._messages[conversation_id][-5:]
        conversation.message_history = [msg.model_dump() for msg in message_history]
        
        # Update conversation last updated time
        conversation.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Merge context updates if provided
        if context:
            for key, value in context.items():
                # Skip special fields that shouldn't be merged as general context
                if key not in ["reference_message_id"]:
                    # Store custom attributes in the attributes dictionary instead of directly on the model
                    if key in ["conversation_id", "topic", "participants", "attributes", 
                             "created_at", "updated_at", "message_count", "message_history"]:
                        setattr(conversation, key, value)
                    else:
                        # Store all other attributes in the attributes dictionary
                        conversation.attributes[key] = value
                else:
                    # Store reference message ID in the conversation
                    conversation.reference_message_id = value
        
        # Save conversation updates
        self._conversations[conversation_id] = conversation
                
        return message_obj
        
    def _resolve_references_in_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """
        Analyze a message to resolve references to previous messages or entities.
        
        Args:
            conversation_id: The conversation identifier
            message: The message content to analyze
            
        Returns:
            Dictionary with resolved reference data or empty dict if no references found
        """
        # Check if we have previous messages for this conversation
        messages = self._messages.get(conversation_id, [])
        if not messages:
            return {}
            
        # Get the previous message
        previous_message = messages[-1]
        
        # Dictionary of common reference words and patterns
        reference_indicators = [
            "it", "its", "that", "this", "those", "these", "they", "them", 
            "the above", "previous", "mentioned", "earlier"
        ]
        
        # Check if message contains any reference indicators
        has_reference = any(indicator in message.lower() for indicator in reference_indicators)
        
        if not has_reference:
            return {}
            
        # Extract potential entities from previous message
        # For this implementation, we'll use a simple approach
        # In a real system, this would use NLP for entity extraction
        previous_content = previous_message.message
        potential_entities = self._extract_entities(previous_content)
        
        # Create the result with reference metadata
        result = {
            "reference_resolved": True,
            "previous_query_reference": True,
            "referenced_message_id": previous_message.message_id,
            "referenced_entity": ", ".join(potential_entities)
        }
        
        return result
        
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract potential entities from text.
        This is a simple implementation that looks for capitalized words.
        A more advanced implementation would use NLP techniques.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of potential entities
        """
        # Split text into words
        words = text.split()
        
        # Look for capitalized words that might be entities
        entities = []
        for word in words:
            # Skip punctuation and short words
            clean_word = word.strip('.,;:!?()"\'')
            if len(clean_word) > 2 and clean_word[0].isupper():
                entities.append(clean_word)
                
        # If no capitalized words found, extract nouns based on common patterns
        if not entities:
            # Simple pattern: words after "about", "regarding", "concerning"
            for pattern in ["about ", "regarding ", "concerning ", "on ", "information about "]:
                if pattern in text.lower():
                    index = text.lower().find(pattern) + len(pattern)
                    remaining = text[index:].split()
                    if remaining:
                        # Take the next couple of words as an entity
                        entity = " ".join(remaining[:min(3, len(remaining))])
                        entities.append(entity.strip('.,;:!?()"\' '))
                        
        return entities

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the message history for a conversation.
        
        Args:
            conversation_id: The conversation identifier
            
        Returns:
            List of messages in the conversation
        """
        if conversation_id not in self._messages:
            return []
            
        return [msg.model_dump() for msg in self._messages[conversation_id]]
    
    def get_conversation_context(self, conversation_id: str) -> ConversationContext:
        """
        Get the conversation context for a conversation.
        
        Args:
            conversation_id: The conversation identifier
            
        Returns:
            The conversation context
        """
        # Get or create the conversation
        if conversation_id not in self._conversations:
            return self._initialize_conversation(conversation_id)
            
        return self._conversations[conversation_id]
    
    def get_message_by_id(self, conversation_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific message by its ID.
        
        Args:
            conversation_id: The conversation identifier
            message_id: The message identifier
            
        Returns:
            The message if found, None otherwise
        """
        if conversation_id not in self._messages:
            return None
            
        for message in self._messages[conversation_id]:
            if message.message_id == message_id:
                return message.model_dump()
                
        return None
    
    async def execute(self, **kwargs) -> CapabilityResult:
        """
        Execute the conversation capability with the provided parameters.
        
        Args:
            **kwargs: Parameters for the conversation capability
            
        Returns:
            CapabilityResult: Result of the execution with conversation data
            
        Raises:
            ValidationError: If the parameters are invalid
            ExecutionError: If there is an error during execution
        """
        start_time = time.time()
        
        # Validate parameters (will raise ValidationError if invalid)
        parameters = self.validate_parameters(**kwargs)
        
        try:
            # Trigger starting event
            self.trigger_event("before_execution", capability_name=self.name, parameters=parameters)
            self.trigger_event(f"{self.name}.started", parameters=parameters)
            
            # Execute the core logic
            message_added = await self._process_message(parameters)
            
            # Get the updated conversation context
            conversation = self.get_conversation_context(parameters["conversation_id"])
            
            # Prepare the result
            result = {
                "status": "success",
                "data": {
                    "message_id": message_added.message_id,
                    "message": message_added.message,
                    "conversation_id": parameters["conversation_id"],
                    "conversation": conversation.model_dump(),
                    "messages_count": len(self._messages.get(parameters["conversation_id"], []))
                }
            }
            
            # Trigger completion events
            execution_time = time.time() - start_time
            self.trigger_event(f"{self.name}.completed", result=result, execution_time=execution_time)
            self.trigger_event("after_execution", capability_name=self.name, result=result, execution_time=execution_time)
            
            return CapabilityResult(
                result=result,
                metadata={
                    "execution_time": execution_time,
                    "conversation_id": parameters["conversation_id"]
                }
            )
            
        except Exception as e:
            # Log the full exception with traceback
            logger.error(f"Error executing conversation capability: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Convert generic exceptions to ExecutionError for consistent handling
            if not isinstance(e, ExecutionError):
                execution_error = ExecutionError(f"Error in conversation execution: {str(e)}", self.name)
            else:
                execution_error = e
                
            error_result = {
                "status": "error",
                "error": str(execution_error),
                "error_type": "execution_error"
            }
            
            self.trigger_event("error", capability_name=self.name, error=str(execution_error), error_type="execution_error")
            
            return CapabilityResult(
                result=error_result,
                metadata={
                    "execution_time": time.time() - start_time,
                    "error": str(execution_error)
                }
            )
            
    async def _process_message(self, parameters: Dict[str, Any]) -> ConversationMessage:
        """
        Process a message and add it to a conversation.
        
        This is a helper method to handle the core logic of the execute method.
        
        Args:
            parameters: Validated parameters for message processing
            
        Returns:
            ConversationMessage: The added message
            
        Raises:
            ExecutionError: If there is an error processing the message
        """
        try:
            # For testing purposes
            if hasattr(self, "_force_error") and self._force_error:
                raise ExecutionError("Forced error for testing", self.name)
                
            # Add cost tracking
            message_length = len(parameters["message"])
            self.record_operation_cost("message_processing", 0.001 * message_length)
            self.record_operation_cost("context_management", 0.002)
            
            # Trigger conversation started event for new conversations
            is_new_conversation = parameters["conversation_id"] not in self._conversations
            if is_new_conversation:
                self.trigger_event("conversation.started", 
                    conversation_id=parameters["conversation_id"],
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
            # Add the message to the conversation
            message_obj = self._add_message_to_conversation(
                parameters["conversation_id"], 
                parameters["message"], 
                parameters.get("metadata"), 
                parameters.get("context"),
                parameters.get("resolve_references", False)
            )
            
            # Trigger message added event
            self.trigger_event("conversation.message_added",
                conversation_id=parameters["conversation_id"],
                message_id=message_obj.message_id,
                timestamp=message_obj.timestamp
            )
            
            return message_obj
            
        except Exception as e:
            # Convert any exceptions during processing to ExecutionError
            if not isinstance(e, ExecutionError):
                raise ExecutionError(f"Error processing message: {str(e)}", self.name) from e
            raise
