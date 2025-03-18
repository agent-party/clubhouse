"""
Translation capability for processing text content between languages.

This module provides a capability for translating text between languages,
supporting various language pairs and formatting options. It follows the
standardized capability pattern with proper parameter validation, event
triggering, and error handling.
"""

import time
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Literal, Union

from pydantic import BaseModel, Field, field_validator

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError

# Configure logging
logger = logging.getLogger(__name__)


class TranslateParameters(BaseModel):
    """Model for translate parameters validation."""
    
    text: str = Field(..., description="The text content to translate")
    target_language: str = Field(
        ...,
        description="The target language code (e.g., 'es', 'fr', 'de')"
    )
    source_language: str = Field(
        default="auto",
        description="The source language code, or 'auto' for automatic detection"
    )
    preserve_formatting: bool = Field(
        default=False,
        description="Whether to preserve formatting (e.g., HTML) in the translation"
    )
    formality: str = Field(
        default="default",
        description="Level of formality for the translation"
    )
    
    @field_validator('target_language')
    def validate_target_language(cls, v: str) -> str:
        """Validate that target_language is a supported language code."""
        # List of supported language codes (ISO 639-1 codes)
        valid_codes = [
            "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", 
            "ko", "ar", "hi", "bn", "pa", "te", "mr", "ta", "ur", "fa", 
            "tr", "pl", "cs", "sv", "da", "no", "fi", "hu", "el", "he", 
            "th", "vi", "id", "ms", "fil", "uk", "ro", "bg", "hr", "sr", 
            "sk", "sl", "et", "lv", "lt", "ca"
        ]
        
        if v not in valid_codes:
            raise ValueError(f"target_language must be a valid language code: {', '.join(valid_codes)}")
        return v
    
    @field_validator('source_language')
    def validate_source_language(cls, v: str) -> str:
        """Validate that source_language is 'auto' or a supported language code."""
        if v == "auto":
            return v
            
        # List of supported language codes (ISO 639-1 codes)
        valid_codes = [
            "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", 
            "ko", "ar", "hi", "bn", "pa", "te", "mr", "ta", "ur", "fa", 
            "tr", "pl", "cs", "sv", "da", "no", "fi", "hu", "el", "he", 
            "th", "vi", "id", "ms", "fil", "uk", "ro", "bg", "hr", "sr", 
            "sk", "sl", "et", "lv", "lt", "ca"
        ]
        
        if v not in valid_codes:
            raise ValueError(f"source_language must be 'auto' or a valid language code: {', '.join(valid_codes)}")
        return v
    
    @field_validator('formality')
    def validate_formality(cls, v: str) -> str:
        """Validate that formality is one of the supported levels."""
        valid_formality = ["default", "formal", "informal"]
        if v not in valid_formality:
            raise ValueError(f"formality must be one of: {', '.join(valid_formality)}")
        return v


class TranslateCapability(BaseCapability):
    """Capability for translating text between languages."""
    
    name = "translate"
    description = "Translate text between languages with support for multiple language pairs and formatting options"
    parameters_schema = TranslateParameters
    
    # Base cost factors
    _base_cost = 0.005
    _cost_per_character = 0.00015
    
    # Language pair cost multipliers
    _language_pair_multipliers = {
        # Common language pairs have lower costs
        "en-es": 1.0, "es-en": 1.0,
        "en-fr": 1.0, "fr-en": 1.0,
        "en-de": 1.0, "de-en": 1.0,
        
        # Less common pairs have higher costs
        "ja-en": 1.5, "en-ja": 1.5,
        "zh-en": 1.5, "en-zh": 1.5,
        "ru-en": 1.3, "en-ru": 1.3,
        "ar-en": 1.4, "en-ar": 1.4,
        
        # Default multiplier for other pairs
        "default": 1.2
    }
    
    # Map of language names for better user experience
    _language_names = {
        "en": "English", "es": "Spanish", "fr": "French", 
        "de": "German", "it": "Italian", "pt": "Portuguese",
        "nl": "Dutch", "ru": "Russian", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
        "hi": "Hindi", "bn": "Bengali", "pa": "Punjabi",
        "te": "Telugu", "mr": "Marathi", "ta": "Tamil",
        "ur": "Urdu", "fa": "Persian", "tr": "Turkish",
        "pl": "Polish", "cs": "Czech", "sv": "Swedish",
        "da": "Danish", "no": "Norwegian", "fi": "Finnish",
        "hu": "Hungarian", "el": "Greek", "he": "Hebrew",
        "th": "Thai", "vi": "Vietnamese", "id": "Indonesian",
        "ms": "Malay", "fil": "Filipino", "uk": "Ukrainian",
        "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
        "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian",
        "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
        "ca": "Catalan"
    }
    
    def __init__(self) -> None:
        """Initialize the TranslateCapability."""
        super().__init__()
        self._operation_costs: Dict[str, float] = {}
    
    def reset_operation_cost(self) -> None:
        """Reset the operation cost tracking."""
        self._operation_costs = {}
    
    def add_operation_cost(self, operation: str, cost: float) -> None:
        """
        Add a cost for a specific operation.
        
        Args:
            operation: The type of operation
            cost: The cost value to add
        """
        if operation in self._operation_costs:
            self._operation_costs[operation] += cost
        else:
            self._operation_costs[operation] = cost
    
    def get_operation_cost(self) -> Dict[str, float]:
        """
        Get the operation costs.
        
        Returns:
            Dictionary with operation types as keys and costs as values,
            plus a 'total' key with the sum of all costs.
        """
        costs = dict(self._operation_costs)
        costs["total"] = sum(costs.values())
        return costs
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters specification for this capability.
        
        Returns:
            Dictionary mapping parameter names to specifications
        """
        return {
            "text": {
                "type": "string",
                "description": "The text content to translate",
                "required": True
            },
            "target_language": {
                "type": "string",
                "description": "The target language code (e.g., 'es', 'fr', 'de')",
                "required": True
            },
            "source_language": {
                "type": "string",
                "description": "The source language code, or 'auto' for automatic detection",
                "required": False,
                "default": "auto"
            },
            "preserve_formatting": {
                "type": "boolean",
                "description": "Whether to preserve formatting (e.g., HTML) in the translation",
                "required": False,
                "default": False
            },
            "formality": {
                "type": "string",
                "description": "Level of formality for the translation (default, formal, informal)",
                "required": False,
                "default": "default"
            }
        }
    
    def get_version(self) -> str:
        """
        Get the version of the capability.
        
        Returns:
            Version string (e.g., "1.0.0")
        """
        return "1.0.0"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this capability.
        
        For backwards compatibility with older capabilities.
        
        Returns:
            The parameters schema dictionary
        """
        return self.parameters
    
    def validate_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Validate the parameters for the translate capability.
        
        This method uses Pydantic for validation to ensure type safety
        and proper constraint checking. It overrides the base class's
        validate_parameters method to add capability-specific validation.
        
        Args:
            **kwargs: The parameters to validate
            
        Returns:
            Dictionary with validated parameters
            
        Raises:
            ValidationError: If parameters fail validation
        """
        try:
            # Use Pydantic model for validation
            params = TranslateParameters(**kwargs)
            return params.model_dump()
        except Exception as e:
            # Convert any Pydantic validation errors to our ValidationError
            error_msg = f"Parameter validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg, self.name)
    
    def _get_language_pair_multiplier(self, source: str, target: str) -> float:
        """
        Get the cost multiplier for a specific language pair.
        
        Args:
            source: Source language code
            target: Target language code
            
        Returns:
            Cost multiplier as a float
        """
        pair_key = f"{source}-{target}"
        return self._language_pair_multipliers.get(pair_key, self._language_pair_multipliers["default"])
    
    def _detect_language(self, text: str) -> str:
        """
        Simulate language detection for the 'auto' source language option.
        
        In a real implementation, this would call a language detection service.
        For demonstration purposes, it's a simple heuristic based on common words.
        
        Args:
            text: Text to analyze for language detection
            
        Returns:
            Detected language code
        """
        # Simple language detection based on common words
        # This is a very naive implementation for demonstration purposes
        text_lower = text.lower()
        
        # Check for common words in different languages
        if any(word in text_lower for word in ["the", "and", "is", "to", "of"]):
            return "en"
        elif any(word in text_lower for word in ["el", "la", "que", "de", "y"]):
            return "es"
        elif any(word in text_lower for word in ["le", "la", "les", "des", "et"]):
            return "fr"
        elif any(word in text_lower for word in ["der", "die", "das", "und", "in"]):
            return "de"
        
        # Default to English if no match
        return "en"
    
    async def _perform_translation(
        self, 
        text: str, 
        target_language: str,
        source_language: str = "auto",
        preserve_formatting: bool = False,
        formality: str = "default"
    ) -> Dict[str, Any]:
        """
        Perform the actual translation operation.
        
        This method would connect to a translation service in a real implementation.
        For demonstration purposes, it's a placeholder that mocks translation results
        and demonstrates proper cost tracking.
        
        Args:
            text: The text to translate
            target_language: Target language code
            source_language: Source language code or "auto"
            preserve_formatting: Whether to preserve formatting
            formality: Level of formality for translation
            
        Returns:
            Dictionary containing translation results
        """
        # If source language is auto, detect it
        detected_source = source_language
        if source_language == "auto":
            detected_source = self._detect_language(text)
        
        # Skip translation if source and target are the same
        if detected_source == target_language:
            return {
                "translated_text": text,
                "source_language": detected_source,
                "target_language": target_language,
                "detected_language": detected_source == source_language,
                "character_count": len(text)
            }
        
        # Simulate processing time based on text length and language pair complexity
        pair_multiplier = self._get_language_pair_multiplier(detected_source, target_language)
        processing_time = len(text) * 0.001 * pair_multiplier
        await asyncio.sleep(min(0.5, processing_time))  # Cap at 0.5 seconds for testing
        
        # Add to operation costs
        self.add_operation_cost("text_length", len(text) * self._cost_per_character)
        self.add_operation_cost(
            "language_pair", 
            self._base_cost * pair_multiplier
        )
        
        # Generate mock translation
        # In a real implementation, this would call a translation API
        translations = {
            # English to Spanish examples
            "en-es": {
                "Hello": "Hola",
                "world": "mundo",
                "How are you?": "¿Cómo estás?",
                "Thank you": "Gracias",
                "Good morning": "Buenos días",
                "Welcome": "Bienvenido"
            },
            # English to French examples
            "en-fr": {
                "Hello": "Bonjour",
                "world": "monde",
                "How are you?": "Comment allez-vous?",
                "Thank you": "Merci",
                "Good morning": "Bonjour",
                "Welcome": "Bienvenue"
            },
            # English to German examples
            "en-de": {
                "Hello": "Hallo",
                "world": "Welt",
                "How are you?": "Wie geht es dir?",
                "Thank you": "Danke",
                "Good morning": "Guten Morgen",
                "Welcome": "Willkommen"
            }
        }
        
        # Try to use predefined translations if available
        pair_key = f"{detected_source}-{target_language}"
        if pair_key in translations:
            words = text.split()
            translated_words = []
            
            for word in words:
                # Check if we have a translation for this word or phrase
                if word in translations[pair_key]:
                    translated_words.append(translations[pair_key][word])
                else:
                    # Mock a translation by adding a language-specific suffix
                    if target_language == "es":
                        translated_words.append(word + "o")
                    elif target_language == "fr":
                        translated_words.append(word + "e")
                    elif target_language == "de":
                        translated_words.append(word + "en")
                    else:
                        translated_words.append(word)
            
            translated_text = " ".join(translated_words)
        else:
            # If no specific translations available, just mock something
            if target_language == "es":
                translated_text = f"[ES] {text}"
            elif target_language == "fr":
                translated_text = f"[FR] {text}"
            elif target_language == "de":
                translated_text = f"[DE] {text}"
            else:
                translated_text = f"[{target_language.upper()}] {text}"
        
        # Apply formality adjustments (in a real system, this would be part of the API call)
        if formality == "formal" and target_language in ["es", "fr", "de"]:
            if target_language == "es":
                translated_text = translated_text.replace("tú", "usted").replace("tu", "su")
            elif target_language == "fr":
                translated_text = translated_text.replace("tu", "vous")
            elif target_language == "de":
                translated_text = translated_text.replace("du", "Sie")
        
        # Create the result structure
        result = {
            "translated_text": translated_text,
            "source_language": detected_source,
            "source_language_name": self._language_names.get(detected_source, detected_source),
            "target_language": target_language,
            "target_language_name": self._language_names.get(target_language, target_language),
            "detected_language": detected_source == source_language,
            "character_count": len(text),
            "preserve_formatting": preserve_formatting,
            "formality": formality
        }
        
        return result
    
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the translate capability.
        
        For backwards compatibility, this method calls execute_with_lifecycle
        which provides the standardized execution flow.
        
        Args:
            **kwargs: The parameters for the translation operation
            
        Returns:
            CapabilityResult containing the translation results and metadata
        """
        # For backwards compatibility, delegate to execute_with_lifecycle
        result = await self.execute_with_lifecycle(**kwargs)
        
        # Return result in CapabilityResult format
        return CapabilityResult(
            result=result,
            metadata={"cost": self.get_operation_cost()}
        )
    
    async def execute_with_lifecycle(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the capability with parameters and standardized lifecycle events.
        
        This method handles parameter validation and triggers standard before/after events.
        
        Args:
            **kwargs: The parameters for the capability
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Reset cost tracking
            self.reset_operation_cost()
            self.add_operation_cost("base", self._base_cost)
            
            # Validate parameters
            validated_params = self.validate_parameters(**kwargs)
            
            # Trigger before_execution event
            self.trigger_event("before_execution", capability_name=self.name, params=validated_params)
            
            # Extract validated parameters
            text = validated_params["text"]
            target_language = validated_params["target_language"]
            source_language = validated_params.get("source_language", "auto")
            preserve_formatting = validated_params.get("preserve_formatting", False)
            formality = validated_params.get("formality", "default")
            
            # Execute the translation
            operation_start_time = time.time()
            translation_results = await self._perform_translation(
                text=text, 
                target_language=target_language,
                source_language=source_language,
                preserve_formatting=preserve_formatting,
                formality=formality
            )
            execution_time = time.time() - operation_start_time
            
            # Format the result
            result_data = {
                "status": "success",
                "results": translation_results,
                "metadata": {
                    "execution_time": execution_time,
                    "input_length": len(text),
                    "cost": self.get_operation_cost()
                }
            }
            
            # Trigger after_execution event
            self.trigger_event("after_execution", 
                capability_name=self.name, 
                result=result_data,
                execution_time=execution_time
            )
            
            return result_data
        except ValidationError as ve:
            error_message = str(ve)
            logger.error(f"Validation error in {self.name}: {error_message}")
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type="ValidationError")
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": "ValidationError"
            }
        except Exception as e:
            # Handle any other exceptions using the ExecutionError framework
            execution_error = ExecutionError(
                f"Error in {self.name} execution: {str(e)}", 
                self.name
            )
            error_message = str(execution_error)
            logger.error(error_message)
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type=type(execution_error).__name__)
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": type(execution_error).__name__
            }
