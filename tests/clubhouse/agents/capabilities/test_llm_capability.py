"""Test for the LLM capability."""
import asyncio
import traceback
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from clubhouse.agents.capabilities.llm_capability import LLMCapability, LLMParameters, LLMProvider


class TestLLMCapability:
    """Test the LLM capability."""

    @pytest.fixture
    def llm_capability(self) -> LLMCapability:
        """Create a LLM capability for testing."""
        return LLMCapability()

    def test_capability_properties(self, llm_capability: LLMCapability):
        """Test the capability properties."""
        assert llm_capability.name == "llm"
        assert llm_capability.description == "Generate text using a Large Language Model"
        assert llm_capability.version == "0.1.0"

    def test_validate_parameters_valid(self, llm_capability: LLMCapability):
        """Test validating parameters with valid input."""
        # Test with basic prompt
        params = {"prompt": "Hello, how are you?"}
        validated = llm_capability.validate_parameters(params)
        assert validated.prompt == "Hello, how are you?"
        assert validated.provider == LLMProvider.ANTHROPIC  # Default provider
        assert validated.model == "claude-3-haiku-20240307"  # Default model
        assert validated.max_tokens == 1000  # Default max tokens
        assert validated.temperature == 0.7  # Default temperature

        # Test with more specific parameters
        params = {
            "prompt": "Write a short story",
            "provider": "openai",
            "model": "gpt-4-turbo",
            "max_tokens": 500,
            "temperature": 0.3,
            "system_prompt": "You are a creative writing assistant",
        }
        validated = llm_capability.validate_parameters(params)
        assert validated.prompt == "Write a short story"
        assert validated.provider == LLMProvider.OPENAI
        assert validated.model == "gpt-4-turbo"
        assert validated.max_tokens == 500
        assert validated.temperature == 0.3
        assert validated.system_prompt == "You are a creative writing assistant"

    def test_validate_parameters_invalid(self, llm_capability: LLMCapability):
        """Test validating parameters with invalid input."""
        # Missing required prompt
        with pytest.raises(ValueError):
            llm_capability.validate_parameters({})
            
        # Invalid provider
        with pytest.raises(ValueError):
            llm_capability.validate_parameters({
                "prompt": "Hello",
                "provider": "invalid_provider"
            })
            
        # Invalid temperature (out of range)
        with pytest.raises(ValueError):
            llm_capability.validate_parameters({
                "prompt": "Hello",
                "temperature": 1.5
            })
            
        # Invalid max_tokens (negative)
        with pytest.raises(ValueError):
            llm_capability.validate_parameters({
                "prompt": "Hello",
                "max_tokens": -100
            })

    @pytest.mark.asyncio
    @patch("clubhouse.agents.capabilities.llm_capability.LLMCapability._call_anthropic")
    async def test_execute_with_anthropic(self, mock_call_anthropic, llm_capability: LLMCapability):
        """Test executing with Anthropic provider."""
        mock_call_anthropic.return_value = "This is a response from Claude."
        
        result = await llm_capability.execute(
            prompt="Tell me about AI",
            provider="anthropic",
            model="claude-3-haiku-20240307"
        )
        
        # Verify the call was made with correct parameters
        mock_call_anthropic.assert_called_once()
        call_args = mock_call_anthropic.call_args[0][0]
        assert call_args.prompt == "Tell me about AI"
        assert call_args.model == "claude-3-haiku-20240307"
        
        # Verify the result
        assert "status" in result.result
        assert result.result["status"] == "success"
        assert "data" in result.result
        assert "response" in result.result["data"]
        assert result.result["data"]["response"] == "This is a response from Claude."

    @pytest.mark.asyncio
    @patch("clubhouse.agents.capabilities.llm_capability.LLMCapability._call_openai")
    async def test_execute_with_openai(self, mock_call_openai, llm_capability: LLMCapability):
        """Test executing with OpenAI provider."""
        mock_call_openai.return_value = "This is a response from GPT."
        
        result = await llm_capability.execute(
            prompt="Tell me about AI",
            provider="openai",
            model="gpt-4-turbo"
        )
        
        # Verify the call was made with correct parameters
        mock_call_openai.assert_called_once()
        call_args = mock_call_openai.call_args[0][0]
        assert call_args.prompt == "Tell me about AI"
        assert call_args.model == "gpt-4-turbo"
        
        # Verify the result
        assert result.result["status"] == "success"
        assert "data" in result.result
        assert "response" in result.result["data"]
        assert result.result["data"]["response"] == "This is a response from GPT."

    @pytest.mark.asyncio
    @patch("clubhouse.agents.capabilities.llm_capability.LLMCapability._call_huggingface")
    async def test_execute_with_huggingface(self, mock_call_huggingface, llm_capability: LLMCapability):
        """Test executing with Hugging Face provider."""
        mock_call_huggingface.return_value = "This is a response from a Hugging Face model."
        
        result = await llm_capability.execute(
            prompt="Tell me about AI",
            provider="huggingface",
            model="mistralai/Mistral-7B-Instruct-v0.2"
        )
        
        # Verify the call was made with correct parameters
        mock_call_huggingface.assert_called_once()
        call_args = mock_call_huggingface.call_args[0][0]
        assert call_args.prompt == "Tell me about AI"
        assert call_args.model == "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Verify the result
        assert result.result["status"] == "success"
        assert "data" in result.result
        assert "response" in result.result["data"]
        assert result.result["data"]["response"] == "This is a response from a Hugging Face model."

    @pytest.mark.asyncio
    async def test_execute_with_invalid_provider(self, llm_capability: LLMCapability):
        """Test executing with an invalid provider (already validated but for coverage)."""
        # This should never happen in practice due to parameter validation,
        # but we test the execution path for completeness
        params = LLMParameters(
            prompt="Test",
            provider="anthropic",  # Will be modified after validation for test
            model="claude-3-haiku-20240307"
        )
        params.provider = "invalid"  # Force an invalid provider after validation
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            await llm_capability._execute(params)

    @pytest.mark.asyncio
    async def test_error_handling(self, llm_capability: LLMCapability):
        """Test error handling during execution."""
        with patch.object(llm_capability, "_call_anthropic", side_effect=Exception("API error")):
            result = await llm_capability.execute(
                prompt="Tell me about AI",
                provider="anthropic"
            )
            
            assert result.result["status"] == "error"
            assert "error" in result.result
            assert "API error" in result.result["error"]

    @pytest.mark.asyncio
    async def test_with_streaming(self, llm_capability: LLMCapability):
        """Test streaming response."""
        with patch.object(llm_capability, "_stream_response", new_callable=AsyncMock) as mock_stream:
            mock_stream.return_value = ["Hello", " world", "!"]
            
            result = await llm_capability.execute(
                prompt="Greet me",
                provider="anthropic",
                stream=True
            )
            
            assert result.result["status"] == "success"
            assert "data" in result.result
            assert "response" in result.result["data"]
            assert result.result["data"]["response"] == "Hello world!"
            assert "stream_finished" in result.result["data"]
            assert result.result["data"]["stream_finished"] is True
            
            # Verify the streaming method was called
            mock_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_integration(self, llm_capability: LLMCapability):
        """Test integration with conversation history."""
        history = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]
        
        with patch.object(llm_capability, "_call_anthropic", return_value="I can help with that!"):
            result = await llm_capability.execute(
                prompt="Can you help me with something?",
                provider="anthropic",
                conversation_history=history
            )
            
            assert result.result["status"] == "success"
            assert result.result["data"]["response"] == "I can help with that!"
