"""
Unit tests for the SummarizeCapability.

This file demonstrates the testing approach for capabilities in the 
Agent Orchestration Platform, following test-driven development principles.
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime

from agent_orchestration_platform.example_implementation.summarize_capability import (
    SummarizeCapability,
    SummarizeParameters,
    SummarizeResponse,
    SummarizeMode,
    CapabilityError
)

from agent_orchestration_platform.core.protocols import (
    AgentProtocol,
    LLMServiceProtocol,
    ServiceRegistry
)


class TestSummarizeCapability(unittest.TestCase):
    """Test cases for the SummarizeCapability."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mocks
        self.agent = MagicMock(spec=AgentProtocol)
        self.agent.id = "test-agent-id"
        self.agent.emit_event = AsyncMock()
        
        self.llm_service = MagicMock(spec=LLMServiceProtocol)
        self.llm_service.generate = AsyncMock()
        
        self.service_registry = MagicMock(spec=ServiceRegistry)
        self.service_registry.get_service.return_value = self.llm_service
        
        # Create capability instance
        self.capability = SummarizeCapability(
            agent=self.agent,
            service_registry=self.service_registry
        )
        
        # Test data
        self.test_text = "This is a long text that needs to be summarized. It contains multiple sentences " \
                         "and covers several topics including AI, machine learning, and data science. " \
                         "The goal is to extract the most important information and present it concisely."
        
        # Default parameters
        self.default_params = SummarizeParameters(
            text=self.test_text
        )
        
        # Mock LLM response
        self.llm_service.generate.return_value = "This is a summary of the text about AI and ML."

    def test_init_missing_llm_service(self):
        """Test initialization with missing LLM service."""
        # Setup service registry to return None for LLM service
        service_registry = MagicMock(spec=ServiceRegistry)
        service_registry.get_service.return_value = None
        
        # Assert that initialization raises CapabilityError
        with self.assertRaises(CapabilityError):
            SummarizeCapability(
                agent=self.agent,
                service_registry=service_registry
            )

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        # Test empty text
        with self.assertRaises(ValueError):
            SummarizeParameters(text="")
        
        # Test whitespace-only text
        with self.assertRaises(ValueError):
            SummarizeParameters(text="   ")
        
        # Test negative max_length
        with self.assertRaises(ValueError):
            SummarizeParameters(text=self.test_text, max_length=-100)

    async def test_execute_success(self):
        """Test successful execution of summarize capability."""
        # Execute capability with default parameters
        response = await self.capability.execute_with_lifecycle(
            self.default_params.dict()
        )
        
        # Verify expected events were emitted
        self.agent.emit_event.assert_any_call("capability_started", {"capability": "summarize"})
        self.agent.emit_event.assert_any_call("summarize_started", {"text_length": len(self.test_text)})
        self.agent.emit_event.assert_any_call("summarize_completed", {"summary_length": 45})
        self.agent.emit_event.assert_any_call("capability_completed", {"capability": "summarize"})
        
        # Verify LLM was called with correct parameters
        self.llm_service.generate.assert_called_once()
        call_args = self.llm_service.generate.call_args[1]
        self.assertEqual(call_args["user_prompt"], self.test_text)
        self.assertEqual(call_args["max_tokens"], 200)  # Default for concise mode
        
        # Verify response structure and content
        self.assertIsInstance(response, dict)
        self.assertEqual(response["summary"], "This is a summary of the text about AI and ML.")
        self.assertEqual(response["original_length"], len(self.test_text))
        self.assertEqual(response["summary_length"], 45)  # Length of mock response
        self.assertIn("metadata", response)
        self.assertEqual(response["metadata"]["capability"], "summarize")

    async def test_execute_with_max_length(self):
        """Test execution with max_length parameter."""
        # Create parameters with max_length
        params = SummarizeParameters(
            text=self.test_text,
            max_length=100
        )
        
        # Execute capability
        await self.capability.execute_with_lifecycle(params.dict())
        
        # Verify LLM was called with correct max_tokens
        call_args = self.llm_service.generate.call_args[1]
        self.assertEqual(call_args["max_tokens"], 35)  # 100/4 + 10 = 35
        
        # Verify system prompt includes length constraint
        self.assertIn("no longer than 100 characters", call_args["system_prompt"])

    async def test_execute_with_focus_areas(self):
        """Test execution with focus_areas parameter."""
        # Create parameters with focus_areas
        params = SummarizeParameters(
            text=self.test_text,
            focus_areas=["AI", "machine learning"]
        )
        
        # Execute capability
        response = await self.capability.execute_with_lifecycle(params.dict())
        
        # Verify system prompt includes focus areas
        call_args = self.llm_service.generate.call_args[1]
        self.assertIn("Focus particularly on these aspects: AI, machine learning", 
                     call_args["system_prompt"])
        
        # Verify focus_coverage in response
        self.assertIn("focus_coverage", response)
        self.assertIsInstance(response["focus_coverage"], dict)
        self.assertIn("AI", response["focus_coverage"])
        self.assertIn("machine learning", response["focus_coverage"])

    async def test_execute_with_different_modes(self):
        """Test execution with different summary modes."""
        for mode in SummarizeMode:
            # Create parameters with specific mode
            params = SummarizeParameters(
                text=self.test_text,
                mode=mode
            )
            
            # Reset mocks
            self.llm_service.generate.reset_mock()
            
            # Execute capability
            await self.capability.execute_with_lifecycle(params.dict())
            
            # Verify system prompt based on mode
            call_args = self.llm_service.generate.call_args[1]
            if mode == SummarizeMode.CONCISE:
                self.assertIn("brief, concise summary", call_args["system_prompt"])
            elif mode == SummarizeMode.DETAILED:
                self.assertIn("detailed summary", call_args["system_prompt"])
            elif mode == SummarizeMode.BULLET_POINTS:
                self.assertIn("bullet-point summary", call_args["system_prompt"])
            elif mode == SummarizeMode.EXECUTIVE:
                self.assertIn("executive summary", call_args["system_prompt"])

    async def test_execute_with_analytics(self):
        """Test execution with analytics enabled."""
        # Create parameters with analytics
        params = SummarizeParameters(
            text=self.test_text,
            include_analytics=True
        )
        
        # Execute capability
        response = await self.capability.execute_with_lifecycle(params.dict())
        
        # Verify analytics in response
        self.assertIn("analytics", response)
        self.assertIsInstance(response["analytics"], dict)
        self.assertIn("word_count", response["analytics"])
        self.assertIn("sentence_count", response["analytics"])
        self.assertIn("avg_words_per_sentence", response["analytics"])

    async def test_llm_service_error(self):
        """Test handling of LLM service errors."""
        # Setup LLM service to raise exception
        self.llm_service.generate.side_effect = Exception("LLM service error")
        
        # Assert that execute raises CapabilityError
        with self.assertRaises(CapabilityError):
            await self.capability.execute_with_lifecycle(self.default_params.dict())
        
        # Verify correct events were emitted
        self.agent.emit_event.assert_any_call("capability_started", {"capability": "summarize"})
        self.agent.emit_event.assert_any_call("summarize_started", {"text_length": len(self.test_text)})
        self.agent.emit_event.assert_any_call("capability_error", {
            "capability": "summarize",
            "error": "Summarization failed: LLM service error"
        })


if __name__ == "__main__":
    # Create event loop and run tests
    loop = asyncio.get_event_loop()
    unittest.main()
