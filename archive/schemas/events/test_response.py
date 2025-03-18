"""Tests for the response event schema."""

import pytest
from uuid import uuid4, UUID
from pydantic import ValidationError

from clubhouse.schemas.events.response import ResponseEvent, ResponseStatus
from tests.schemas.events.test_utils import assert_serialization_roundtrip, create_minimal_event_data


class TestResponseEvent:
    """Tests for the ResponseEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(ResponseEvent)
        response_id = uuid4()
        command_id = uuid4()
        session_id = uuid4()
        # Add required fields specific to ResponseEvent
        event_data.update({
            "response_id": str(response_id),
            "command_id": str(command_id),
            "session_id": str(session_id),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value
        })
        
        event = ResponseEvent.model_validate(event_data)
        
        assert event.response_id == response_id
        assert event.command_id == command_id
        assert event.session_id == session_id
        assert event.responder_id == "agent-456"
        assert event.recipient_id == "user-123"
        assert event.status == ResponseStatus.SUCCESS
        assert event.event_type == "response"
        assert event.result == {}  # Default empty dict
        
    def test_with_result_data(self):
        """Test creating a response with result data."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value,
            "result": {
                "search_results": [
                    {"title": "Introduction to ML", "url": "https://example.com/ml-intro"},
                    {"title": "Advanced ML", "url": "https://example.com/ml-advanced"}
                ],
                "total_matches": 128
            }
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert event.result == {
            "search_results": [
                {"title": "Introduction to ML", "url": "https://example.com/ml-intro"},
                {"title": "Advanced ML", "url": "https://example.com/ml-advanced"}
            ],
            "total_matches": 128
        }
        
    def test_all_status_types(self):
        """Test creating responses with all status types."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123"
        })
        
        for status in ResponseStatus:
            event_data["status"] = status.value
            event = ResponseEvent.model_validate(event_data)
            assert event.status == status
            
    def test_error_information(self):
        """Test creating a response with error information."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.ERROR.value,
            "error_code": "capability_error_001",
            "error_message": "Failed to execute search capability due to invalid parameters"
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert event.status == ResponseStatus.ERROR
        assert event.error_code == "capability_error_001"
        assert event.error_message == "Failed to execute search capability due to invalid parameters"
        
    def test_streaming_and_chunking(self):
        """Test creating streaming and chunked responses."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value,
            "is_streaming": True,
            "is_final": False,
            "sequence_number": 1,
            "total_chunks": 3
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert event.is_streaming is True
        assert event.is_final is False
        assert event.sequence_number == 1
        assert event.total_chunks == 3
        
    def test_performance_metrics(self):
        """Test creating a response with performance metrics."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value,
            "execution_time_ms": 450,
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 350,
                "total_tokens": 470
            }
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert event.execution_time_ms == 450
        assert event.token_usage == {
            "prompt_tokens": 120,
            "completion_tokens": 350,
            "total_tokens": 470
        }
        
    def test_additional_metadata(self):
        """Test creating a response with additional metadata."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value,
            "metadata": {
                "search_engine": "semantic_search_v2",
                "query_processing_time_ms": 50,
                "data_sources": ["knowledge_base", "web"]
            }
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert event.metadata == {
            "search_engine": "semantic_search_v2",
            "query_processing_time_ms": 50,
            "data_sources": ["knowledge_base", "web"]
        }
        
    def test_serialization_roundtrip(self):
        """Test that events can be serialized to JSON and back."""
        event_data = create_minimal_event_data(ResponseEvent)
        event_data.update({
            "response_id": str(uuid4()),
            "command_id": str(uuid4()),
            "session_id": str(uuid4()),
            "responder_id": "agent-456",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value,
            "result": {"key": "value"}
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert_serialization_roundtrip(event)
