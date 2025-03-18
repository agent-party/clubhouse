"""Tests for the response event schema."""

import uuid
import pytest
import datetime
from pydantic import ValidationError

from clubhouse.schemas.events.response import ResponseEvent, ResponseStatus
from tests.clubhouse.schemas.events.test_utils import create_minimal_event_data, assert_serialization_roundtrip


class TestResponseEvent:
    """Tests for the ResponseEvent class."""
    
    def test_minimal_valid_event(self):
        """Test creating a minimal valid event."""
        event_data = create_minimal_event_data(ResponseEvent)
        command_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Add required fields specific to ResponseEvent
        event_data.update({
            "command_id": command_id,
            "response_id": response_id,
            "session_id": session_id,
            "responder_id": "test_responder",
            "recipient_id": "test_recipient",
            "status": ResponseStatus.SUCCESS.value,
            "result": {"data": {}},
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        })
        
        event = ResponseEvent.model_validate(event_data)
        
        assert str(event.command_id) == command_id
        assert str(event.response_id) == response_id
        assert str(event.session_id) == session_id
        assert event.responder_id == "test_responder"
        assert event.recipient_id == "test_recipient"
        assert event.status == ResponseStatus.SUCCESS
        assert event.token_usage is not None
        assert event.token_usage["prompt_tokens"] == 100
        assert event.token_usage["completion_tokens"] == 50
        assert event.token_usage["total_tokens"] == 150
        
    def test_with_result_data(self):
        """Test creating a response with result data."""
        event_data = create_minimal_event_data(ResponseEvent)
        command_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        event_data.update({
            "command_id": command_id,
            "response_id": response_id,
            "session_id": session_id,
            "responder_id": "search_capability",
            "recipient_id": "user-123",
            "status": ResponseStatus.SUCCESS.value,
            "result": {
                "data": {
                    "search_results": [
                        {"title": "Introduction to Machine Learning", "url": "https://example.com/ml-intro"},
                        {"title": "Advanced ML Techniques", "url": "https://example.com/ml-advanced"}
                    ],
                    "total_matches": 128
                }
            },
            "execution_time_ms": 450,
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 350,
                "total_tokens": 470
            }
        })
        
        event = ResponseEvent.model_validate(event_data)
        
        assert str(event.command_id) == command_id
        assert str(event.response_id) == response_id
        assert str(event.session_id) == session_id
        assert event.responder_id == "search_capability"
        assert event.recipient_id == "user-123"
        assert event.status == ResponseStatus.SUCCESS
        assert event.result["data"]["search_results"][0]["title"] == "Introduction to Machine Learning"
        assert event.result["data"]["total_matches"] == 128
        assert event.execution_time_ms == 450
        assert event.token_usage["prompt_tokens"] == 120
        assert event.token_usage["completion_tokens"] == 350
        assert event.token_usage["total_tokens"] == 470
        
    def test_all_status_types(self):
        """Test creating responses with all status types."""
        event_data = create_minimal_event_data(ResponseEvent)
        command_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Add all required fields for ResponseEvent
        event_data.update({
            "response_id": response_id,
            "command_id": command_id,
            "session_id": session_id,
            "responder_id": "test_responder",
            "recipient_id": "test_recipient",
            "execution_time_ms": 200,
            "token_usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            }
        })
        
        for status in ResponseStatus:
            event_data["status"] = status.value
            if status == ResponseStatus.SUCCESS:
                event_data["result"] = {"data": {}}
            elif status == ResponseStatus.ERROR:
                event_data["error"] = {"code": "TEST_ERROR", "message": "Test error message"}
            event = ResponseEvent.model_validate(event_data)
            assert event.status == status
            
    def test_error_information(self):
        """Test creating a response with error information."""
        event_data = create_minimal_event_data(ResponseEvent)
        response_id = str(uuid.uuid4())
        command_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Add all required fields for ResponseEvent
        event_data.update({
            "response_id": response_id,
            "command_id": command_id,
            "session_id": session_id,
            "responder_id": "error_responder",
            "recipient_id": "error_recipient",
            "execution_time_ms": 150,
            "token_usage": {
                "prompt_tokens": 40,
                "completion_tokens": 10,
                "total_tokens": 50
            },
            "status": ResponseStatus.ERROR.value,
            "error": {
                "code": "capability_error_001",
                "message": "Failed to execute search capability due to invalid parameters"
            }
        })
        
        event = ResponseEvent.model_validate(event_data)
        assert event.status == ResponseStatus.ERROR
        
    def test_serialization_roundtrip(self):
        """Test that events can be serialized to JSON and back."""
        event_data = create_minimal_event_data(ResponseEvent)
        response_id = str(uuid.uuid4())
        command_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Add all required fields for ResponseEvent
        event_data.update({
            "response_id": response_id,
            "command_id": command_id,
            "session_id": session_id,
            "responder_id": "test_responder",
            "recipient_id": "test_recipient",
            "execution_time_ms": 120,
            "token_usage": {
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "total_tokens": 50
            },
            "status": ResponseStatus.SUCCESS.value,
            "result": {"data": {"key": "value"}}
        })
        
        # Test serialization roundtrip with event class and event data
        assert_serialization_roundtrip(ResponseEvent, event_data)

    def test_create_from_dict(self):
        """Test creating an event from a dictionary."""
        command_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        data = {
            "event_id": str(uuid.uuid4()),
            "event_type": "response",
            "event_version": "1.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "producer_id": "test_producer",
            "response_id": response_id,
            "command_id": command_id,
            "session_id": session_id,
            "responder_id": "dict_responder",
            "recipient_id": "dict_recipient",
            "execution_time_ms": 250,
            "token_usage": {
                "prompt_tokens": 60,
                "completion_tokens": 40,
                "total_tokens": 100
            },
            "status": "success",
            "result": {
                "data": {"search_results": ["result1", "result2"]},
                "metadata": {"total_results": 2, "query_time_ms": 150}
            }
        }
        
        event = ResponseEvent.model_validate(data)
        
        assert str(event.command_id) == command_id
        assert event.status == ResponseStatus.SUCCESS
        assert event.result["data"] == {"search_results": ["result1", "result2"]}
        assert event.result["metadata"] == {"total_results": 2, "query_time_ms": 150}
        
    def test_minimal_data_creation(self):
        """Test creating an event from minimal data."""
        data = create_minimal_event_data(ResponseEvent)
        
        # Add required fields specific to ResponseEvent
        data.update({
            "response_id": str(uuid.uuid4()),
            "command_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "responder_id": "minimal_responder",
            "recipient_id": "minimal_recipient",
            "execution_time_ms": 150,
            "token_usage": {
                "prompt_tokens": 25,
                "completion_tokens": 15,
                "total_tokens": 40
            },
            "status": "success",
            "result": {
                "data": {"test": "data"}
            }
        })
        
        event = ResponseEvent.model_validate(data)
        
        assert event.event_id is not None
        assert event.producer_id == "test_producer"
        assert event.command_id is not None
        assert event.status == ResponseStatus.SUCCESS
        assert event.result["data"] == {"test": "data"}
