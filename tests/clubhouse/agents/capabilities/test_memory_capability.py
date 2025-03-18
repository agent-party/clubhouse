"""
Tests for the MemoryCapability implementation.

This module tests the functionality of the MemoryCapability, verifying that
it correctly handles parameter validation, execution, error handling,
and event triggering during memory operations.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional

from clubhouse.agents.errors import ValidationError, ExecutionError
from clubhouse.agents.capability import CapabilityResult

# This will be implemented later
pytest.importorskip("clubhouse.agents.capabilities.memory_capability")
from clubhouse.agents.capabilities.memory_capability import MemoryCapability, MemoryOperation


class TestMemoryCapability:
    """Test suite for MemoryCapability."""
    
    def test_initialization(self):
        """Test capability initialization."""
        capability = MemoryCapability()
        assert capability.name == "memory"
        assert "memory" in capability.description.lower()
        assert isinstance(capability.parameters, dict)
        assert capability._operation_costs == {}
    
    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        capability = MemoryCapability()
        
        # Test create operation
        create_params = {
            "operation": "create",
            "content": "This is a test memory content.",
            "metadata": {
                "user_id": "user123",
                "importance": "high"
            }
        }
        
        validated = capability.validate_parameters(**create_params)
        assert validated["operation"] == MemoryOperation.CREATE
        assert validated["content"] == create_params["content"]
        assert validated["metadata"] == create_params["metadata"]
        
        # Test retrieve operation
        retrieve_params = {
            "operation": "retrieve",
            "query": "test memory",
            "filters": {
                "user_id": "user123"
            },
            "limit": 5
        }
        
        validated = capability.validate_parameters(**retrieve_params)
        assert validated["operation"] == MemoryOperation.RETRIEVE
        assert validated["query"] == retrieve_params["query"]
        assert validated["filters"] == retrieve_params["filters"]
        assert validated["limit"] == retrieve_params["limit"]
        
        # Test update operation
        update_params = {
            "operation": "update",
            "memory_id": "mem123",
            "content": "Updated memory content",
            "metadata": {
                "importance": "medium"
            }
        }
        
        validated = capability.validate_parameters(**update_params)
        assert validated["operation"] == MemoryOperation.UPDATE
        assert validated["memory_id"] == update_params["memory_id"]
        assert validated["content"] == update_params["content"]
        assert validated["metadata"] == update_params["metadata"]
        
        # Test delete operation
        delete_params = {
            "operation": "delete",
            "memory_id": "mem123"
        }
        
        validated = capability.validate_parameters(**delete_params)
        assert validated["operation"] == MemoryOperation.DELETE
        assert validated["memory_id"] == delete_params["memory_id"]
    
    def test_parameter_validation_failure(self):
        """Test parameter validation with invalid parameters."""
        capability = MemoryCapability()
        
        # Missing required operation
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(content="Test content")
        assert "operation" in str(exc_info.value).lower()
        
        # Invalid operation
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(operation="invalid_op")
        assert "operation" in str(exc_info.value).lower()
        
        # Missing required parameters for create
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(operation="create")
        assert "content" in str(exc_info.value).lower()
        
        # Missing required parameters for retrieve
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(operation="retrieve")
        assert "query" in str(exc_info.value).lower()
        
        # Missing required parameters for update
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(operation="update", content="Updated content")
        assert "memory_id" in str(exc_info.value).lower()
        
        # Missing required parameters for delete
        with pytest.raises(ValidationError) as exc_info:
            capability.validate_parameters(operation="delete")
        assert "memory_id" in str(exc_info.value).lower()
    
    def test_cost_tracking(self):
        """Test operation cost tracking."""
        capability = MemoryCapability()
        
        # Add some costs
        capability.add_operation_cost("base", 0.01)
        capability.add_operation_cost("content_length", 0.05)
        capability.add_operation_cost("storage", 0.02)
        
        # Get the costs
        costs = capability.get_operation_cost()
        assert costs["base"] == 0.01
        assert costs["content_length"] == 0.05
        assert costs["storage"] == 0.02
        assert costs["total"] == 0.08
        
        # Reset the costs
        capability.reset_operation_cost()
        costs = capability.get_operation_cost()
        assert costs["total"] == 0
    
    @pytest.mark.asyncio
    async def test_event_handlers(self):
        """Test event handler registration and triggering."""
        capability = MemoryCapability()
        
        # Mock event handlers
        before_handler = MagicMock()
        after_handler = MagicMock()
        
        # Register handlers
        capability.register_event_handler("before_execution", before_handler)
        capability.register_event_handler("after_execution", after_handler)
        
        # Execute with mocked memory operation function
        with patch.object(capability, '_create_memory', return_value={"id": "mem123", "status": "created"}):
            result = await capability.execute_with_lifecycle(
                operation="create", 
                content="Test memory"
            )
        
        # Check that the handlers were called with correct arguments
        before_handler.assert_called_once()
        after_handler.assert_called_once()
        
        # Verify correct parameters were passed to before_execution
        before_args = before_handler.call_args[1]
        assert before_args["capability_name"] == "memory"
        assert "params" in before_args
        
        # Verify result was passed to after_execution
        after_args = after_handler.call_args[1]
        assert after_args["capability_name"] == "memory"
        assert "result" in after_args
    
    @pytest.mark.asyncio
    async def test_create_memory(self):
        """Test creating a new memory."""
        capability = MemoryCapability()
        
        # Test parameters
        content = "This is a test memory"
        metadata = {"user_id": "user123", "tags": ["test", "memory"]}
        
        # Call the internal method directly
        result = await capability._create_memory(content=content, metadata=metadata)
        
        # Check the result structure
        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert result["content"] == content
        assert "metadata" in result
        assert result["metadata"] == metadata
        assert "created_at" in result
    
    @pytest.mark.asyncio
    async def test_retrieve_memory(self):
        """Test retrieving memories."""
        capability = MemoryCapability()
        
        # Test parameters
        query = "test memory"
        filters = {"user_id": "user123"}
        limit = 5
        
        # Call the internal method directly
        result = await capability._retrieve_memory(
            query=query, 
            filters=filters, 
            limit=limit
        )
        
        # Check the result structure
        assert isinstance(result, dict)
        assert "memories" in result
        assert isinstance(result["memories"], list)
        assert "count" in result
        assert isinstance(result["count"], int)
    
    @pytest.mark.asyncio
    async def test_update_memory(self):
        """Test updating an existing memory."""
        capability = MemoryCapability()
        
        # First create a memory
        create_result = await capability._create_memory(
            content="Original content",
            metadata={"user_id": "user123"}
        )
        
        memory_id = create_result["id"]
        
        # Now update it
        updated_content = "Updated content"
        updated_metadata = {"user_id": "user123", "importance": "high"}
        
        update_result = await capability._update_memory(
            memory_id=memory_id,
            content=updated_content,
            metadata=updated_metadata
        )
        
        # Check the result structure
        assert isinstance(update_result, dict)
        assert "id" in update_result
        assert update_result["id"] == memory_id
        assert "content" in update_result
        assert update_result["content"] == updated_content
        assert "metadata" in update_result
        assert update_result["metadata"] == updated_metadata
        assert "updated_at" in update_result
    
    @pytest.mark.asyncio
    async def test_delete_memory(self):
        """Test deleting a memory."""
        capability = MemoryCapability()
        
        # First create a memory
        create_result = await capability._create_memory(
            content="Memory to delete",
            metadata={"user_id": "user123"}
        )
        
        memory_id = create_result["id"]
        
        # Now delete it
        delete_result = await capability._delete_memory(memory_id=memory_id)
        
        # Check the result structure
        assert isinstance(delete_result, dict)
        assert "id" in delete_result
        assert delete_result["id"] == memory_id
        assert "status" in delete_result
        assert delete_result["status"] == "deleted"
    
    @pytest.mark.asyncio
    async def test_create_memory_execution_success(self):
        """Test successful execution of memory creation."""
        capability = MemoryCapability()
        
        # Execute with actual parameters
        result = await capability.execute_with_lifecycle(
            operation="create",
            content="This is a test memory for execution",
            metadata={"user_id": "user123", "tags": ["test"]}
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        assert "metadata" in result
        
        # Check specific result content
        results = result["results"]
        assert "id" in results
        assert "content" in results
        assert "metadata" in results
        assert results["metadata"]["user_id"] == "user123"
        
        # Check metadata
        metadata = result["metadata"]
        assert "execution_time" in metadata
        assert "operation" in metadata
        assert metadata["operation"] == "create"
    
    @pytest.mark.asyncio
    async def test_retrieve_memory_execution_success(self):
        """Test successful execution of memory retrieval."""
        capability = MemoryCapability()
        
        # First create some memories to retrieve
        await capability.execute_with_lifecycle(
            operation="create",
            content="Memory content 1",
            metadata={"user_id": "user123", "tags": ["test"]}
        )
        
        await capability.execute_with_lifecycle(
            operation="create",
            content="Memory content 2",
            metadata={"user_id": "user123", "tags": ["test"]}
        )
        
        # Now retrieve them
        result = await capability.execute_with_lifecycle(
            operation="retrieve",
            query="memory content",
            filters={"user_id": "user123"},
            limit=5
        )
        
        # Check result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "success"
        assert "results" in result
        
        # Check specific result content
        results = result["results"]
        assert "memories" in results
        assert isinstance(results["memories"], list)
        assert len(results["memories"]) > 0
        assert "count" in results
    
    @pytest.mark.asyncio
    async def test_execution_with_validation_error(self):
        """Test execution with validation error."""
        capability = MemoryCapability()
        
        # Execute with missing required parameter
        result = await capability.execute_with_lifecycle(operation="create")
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "content" in result["error"].lower()
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_execution_with_unexpected_error(self):
        """Test execution with unexpected runtime error."""
        capability = MemoryCapability()
        
        # Mock the create_memory function to raise an exception
        error_message = "Memory service unavailable"
        with patch.object(capability, '_create_memory', side_effect=Exception(error_message)):
            result = await capability.execute_with_lifecycle(
                operation="create",
                content="This should fail during creation"
            )
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert error_message in result["error"]
        assert "error_type" in result
    
    @pytest.mark.asyncio
    async def test_memory_not_found_error(self):
        """Test error handling when a memory is not found."""
        capability = MemoryCapability()
        
        # Try to update a non-existent memory
        non_existent_id = "non_existent_memory_id"
        result = await capability.execute_with_lifecycle(
            operation="update",
            memory_id=non_existent_id,
            content="Updated content"
        )
        
        # Check error result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "error"
        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "error_type" in result
