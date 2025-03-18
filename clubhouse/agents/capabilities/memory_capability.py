"""
Memory capability for managing agent memories.

This module provides a capability for performing memory operations like
creating, retrieving, updating, and deleting memories. It follows the
standardized capability pattern with proper parameter validation,
event triggering, and error handling.
"""

import time
import uuid
import logging
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Literal, Union, Set

from pydantic import BaseModel, Field, field_validator, model_validator

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError

# Configure logging
logger = logging.getLogger(__name__)


class MemoryOperation(str, Enum):
    """Enum for memory operations."""
    CREATE = "create"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"


class MemoryParameters(BaseModel):
    """Base model for memory parameters."""
    operation: str = Field(
        ..., 
        description="Type of memory operation to perform",
        json_schema_extra={"enum": [op.value for op in MemoryOperation]}
    )
    content: Optional[str] = Field(
        default=None,
        description="Content of the memory (for create/update operations)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata for the memory (for create/update operations)"
    )
    memory_id: Optional[str] = Field(
        default=None,
        description="ID of the memory (for update/delete operations)"
    )
    query: Optional[str] = Field(
        default=None,
        description="Query to search for relevant memories (for retrieve operation)"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters to apply to memory search (for retrieve operation)"
    )
    limit: Optional[int] = Field(
        default=10,
        description="Maximum number of memories to retrieve (for retrieve operation)"
    )
    
    @model_validator(mode='after')
    def validate_operation_parameters(self) -> 'MemoryParameters':
        """Validate parameters based on the operation type."""
        operation = self.operation
        
        if operation == MemoryOperation.CREATE.value:
            if not self.content:
                raise ValueError("content is required for create operation")
        elif operation == MemoryOperation.RETRIEVE.value:
            if not self.query:
                raise ValueError("query is required for retrieve operation")
        elif operation == MemoryOperation.UPDATE.value:
            if not self.memory_id:
                raise ValueError("memory_id is required for update operation")
            if self.content is None and self.metadata is None:
                raise ValueError("Either content or metadata must be provided for update operation")
        elif operation == MemoryOperation.DELETE.value:
            if not self.memory_id:
                raise ValueError("memory_id is required for delete operation")
        
        return self


class BaseMemoryParameters(BaseModel):
    """Base model for memory parameters."""
    operation: MemoryOperation = Field(
        ...,
        description="Type of memory operation to perform"
    )


class CreateMemoryParameters(BaseMemoryParameters):
    """Model for create memory parameters."""
    operation: Literal[MemoryOperation.CREATE]
    content: str = Field(
        ...,
        description="Content of the memory to create"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for the memory"
    )
    

class RetrieveMemoryParameters(BaseMemoryParameters):
    """Model for retrieve memory parameters."""
    operation: Literal[MemoryOperation.RETRIEVE]
    query: str = Field(
        ...,
        description="Query to search for relevant memories"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters to apply to the search"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of memories to retrieve"
    )
    

class UpdateMemoryParameters(BaseMemoryParameters):
    """Model for update memory parameters."""
    operation: Literal[MemoryOperation.UPDATE]
    memory_id: str = Field(
        ...,
        description="ID of the memory to update"
    )
    content: Optional[str] = Field(
        default=None,
        description="New content for the memory"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="New metadata for the memory"
    )
    
    @model_validator(mode='after')
    def validate_update_fields(self) -> 'UpdateMemoryParameters':
        """Ensure at least one update field is provided."""
        if self.content is None and self.metadata is None:
            raise ValueError("Either content or metadata must be provided for update operation")
        return self


class DeleteMemoryParameters(BaseMemoryParameters):
    """Model for delete memory parameters."""
    operation: Literal[MemoryOperation.DELETE]
    memory_id: str = Field(
        ...,
        description="ID of the memory to delete"
    )


class MemoryCapability(BaseCapability):
    """Capability for managing agent memories."""
    
    name = "memory"
    description = "Memory capability for managing agent memories including creation, retrieval, update, and deletion"
    
    # Define parameters_schema for compliance with capability standards
    parameters_schema = MemoryParameters
    
    # Base cost factors
    _base_cost = 0.01
    _cost_per_character = 0.0002
    _storage_cost = 0.005
    _retrieval_cost = 0.01
    
    def __init__(self) -> None:
        """Initialize the MemoryCapability."""
        super().__init__()
        self._operation_costs: Dict[str, float] = {}
        self._memory_store: Dict[str, Dict[str, Any]] = {}  # In-memory store for testing
    
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
            "operation": {
                "type": "string",
                "description": "Type of memory operation to perform (create, retrieve, update, delete)",
                "required": True,
                "enum": [op.value for op in MemoryOperation]
            },
            "content": {
                "type": "string",
                "description": "Content of the memory (for create/update operations)",
                "required": False
            },
            "metadata": {
                "type": "object",
                "description": "Metadata for the memory (for create/update operations)",
                "required": False
            },
            "memory_id": {
                "type": "string",
                "description": "ID of the memory (for update/delete operations)",
                "required": False
            },
            "query": {
                "type": "string",
                "description": "Query to search for relevant memories (for retrieve operation)",
                "required": False
            },
            "filters": {
                "type": "object",
                "description": "Filters to apply to memory search (for retrieve operation)",
                "required": False
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to retrieve (for retrieve operation)",
                "required": False,
                "default": 10
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
        Validate the parameters for the memory capability.
        
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
            # Check if operation is provided
            if "operation" not in kwargs:
                raise ValueError("Missing required parameter: operation")
            
            # Validate operation and ensure it's a valid MemoryOperation
            operation = kwargs.get("operation")
            if operation is not None:
                try:
                    operation_value = str(operation).lower()
                    operation = MemoryOperation(operation_value)
                except (ValueError, AttributeError):
                    valid_ops = [op.value for op in MemoryOperation]
                    raise ValueError(f"Invalid operation: {operation}. Valid operations are: {', '.join(valid_ops)}")
            else:
                raise ValueError("Operation cannot be None")
            
            # Use the appropriate Pydantic model based on operation
            model: Union[CreateMemoryParameters, RetrieveMemoryParameters, UpdateMemoryParameters, DeleteMemoryParameters]
            
            if operation == MemoryOperation.CREATE:
                model = CreateMemoryParameters(**kwargs)
            elif operation == MemoryOperation.RETRIEVE:
                model = RetrieveMemoryParameters(**kwargs)
            elif operation == MemoryOperation.UPDATE:
                model = UpdateMemoryParameters(**kwargs)
            elif operation == MemoryOperation.DELETE:
                model = DeleteMemoryParameters(**kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return model.model_dump()
        except Exception as e:
            # Convert any Pydantic validation errors to our ValidationError
            error_msg = f"Parameter validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg, self.name)
    
    async def _create_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new memory.
        
        Args:
            content: Content of the memory
            metadata: Optional metadata for the memory
            
        Returns:
            Dictionary containing the created memory details
        """
        # Track costs
        self.add_operation_cost("base", self._base_cost)
        self.add_operation_cost("content_length", len(content) * self._cost_per_character)
        self.add_operation_cost("storage", self._storage_cost)
        
        # Simulate some processing time
        await asyncio.sleep(0.2)
        
        # Generate a unique ID
        memory_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Create the memory
        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": created_at,
            "updated_at": None
        }
        
        # Store the memory (in a real implementation, this would use the memory service)
        self._memory_store[memory_id] = memory
        
        return memory
    
    async def _retrieve_memory(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None, 
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve memories based on query and filters.
        
        Args:
            query: Query to search for relevant memories
            filters: Optional filters to apply to the search
            limit: Maximum number of memories to retrieve
            
        Returns:
            Dictionary containing the retrieved memories
        """
        # Track costs
        self.add_operation_cost("base", self._base_cost)
        self.add_operation_cost("query_length", len(query) * self._cost_per_character)
        self.add_operation_cost("retrieval", self._retrieval_cost)
        
        # Simulate some processing time
        await asyncio.sleep(0.3)
        
        # In a real implementation, this would use a vector search
        # For demo, just do a simple text search in our in-memory store
        memories = []
        
        for memory_id, memory in self._memory_store.items():
            # Check if the query matches the content
            if query.lower() in memory["content"].lower():
                # Apply filters if provided
                if filters:
                    matches_filters = True
                    for key, value in filters.items():
                        if key in memory["metadata"]:
                            if memory["metadata"][key] != value:
                                matches_filters = False
                                break
                        else:
                            matches_filters = False
                            break
                    
                    if not matches_filters:
                        continue
                
                memories.append(memory)
        
        # Sort by recency (created_at)
        memories.sort(key=lambda m: m["created_at"], reverse=True)
        
        # Apply limit
        memories = memories[:limit]
        
        return {
            "memories": memories,
            "count": len(memories),
            "query": query,
            "filters": filters or {}
        }
    
    async def _update_memory(
        self, 
        memory_id: str, 
        content: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: Optional new content for the memory
            metadata: Optional new metadata for the memory
            
        Returns:
            Dictionary containing the updated memory details
            
        Raises:
            ExecutionError: If the memory with the given ID is not found
        """
        # Track costs
        self.add_operation_cost("base", self._base_cost)
        if content:
            self.add_operation_cost("content_length", len(content) * self._cost_per_character)
        
        # Simulate some processing time
        await asyncio.sleep(0.2)
        
        # Check if the memory exists
        if memory_id not in self._memory_store:
            raise ExecutionError(f"Memory with ID {memory_id} not found", self.name)
        
        # Get the existing memory
        memory = self._memory_store[memory_id]
        
        # Update the memory
        if content is not None:
            memory["content"] = content
        
        if metadata is not None:
            # In a real implementation, might want to merge metadata instead of replace
            memory["metadata"] = metadata
        
        # Update the timestamp
        memory["updated_at"] = datetime.now().isoformat()
        
        # Store the updated memory
        self._memory_store[memory_id] = memory
        
        return memory
    
    async def _delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            Dictionary containing the deletion status
            
        Raises:
            ExecutionError: If the memory with the given ID is not found
        """
        # Track costs
        self.add_operation_cost("base", self._base_cost)
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Check if the memory exists
        if memory_id not in self._memory_store:
            raise ExecutionError(f"Memory with ID {memory_id} not found", self.name)
        
        # Get the memory for the response
        memory = self._memory_store[memory_id]
        
        # Delete the memory
        del self._memory_store[memory_id]
        
        # Return deletion status
        return {
            "id": memory_id,
            "status": "deleted",
            "deleted_at": datetime.now().isoformat()
        }
    
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the memory capability.
        
        For backwards compatibility, this method calls execute_with_lifecycle
        which provides the standardized execution flow.
        
        Args:
            **kwargs: The parameters for the memory operation
            
        Returns:
            CapabilityResult containing the memory operation results and metadata
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
            
            # Validate parameters
            validated_params = self.validate_parameters(**kwargs)
            
            # Trigger before_execution event
            self.trigger_event("before_execution", capability_name=self.name, params=validated_params)
            
            # Extract operation type
            operation = validated_params["operation"]
            
            # Execute the appropriate operation
            operation_start_time = time.time()
            
            if operation == MemoryOperation.CREATE:
                result = await self._create_memory(
                    content=validated_params["content"],
                    metadata=validated_params.get("metadata")
                )
            elif operation == MemoryOperation.RETRIEVE:
                result = await self._retrieve_memory(
                    query=validated_params["query"],
                    filters=validated_params.get("filters"),
                    limit=validated_params.get("limit", 10)
                )
            elif operation == MemoryOperation.UPDATE:
                result = await self._update_memory(
                    memory_id=validated_params["memory_id"],
                    content=validated_params.get("content"),
                    metadata=validated_params.get("metadata")
                )
            elif operation == MemoryOperation.DELETE:
                result = await self._delete_memory(
                    memory_id=validated_params["memory_id"]
                )
            else:
                # This should never happen due to validation
                raise ValueError(f"Unsupported operation: {operation}")
            
            execution_time = time.time() - operation_start_time
            
            # Format the result
            result_data = {
                "status": "success",
                "results": result,
                "metadata": {
                    "execution_time": execution_time,
                    "operation": operation,
                    "cost": self.get_operation_cost()
                }
            }
            
            # Trigger after_execution event
            self.trigger_event("after_execution", 
                capability_name=self.name, 
                result=result_data,
                execution_time=execution_time
            )
            
            # Also trigger operation-specific event
            self.trigger_event(f"memory.{operation}", 
                capability_name=self.name,
                result=result
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
        except ExecutionError as ee:
            error_message = str(ee)
            logger.error(f"Execution error in {self.name}: {error_message}")
            
            # Trigger error event
            self.trigger_event(f"{self.name}.error", error=error_message, error_type="ExecutionError")
            
            return {
                "status": "error", 
                "error": error_message,
                "error_type": "ExecutionError"
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
