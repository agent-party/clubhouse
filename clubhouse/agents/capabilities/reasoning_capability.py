"""
Reasoning capability for performing multi-step reasoning on complex queries.

This module provides a capability for executing reasoning tasks, supporting
various reasoning types like general, logical, business, strategic, and financial.
It follows the standardized capability pattern with proper parameter validation,
event triggering, and error handling.
"""

import time
import random
import logging
import asyncio
from typing import Dict, List, Any, Optional, Literal, Union, Set

from pydantic import BaseModel, Field, field_validator

from clubhouse.agents.capability import BaseCapability, CapabilityResult
from clubhouse.agents.errors import ValidationError, ExecutionError

# Configure logging
logger = logging.getLogger(__name__)


class ReasoningParameters(BaseModel):
    """Model for reasoning parameters validation."""
    
    query: str = Field(..., description="The question or problem to reason about")
    reasoning_type: str = Field(
        default="general",
        description="Type of reasoning to perform (general, logical, business, etc.)"
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context or background information for the reasoning task"
    )
    constraints: Optional[List[str]] = Field(
        default=None,
        description="Optional list of constraints or requirements for the reasoning process"
    )
    max_steps: int = Field(
        default=5,
        description="Maximum number of reasoning steps to perform"
    )
    step_by_step: bool = Field(
        default=False,
        description="Whether to include step-by-step reasoning in the output"
    )
    
    @field_validator('reasoning_type')
    def validate_reasoning_type(cls, v: str) -> str:
        """Validate that reasoning_type is one of the supported types."""
        valid_types = ["general", "logical", "business", "strategic", "financial", "scientific", "ethical", "custom"]
        if v.lower() not in [t.lower() for t in valid_types]:
            raise ValueError(f"reasoning_type must be one of: {', '.join(valid_types)}")
        return v.lower()
    
    @field_validator('max_steps')
    def validate_max_steps(cls, v: int) -> int:
        """Validate that max_steps is within reasonable limits."""
        if v < 1:
            raise ValueError("max_steps must be at least 1")
        if v > 10:
            raise ValueError("max_steps cannot exceed 10 to prevent excessive computation")
        return v
    
    @field_validator('constraints')
    def validate_constraints(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate constraints list if provided."""
        if v is not None:
            # Check if constraints is empty
            if len(v) == 0:
                return None
                
            # Check for duplicate constraints
            if len(v) != len(set(v)):
                # Remove duplicates instead of raising an error
                return list(dict.fromkeys(v))
        
        return v


class ReasoningCapability(BaseCapability):
    """Capability for performing multi-step reasoning on complex queries."""
    
    name = "reasoning"
    description = "Perform multi-step reasoning on complex questions or problems"
    parameters_schema = ReasoningParameters
    
    # Base cost factors
    _base_cost = 0.02
    _cost_per_character = 0.0002
    _cost_per_step = 0.01
    _cost_multipliers = {
        "general": 1.0,
        "logical": 1.1,
        "business": 1.3,
        "strategic": 1.4,
        "financial": 1.5,
        "scientific": 1.6,
        "ethical": 1.2,
        "custom": 1.4
    }
    
    # Template reasoning steps for different reasoning types
    _reasoning_templates = {
        "general": [
            "Understand the core question or problem",
            "Identify key variables and factors",
            "Consider different perspectives",
            "Evaluate potential outcomes",
            "Synthesize insights into a coherent answer"
        ],
        "logical": [
            "Break down the premises of the problem",
            "Identify logical relationships between elements",
            "Apply formal logical rules to derive implications",
            "Check for logical fallacies or contradictions",
            "Construct a valid logical argument leading to the conclusion"
        ],
        "business": [
            "Analyze the business context and market conditions",
            "Identify key stakeholders and their interests",
            "Assess potential costs, benefits, and risks",
            "Consider competitive positioning and market trends",
            "Formulate a business strategy that optimizes for stated objectives"
        ],
        "strategic": [
            "Define the strategic objectives and success criteria",
            "Map the current situation and available resources",
            "Identify potential pathways and their trade-offs",
            "Anticipate obstacles and competitive responses",
            "Formulate a coherent strategy with implementation steps"
        ],
        "financial": [
            "Quantify the financial variables and constraints",
            "Model cash flows and financial impacts",
            "Apply relevant financial principles and metrics",
            "Assess risks and perform sensitivity analysis",
            "Provide financial recommendations based on optimal outcomes"
        ],
        "scientific": [
            "Define the scientific question or hypothesis",
            "Review relevant scientific principles and evidence",
            "Analyze causal relationships and mechanisms",
            "Apply scientific methods to evaluate the hypothesis",
            "Draw conclusions based on evidence and scientific reasoning"
        ],
        "ethical": [
            "Identify the ethical dimensions of the situation",
            "Consider relevant ethical principles and frameworks",
            "Evaluate the rights and interests of all stakeholders",
            "Weigh competing ethical considerations",
            "Formulate an ethically justified conclusion or recommendation"
        ]
    }
    
    def __init__(self) -> None:
        """Initialize the ReasoningCapability."""
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
            "query": {
                "type": "string",
                "description": "The question or problem to reason about",
                "required": True
            },
            "reasoning_type": {
                "type": "string",
                "description": "Type of reasoning to perform (general, logical, business, etc.)",
                "required": False,
                "default": "general"
            },
            "context": {
                "type": "string",
                "description": "Optional context or background information for the reasoning task",
                "required": False
            },
            "constraints": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Optional list of constraints or requirements for the reasoning process",
                "required": False
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum number of reasoning steps to perform",
                "required": False,
                "default": 5
            },
            "step_by_step": {
                "type": "boolean",
                "description": "Whether to include step-by-step reasoning in the output",
                "required": False,
                "default": False
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
        Validate the parameters for the reasoning capability.
        
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
            params = ReasoningParameters(**kwargs)
            
            # Additional validation for reasoning_type and constraints
            reasoning_type = params.reasoning_type
            
            # Handle empty constraints list
            if params.constraints is not None and len(params.constraints) == 0:
                params.constraints = None
            
            return params.model_dump()
        except Exception as e:
            # Convert any Pydantic validation errors to our ValidationError
            error_msg = f"Parameter validation failed: {str(e)}"
            logger.error(error_msg)
            raise ValidationError(error_msg, self.name)
    
    def _get_reasoning_steps(self, reasoning_type: str, max_steps: int) -> List[str]:
        """
        Get appropriate reasoning steps for the specified reasoning type.
        
        Args:
            reasoning_type: The type of reasoning
            max_steps: Maximum number of steps to return
            
        Returns:
            List of reasoning steps
        """
        # Use template reasoning steps for known reasoning types
        if reasoning_type in self._reasoning_templates:
            steps = self._reasoning_templates[reasoning_type]
            return steps[:max_steps]
        
        # For custom reasoning, generate generic steps
        return [f"Step {i+1}: Custom reasoning process" for i in range(max_steps)]
    
    async def _perform_reasoning(
        self, 
        query: str, 
        reasoning_type: str,
        context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        max_steps: int = 5,
        step_by_step: bool = False
    ) -> Dict[str, Any]:
        """
        Perform the actual reasoning operation.
        
        This method would connect to actual reasoning services in a real implementation.
        For demonstration purposes, it's a placeholder that mocks reasoning results
        and demonstrates proper cost tracking.
        
        Args:
            query: The question or problem to reason about
            reasoning_type: Type of reasoning to perform
            context: Optional context information
            constraints: Optional list of constraints
            max_steps: Maximum number of reasoning steps
            step_by_step: Whether to include step-by-step reasoning
            
        Returns:
            Dictionary containing reasoning results
        """
        # Ensure reasoning type is valid
        if reasoning_type not in self._cost_multipliers:
            reasoning_type = "general"
        
        # Get appropriate reasoning steps
        template_steps = self._get_reasoning_steps(reasoning_type, max_steps)
        
        # Calculate query complexity based on length and any provided context
        query_length = len(query)
        context_length = len(context) if context else 0
        total_input_length = query_length + context_length
        
        # Simulate processing time based on query complexity and reasoning type
        reasoning_complexity = total_input_length * 0.0005 * self._cost_multipliers.get(reasoning_type, 1.0)
        reasoning_complexity += max_steps * 0.1  # More steps take more time
        await asyncio.sleep(min(0.5, reasoning_complexity))  # Cap at 0.5 seconds for testing
        
        # Add to operation costs
        self.add_operation_cost("base", self._base_cost)
        self.add_operation_cost("query_complexity", total_input_length * self._cost_per_character)
        self.add_operation_cost(
            "reasoning_type", 
            self._base_cost * self._cost_multipliers.get(reasoning_type, 1.0)
        )
        self.add_operation_cost("steps", max_steps * self._cost_per_step)
        
        # Generate specific reasoning steps based on the query and reasoning type
        actual_steps = []
        
        # Apply constraints to reasoning if provided
        constraint_text = ""
        if constraints and len(constraints) > 0:
            constraint_text = f" (considering constraints: {', '.join(constraints)})"
        
        # Generate reasoning steps with some randomization for demonstration
        for i, template in enumerate(template_steps):
            # Make the step more specific to the query
            step = f"{template}{constraint_text}"
            
            # Add some variation based on position in sequence
            if i == 0:
                # First step often involves understanding the question
                step = f"Analyze the question: '{query[:50]}...' within {reasoning_type} reasoning framework"
            elif i == len(template_steps) - 1:
                # Last step involves forming a conclusion
                step = f"Synthesize insights into a {reasoning_type} conclusion"
            
            actual_steps.append(step)
        
        # Generate a mock conclusion based on the reasoning type
        conclusion_templates = {
            "general": f"Based on careful consideration of all factors, the answer to '{query[:30]}...' is...",
            "logical": "Following logical reasoning principles, we can conclude that...",
            "business": "From a business perspective, the optimal approach would be to...",
            "strategic": "The strategic analysis indicates that the best course of action is to...",
            "financial": "Financial analysis suggests that the most advantageous decision would be to...",
            "scientific": "Scientific reasoning and evidence points to the conclusion that...",
            "ethical": "From an ethical standpoint, the most justifiable position is..."
        }
        
        conclusion = conclusion_templates.get(
            reasoning_type, 
            f"After {reasoning_type} analysis, the answer to the query is..."
        )
        
        # Add specific query details to make conclusion more relevant
        conclusion += f" This addresses the original question about {query[:50]}..."
        
        # Build detailed result
        result = {
            "reasoning_type": reasoning_type,
            "steps": actual_steps if step_by_step else [],
            "conclusion": conclusion,
            "query_length": query_length,
            "context_provided": context is not None,
            "constraints_applied": constraints is not None,
            "step_count": len(actual_steps)
        }
        
        return result
    
    async def execute(self, **kwargs: Any) -> CapabilityResult:
        """
        Execute the reasoning capability.
        
        For backwards compatibility, this method calls execute_with_lifecycle
        which provides the standardized execution flow.
        
        Args:
            **kwargs: The parameters for the reasoning operation
            
        Returns:
            CapabilityResult containing the reasoning results and metadata
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
            
            # Extract validated parameters
            query = validated_params["query"]
            reasoning_type = validated_params.get("reasoning_type", "general")
            context = validated_params.get("context")
            constraints = validated_params.get("constraints")
            max_steps = validated_params.get("max_steps", 5)
            step_by_step = validated_params.get("step_by_step", False)
            
            # Execute the reasoning
            operation_start_time = time.time()
            reasoning_results = await self._perform_reasoning(
                query=query, 
                reasoning_type=reasoning_type,
                context=context,
                constraints=constraints,
                max_steps=max_steps,
                step_by_step=step_by_step
            )
            execution_time = time.time() - operation_start_time
            
            # Format the result
            result_data = {
                "status": "success",
                "results": reasoning_results,
                "metadata": {
                    "execution_time": execution_time,
                    "query_length": len(query),
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
