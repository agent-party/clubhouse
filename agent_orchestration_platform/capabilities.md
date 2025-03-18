# Agent Capabilities Framework

## Overview

The Capabilities Framework defines the core functionality units that agents can utilize to perform tasks. Each capability represents a distinct skill or function that agents can leverage individually or in combination to solve problems across various domains.

## Core Principles

1. **Modularity**: Each capability functions as an independent, reusable component
2. **Composability**: Capabilities can be combined to address complex requirements
3. **Testability**: Each capability has clear boundaries and interfaces for testing
4. **Extensibility**: New capabilities can be added without disrupting existing ones
5. **Cost-Awareness**: Capabilities track resource consumption and operate within budgets

## Capability Structure

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import uuid
from datetime import datetime


class CapabilityStatus(str, Enum):
    """Status of a capability execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class CapabilityMetrics(BaseModel):
    """Metrics tracked for capability execution."""
    duration_ms: int = 0
    token_count: Dict[str, int] = Field(default_factory=dict)  # input/output tokens
    token_cost: float = 0.0
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    
    def add_tokens(self, input_tokens: int, output_tokens: int, model: str):
        """Add token usage and calculate costs."""
        self.token_count["input"] = self.token_count.get("input", 0) + input_tokens
        self.token_count["output"] = self.token_count.get("output", 0) + output_tokens
        
        # Calculate cost based on model-specific pricing
        # This would reference a pricing table in production
        input_cost_per_1k = 0.0015  # Example rate for input tokens
        output_cost_per_1k = 0.002  # Example rate for output tokens
        
        if model.startswith("gpt-4"):
            input_cost_per_1k = 0.03
            output_cost_per_1k = 0.06
        
        new_cost = (input_tokens / 1000 * input_cost_per_1k) + \
                   (output_tokens / 1000 * output_cost_per_1k)
        
        self.token_cost += new_cost


class CapabilityResult(BaseModel):
    """Result of a capability execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    capability_id: str
    status: CapabilityStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    metrics: CapabilityMetrics = Field(default_factory=CapabilityMetrics)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CapabilityParameters(BaseModel):
    """Base class for capability parameters."""
    budget_limit: Optional[float] = None


class BaseCapability:
    """Base class for all capabilities."""
    
    def __init__(self, 
                llm_service, 
                event_bus, 
                storage_service=None,
                cost_manager=None):
        self.llm_service = llm_service
        self.event_bus = event_bus
        self.storage_service = storage_service
        self.cost_manager = cost_manager
        self.id = str(uuid.uuid4())
    
    async def execute(self, params: CapabilityParameters) -> CapabilityResult:
        """Execute the capability with parameters."""
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement execute()")
    
    async def execute_with_lifecycle(self, 
                                   params: CapabilityParameters, 
                                   context: Dict[str, Any] = None) -> CapabilityResult:
        """Execute capability with lifecycle events and metrics tracking."""
        # Start metrics tracking
        start_time = datetime.utcnow()
        metrics = CapabilityMetrics()
        
        # Check budget before execution
        if params.budget_limit is not None and self.cost_manager:
            available_budget = await self.cost_manager.check_available_budget(
                capability_id=self.id,
                context=context
            )
            if available_budget < params.budget_limit:
                return CapabilityResult(
                    capability_id=self.id,
                    status=CapabilityStatus.ABORTED,
                    error=f"Budget limit exceeded. Available: {available_budget}, Required: {params.budget_limit}",
                    metrics=metrics
                )
        
        # Send started event
        await self.event_bus.publish(
            f"capability.{self.__class__.__name__.lower()}.started",
            {
                "capability_id": self.id,
                "params": params.dict() if hasattr(params, "dict") else params,
                "context": context
            }
        )
        
        try:
            # Execute capability
            result = await self.execute(params)
            status = CapabilityStatus.COMPLETED
            error = None
        except Exception as e:
            # Handle failure
            result = None
            status = CapabilityStatus.FAILED
            error = str(e)
        
        # Complete metrics
        end_time = datetime.utcnow()
        metrics.duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Update budget usage if cost manager exists
        if self.cost_manager:
            await self.cost_manager.record_usage(
                capability_id=self.id,
                metrics=metrics,
                context=context
            )
        
        # Create result object
        capability_result = CapabilityResult(
            capability_id=self.id,
            status=status,
            result=result,
            error=error,
            metrics=metrics
        )
        
        # Send completed/failed event
        event_type = "completed" if status == CapabilityStatus.COMPLETED else "failed"
        await self.event_bus.publish(
            f"capability.{self.__class__.__name__.lower()}.{event_type}",
            {
                "capability_id": self.id,
                "result": capability_result.dict(),
                "context": context
            }
        )
        
        return capability_result
```

## Standard Capabilities

### 1. Search Capability

Enables agents to retrieve and analyze information from various sources.

```python
class SearchParameters(CapabilityParameters):
    """Parameters for search capability."""
    query: str
    sources: List[str]
    max_results: int = 10
    filters: Optional[Dict[str, Any]] = None
    depth: Literal["basic", "standard", "deep"] = "standard"


class SearchCapability(BaseCapability):
    """Capability for searching information sources."""
    
    def __init__(self, llm_service, event_bus, search_service, storage_service=None, cost_manager=None):
        super().__init__(llm_service, event_bus, storage_service, cost_manager)
        self.search_service = search_service
    
    async def execute(self, params: SearchParameters) -> Dict[str, Any]:
        """Execute search with given parameters."""
        # Validate parameters
        if not params.query:
            raise ValueError("Search query cannot be empty")
        
        # Estimate token usage for budget tracking
        estimated_tokens = len(params.query.split()) * 2  # Rough estimation
        if params.depth == "deep":
            estimated_tokens *= 3
        elif params.depth == "basic":
            estimated_tokens /= 2
            
        # Record token estimation in metrics
        metrics = CapabilityMetrics()
        metrics.add_tokens(estimated_tokens, 0, "text-embedding-ada-002")
        
        # Execute search across specified sources
        results = []
        for source in params.sources:
            source_results = await self.search_service.search(
                query=params.query,
                source=source,
                max_results=params.max_results,
                filters=params.filters,
                depth=params.depth
            )
            results.extend(source_results)
        
        # Process and enhance results with LLM if needed
        if params.depth != "basic" and results:
            # Use LLM to analyze and enhance search results
            enhancement_prompt = f"""
            Analyze these search results for the query: "{params.query}"
            
            Results:
            {results}
            
            Please provide:
            1. The most relevant information addressing the query
            2. Any contradictions or inconsistencies in the results
            3. Key insights that might not be explicitly stated
            """
            
            enhancement_response = await self.llm_service.generate(
                prompt=enhancement_prompt,
                max_tokens=500
            )
            
            # Update metrics with token usage
            input_tokens = len(enhancement_prompt.split())
            output_tokens = len(enhancement_response.split())
            metrics.add_tokens(input_tokens, output_tokens, "gpt-3.5-turbo")
            
            # Add enhanced analysis to results
            results = {
                "raw_results": results,
                "enhanced_analysis": enhancement_response
            }
        
        return results
```

### 2. Summarize Capability

Enables agents to condense and extract key information from longer content.

```python
class SummarizeParameters(CapabilityParameters):
    """Parameters for summarize capability."""
    content: str
    length: Literal["short", "medium", "long"] = "medium"
    focus: Optional[str] = None
    format: Literal["bullets", "paragraphs", "structured"] = "paragraphs"
    include_metadata: bool = False


class SummarizeCapability(BaseCapability):
    """Capability for summarizing content."""
    
    async def execute(self, params: SummarizeParameters) -> Dict[str, Any]:
        """Execute summarization with given parameters."""
        # Validate parameters
        if not params.content:
            raise ValueError("Content cannot be empty")
        
        # Prepare metrics
        metrics = CapabilityMetrics()
        
        # Determine target length based on parameter
        length_tokens = {
            "short": 100,
            "medium": 250,
            "long": 500
        }
        target_length = length_tokens[params.length]
        
        # Construct summarization prompt
        focus_instruction = f"Focus on aspects related to: {params.focus}" if params.focus else ""
        format_instruction = {
            "bullets": "Format the summary as bullet points.",
            "paragraphs": "Format the summary as concise paragraphs.",
            "structured": "Format the summary with clear headings and sections."
        }[params.format]
        
        prompt = f"""
        Summarize the following content in approximately {target_length} words.
        {focus_instruction}
        {format_instruction}
        
        Content to summarize:
        {params.content}
        """
        
        # Generate summary
        summary_response = await self.llm_service.generate(
            prompt=prompt,
            max_tokens=target_length * 1.5  # Allow some flexibility
        )
        
        # Update metrics with token usage
        input_tokens = len(params.content.split())
        output_tokens = len(summary_response.split())
        metrics.add_tokens(input_tokens, output_tokens, "gpt-3.5-turbo")
        
        # Generate metadata if requested
        metadata = {}
        if params.include_metadata:
            metadata_prompt = f"""
            For the following content, extract key metadata such as:
            - Main topics covered
            - Key entities mentioned
            - Estimated reading time of original
            - Tone and perspective
            
            Content:
            {params.content}
            """
            
            metadata_response = await self.llm_service.generate_with_json_output(
                prompt=metadata_prompt
            )
            
            # Update metrics with additional token usage
            metadata_input_tokens = len(metadata_prompt.split())
            metadata_output_tokens = len(str(metadata_response).split())
            metrics.add_tokens(metadata_input_tokens, metadata_output_tokens, "gpt-3.5-turbo")
            
            metadata = metadata_response
        
        # Return results
        return {
            "summary": summary_response,
            "metadata": metadata,
            "original_length": len(params.content.split()),
            "summary_length": len(summary_response.split()),
            "compression_ratio": len(summary_response.split()) / len(params.content.split())
        }
```

### 3. Reasoning Capability

Enables agents to perform step-by-step analysis and logical problem-solving.

```python
class ReasoningParameters(CapabilityParameters):
    """Parameters for reasoning capability."""
    problem: str
    context: Optional[str] = None
    reasoning_depth: Literal["basic", "standard", "deep"] = "standard"
    chain_of_thought: bool = True
    verification: bool = False


class ReasoningCapability(BaseCapability):
    """Capability for logical reasoning and problem-solving."""
    
    async def execute(self, params: ReasoningParameters) -> Dict[str, Any]:
        """Execute reasoning with given parameters."""
        # Validate parameters
        if not params.problem:
            raise ValueError("Problem statement cannot be empty")
        
        # Prepare metrics
        metrics = CapabilityMetrics()
        
        # Set up model based on reasoning depth
        model = {
            "basic": "gpt-3.5-turbo",
            "standard": "gpt-3.5-turbo-16k",
            "deep": "gpt-4"
        }[params.reasoning_depth]
        
        # Construct reasoning prompt
        context_info = f"\nContext Information:\n{params.context}" if params.context else ""
        cot_instruction = "Think step-by-step and explain your reasoning process in detail." if params.chain_of_thought else ""
        
        prompt = f"""
        Problem: {params.problem}
        {context_info}
        
        {cot_instruction}
        
        Analyze this problem and provide a well-reasoned solution.
        """
        
        # Generate reasoning
        reasoning_response = await self.llm_service.generate(
            prompt=prompt,
            model=model,
            temperature=0.3,  # Lower temperature for logical reasoning
            max_tokens=2000
        )
        
        # Update metrics with token usage
        input_tokens = len(prompt.split())
        output_tokens = len(reasoning_response.split())
        metrics.add_tokens(input_tokens, output_tokens, model)
        
        # Perform verification if requested
        verification_result = None
        if params.verification:
            verification_prompt = f"""
            Review the following solution to this problem:
            
            Problem: {params.problem}
            
            Solution:
            {reasoning_response}
            
            Please verify this solution by:
            1. Checking for logical errors or inconsistencies
            2. Validating the key assumptions
            3. Testing the solution against edge cases
            4. Providing a confidence score from 0-100%
            
            Format your response as a JSON with these fields:
            - is_correct: boolean
            - issues: array of strings
            - confidence_score: integer
            - improved_solution: string (optional)
            """
            
            verification_result = await self.llm_service.generate_with_json_output(
                prompt=verification_prompt,
                model=model
            )
            
            # Update metrics with additional token usage
            verification_input_tokens = len(verification_prompt.split())
            verification_output_tokens = len(str(verification_result).split())
            metrics.add_tokens(verification_input_tokens, verification_output_tokens, model)
        
        # Return results
        return {
            "reasoning": reasoning_response,
            "verification": verification_result,
            "model_used": model,
            "has_chain_of_thought": params.chain_of_thought
        }
```

### 4. Tool Use Capability

Enables agents to interact with external tools and APIs to accomplish tasks.

```python
class ToolParameters(CapabilityParameters):
    """Parameters for tool use capability."""
    tool_name: str
    inputs: Dict[str, Any]
    safety_check: bool = True
    max_retries: int = 3
    timeout_seconds: int = 30
    asynchronous: bool = False


class ToolUseCapability(BaseCapability):
    """Capability for using external tools and APIs."""
    
    def __init__(self, llm_service, event_bus, tool_registry, storage_service=None, cost_manager=None):
        super().__init__(llm_service, event_bus, storage_service, cost_manager)
        self.tool_registry = tool_registry
    
    async def execute(self, params: ToolParameters) -> Dict[str, Any]:
        """Execute tool with given parameters."""
        # Validate parameters
        if not params.tool_name:
            raise ValueError("Tool name cannot be empty")
        
        # Check if tool exists in registry
        if not await self.tool_registry.has_tool(params.tool_name):
            raise ValueError(f"Tool '{params.tool_name}' not found in registry")
        
        # Get tool definition
        tool_def = await self.tool_registry.get_tool(params.tool_name)
        
        # Validate inputs against tool schema
        validation_result = await self.tool_registry.validate_inputs(
            tool_name=params.tool_name,
            inputs=params.inputs
        )
        
        if not validation_result.valid:
            raise ValueError(f"Invalid inputs: {validation_result.errors}")
        
        # Perform safety check if required
        if params.safety_check:
            safety_result = await self.tool_registry.check_safety(
                tool_name=params.tool_name,
                inputs=params.inputs
            )
            
            if not safety_result.safe:
                raise ValueError(f"Safety check failed: {safety_result.reason}")
        
        # Execute tool with retries
        retry_count = 0
        last_error = None
        
        while retry_count < params.max_retries:
            try:
                # Execute tool
                tool_result = await self.tool_registry.execute_tool(
                    tool_name=params.tool_name,
                    inputs=params.inputs,
                    timeout=params.timeout_seconds,
                    asynchronous=params.asynchronous
                )
                
                # Return result
                return {
                    "tool_name": params.tool_name,
                    "result": tool_result,
                    "retries": retry_count,
                    "status": "success",
                    "asynchronous": params.asynchronous,
                    "job_id": tool_result.get("job_id") if params.asynchronous else None
                }
            except Exception as e:
                last_error = str(e)
                retry_count += 1
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** retry_count)
        
        # If we get here, all retries failed
        return {
            "tool_name": params.tool_name,
            "result": None,
            "retries": retry_count,
            "status": "failed",
            "error": last_error
        }
```

### 5. Generation Capability

Enables agents to create original content across various formats and styles.

```python
class GenerationParameters(CapabilityParameters):
    """Parameters for content generation capability."""
    prompt: str
    type: Literal["text", "code", "creative", "technical", "conversational"] = "text"
    length: Literal["short", "medium", "long"] = "medium"
    style_guide: Optional[str] = None
    reference_examples: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None
    model_preference: Optional[str] = None


class GenerationCapability(BaseCapability):
    """Capability for generating various types of content."""
    
    async def execute(self, params: GenerationParameters) -> Dict[str, Any]:
        """Execute generation with given parameters."""
        # Validate parameters
        if not params.prompt:
            raise ValueError("Generation prompt cannot be empty")
        
        # Prepare metrics
        metrics = CapabilityMetrics()
        
        # Determine output length in tokens
        length_tokens = {
            "short": 200,
            "medium": 500,
            "long": 1500
        }
        target_length = length_tokens[params.length]
        
        # Determine appropriate model based on type and preference
        model = params.model_preference or {
            "text": "gpt-3.5-turbo",
            "code": "gpt-3.5-turbo",
            "creative": "gpt-4",
            "technical": "gpt-4",
            "conversational": "gpt-3.5-turbo"
        }[params.type]
        
        # Adjust temperature based on content type
        temperature = {
            "text": 0.7,
            "code": 0.2,
            "creative": 0.9,
            "technical": 0.3,
            "conversational": 0.8
        }[params.type]
        
        # Construct generation prompt
        style_instructions = f"\nStyle Guide:\n{params.style_guide}" if params.style_guide else ""
        
        examples_text = ""
        if params.reference_examples:
            examples_text = "\nReference Examples:\n"
            for i, example in enumerate(params.reference_examples):
                examples_text += f"Example {i+1}:\n{example}\n\n"
        
        constraints_text = ""
        if params.constraints:
            constraints_text = "\nConstraints:\n"
            for key, value in params.constraints.items():
                constraints_text += f"- {key}: {value}\n"
        
        prompt = f"""
        Generate {params.type} content based on the following prompt:
        
        {params.prompt}
        
        {style_instructions}
        {examples_text}
        {constraints_text}
        
        Target length: Approximately {target_length} words.
        """
        
        # Generate content
        generation_response = await self.llm_service.generate(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=target_length * 1.5
        )
        
        # Update metrics with token usage
        input_tokens = len(prompt.split())
        output_tokens = len(generation_response.split())
        metrics.add_tokens(input_tokens, output_tokens, model)
        
        # Return results
        return {
            "content": generation_response,
            "model_used": model,
            "type": params.type,
            "length": len(generation_response.split()),
            "token_count": output_tokens
        }
```

## Capability Registry

The CapabilityRegistry manages available capabilities and provides discovery services.

```python
class CapabilityRegistry:
    """Registry for managing available capabilities."""
    
    def __init__(self, neo4j_service, event_bus):
        self.neo4j_service = neo4j_service
        self.event_bus = event_bus
        self.capabilities = {}
        
    async def register_capability(self, capability_class, metadata):
        """Register a capability with metadata."""
        capability_name = capability_class.__name__
        
        # Store in registry
        self.capabilities[capability_name] = {
            "class": capability_class,
            "metadata": metadata
        }
        
        # Store in database for persistence
        await self.neo4j_service.register_capability(
            name=capability_name,
            metadata=metadata
        )
        
        # Publish event
        await self.event_bus.publish(
            "capability.registry.capability_registered",
            {
                "capability_name": capability_name,
                "metadata": metadata
            }
        )
    
    async def get_capability(self, name):
        """Get a capability by name."""
        if name not in self.capabilities:
            raise ValueError(f"Capability '{name}' not found in registry")
        
        return self.capabilities[name]
    
    async def list_capabilities(self, filters=None):
        """List all registered capabilities, optionally filtered."""
        if not filters:
            return self.capabilities
        
        # Apply filters
        filtered = {}
        for name, capability in self.capabilities.items():
            match = True
            for key, value in filters.items():
                if key not in capability["metadata"] or capability["metadata"][key] != value:
                    match = False
                    break
            
            if match:
                filtered[name] = capability
        
        return filtered
    
    async def instantiate_capability(self, name, services):
        """Instantiate a capability with required services."""
        if name not in self.capabilities:
            raise ValueError(f"Capability '{name}' not found in registry")
        
        capability_class = self.capabilities[name]["class"]
        
        # Check for required services
        required_services = getattr(capability_class, "required_services", [])
        for service_name in required_services:
            if service_name not in services:
                raise ValueError(f"Required service '{service_name}' not provided")
        
        # Instantiate capability
        capability_instance = capability_class(**services)
        
        return capability_instance
```

## Cost Management Integration

All capabilities integrate with the cost management system to:

1. **Track resource usage**: Monitor token consumption, API calls, and compute time
2. **Enforce budgets**: Abort operations that would exceed allocated budgets
3. **Optimize for efficiency**: Choose appropriate models based on task complexity
4. **Report costs**: Provide transparent cost reporting for all operations

```python
class CostManager:
    """Service for managing and tracking capability costs."""
    
    def __init__(self, neo4j_service, event_bus):
        self.neo4j_service = neo4j_service
        self.event_bus = event_bus
    
    async def check_available_budget(self, capability_id, context=None):
        """Check available budget for a capability execution."""
        # Get budget scope from context
        scope = context.get("budget_scope", "default") if context else "default"
        
        # Retrieve budget allocation
        budget_allocation = await self.neo4j_service.get_budget_allocation(scope)
        
        # Retrieve current usage
        current_usage = await self.neo4j_service.get_budget_usage(scope)
        
        # Calculate available budget
        available = budget_allocation - current_usage
        
        return available
    
    async def record_usage(self, capability_id, metrics, context=None):
        """Record resource usage for a capability execution."""
        # Get budget scope from context
        scope = context.get("budget_scope", "default") if context else "default"
        
        # Store usage in database
        await self.neo4j_service.record_budget_usage(
            scope=scope,
            capability_id=capability_id,
            token_cost=metrics.token_cost,
            resource_usage=metrics.resource_usage,
            context=context
        )
        
        # Publish usage event
        await self.event_bus.publish(
            "cost.usage_recorded",
            {
                "capability_id": capability_id,
                "scope": scope,
                "token_cost": metrics.token_cost,
                "token_count": metrics.token_count,
                "resource_usage": metrics.resource_usage,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Check if approaching budget limit
        allocation = await self.neo4j_service.get_budget_allocation(scope)
        total_usage = await self.neo4j_service.get_budget_usage(scope)
        
        # If usage exceeds 80% of allocation, send warning
        if total_usage >= 0.8 * allocation:
            await self.event_bus.publish(
                "cost.budget_warning",
                {
                    "scope": scope,
                    "allocation": allocation,
                    "usage": total_usage,
                    "percentage": (total_usage / allocation) * 100,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
```

## Testing Strategy

Testing for capabilities focuses on these key areas:

1. **Unit Tests**:
   - Parameter validation logic
   - Result formatting
   - Cost calculation accuracy

2. **Integration Tests**:
   - Interaction with services
   - Event emission
   - Budget enforcement

3. **Performance Tests**:
   - Response time under load
   - Resource usage patterns
   - Scaling behavior

4. **Cost Efficiency Tests**:
   - Token optimizations
   - Budget adherence
   - Cost vs. quality tradeoffs

## Implementation Guidelines

When implementing new capabilities:

1. Always inherit from BaseCapability
2. Use Pydantic models for parameters
3. Implement proper error handling
4. Track all resource usage in metrics
5. Follow event-driven patterns
6. Consider cost implications of implementation choices
7. Prefer composition over deep inheritance
8. Write comprehensive tests for all functionality

## Conclusion

The Capabilities Framework provides a standardized approach to agent functionality that emphasizes modularity, testability, and cost awareness. By decomposing complex agent behaviors into discrete, reusable capabilities, the system enables efficient development, testing, and evolution of agent functionality while maintaining strict budget controls.
