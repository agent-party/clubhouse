# Agent Capabilities

## Overview

The Agent Orchestration Platform implements a capability-based architecture where agent functionality is encapsulated in modular, reusable capabilities. This approach enables flexible agent composition and clear separation of concerns, following SOLID principles and clean code practices.

## BaseCapability

All capabilities extend from the `BaseCapability` abstract base class, which provides core functionality for parameter validation, error handling, and event lifecycle management:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar, cast

from pydantic import BaseModel, ValidationError

from clubhouse.agents.events import EventType
from clubhouse.common.errors import CapabilityError, ParameterValidationError

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

class BaseCapability(Generic[T, R], ABC):
    """Base class for all agent capabilities.
    
    Provides standard functionality for parameter validation,
    error handling, and event lifecycle management.
    """
    
    class Parameters(BaseModel):
        """Base parameters model for capability execution."""
        pass
        
    class Response(BaseModel):
        """Base response model for capability results."""
        status: str = "success"
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        
    def __init__(self, agent):
        """Initialize the capability with its parent agent."""
        self.agent = agent
        
    def validate_parameters(self, parameters: Dict[str, Any]) -> T:
        """Validate and convert parameters dictionary to Parameters model."""
        try:
            # Get the concrete Parameters class from the subclass
            params_class = self.get_parameters_class()
            # Validate and convert to Parameters model
            return params_class(**parameters)
        except ValidationError as e:
            # Convert Pydantic validation error to our custom error
            raise ParameterValidationError(str(e)) from e
        except Exception as e:
            # Handle any other errors during validation
            raise CapabilityError(f"Parameter validation failed: {str(e)}") from e
            
    def get_parameters_class(self) -> Type[T]:
        """Get the concrete Parameters class for this capability."""
        # Get the specific Parameters class from the subclass
        return cast(Type[T], self.__class__.Parameters)
        
    def get_response_class(self) -> Type[R]:
        """Get the concrete Response class for this capability."""
        # Get the specific Response class from the subclass
        return cast(Type[R], self.__class__.Response)
        
    async def execute_with_lifecycle(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the capability with full lifecycle event handling.
        
        This method handles:
        1. Parameter validation
        2. Pre-execution events
        3. Execution
        4. Post-execution events
        5. Error handling
        """
        try:
            # Validate parameters
            validated_params = self.validate_parameters(parameters)
            
            # Trigger pre-execution event
            await self.agent.trigger_event(
                EventType.CAPABILITY_STARTED,
                capability_name=self.__class__.__name__,
                parameters=parameters
            )
            
            # Execute the capability
            result = await self.execute(validated_params)
            
            # Ensure result is properly formatted
            response_class = self.get_response_class()
            if not isinstance(result, response_class):
                # Convert dict or other result to Response model
                if isinstance(result, dict):
                    result = response_class(data=result)
                else:
                    result = response_class(data={"result": result})
            
            # Trigger post-execution event
            await self.agent.trigger_event(
                EventType.CAPABILITY_COMPLETED,
                capability_name=self.__class__.__name__,
                parameters=parameters,
                result=result.dict()
            )
            
            return result.dict()
            
        except ParameterValidationError as e:
            # Handle parameter validation errors
            error_response = self.get_response_class()(
                status="error",
                error=f"Parameter validation failed: {str(e)}"
            )
            
            # Trigger error event
            await self.agent.trigger_event(
                EventType.CAPABILITY_ERROR,
                capability_name=self.__class__.__name__,
                parameters=parameters,
                error=str(e)
            )
            
            return error_response.dict()
            
        except Exception as e:
            # Handle any other errors
            error_response = self.get_response_class()(
                status="error",
                error=f"Capability execution failed: {str(e)}"
            )
            
            # Trigger error event
            await self.agent.trigger_event(
                EventType.CAPABILITY_ERROR,
                capability_name=self.__class__.__name__,
                parameters=parameters,
                error=str(e)
            )
            
            return error_response.dict()
    
    @abstractmethod
    async def execute(self, parameters: T) -> R:
        """Execute the capability with validated parameters.
        
        This method must be implemented by all capability subclasses.
        """
        pass
```

## Core Capabilities

The platform includes several core capabilities that are essential for basic agent functionality:

### SearchCapability

The `SearchCapability` enables agents to search for information:

```python
class SearchCapability(BaseCapability):
    """Capability for searching and retrieving information."""
    
    class Parameters(BaseModel):
        """Parameters for search capability."""
        query: str
        max_results: Optional[int] = 10
        sources: Optional[List[str]] = None
        filters: Optional[Dict[str, Any]] = None
        
    class Response(BaseCapability.Response):
        """Response for search capability."""
        results: Optional[List[Dict[str, Any]]] = None
        total_found: Optional[int] = None
        
    async def execute(self, parameters: Parameters) -> Response:
        """Execute the search with validated parameters."""
        try:
            # Get search service from registry
            search_service = self.agent.service_registry.get_service(SearchServiceProtocol)
            
            # Perform search operation
            search_results = await search_service.search(
                query=parameters.query,
                max_results=parameters.max_results,
                sources=parameters.sources,
                filters=parameters.filters
            )
            
            # Return formatted response
            return self.Response(
                status="success",
                data={
                    "results": search_results.items,
                    "total_found": search_results.total
                },
                results=search_results.items,
                total_found=search_results.total
            )
            
        except Exception as e:
            # Handle exceptions
            return self.Response(
                status="error",
                error=f"Search failed: {str(e)}"
            )
```

### SummarizeCapability

The `SummarizeCapability` enables agents to summarize content. This capability is being refactored to follow the improved patterns from SearchCapability:

```python
class SummarizeCapability(BaseCapability):
    """Capability for summarizing content."""
    
    class Parameters(BaseModel):
        """Parameters for summarize capability."""
        content: str
        max_length: Optional[int] = 200
        format: Optional[str] = "paragraph"  # "paragraph", "bullets", "key_points"
        focus: Optional[List[str]] = None  # Aspects to focus on
        
    class Response(BaseCapability.Response):
        """Response for summarize capability."""
        summary: Optional[str] = None
        original_length: Optional[int] = None
        summary_length: Optional[int] = None
        
    async def execute(self, parameters: Parameters) -> Response:
        """Execute the summarization with validated parameters."""
        try:
            # Get the NLP service from registry
            nlp_service = self.agent.service_registry.get_service(NLPServiceProtocol)
            
            # Track original content length
            original_length = len(parameters.content)
            
            # Generate summary using NLP service
            summary_result = await nlp_service.summarize(
                content=parameters.content,
                max_length=parameters.max_length,
                format=parameters.format,
                focus=parameters.focus
            )
            
            # Trigger custom events for backward compatibility
            await self.agent.trigger_event(
                "summarize_completed",
                summary=summary_result.text,
                original_length=original_length,
                summary_length=len(summary_result.text)
            )
            
            # Return formatted response
            return self.Response(
                status="success",
                data={
                    "summary": summary_result.text,
                    "original_length": original_length,
                    "summary_length": len(summary_result.text)
                },
                summary=summary_result.text,
                original_length=original_length,
                summary_length=len(summary_result.text)
            )
            
        except Exception as e:
            # Handle exceptions
            return self.Response(
                status="error",
                error=f"Summarization failed: {str(e)}"
            )
```

## Evolutionary Framework Capabilities

These specialized capabilities support the evolutionary framework:

### GeneratorCapability

The `GeneratorCapability` enables agents to generate creative solutions to problems:

```python
class GeneratorCapability(BaseCapability):
    """Capability for generating creative solutions to problems."""
    
    class Parameters(BaseModel):
        """Parameters for generator capability."""
        problem_statement: str
        constraints: Optional[List[str]] = None
        previous_attempts: Optional[List[Dict[str, Any]]] = None
        creativity_level: Optional[float] = 0.7
        
    class Response(BaseCapability.Response):
        """Response for generator capability."""
        solutions: Optional[List[Dict[str, Any]]] = None
        rationale: Optional[str] = None
        
    async def execute(self, parameters: Parameters) -> Response:
        """Execute the solution generation with validated parameters."""
        try:
            # Get the LLM service from registry
            llm_service = self.agent.service_registry.get_service(LLMServiceProtocol)
            
            # Prepare generation prompt
            prompt = self._build_generation_prompt(
                problem_statement=parameters.problem_statement,
                constraints=parameters.constraints,
                previous_attempts=parameters.previous_attempts
            )
            
            # Generate solutions using LLM service
            solutions = await llm_service.generate_solutions(
                prompt=prompt,
                num_solutions=5,  # Default to 5 solutions
                temperature=parameters.creativity_level
            )
            
            # Return formatted response
            return self.Response(
                status="success",
                data={
                    "solutions": solutions.items,
                    "rationale": solutions.rationale
                },
                solutions=solutions.items,
                rationale=solutions.rationale
            )
            
        except Exception as e:
            # Handle exceptions
            return self.Response(
                status="error",
                error=f"Solution generation failed: {str(e)}"
            )
            
    def _build_generation_prompt(
        self, 
        problem_statement: str,
        constraints: Optional[List[str]] = None,
        previous_attempts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build the generation prompt based on parameters."""
        # Implementation of prompt engineering
        # ...
```

### CriticCapability

The `CriticCapability` enables agents to critically analyze solutions and provide feedback:

```python
class CriticCapability(BaseCapability):
    """Capability for critically analyzing solutions."""
    
    class Parameters(BaseModel):
        """Parameters for critic capability."""
        solution: Dict[str, Any]
        evaluation_criteria: Optional[List[str]] = None
        previous_critiques: Optional[List[Dict[str, Any]]] = None
        critique_depth: Optional[str] = "standard"  # "light", "standard", "deep"
        
    class Response(BaseCapability.Response):
        """Response for critic capability."""
        critiques: Optional[List[Dict[str, str]]] = None  # aspect: critique
        questions: Optional[List[str]] = None
        strengths: Optional[List[str]] = None
        improvement_suggestions: Optional[List[str]] = None
        
    async def execute(self, parameters: Parameters) -> Response:
        """Execute the critical analysis with validated parameters."""
        # Implementation of critical analysis
        # ...
```

## Agent Types and Capability Composition

The platform supports various agent types through capability composition:

### AssistantAgent

The `AssistantAgent` provides general-purpose assistance:

```python
class AssistantAgent(BaseAgent):
    """General-purpose assistant agent."""
    
    def __init__(self, agent_id: str, service_registry: ServiceRegistry):
        """Initialize the assistant agent with required capabilities."""
        super().__init__(agent_id, service_registry)
        
        # Register core capabilities
        self.register_capability("search", SearchCapability(self))
        self.register_capability("summarize", SummarizeCapability(self))
        
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages and route to appropriate capabilities."""
        try:
            # Extract message components
            message_type = message.get("type", "")
            content = message.get("content", {})
            sender = message.get("sender", "")
            
            # Route to appropriate capability based on message type
            if message_type == "search":
                return await self.execute_capability("search", content)
                
            elif message_type == "summarize":
                return await self.execute_capability("summarize", content)
                
            else:
                # Handle unknown message types
                return {
                    "status": "error",
                    "error": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            # Handle any exceptions during message processing
            return {
                "status": "error",
                "error": f"Message processing failed: {str(e)}"
            }
```

### CreativeAgent

The `CreativeAgent` specializes in creative generation and refinement:

```python
class CreativeAgent(BaseAgent):
    """Agent specialized for creative generation and refinement."""
    
    def __init__(self, agent_id: str, service_registry: ServiceRegistry):
        """Initialize the creative agent with required capabilities."""
        super().__init__(agent_id, service_registry)
        
        # Register evolutionary capabilities
        self.register_capability("generate", GeneratorCapability(self))
        self.register_capability("refine", RefinerCapability(self))
        
    # Implementation of creative agent-specific methods
    # ...
```

## Capability Registry

The `CapabilityRegistry` manages discovery and access to capabilities:

```python
class CapabilityRegistry:
    """Registry for discovering and managing agent capabilities."""
    
    def __init__(self, service_registry: ServiceRegistry):
        """Initialize the capability registry."""
        self.service_registry = service_registry
        self.capabilities = {}
        
    def register_capability(
        self, 
        capability_name: str, 
        capability_class: Type[BaseCapability],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a capability type in the registry."""
        self.capabilities[capability_name] = {
            "class": capability_class,
            "metadata": metadata or {}
        }
        
    def get_capability_class(self, capability_name: str) -> Type[BaseCapability]:
        """Get a capability class by name."""
        if capability_name not in self.capabilities:
            raise ValueError(f"Capability not found: {capability_name}")
            
        return self.capabilities[capability_name]["class"]
        
    def create_capability_instance(
        self,
        capability_name: str,
        agent: BaseAgent
    ) -> BaseCapability:
        """Create an instance of a capability for an agent."""
        capability_class = self.get_capability_class(capability_name)
        return capability_class(agent)
        
    def list_available_capabilities(self) -> List[Dict[str, Any]]:
        """List all available capabilities with metadata."""
        return [
            {
                "name": name,
                "metadata": info["metadata"]
            }
            for name, info in self.capabilities.items()
        ]
```

## Capability Compatibility and Versioning

To ensure proper compatibility between agents and capabilities, the platform implements a versioning system:

```python
class CapabilityVersionManager:
    """Manages capability versioning and compatibility."""
    
    def __init__(self):
        """Initialize the capability version manager."""
        self.capability_versions = {}
        
    def register_capability_version(
        self,
        capability_name: str,
        version: str,
        capability_class: Type[BaseCapability]
    ) -> None:
        """Register a specific version of a capability."""
        if capability_name not in self.capability_versions:
            self.capability_versions[capability_name] = {}
            
        self.capability_versions[capability_name][version] = capability_class
        
    def get_capability_version(
        self,
        capability_name: str,
        version: str
    ) -> Type[BaseCapability]:
        """Get a specific version of a capability."""
        if capability_name not in self.capability_versions:
            raise ValueError(f"Capability not found: {capability_name}")
            
        if version not in self.capability_versions[capability_name]:
            raise ValueError(f"Version {version} not found for capability {capability_name}")
            
        return self.capability_versions[capability_name][version]
        
    def get_latest_version(self, capability_name: str) -> str:
        """Get the latest version of a capability."""
        if capability_name not in self.capability_versions:
            raise ValueError(f"Capability not found: {capability_name}")
            
        # Sort versions using semantic versioning
        versions = list(self.capability_versions[capability_name].keys())
        versions.sort(key=lambda v: [int(x) for x in v.split(".")])
        
        return versions[-1]
```

## Capability Discovery and Adaptation

The platform enables dynamic capability discovery and adaptation:

```python
class CapabilityDiscovery:
    """Discovers capabilities and adapts agents accordingly."""
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        capability_registry: CapabilityRegistry
    ):
        """Initialize the capability discovery service."""
        self.service_registry = service_registry
        self.capability_registry = capability_registry
        
    async def discover_capabilities(self, agent_id: str) -> List[Dict[str, Any]]:
        """Discover available capabilities for an agent."""
        # Implementation of capability discovery
        # ...
        
    async def adapt_agent(
        self,
        agent: BaseAgent,
        required_capabilities: List[str]
    ) -> bool:
        """Adapt an agent by adding required capabilities."""
        # Implementation of agent adaptation
        # ...
```

## Testing Strategy

The testing strategy for capabilities follows Test-Driven Development principles:

```python
# Example test for SummarizeCapability
class TestSummarizeCapability:
    """Tests for the SummarizeCapability."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        mock_agent = MagicMock()
        mock_agent.service_registry = MagicMock()
        mock_agent.trigger_event = AsyncMock()
        return mock_agent
        
    @pytest.fixture
    def mock_nlp_service(self):
        """Create a mock NLP service for testing."""
        mock_service = MagicMock()
        mock_service.summarize = AsyncMock()
        return mock_service
        
    @pytest.fixture
    def summarize_capability(self, mock_agent, mock_nlp_service):
        """Create a SummarizeCapability instance for testing."""
        capability = SummarizeCapability(mock_agent)
        mock_agent.service_registry.get_service.return_value = mock_nlp_service
        return capability
        
    @pytest.mark.asyncio
    async def test_validate_parameters_valid(self, summarize_capability):
        """Test parameter validation with valid parameters."""
        # Arrange
        params = {
            "content": "This is a test content that needs to be summarized.",
            "max_length": 50
        }
        
        # Act
        validated = summarize_capability.validate_parameters(params)
        
        # Assert
        assert validated.content == params["content"]
        assert validated.max_length == params["max_length"]
        assert validated.format == "paragraph"  # Default value
        
    @pytest.mark.asyncio
    async def test_validate_parameters_invalid(self, summarize_capability):
        """Test parameter validation with invalid parameters."""
        # Arrange
        params = {
            "max_length": 50  # Missing required 'content'
        }
        
        # Act & Assert
        with pytest.raises(ParameterValidationError):
            summarize_capability.validate_parameters(params)
            
    @pytest.mark.asyncio
    async def test_execute_success(self, summarize_capability, mock_nlp_service):
        """Test successful execution of summarize capability."""
        # Arrange
        params = SummarizeCapability.Parameters(
            content="This is a test content that needs to be summarized.",
            max_length=50
        )
        
        summary_result = MagicMock()
        summary_result.text = "This is a summary."
        mock_nlp_service.summarize.return_value = summary_result
        
        # Act
        result = await summarize_capability.execute(params)
        
        # Assert
        assert result.status == "success"
        assert result.summary == "This is a summary."
        assert result.original_length == len(params.content)
        assert result.summary_length == len("This is a summary.")
        
        # Verify service call
        mock_nlp_service.summarize.assert_called_once_with(
            content=params.content,
            max_length=params.max_length,
            format=params.format,
            focus=params.focus
        )
        
    @pytest.mark.asyncio
    async def test_execute_with_lifecycle(self, summarize_capability, mock_agent, mock_nlp_service):
        """Test execution with lifecycle events."""
        # Arrange
        params = {
            "content": "This is a test content that needs to be summarized.",
            "max_length": 50
        }
        
        summary_result = MagicMock()
        summary_result.text = "This is a summary."
        mock_nlp_service.summarize.return_value = summary_result
        
        # Act
        result = await summarize_capability.execute_with_lifecycle(params)
        
        # Assert
        assert result["status"] == "success"
        assert result["data"]["summary"] == "This is a summary."
        
        # Verify events
        mock_agent.trigger_event.assert_any_call(
            EventType.CAPABILITY_STARTED,
            capability_name="SummarizeCapability",
            parameters=params
        )
        
        mock_agent.trigger_event.assert_any_call(
            "summarize_completed",
            summary="This is a summary.",
            original_length=len(params["content"]),
            summary_length=len("This is a summary.")
        )
        
        mock_agent.trigger_event.assert_any_call(
            EventType.CAPABILITY_COMPLETED,
            capability_name="SummarizeCapability",
            parameters=params,
            result=ANY  # Check the structure more specifically if needed
        )
```

## Conclusion

The capability-based architecture provides a flexible foundation for agent functionality, enabling:

1. Clear separation of concerns through modular capabilities
2. Consistent parameter validation using Pydantic models
3. Standardized error handling through the centralized error framework
4. Uniform event lifecycle management
5. Extensibility through capability composition

This architecture supports the evolutionary framework by providing specialized capabilities for generation, critique, refinement, and evaluation, while maintaining the core platform principles of reliability, testability, and maintainability.
