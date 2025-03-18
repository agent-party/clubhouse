# OpenAI Agent Implementation

## Overview

The OpenAI Agent Implementation provides a standardized framework for integrating OpenAI's large language models into the Agent Orchestration Platform. This document outlines the architecture, implementation patterns, and integration details for leveraging OpenAI's capabilities effectively and cost-efficiently.

## Core Principles

1. **Model Abstraction**: Abstract OpenAI model specifics to allow flexible model selection and fallbacks
2. **Prompt Engineering**: Implement systematic prompt construction with templating and versioning
3. **Token Optimization**: Maximize information density while minimizing token usage
4. **Cost Management**: Implement fine-grained tracking and budgeting for API usage
5. **Response Quality**: Ensure consistent, high-quality outputs through prompt design and validation

## Architecture Components

### 1. OpenAI Service Layer

```
┌────────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                    │       │                 │       │                 │
│ Agent Capability   │──────▶│ OpenAI Service  │──────▶│ OpenAI API      │
│                    │       │ Protocol        │       │                 │
└────────────────────┘       └─────────────────┘       └─────────────────┘
```

The OpenAI Service Layer provides a clean abstraction for model interactions:

- **OpenAI Service Protocol**: Defines interface for LLM operations
- **OpenAI Service Implementation**: Handles API communication and error handling
- **Model Configuration**: Manages model-specific settings and parameters

### 2. Prompt Management System

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│ Prompt Templates│────▶│  Context Builder  │────▶│   Template        │
│                 │     │                   │     │   Processor       │
└─────────────────┘     └───────────────────┘     └───────────────────┘
```

The Prompt Management System ensures consistent and effective prompts:

- **Prompt Templates**: Versioned templates for different use cases
- **Context Builder**: Assembles relevant context for prompts
- **Template Processor**: Fills templates with context and parameters

### 3. Response Processing

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Response Parsing  │────▶│ Response Validation│───▶│ Response          │
│                   │     │                   │     │ Transformation    │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Response Processing system handles model outputs:

- **Response Parsing**: Extracts structured data from responses
- **Response Validation**: Ensures responses meet quality criteria
- **Response Transformation**: Converts responses to domain-specific formats

### 4. Token Management

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Token Counter     │────▶│ Budget Enforcer   │────▶│ Usage Reporter    │
│                   │     │                   │     │                   │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Token Management system tracks and controls token usage:

- **Token Counter**: Estimates and tracks actual token usage
- **Budget Enforcer**: Enforces token budget constraints
- **Usage Reporter**: Reports token usage for billing and analytics

## Implementation Details

### Data Models

```python
class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API."""
    api_key: str
    organization: Optional[str] = None
    default_model: str = "gpt-4o"
    timeout: int = 60  # seconds
    max_retries: int = 3
    
class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    model_id: str
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 1000
    stop_sequences: Optional[List[str]] = None
    
class PromptTemplate(BaseModel):
    """Template for constructing prompts."""
    id: str
    version: str
    template: str
    variables: List[str]
    description: str
    model_defaults: Dict[str, Dict[str, Any]] = {}
    token_estimate: Optional[int] = None
    
class TokenUsage(BaseModel):
    """Token usage for a single OpenAI API call."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_id: str
    timestamp: datetime
    operation_id: str
    agent_id: str
    capability_type: str
```

### Service Interface

```python
class OpenAIServiceProtocol(Protocol):
    """Protocol for OpenAI service operations."""
    
    async def completion(
        self,
        prompt: str,
        model_config: Optional[ModelConfig] = None,
        operation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        capability_type: Optional[str] = None
    ) -> Tuple[str, TokenUsage]:
        """Generate a completion for the given prompt."""
        ...
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_config: Optional[ModelConfig] = None,
        operation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        capability_type: Optional[str] = None
    ) -> Tuple[str, TokenUsage]:
        """Generate a chat completion for the given messages."""
        ...
    
    async def structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        model_config: Optional[ModelConfig] = None,
        operation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        capability_type: Optional[str] = None
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        """Generate a structured output for the given prompt."""
        ...
    
    async def embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
        operation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        capability_type: Optional[str] = None
    ) -> Tuple[List[float], TokenUsage]:
        """Generate an embedding for the given text."""
        ...
```

### Prompt Template Management

```python
class PromptManager:
    """Manages prompt templates and rendering."""
    
    def __init__(self, template_repository: PromptTemplateRepository):
        """Initialize the prompt manager."""
        self.template_repository = template_repository
        self.template_cache: Dict[str, Dict[str, PromptTemplate]] = {}
    
    async def get_template(self, template_id: str, version: Optional[str] = None) -> PromptTemplate:
        """Get a prompt template by ID and version."""
        # Check cache first
        if template_id in self.template_cache and (
            not version or version in self.template_cache[template_id]
        ):
            return self.template_cache[template_id][version or "latest"]
        
        # Fetch from repository
        template = await self.template_repository.get(template_id, version)
        
        # Update cache
        if template_id not in self.template_cache:
            self.template_cache[template_id] = {}
        
        self.template_cache[template_id][template.version] = template
        if not version:
            self.template_cache[template_id]["latest"] = template
        
        return template
    
    async def render_prompt(
        self,
        template_id: str,
        variables: Dict[str, Any],
        version: Optional[str] = None
    ) -> str:
        """Render a prompt template with variables."""
        template = await self.get_template(template_id, version)
        
        # Validate variables
        missing_vars = [var for var in template.variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing variables for template: {', '.join(missing_vars)}")
        
        # Render template
        prompt = template.template
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(var_value))
        
        return prompt
```

### Token Budget Management

```python
class TokenBudgetManager:
    """Manages token budgets for OpenAI API calls."""
    
    def __init__(
        self,
        token_usage_repository: TokenUsageRepository,
        budget_repository: BudgetRepository
    ):
        """Initialize the token budget manager."""
        self.token_usage_repository = token_usage_repository
        self.budget_repository = budget_repository
    
    async def check_budget(
        self,
        agent_id: str,
        estimated_tokens: int,
        model_id: str
    ) -> bool:
        """Check if the estimated token usage is within budget."""
        # Get budget for agent
        budget = await self.budget_repository.get_for_agent(agent_id)
        if not budget:
            # No budget defined, allow by default
            return True
        
        # Get current usage
        current_usage = await self.token_usage_repository.get_agent_usage(
            agent_id,
            start_date=budget.period_start,
            end_date=budget.period_end
        )
        
        # Calculate cost of estimated tokens
        cost_calculator = ModelCostCalculator()
        estimated_cost = cost_calculator.calculate_cost(
            model_id,
            estimated_tokens,
            0  # Completion tokens unknown at this stage
        )
        
        # Check if within budget
        return current_usage.total_cost + estimated_cost <= budget.amount
    
    async def record_usage(self, token_usage: TokenUsage) -> None:
        """Record token usage."""
        # Store usage
        await self.token_usage_repository.create(token_usage)
        
        # Calculate cost
        cost_calculator = ModelCostCalculator()
        cost = cost_calculator.calculate_cost(
            token_usage.model_id,
            token_usage.prompt_tokens,
            token_usage.completion_tokens
        )
        
        # Create cost entry
        cost_entry = CostEntry(
            token_usage_id=token_usage.operation_id,
            prompt_cost=cost.prompt_cost,
            completion_cost=cost.completion_cost,
            total_cost=cost.total_cost,
            currency="USD"
        )
        
        # Store cost entry
        await self.cost_repository.create(cost_entry)
        
        # Check if budget warning needed
        await self._check_budget_warning(token_usage.agent_id)
    
    async def _check_budget_warning(self, agent_id: str) -> None:
        """Check if a budget warning should be issued."""
        # Get budget for agent
        budget = await self.budget_repository.get_for_agent(agent_id)
        if not budget:
            return
        
        # Get current usage
        current_usage = await self.token_usage_repository.get_agent_usage(
            agent_id,
            start_date=budget.period_start,
            end_date=budget.period_end
        )
        
        # Check warning threshold
        usage_percentage = current_usage.total_cost / budget.amount
        if usage_percentage >= budget.warning_threshold:
            # Publish warning event
            await self.event_bus.publish(
                topic="budget.warning",
                value={
                    "agent_id": agent_id,
                    "budget_id": budget.budget_id,
                    "current_usage": current_usage.total_cost,
                    "budget_amount": budget.amount,
                    "percentage_used": usage_percentage,
                    "timestamp": datetime.now().isoformat()
                },
                key=agent_id
            )
```

## Model-Specific Optimizations

### GPT-4 Optimizations

```python
class GPT4Optimizer:
    """Optimizations specific to GPT-4 model."""
    
    @staticmethod
    def optimize_prompt(prompt: str) -> str:
        """Optimize a prompt for GPT-4."""
        # Remove redundant whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = re.sub(r'\n\s*\n', '\n\n', prompt)
        
        # Add focus markers for important sections
        prompt = re.sub(r'(#+ .+)\n', r'***\1***\n', prompt)
        
        return prompt
    
    @staticmethod
    def get_recommended_params(task_type: str) -> Dict[str, Any]:
        """Get recommended parameters for GPT-4 based on task type."""
        params = {
            "creative": {
                "temperature": 0.9,
                "top_p": 1.0,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.5,
            },
            "analytical": {
                "temperature": 0.2,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "factual": {
                "temperature": 0.1,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "code": {
                "temperature": 0.2,
                "top_p": 0.95,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
        }
        
        return params.get(task_type, params["analytical"])
```

### OpenAI Function Calling

```python
class FunctionCallingHandler:
    """Handles OpenAI function calling."""
    
    def __init__(self, function_registry: Dict[str, Callable]):
        """Initialize the function calling handler."""
        self.function_registry = function_registry
    
    def prepare_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare functions for OpenAI API."""
        return [
            {
                "name": func["name"],
                "description": func["description"],
                "parameters": {
                    "type": "object",
                    "properties": func["parameters"],
                    "required": func.get("required", [])
                }
            }
            for func in functions
        ]
    
    async def handle_function_call(
        self,
        function_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a function call from OpenAI."""
        function_name = function_call["name"]
        arguments = json.loads(function_call["arguments"])
        
        if function_name not in self.function_registry:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Execute function
        function = self.function_registry[function_name]
        result = await function(**arguments)
        
        return {
            "function_name": function_name,
            "result": result
        }
```

## Integration with Existing Components

### 1. Capability Implementation

OpenAI integration with agent capabilities:

```python
class ReasoningCapability(BaseCapability):
    """Capability for reasoning using OpenAI models."""
    
    def __init__(
        self,
        agent_id: str,
        service_registry: ServiceRegistry,
        event_bus: EventBusProtocol,
        capability_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the reasoning capability."""
        super().__init__(agent_id, service_registry, event_bus, capability_config)
        self.openai_service = service_registry.get(OpenAIServiceProtocol)
        self.prompt_manager = service_registry.get(PromptManager)
        self.token_budget_manager = service_registry.get(TokenBudgetManager)
    
    async def execute(self, params: ReasoningParams) -> ReasoningResult:
        """Execute reasoning capability."""
        operation_id = str(uuid.uuid4())
        
        # Create model config
        model_config = ModelConfig(
            model_id=params.model or self.config.get("default_model", "gpt-4o"),
            temperature=params.temperature or 0.7,
            max_tokens=params.max_tokens or 1000
        )
        
        # Estimate token usage
        tokenizer = Tokenizer()
        estimated_tokens = tokenizer.count_tokens(params.prompt)
        
        # Check budget
        within_budget = await self.token_budget_manager.check_budget(
            self.agent_id, estimated_tokens, model_config.model_id
        )
        
        if not within_budget:
            raise BudgetExceededException("Token budget exceeded")
        
        # Generate completion
        response, token_usage = await self.openai_service.completion(
            prompt=params.prompt,
            model_config=model_config,
            operation_id=operation_id,
            agent_id=self.agent_id,
            capability_type="reasoning"
        )
        
        # Record token usage
        await self.token_budget_manager.record_usage(token_usage)
        
        # Create result
        return ReasoningResult(
            operation_id=operation_id,
            reasoning=response,
            model=model_config.model_id,
            usage=TokenUsageInfo(
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens
            )
        )
```

### 2. Event Bus Integration

OpenAI-related events are published to the event bus:

```python
# Event types
MODEL_CALL_STARTED = "openai.model.call_started"
MODEL_CALL_COMPLETED = "openai.model.call_completed"
MODEL_CALL_FAILED = "openai.model.call_failed"
BUDGET_WARNING = "openai.budget.warning"
BUDGET_EXCEEDED = "openai.budget.exceeded"

# Publishing a model call event
async def _publish_model_call_event(
    self,
    event_type: str,
    model_id: str,
    operation_id: str,
    agent_id: Optional[str],
    capability_type: Optional[str],
    token_usage: Optional[TokenUsage] = None,
    error: Optional[Exception] = None
) -> None:
    """Publish a model call event."""
    event_data = {
        "model_id": model_id,
        "operation_id": operation_id,
        "agent_id": agent_id,
        "capability_type": capability_type,
        "timestamp": datetime.now().isoformat()
    }
    
    if token_usage:
        event_data["token_usage"] = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens
        }
    
    if error:
        event_data["error"] = str(error)
        event_data["error_type"] = type(error).__name__
    
    await self.event_bus.publish(
        topic=event_type,
        value=event_data,
        key=operation_id
    )
```

### 3. Cost Management Integration

OpenAI cost tracking is integrated with the cost management system:

```python
class ModelCostCalculator:
    """Calculates cost for OpenAI model usage."""
    
    # Current pricing as of 2023
    MODEL_PRICING = {
        "gpt-4o": {
            "prompt": 0.01 / 1000,  # $0.01 per 1K tokens
            "completion": 0.03 / 1000  # $0.03 per 1K tokens
        },
        "gpt-4": {
            "prompt": 0.03 / 1000,  # $0.03 per 1K tokens
            "completion": 0.06 / 1000  # $0.06 per 1K tokens
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0015 / 1000,  # $0.0015 per 1K tokens
            "completion": 0.002 / 1000  # $0.002 per 1K tokens
        },
        "text-embedding-ada-002": {
            "prompt": 0.0001 / 1000,  # $0.0001 per 1K tokens
            "completion": 0.0  # No completion tokens for embeddings
        }
    }
    
    def calculate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> CostDetails:
        """Calculate cost for token usage."""
        # Get pricing for model
        model_pricing = self.MODEL_PRICING.get(model_id, self.MODEL_PRICING["gpt-3.5-turbo"])
        
        # Calculate costs
        prompt_cost = prompt_tokens * model_pricing["prompt"]
        completion_cost = completion_tokens * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        return CostDetails(
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost
        )
```

### 4. Agent Prompt Integration

System prompts are tailored for cost efficiency:

```
You are an AI assistant operating with budget constraints. To use resources efficiently:

1. Be concise - avoid unnecessary elaboration
2. Use the minimum context window necessary to complete tasks
3. Prioritize resource-efficient capabilities when multiple options exist
4. Consider cost-benefit trade-offs for computationally expensive operations

When generating content:
1. Focus on addressing the core request first
2. Avoid repeating information already provided
3. Use efficient formatting and structure
4. Minimize token usage while maintaining quality
```

## Error Handling and Resilience

### 1. Retry Strategy

```python
class OpenAIRetryStrategy:
    """Retry strategy for OpenAI API calls."""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1
    ):
        """Initialize the retry strategy."""
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    async def execute_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        retryable_errors: List[Type[Exception]] = None
    ) -> T:
        """Execute an operation with retries."""
        if retryable_errors is None:
            retryable_errors = [
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError
            ]
        
        last_exception = None
        backoff = self.initial_backoff
        
        for retry_count in range(self.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not any(isinstance(e, err_type) for err_type in retryable_errors):
                    logger.warning(f"Non-retryable error: {e}")
                    raise
                
                if retry_count == self.max_retries:
                    logger.error(f"Maximum retries reached: {e}")
                    raise
                
                # Apply jitter to backoff
                jitter_amount = backoff * self.jitter * (random.random() * 2 - 1)
                sleep_time = backoff + jitter_amount
                
                logger.warning(
                    f"Retryable error on attempt {retry_count + 1}/{self.max_retries + 1}: "
                    f"{e}. Retrying in {sleep_time:.2f}s"
                )
                
                await asyncio.sleep(sleep_time)
                backoff *= self.backoff_factor
        
        # This should never happen, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic failed in an unexpected way")
```

### 2. Error Classification

```python
class OpenAIErrorClassifier:
    """Classifies OpenAI API errors."""
    
    ERROR_CATEGORIES = {
        "rate_limit": [
            openai.RateLimitError,
        ],
        "timeout": [
            openai.APITimeoutError,
            asyncio.TimeoutError
        ],
        "connection": [
            openai.APIConnectionError,
            ConnectionError,
            socket.error
        ],
        "validation": [
            openai.InvalidRequestError,
            ValueError
        ],
        "authentication": [
            openai.AuthenticationError
        ],
        "permission": [
            openai.PermissionError
        ],
        "server": [
            openai.APIError
        ]
    }
    
    @classmethod
    def classify_error(cls, error: Exception) -> str:
        """Classify an error into a category."""
        for category, error_types in cls.ERROR_CATEGORIES.items():
            if any(isinstance(error, err_type) for err_type in error_types):
                return category
        
        return "unknown"
    
    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        """Determine if an error is retryable."""
        category = cls.classify_error(error)
        return category in ["rate_limit", "timeout", "connection", "server"]
    
    @classmethod
    def get_user_message(cls, error: Exception) -> str:
        """Get a user-friendly message for an error."""
        category = cls.classify_error(error)
        
        messages = {
            "rate_limit": "The AI service is currently experiencing high demand. Please try again in a moment.",
            "timeout": "The request to the AI service timed out. Please try again.",
            "connection": "There was a problem connecting to the AI service. Please check your internet connection.",
            "validation": "The request to the AI service was invalid. Please check your inputs.",
            "authentication": "There was a problem authenticating with the AI service.",
            "permission": "You don't have permission to use this AI service feature.",
            "server": "The AI service is currently experiencing technical difficulties. Please try again later.",
            "unknown": "An unexpected error occurred with the AI service. Please try again."
        }
        
        return messages.get(category, str(error))
```

## Testing Strategy

Following our test-driven development approach, we implement:

1. **Unit Tests**:
   - Test prompt rendering with different templates
   - Validate token counting accuracy
   - Test error handling and retry logic

2. **Integration Tests**:
   - Test against OpenAI API sandbox environments
   - Verify end-to-end capability execution
   - Test budget enforcement with simulated constraints

3. **Performance Tests**:
   - Measure latency under different loads
   - Test concurrent API calls with connection pooling
   - Validate token counting performance with large inputs

## Future Enhancements

1. **Advanced Prompt Techniques**:
   - Implement Chain-of-thought prompting
   - Add few-shot learning templates
   - Develop self-critique prompting patterns

2. **Model Evaluation Framework**:
   - Automated evaluation of model responses
   - A/B testing of prompt templates
   - Systematic model comparison

3. **Cross-Model Compatibility**:
   - Abstract interfaces for multiple LLM providers
   - Prompt adaptation for different models
   - Model-specific optimization strategies

## Conclusion

The OpenAI Agent Implementation provides a robust and flexible foundation for integrating OpenAI's language models into the Agent Orchestration Platform. By implementing proper abstraction, prompt management, token optimization, and error handling, the system ensures reliable, cost-effective, and high-quality AI capabilities.
