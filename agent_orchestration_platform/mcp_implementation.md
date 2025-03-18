# MCP (Machine Coding Project) Implementation

## Overview

The MCP implementation provides integration with GitHub services, enabling agents to interact with code repositories as part of their capabilities. This document outlines the architecture, implementation details, and integration patterns for the MCP system within the Agent Orchestration Platform.

## Core Principles

1. **Security First**: All GitHub interactions adhere to strict security protocols
2. **Least Privilege**: Agents only receive the permissions they need for their specific tasks
3. **Auditability**: All code changes are tracked and attributable
4. **Graceful Degradation**: System handles API limits and failures robustly
5. **Version Control Awareness**: Agents understand Git workflows and best practices

## Architecture Components

### 1. GitHub Interface Layer

```
┌────────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                    │       │                 │       │                 │
│ Agent Capability   │──────▶│ MCP Protocol    │──────▶│ GitHub API      │
│                    │       │                 │       │ Client          │
└────────────────────┘       └─────────────────┘       └─────────────────┘
```

The GitHub Interface Layer manages all interactions with GitHub:

- **MCP Protocol**: Defines the interface for GitHub operations
- **GitHub API Client**: Handles authentication and rate limiting
- **Operation Handlers**: Specialized modules for different GitHub operations

### 2. Code Understanding System

```
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                 │     │                   │     │                   │
│ Code Fetcher    │────▶│  Parser & Indexer │────▶│   Code Context    │
│                 │     │                   │     │   Builder         │
└─────────────────┘     └───────────────────┘     └───────────────────┘
```

The Code Understanding System processes repository content:

- **Code Fetcher**: Retrieves files and directories from repositories
- **Parser & Indexer**: Creates searchable indexes of code artifacts
- **Code Context Builder**: Constructs relevant context for agent operations

### 3. Version Control Management

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Branch Management │────▶│ Commit Generation │────▶│ Pull Request      │
│                   │     │                   │     │ Handler           │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Version Control Management system handles Git operations:

- **Branch Management**: Creates and tracks branches for code changes
- **Commit Generation**: Generates meaningful commit messages and changes
- **Pull Request Handler**: Creates and manages pull requests

### 4. Code Quality Assurance

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ Linting & Static  │────▶│ Test Generation   │────▶│ CI Integration    │
│ Analysis          │     │ & Execution       │     │                   │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
```

The Code Quality Assurance system ensures code quality:

- **Linting & Static Analysis**: Performs code quality checks
- **Test Generation & Execution**: Creates and runs tests for code changes
- **CI Integration**: Interfaces with CI/CD systems to validate changes

## Implementation Details

### Data Models

```python
class GitHubAuth(BaseModel):
    """Authentication details for GitHub."""
    access_token: str
    installation_id: Optional[int] = None
    auth_type: Literal["personal", "app", "oauth"] = "personal"
    
class RepositoryReference(BaseModel):
    """Reference to a GitHub repository."""
    owner: str
    repo: str
    branch: Optional[str] = None
    
class FileChange(BaseModel):
    """Represents a change to a file in a repository."""
    path: str
    content: str
    operation: Literal["create", "update", "delete"] = "update"
    message: Optional[str] = None
    
class PullRequestDetails(BaseModel):
    """Details for creating a pull request."""
    title: str
    body: str
    base_branch: str
    head_branch: str
    draft: bool = False
    maintainer_can_modify: bool = True
```

### Service Interfaces

```python
class MCPServiceProtocol(Protocol):
    """Protocol for interacting with GitHub repositories."""
    
    async def get_file_contents(
        self, repo_ref: RepositoryReference, path: str
    ) -> str:
        """Get the contents of a file from a repository."""
        ...
    
    async def create_or_update_file(
        self, repo_ref: RepositoryReference, file_change: FileChange
    ) -> str:
        """Create or update a file in a repository."""
        ...
    
    async def create_branch(
        self, repo_ref: RepositoryReference, branch_name: str, from_branch: Optional[str] = None
    ) -> bool:
        """Create a new branch in a repository."""
        ...
    
    async def create_pull_request(
        self, repo_ref: RepositoryReference, pr_details: PullRequestDetails
    ) -> str:
        """Create a pull request in a repository."""
        ...
    
    async def create_issue(
        self, repo_ref: RepositoryReference, title: str, body: str, labels: Optional[List[str]] = None
    ) -> int:
        """Create an issue in a repository."""
        ...
```

### Neo4j Schema

```
(:GitHubRepository {
    owner: string,
    repo: string,
    default_branch: string,
    description: string,
    is_private: boolean,
    last_accessed: datetime
})

(:GitHubBranch {
    name: string,
    sha: string,
    is_default: boolean,
    created_at: datetime,
    last_updated: datetime
})

(:GitHubPullRequest {
    pr_number: int,
    title: string,
    body: string,
    state: string,
    created_at: datetime,
    updated_at: datetime,
    merge_state: string
})

(:GitHubCommit {
    sha: string,
    message: string,
    author: string,
    created_at: datetime
})

// RELATIONSHIPS
(a:Agent)-[:CREATED]->(b:GitHubBranch)
(a:Agent)-[:AUTHORED]->(c:GitHubCommit)
(a:Agent)-[:OPENED]->(p:GitHubPullRequest)
(b:GitHubBranch)-[:BELONGS_TO]->(r:GitHubRepository)
(c:GitHubCommit)-[:COMMITTED_TO]->(b:GitHubBranch)
(p:GitHubPullRequest)-[:TARGETS]->(r:GitHubRepository)
(p:GitHubPullRequest)-[:MERGES]->(b:GitHubBranch)
```

## Integration with Existing Components

### 1. Capability Integration

The MCP system provides specialized capabilities for repository interaction:

```python
class GitHubRepositoryCapability(BaseCapability):
    """Capability for interacting with GitHub repositories."""
    
    async def execute(self, params: GitHubRepositoryParams) -> GitHubRepositoryResult:
        # Track token usage and cost
        token_tracker = self.service_registry.get(TokenTrackingService)
        tracking_id = token_tracker.start_tracking(
            agent_id=self.agent_id,
            capability_type="github_repository"
        )
        
        try:
            # Get MCP service
            mcp_service = self.service_registry.get(MCPServiceProtocol)
            
            # Execute repository operation
            result = await self._perform_repository_operation(
                mcp_service, params
            )
            
            # Record token usage
            token_tracker.record_usage(
                tracking_id=tracking_id,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                model_id=params.model
            )
            
            return result
        except Exception as e:
            # Record error
            token_tracker.record_error(tracking_id)
            raise
```

### 2. Event Bus Integration

MCP-related events are published to the event bus:

```python
# Event types
REPOSITORY_CLONED = "github.repository.cloned"
BRANCH_CREATED = "github.branch.created"
PULL_REQUEST_CREATED = "github.pull_request.created"
CODE_CHANGE_COMMITTED = "github.code.committed"

# Publishing a code change event
event_bus.publish(
    topic=CODE_CHANGE_COMMITTED,
    value={
        "repository": {
            "owner": repo_ref.owner,
            "repo": repo_ref.repo,
        },
        "branch": branch_name,
        "files_changed": [file.path for file in file_changes],
        "commit_sha": commit_result.sha,
        "commit_message": commit_message,
        "agent_id": agent_id,
        "timestamp": datetime.now().isoformat()
    }
)
```

### 3. Agent System Prompt Integration

GitHub awareness is embedded in agent system prompts:

```
When interacting with GitHub repositories:

1. Follow Git best practices for branching and committing
2. Create meaningful commit messages that explain WHY changes were made
3. Keep pull requests focused on a single concern
4. Include comprehensive test coverage for code changes
5. Follow the repository's contribution guidelines
6. Consider code readability and maintainability in all changes
```

### 4. Cost Management Integration

GitHub API usage is tracked for cost management:

```python
class GitHubAPIUsage(BaseModel):
    """Tracks GitHub API usage for cost accounting."""
    operation_type: str  # e.g., "get_file", "create_branch"
    api_calls: int
    resource_path: str  # e.g., "/repos/{owner}/{repo}/contents/{path}"
    agent_id: str
    timestamp: datetime
    
# Recording API usage
cost_tracker = service_registry.get(CostTrackingService)
cost_tracker.record_api_usage(
    GitHubAPIUsage(
        operation_type="create_pull_request",
        api_calls=1,
        resource_path=f"/repos/{repo_ref.owner}/{repo_ref.repo}/pulls",
        agent_id=agent_id,
        timestamp=datetime.now()
    )
)
```

## Security Considerations

1. **Token Management**:
   - Tokens stored in encrypted format
   - Rotation policies for access tokens
   - Least privilege access to repositories

2. **Access Control**:
   - Fine-grained permissions for repository operations
   - Approval workflows for sensitive operations
   - Audit logs for all GitHub interactions

3. **Data Protection**:
   - No storage of sensitive repository content
   - Scanning for accidental credential commits
   - Secure handling of private repository content

## Rate Limiting and Resilience

1. **Smart Retries**:
   - Exponential backoff for rate limit errors
   - Circuit breakers for failing endpoints
   - Request prioritization during limited availability

2. **Usage Tracking**:
   - Monitoring of rate limit consumption
   - Predictive throttling to prevent limit exhaustion
   - Fair allocation of API quota across agents

3. **Graceful Degradation**:
   - Alternative operation modes during limited availability
   - Queuing mechanisms for non-time-sensitive operations
   - Clear feedback on rate limit status

## Testing Strategy

Following our test-driven development approach, we implement:

1. **Unit Tests**:
   - Mock GitHub API responses
   - Test handling of various API error conditions
   - Validate proper construction of API requests

2. **Integration Tests**:
   - Test against GitHub API sandbox environments
   - Verify end-to-end workflows using test repositories
   - Test rate limit handling with simulated constraints

3. **Security Tests**:
   - Validate token handling and storage
   - Test permission enforcement
   - Verify proper error handling of unauthorized operations

## Future Enhancements

1. **Advanced Code Understanding**:
   - Semantic code understanding capabilities
   - Repository-specific style and pattern learning
   - Automated code review suggestions

2. **Workflow Integration**:
   - GitHub Actions integration
   - Project management feature integration
   - Release management automation

3. **Multi-Provider Support**:
   - Extension to GitLab, Bitbucket, and other platforms
   - Unified interface across Git providers
   - Provider-specific optimization

## Conclusion

The MCP implementation enables secure and efficient interaction with GitHub repositories, allowing agents to participate in code development workflows. By providing a standardized interface for repository operations with proper security, rate limiting, and error handling, the system ensures reliable and cost-effective integration with GitHub services.
