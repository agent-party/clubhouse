"""
Database configuration models for the Clubhouse platform.

This module defines configuration models for various database systems, including
Neo4j, that the platform can connect to. These models can be used in conjunction
with the configuration system to validate and access database connection parameters.
"""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import cast, List, Dict, Any, Type


class DatabaseType(str, Enum):
    """Supported database types in the platform."""

    NEO4J = "neo4j"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
    MEMORY = "memory"  # In-memory database for testing


class DatabaseAuthType(str, Enum):
    """Authentication types for database connections."""

    BASIC = "basic"  # Username/password authentication
    TOKEN = "token"  # Token-based authentication
    NONE = "none"  # No authentication (e.g., for local development)
    CERT = "cert"  # Certificate-based authentication


class SSLConfig(BaseModel):
    """SSL configuration for secure database connections."""

    enabled: bool = Field(
        default=True, description="Whether SSL is enabled for this connection"
    )
    verify: bool = Field(
        default=True, description="Whether to verify SSL certificates"
    )
    ca_cert_path: Optional[str] = Field(
        default=None, description="Path to CA certificate file"
    )
    client_cert_path: Optional[str] = Field(
        default=None, description="Path to client certificate file"
    )
    client_key_path: Optional[str] = Field(
        default=None, description="Path to client key file"
    )

    @field_validator("ca_cert_path", "client_cert_path", "client_key_path")
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        """Validate certificate file paths."""
        if v is not None and not v:
            raise ValueError("Certificate paths cannot be empty strings")
        return v

    model_config = ConfigDict(
        extra="forbid"
    )


class ConnectionPoolConfig(BaseModel):
    """
    Connection pool configuration for database connections.
    
    This model defines settings for database connection pooling,
    which is important for managing resources efficiently.
    """

    max_size: int = Field(
        default=10,
        description="Maximum number of connections in the pool",
        ge=1,
        le=1000,
    )
    min_size: int = Field(
        default=1,
        description="Minimum number of connections to maintain in the pool",
        ge=0,
    )
    max_idle_time_seconds: int = Field(
        default=600,
        description="Maximum time in seconds a connection can remain idle before being closed",
        ge=0,
    )
    connection_timeout_seconds: int = Field(
        default=30,
        description="Maximum time in seconds to wait for a connection from the pool",
        ge=1,
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed connection acquisitions",
        ge=0,
    )
    
    @field_validator("min_size")
    @classmethod
    def validate_min_size(cls, v: int, values: Dict) -> int:
        """Validate that min_size is less than or equal to max_size."""
        if "max_size" in values.data and v > values.data["max_size"]:  # type: ignore[missing_attribute]
            raise ValueError("min_size must be less than or equal to max_size")
        return v

    model_config = ConfigDict(
        extra="forbid"
    )


class BaseDatabaseConfig(BaseModel):
    """
    Base configuration for database connections.
    
    This abstract base model defines common properties for all database types.
    Specific database implementations should inherit from this class.
    """

    type: DatabaseType = Field(..., description="Type of the database")
    name: str = Field(..., description="Name of this database configuration")
    description: Optional[str] = Field(
        default=None, description="Description of this database configuration"
    )
    enabled: bool = Field(
        default=True, description="Whether this database connection is enabled"
    )
    auth_type: DatabaseAuthType = Field(
        default=DatabaseAuthType.BASIC,
        description="Authentication method for this database",
    )
    username: Optional[str] = Field(
        default=None, description="Username for database authentication"
    )
    password: Optional[str] = Field(
        default=None, description="Password for database authentication"
    )
    auth_token: Optional[str] = Field(
        default=None, description="Authentication token for database authentication"
    )
    connection_pool: ConnectionPoolConfig = Field(
        default_factory=ConnectionPoolConfig,
        description="Connection pool configuration",
    )
    ssl: SSLConfig = Field(
        default_factory=SSLConfig, description="SSL configuration"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate the database configuration name."""
        if not v:
            raise ValueError("Database configuration name cannot be empty")
        if len(v) < 3:
            raise ValueError("Database configuration name must be at least 3 characters")
        return v

    @field_validator("username", "password")
    @classmethod
    def validate_credentials(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Validate credentials based on auth_type."""
        if values.data.get("auth_type") == DatabaseAuthType.BASIC:  # type: ignore[missing_attribute]
            if v is None:
                raise ValueError(
                    "Username and password must be provided for basic authentication"
                )
        return v

    @field_validator("auth_token")
    @classmethod
    def validate_token(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Validate token based on auth_type."""
        if values.data.get("auth_type") == DatabaseAuthType.TOKEN:  # type: ignore[missing_attribute]
            if v is None:
                raise ValueError("Auth token must be provided for token authentication")
        return v

    model_config = ConfigDict(
        extra="forbid"
    )


class Neo4jDatabaseConfig(BaseDatabaseConfig):
    """
    Neo4j specific database configuration.
    
    This model extends the base database configuration with Neo4j specific
    properties such as bolt URL, protocol version, and query timeout.
    """

    type: DatabaseType = Field(
        default=DatabaseType.NEO4J, description="Type of the database"
    )
    hosts: List[str] = Field(
        ..., description="List of Neo4j host addresses (with optional port)"
    )
    database: str = Field(default="neo4j", description="Neo4j database name")
    protocol_version: Optional[float] = Field(
        default=None, description="Neo4j Bolt protocol version"
    )
    query_timeout_seconds: int = Field(
        default=60, description="Timeout for Neo4j queries in seconds", ge=1
    )
    max_transaction_retry_time_seconds: int = Field(
        default=30,
        description="Maximum time to retry transactions on transient errors",
        ge=0,
    )
    encryption: bool = Field(
        default=True, description="Whether to use encryption for the connection"
    )
    trust: str = Field(
        default="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES",
        description="Trust strategy for certificates",
    )
    keep_alive: bool = Field(
        default=True, description="Whether to use TCP keep alive for the connection"
    )
    
    @field_validator("hosts")
    @classmethod
    def validate_hosts(cls, v: List[str]) -> List[str]:
        """Validate Neo4j host addresses."""
        if not v:
            raise ValueError("At least one Neo4j host must be provided")
        for host in v:
            if not host:
                raise ValueError("Neo4j host cannot be empty")
        return v

    model_config = ConfigDict(
        extra="forbid"
    )


class DatabaseConfig(BaseModel):
    """
    Configuration for all database connections in the system.
    
    This model defines a collection of database configurations,
    each with a unique name that can be referenced by services.
    """

    databases: Dict[str, Union[Neo4jDatabaseConfig]] = Field(
        default_factory=dict,
        description="Dictionary of database configurations by name",
    )
    default_database: Optional[str] = Field(
        default=None, description="Name of the default database configuration to use"
    )
    
    @field_validator("default_database")
    @classmethod
    def validate_default_database(cls, v: Optional[str], values: Dict) -> Optional[str]:
        """Validate that the default database exists in the databases dictionary."""
        if v is not None and v not in values.data.get("databases", {}):  # type: ignore[missing_attribute]
            raise ValueError(f"Default database '{v}' not found in configured databases")
        return v

    model_config = ConfigDict(
        extra="forbid"
    )


# Example of usage:
"""
database_config = DatabaseConfig(
    databases={
        "neo4j_main": Neo4jDatabaseConfig(
            name="neo4j_main",
            description="Main Neo4j database for agent data",
            hosts=["localhost:7687"],
            database="neo4j",
            username="neo4j",
            password="password",
            connection_pool=ConnectionPoolConfig(
                max_size=20,
                min_size=2,
            ),
        ),
    },
    default_database="neo4j_main",
)
"""