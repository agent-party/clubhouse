"""
Main application configuration model.
"""

from pydantic import BaseModel, ConfigDict, Field

from clubhouse.core.config.models.kafka_config import KafkaConfig
from clubhouse.core.config.models.mcp_config import MCPConfig
from clubhouse.core.config.models.schema_registry_config import SchemaRegistryConfig


class AppConfig(BaseModel):
    """
    Main application configuration.

    This is the root configuration object that contains all other configuration sections.

    Attributes:
        mcp: MCP server configuration
        kafka: Kafka integration configuration
        schema_registry: Schema Registry integration configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )

    mcp: MCPConfig = Field(default_factory=MCPConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    schema_registry: SchemaRegistryConfig = Field(default_factory=SchemaRegistryConfig)
