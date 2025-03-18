"""
Schema Registry configuration model.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing import cast, List, Dict, Any, Type


class SchemaRegistryConfig(BaseModel):
    """
    Configuration for Schema Registry integration.

    Attributes:
        url: URL of the Schema Registry server
        basic_auth_user_info: Basic auth credentials in format username:password
    """

    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )

    url: str = Field(
        default="http://localhost:8081", description="URL of the Schema Registry server"
    )
    basic_auth_user_info: Optional[str] = Field(
        default=None, description="Basic auth credentials in format username:password"
    )

    def to_client_config(self) -> Dict[str, Any]:
        """
        Convert this configuration to a Schema Registry client configuration dictionary.

        Returns:
            A dictionary suitable for initializing a Schema Registry client
        """
        config = {
            "url": self.url,
        }

        if self.basic_auth_user_info:
            config["basic.auth.user.info"] = self.basic_auth_user_info

        return config
