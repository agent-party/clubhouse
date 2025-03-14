"""
Kafka configuration model.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


class KafkaConfig(BaseModel):
    """
    Configuration for Kafka integration.
    
    Attributes:
        bootstrap_servers: Comma-separated list of Kafka broker addresses
        topic_prefix: Prefix for all Kafka topics
        group_id: Consumer group ID
        client_id: Client ID for Kafka producer/consumer
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
    )
    
    bootstrap_servers: str = Field(
        default="localhost:9092", 
        description="Comma-separated list of Kafka broker addresses"
    )
    topic_prefix: str = Field(
        default="", 
        description="Prefix for all Kafka topics"
    )
    group_id: Optional[str] = Field(
        default=None, 
        description="Consumer group ID"
    )
    client_id: Optional[str] = Field(
        default=None, 
        description="Client ID for Kafka producer/consumer"
    )
    auto_offset_reset: str = Field(
        default="earliest", 
        description="Auto offset reset policy",
    )
    
    @field_validator('auto_offset_reset')
    @classmethod
    def validate_auto_offset_reset(cls, v: str) -> str:
        """Ensure auto_offset_reset is a valid value.
        
        Args:
            v: The auto offset reset value to validate.
            
        Returns:
            str: The validated auto offset reset value.
            
        Raises:
            ValueError: If the auto offset reset value is not valid.
        """
        valid_values = ["earliest", "latest", "none"]
        if v not in valid_values:
            raise ValueError(f"Invalid auto_offset_reset value. Must be one of: {', '.join(valid_values)}")
        return v
        
    def get_consumer_config(self) -> Dict[str, Any]:
        """
        Get the Kafka consumer configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: The Kafka consumer configuration.
        """
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "auto.offset.reset": self.auto_offset_reset,
        }
        
        if self.group_id:
            config["group.id"] = self.group_id
            
        if self.client_id:
            config["client.id"] = self.client_id
            
        return config
    
    def get_producer_config(self) -> Dict[str, Any]:
        """
        Get the Kafka producer configuration as a dictionary.
        
        Returns:
            Dict[str, Any]: The Kafka producer configuration.
        """
        config = {
            "bootstrap.servers": self.bootstrap_servers,
        }
        
        if self.client_id:
            config["client.id"] = self.client_id
            
        return config
