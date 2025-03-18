"""
Base Service Implementation.

This module provides the base class for all services in the agent orchestration platform.
"""

from abc import ABC
from typing import Dict, Any


class ServiceBase(ABC):
    """Base class for all services in the platform."""
    
    def __init__(self):
        """Initialize service."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.
        
        Returns:
            Dictionary containing service status information
        """
        return {
            "status": "operational",
            "service_type": self.__class__.__name__
        }
