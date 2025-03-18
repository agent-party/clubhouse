"""
Clubhouse agent capabilities package.

This package contains implementations of various agent capabilities
that follow the standardized BaseCapability interface with enhanced 
error handling and validation.
"""

from clubhouse.agents.capabilities.search_capability import SearchCapability
from typing import cast, List, Dict, Any, Type

__all__ = ["SearchCapability"]
