"""
Tests for enforcing capability implementation standards.

This module contains tests that validate capabilities follow the standardized
implementation patterns defined in the capability development guide.
"""

import pytest
import inspect
import os
import importlib
import pkgutil
from pathlib import Path
from typing import List, Type, Set

from clubhouse.agents.capability import BaseCapability
import clubhouse.agents.capabilities


def get_all_capability_classes() -> List[Type[BaseCapability]]:
    """
    Discover all capability classes in the capabilities package.
    
    Returns:
        List of all capability classes that inherit from BaseCapability
    """
    capabilities = []
    capabilities_package = clubhouse.agents.capabilities
    package_path = Path(capabilities_package.__file__).parent
    
    # Get all Python modules in the capabilities package
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if module_name.startswith('__') or module_name.startswith('test_'):
            continue
            
        # Import the module
        module = importlib.import_module(f"clubhouse.agents.capabilities.{module_name}")
        
        # Find all classes in the module that inherit from BaseCapability
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseCapability) and 
                obj is not BaseCapability):
                capabilities.append(obj)
    
    return capabilities


class TestCapabilityCompliance:
    """Tests to ensure capabilities follow the standardized implementation pattern."""
    
    def test_all_capabilities_have_required_methods(self):
        """Test that all capabilities implement required methods and properties."""
        capabilities = get_all_capability_classes()
        
        # Make sure we found at least some capabilities
        assert len(capabilities) > 0, "No capability classes were found"
        
        for capability_class in capabilities:
            # Create an instance for testing
            capability = capability_class()
            
            # Check required properties
            assert hasattr(capability, 'name'), f"{capability_class.__name__} missing 'name' property"
            assert hasattr(capability, 'description'), f"{capability_class.__name__} missing 'description' property"
            assert hasattr(capability, 'parameters_schema'), f"{capability_class.__name__} missing 'parameters_schema' property"
            
            # Check required methods
            assert hasattr(capability, 'validate_parameters'), f"{capability_class.__name__} missing 'validate_parameters' method"
            assert hasattr(capability, 'execute'), f"{capability_class.__name__} missing 'execute' method"
            # Either have execute_with_lifecycle or a method that calls it
            assert (hasattr(capability, 'execute_with_lifecycle') or 
                    hasattr(capability, 'execute_and_handle_lifecycle')), \
                f"{capability_class.__name__} missing 'execute_with_lifecycle' method"
    
    def test_all_capabilities_handle_standard_events(self):
        """Test that all capabilities use the standard event pattern."""
        capabilities = get_all_capability_classes()
        
        for capability_class in capabilities:
            # Create an instance for testing
            capability = capability_class()
            
            # Check the source code of execute method to verify event triggering pattern
            execute_source = inspect.getsource(capability.execute)
            
            # The execute method should either:
            # 1. Call execute_with_lifecycle directly, or
            # 2. Trigger before/after execution events
            standard_patterns_used = (
                'self.execute_with_lifecycle' in execute_source or
                'self.trigger_event("before_execution"' in execute_source or
                'self.trigger_event("after_execution"' in execute_source or
                # For backward compatibility in transition period:
                'self.trigger_event(f"{self.name}.started"' in execute_source or
                'self.trigger_event(f"{self.name}.completed"' in execute_source
            )
            
            assert standard_patterns_used, \
                f"{capability_class.__name__}.execute() doesn't follow standard event pattern"
    
    def test_capabilities_use_pydantic_validation(self):
        """Test that all capabilities use Pydantic for parameter validation."""
        capabilities = get_all_capability_classes()
        
        for capability_class in capabilities:
            # Check if the capability has a parameters_schema that is a Pydantic model
            assert hasattr(capability_class, 'parameters_schema'), \
                f"{capability_class.__name__} missing 'parameters_schema' property"
            
            schema = getattr(capability_class, 'parameters_schema')
            # Pydantic models have a model_validate or parse_obj method
            has_pydantic_methods = (
                hasattr(schema, 'model_validate') or 
                hasattr(schema, 'parse_obj') or
                hasattr(schema, 'model_dump')
            )
            
            assert has_pydantic_methods, \
                f"{capability_class.__name__} doesn't use Pydantic for parameter validation"
    
    def test_capabilities_handle_errors_consistently(self):
        """Test that all capabilities handle errors using the centralized error framework."""
        capabilities = get_all_capability_classes()
        
        for capability_class in capabilities:
            # Create an instance for testing
            capability = capability_class()
            
            # Check the source code of execute and validate_parameters methods
            execute_source = inspect.getsource(capability.execute)
            validate_source = inspect.getsource(capability.validate_parameters)
            
            # Look for proper error handling patterns
            uses_validation_error = 'ValidationError' in validate_source
            uses_execution_error = 'ExecutionError' in execute_source or 'self._handle_execution_error' in execute_source
            
            assert uses_validation_error, \
                f"{capability_class.__name__}.validate_parameters() doesn't use ValidationError"
            
            # Some capabilities might handle errors through execute_with_lifecycle
            lifecycle_delegation = 'self.execute_with_lifecycle' in execute_source
            
            assert uses_execution_error or lifecycle_delegation, \
                f"{capability_class.__name__}.execute() doesn't handle errors consistently"
