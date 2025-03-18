"""
Datetime utilities for cross-version compatibility.

This module provides utilities for working with datetimes consistently
across different Python versions.
"""
from datetime import datetime, timezone

# Create a UTC timezone instance that works across Python versions
# In Python 3.11+, we can use datetime.UTC directly
# For older versions, we use timezone.utc
try:
    # Try to access UTC (available in Python 3.11+)
    UTC = datetime.UTC
except AttributeError:
    # Fall back to timezone.utc for older Python versions
    UTC = timezone.utc


def utc_now() -> datetime:
    """
    Get the current datetime in UTC.

    This is a cross-version compatible replacement for datetime.utcnow(),
    which is deprecated in newer Python versions.

    Returns:
        Current datetime with UTC timezone
    """
    return datetime.now(UTC)
