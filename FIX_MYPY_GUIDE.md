# Systematic mypy Error Resolution Guide

This guide outlines the approach to systematically eliminate all mypy errors from the codebase, following our quality-first development approach and SOLID principles.

## Tools Available

We've created two scripts to help fix mypy errors:

1. `fix_mypy_errors.py` - Automatically fixes common syntax errors and applies type ignores where appropriate
2. `fix_mypy_category.py` - Focuses on fixing specific categories of errors

## Step 1: Fix Syntax and Import Errors

Run the general fixer script first to clean up basic issues:

```bash
python3 fix_mypy_errors.py
```

## Step 2: Fix Protocol Instantiation Issues

Protocol classes cannot be directly instantiated. Instead, use the concrete implementations:

```bash
python3 fix_mypy_category.py 1
```

## Step 3: Fix Agent Capability Access Patterns

Fix issues with accessing AgentCapability:

```bash
python3 fix_mypy_category.py 2
```

## Step 4: Fix get_protocol calls

Fix service registry get_protocol call parameters:

```bash
python3 fix_mypy_category.py 3
```

## Step 5: Add Missing Type Annotations

Add type annotations to function parameters and return values:

```bash
python3 fix_mypy_category.py 4
```

## Step 6: Fix Dict-like Attribute Access

Fix issues with dictionary-like object attribute access:

```bash
python3 fix_mypy_category.py 5
```

## Step 7: Manual Review and Cleanup

After running the automatic fixers, perform manual review of any remaining errors:

1. Check for `# type: ignore` comments that can be removed
2. Look for specific domain knowledge issues that automated tools couldn't resolve
3. Ensure all new type annotations make logical sense

## Principles to Follow

During manual fixes, adhere to these principles:

1. **Type Safety** - Always prefer explicit typing over Any
2. **Proper Protocol Usage** - Never instantiate Protocol classes directly
3. **Consistent Pattern Usage** - Follow established patterns for service access and state management
4. **Docstrings** - Add or update docstrings when modifying functions
5. **Test Coverage** - Ensure tests still pass after making changes

## Most Common Error Categories

1. **Protocol Instantiation** - Can't instantiate protocol classes directly
2. **Attribute Access** - Dict-like objects need subscription not attribute access
3. **Type Incompatibility** - Mismatched types in assignments and return values
4. **Missing Annotations** - Function parameters without type annotations
5. **Unreachable Code** - Code that can never be executed

## Long-term Maintenance

To prevent future mypy errors:

1. Run mypy as part of CI/CD pipeline
2. Make mypy checks mandatory before merging PRs
3. Document common patterns for type safety in the project wiki
4. Periodically review and optimize type annotations
