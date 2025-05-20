# 📊 High-Level Design Principles

## 1. Python Version Requirement
- This project requires Python 3.12 or newer
- Leverage Python 3.12 features including enhanced typing support, improved f-string performance, and type parameter syntax
- All code should be written with Python 3.12 compatibility in mind, without backward compatibility concerns

## 2. Modular Architecture
- Organize code into well-defined modules and packages.
- Each module should have a single responsibility, adhering to the principle of separation of concerns.
- Avoid circular dependencies by carefully managing imports and module interactions.

## 3. Clear API Boundaries
- Define explicit interfaces for each component.
- This practice enhances readability and facilitates easier integration and testing.

---

# 🧑‍💻 Code Practices

## 1. Adherence to PEP 8 with Practical Adjustments
- Follow PEP 8 guidelines for code style, including naming conventions, indentation, and spacing.
- Use relaxed line length of 100 characters.
- We'll consider using Black or ruff for consistent formatting.

## 2. Prefer Comprehensions Over Loops
- Utilize list and dictionary comprehensions for concise data processing without introducing additional variables.
- Example:

```python
squares = [x * x for x in range(10)]
```

## 3. Function and Method Design
- Keep functions and methods short and focused on a single task.
- If a function becomes too long or handles multiple responsibilities, consider refactoring into smaller functions.

## 4. Consistent Naming Conventions
- Use descriptive and consistent naming for variables, functions, classes, and modules.
- Follow standard Python naming conventions (e.g., `snake_case` for functions and variables, `CamelCase` for classes).

## 5. Comprehensive Documentation
- Include docstrings for all public modules, classes, functions, and methods, following PEP 257 conventions.
- Provide clear descriptions of the purpose, parameters, return values, and exceptions.

## 6. Linting and Static Type Checking
- Use ruff for linting Python code
- Use mypy for static type checking
- These tools are configured to run automatically via pre-commit hooks

### Running Ruff
```bash
# Check your code with ruff
python -m ruff check src/

# Fix issues automatically when possible
python -m ruff check --fix src/
```

### Running Mypy
```bash
# Run mypy for type checking
python -m mypy src/
```

### Pre-commit Hooks
The project uses pre-commit hooks to automatically run linting and type checking when you commit changes. This ensures code quality standards are maintained without requiring manual intervention.

```bash
# First-time setup (only needed once)
pre-commit install

# Manual run (typically not needed, but useful for troubleshooting)
pre-commit run --all-files
```

Once installed, the hooks run automatically during each commit. If issues are found, the commit will be rejected with details about what needs to be fixed.

## 7. Additional decisions
- Package management: Use poetry for dependency management
- logging: Use the standard logging module
- testing: Use pytest for testing
