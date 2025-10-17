# Contributing to Lotus

Thank you for your interest in contributing to Lotus! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Issue Templates](#issue-templates)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Getting Help](#getting-help)

## Getting Started

Before contributing, please:

1. Read this contributing guide
2. Check existing issues and pull requests to avoid duplicates
3. Join our community discussions
4. Familiarize yourself with the codebase

## Development Setup

### Prerequisites

- Python 3.10
- Git
- Conda (recommended) or virtual environment

### Setup Instructions

```bash
# Create and activate conda environment
conda create -n lotus python=3.10 -y
conda activate lotus

# Clone the repository
git clone git@github.com:lotus-data/lotus.git
cd lotus

# Install lotus in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Contribution Workflow

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally
3. Add the upstream repository as a remote

```bash
git remote add upstream git@github.com:lotus-data/lotus.git
```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

- Follow the code style guidelines
- Write tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run the test suite
pytest

# Run linting
pre-commit run --all-files

# Run type checking
mypy lotus/
```

### 5. Commit Your Changes

Use conventional commit messages:

```
type(scope): description

Examples:
feat(models): add support for new model provider
fix(api): resolve authentication issue
docs(readme): update installation instructions
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request using our template.


## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the project's style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No new warnings are generated
- [ ] Self-review of your code

### PR Template

Please include the following in your PR:

- **Purpose**: Clear description of what the PR accomplishes
- **Test Plan**: How you tested your changes
- **Test Results**: Results of your testing
- **Documentation Updates**: Any documentation changes needed
- **Type of Change**: Bug fix, feature, breaking change, etc.
- **Checklist**: Quality assurance items

### Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer must approve
3. All conversations must be resolved
4. Documentation updates may be required

## Code Style and Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Keep functions and classes focused and well-documented
- Use meaningful variable and function names

### Pre-commit Hooks

We use pre-commit hooks to maintain code quality:

- **ruff**: Linting and code formatting
- **mypy**: Type checking

### Running Code Quality Checks
```
bash
# Install pre-commit if you haven't already
pip install pre-commit

# Install the pre-commit hooks defined in .pre-commit-config.yaml
pre-commit install

# Run all pre-commit hooks on all files
pre-commit run --all-files

# To run a specific hook (e.g., ruff)
pre-commit run ruff --all-files

# To run pre-commit checks before every commit (recommended), just commit as usual:
git commit -m "Your commit message"
# The hooks will run automatically

```

## Testing Guidelines

### Writing Tests

We maintain two test suites:
- lotus/.github/tests: essential tests for CI/CD to ensure core functionality
- lotus/tests: additional tests for comprehensive testing of non-core functionality and integrations

If you are unsure where to add your new tests, we recommend starting them within lotus/tests and highlighting your question in your PR, so that the maintainers can respond with their suggestions.

You can find useful documentation, conceptual explanations, and best practices for writing pytests [here](https://docs.pytest.org/en/stable/getting-started.html).

Our general guidelines for testing include the following:
- Write tests for new functionality, ensuring full coverage of possible code paths and edge cases
- Avoid writing tests that depend on specific model behaviors. For example, when writing a `sem_map` test, we would avoid assertions on the exact projection output, and instead write assertions that the expected column exists in the resulting dataframe with non-empty string attributes.
- Use descriptive test names
- Mock external dependencies

### Running Tests

- first export the following enviorment variables:

```
export ENABLE_OPENAI_TESTS="true"
export ENABLE_LOCAL_TESTS="true"
export OPENAI_API_KEY="<your key>"
```


- then run your pytest

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run tests in parallel
pytest -n auto
```

## Documentation

### Documentation Standards

- Keep documentation up to date
- Use clear, concise language
- Include code examples
- Update README.md for significant changes

### Documentation Structure

- `README.md`: Project overview and quick start
- `docs/`: Detailed documentation
- `examples/`: Code examples
- Inline code comments for complex logic

## Running Models

Lotus uses the `litellm` library to interface with various model providers. Here are some examples:

### GPT-4o Example

```python
from lotus.models import LM

lm = LM(model="gpt-4o")
```

### Ollama Example

```python
from lotus.models import LM

lm = LM(model="ollama/llama3.2")
```

### vLLM Example

```python
from lotus.models import LM

lm = LM(
    model='hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct',
    api_base='http://localhost:8000/v1',
    max_ctx_len=8000,
    max_tokens=1000
)
```

## Getting Help

### Community Resources

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: Check the README and examples folder

### Before Asking for Help

1. Check existing issues and discussions
2. Read the documentation
3. Try to reproduce the issue in a minimal environment
4. Provide clear, detailed information about your problem

### Contact Information

- **Repository**: https://github.com/lotus-data/lotus
- **Discussions**: https://github.com/lotus-data/lotus/discussions
- **Issues**: https://github.com/lotus-data/lotus/issues

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.


---

Thank you for contributing to Lotus! 🚀
