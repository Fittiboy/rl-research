# Contributing to RL Research Framework

Thank you for your interest in contributing to the RL Research Framework! This document provides guidelines and instructions for contributing.

## Code Style

- Follow PEP 8 guidelines for Python code
- Use type hints for function parameters and return values
- Write docstrings for all public functions, classes, and modules
- Keep functions focused and single-purpose
- Use meaningful variable and function names

## Development Setup

1. Fork the repository
2. Create a virtual environment:
```bash
conda create -n rl-research python=3.10
conda activate rl-research
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Testing

- Write tests for all new functionality
- Run tests before submitting PRs:
```bash
pytest tests/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Update documentation if needed
5. Run tests locally
6. Submit PR with clear description of changes

## Documentation

- Update relevant documentation for any code changes
- Add docstrings for new functions and classes
- Update README.md if adding new features
- Add examples for significant new functionality

## Questions?

Feel free to open an issue for any questions about contributing! 