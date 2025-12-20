# Contributing to NVO Rankings Predictor

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nvo-7mi-klas-rankings.git
   cd nvo-7mi-klas-rankings
   ```
3. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Test your changes:
   ```bash
   # Run predictions
   nvo predict --year 2025 --gender female
   
   # Validate against actual data
   nvo validate --test-year 2025 --gender female
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "Add: description of your change"
   ```

5. Push and create a pull request

## Code Style

- Use Python 3.12+ features
- Follow PEP 8 guidelines
- Add type hints to function signatures
- Include docstrings for public functions
- Keep functions focused and small

## Project Structure

```
nvo/
├── cli/           # CLI commands (Click)
├── data/          # Data loading and processing
├── models/        # ML training and prediction
├── services/      # Business logic layer
├── display/       # Output formatting
└── utils/         # Logging, config utilities
```

## Areas for Contribution

### High Priority
- [ ] Support for other Bulgarian cities (not just Sofia)
- [ ] Improved model for male predictions
- [ ] Better handling of new schools (cold start)

### Medium Priority
- [ ] Unit tests for core modules
- [ ] API endpoint for programmatic access
- [ ] Historical data visualization improvements

### Nice to Have
- [ ] Bulgarian language support in UI
- [ ] Mobile-friendly Streamlit layout
- [ ] Export to PDF reports

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output

## Questions?

Feel free to open an issue for questions or discussions.
