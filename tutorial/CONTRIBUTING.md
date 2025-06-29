# Contributing to NLP-Tutorials

Thank you for your interest in contributing to NLP-Tutorials! This document provides guidelines and information for contributors.

## How to Contribute

We welcome contributions from the community! Here are several ways you can contribute:

### 1. Reporting Issues

If you find a bug or have a suggestion for improvement, please:

- Check if the issue has already been reported
- Use the appropriate issue template
- Provide a clear and descriptive title
- Include steps to reproduce the issue
- Specify your environment (OS, Python version, PyTorch version)
- Include error messages and stack traces if applicable

### 2. Suggesting New Tutorials

We're always looking for new tutorial ideas! To suggest a new tutorial:

- Open an issue with the "New Tutorial" label
- Describe the NLP concept or technique you'd like to cover
- Explain why this would be valuable for learners
- Provide any relevant resources or papers

### 3. Improving Existing Tutorials

You can help improve existing tutorials by:

- Fixing typos or grammatical errors
- Clarifying explanations or code comments
- Adding more detailed explanations for complex concepts
- Improving code examples or adding better comments
- Updating dependencies or requirements
- Adding additional learning resources

### 4. Adding New Features

For larger contributions like new features:

- Open an issue first to discuss the proposed changes
- Fork the repository and create a feature branch
- Implement your changes with clear documentation
- Add tests if applicable
- Update relevant documentation

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Jupyter Notebook or JupyterLab
- PyTorch (latest stable version recommended)

### Local Development

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/darinz/NLP-Tutorials.git
   cd NLP-Tutorials
   ```

3. Set up the development environment:
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt  # If available
   pip install jupyter notebook
   ```

4. Make your changes
5. Test your changes by running the notebooks
6. Commit your changes with clear commit messages

## Code Style and Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and concise
- Add comments for complex logic

### Jupyter Notebooks

- Keep cells focused and well-organized
- Add markdown explanations between code cells
- Include clear section headers
- Ensure all code cells run without errors
- Add comments explaining complex operations

### Documentation

- Write clear, concise explanations
- Use proper markdown formatting
- Include code examples where helpful
- Link to relevant resources and papers
- Keep README files up to date

## Pull Request Process

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch for your changes
3. **Make Changes**: Implement your changes following the guidelines above
4. **Test**: Ensure all notebooks run correctly and tests pass
5. **Commit**: Make clear, descriptive commit messages
6. **Push**: Push your changes to your fork
7. **Submit PR**: Create a pull request with a clear description

### Pull Request Guidelines

- Use a clear, descriptive title
- Provide a detailed description of changes
- Reference any related issues
- Include screenshots or examples if applicable
- Ensure all CI checks pass

### Commit Message Format

Use clear, descriptive commit messages:

```
feat: add new attention mechanism tutorial
fix: correct typo in RNN classification README
docs: update installation instructions
style: format code according to PEP 8
refactor: improve code organization in transformer tutorial
```

## Review Process

- All pull requests require review before merging
- Reviews focus on code quality, clarity, and educational value
- Constructive feedback will be provided for improvements
- Maintainers may request changes before merging

## Getting Help

If you need help with contributing:

- Check existing issues and pull requests
- Join our community discussions
- Ask questions in issues with the "question" label
- Reach out to maintainers directly

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate of others
- Use inclusive language
- Be open to constructive feedback
- Help others learn and grow

## Recognition

Contributors will be recognized in:

- The project README
- Release notes
- Contributor acknowledgments

## License

By contributing to NLP-Tutorials, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to NLP-Tutorials! Your contributions help make NLP education more accessible to everyone. 