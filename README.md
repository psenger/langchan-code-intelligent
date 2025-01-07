# Langchain Code Intelligent

A Python-based code analysis tool that uses LangChain, Ollama, and vector embeddings to provide intelligent code assistance, refactoring suggestions, and feature implementation guidance for both Python and JavaScript/TypeScript codebases.

## Features

- **Code Analysis**: Analyze Python and JavaScript/TypeScript codebases for patterns and improvement opportunities
- **Intelligent Refactoring**: Get context-aware refactoring suggestions
- **Feature Implementation**: Receive guidance on implementing new features
- **Bug Analysis**: Identify potential issues and get fix recommendations
- **Vector Search**: Efficiently search through your codebase using semantic similarity
- **Multiple Embedding Options**: Support for instructor embeddings and Ollama embeddings
- **Code Chunking**: Smart code splitting for better analysis
- **Progress Tracking**: Visual feedback during processing

## Prerequisites

- Python 3.8+
- Ollama installed locally (for LLM support)
- Git (for version control)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langchain-code-intelligent.git
cd langchain-code-intelligent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Ollama:
- Follow the installation instructions at [Ollama's website](https://ollama.ai)
- Pull required models:
```bash
ollama pull codellama:13b
```

## Project Structure

The project consists of two main Python files:

### 1. Python CLI (`python-cli.py`)
- Specialized for Python codebase analysis
- Uses vector embeddings for semantic code search
- Provides Python-specific refactoring and feature suggestions
- Optimized for Python best practices and patterns

### 2. JavaScript CLI (`js-cli.py`)
- Focused on JavaScript/TypeScript codebase analysis
- Smart code chunking and processing
- JavaScript/TypeScript specific recommendations
- Framework-aware suggestions (React, Vue, Angular)

## Usage

1. Start the Ollama service on your machine

2. Run the appropriate CLI based on your codebase:
```bash
# For Python projects
python python-cli.py

# For JavaScript/TypeScript projects
python js-cli.py
```

3. Follow the interactive prompts:
   - Enter the path to your codebase
   - Choose the type of analysis you want to perform:
     1. Code Refactoring
     2. Feature Addition
     3. Bug Analysis
     4. Custom Query

4. Provide additional information when prompted

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
langchain
langchain-community
lancedb
numpy
pydantic
aiohttp
pandas
transformers
langchain-ollama
--extra-index-url https://download.pytorch.org/whl/nightly/cpu
--pre
torch
onnxruntime
InstructorEmbedding
```

## Configuration

The tool can be configured through the following parameters in both versions:

- Database path (default: ".lancedb")
- Supported file extensions
  - Python CLI: .py, .pyi, .pyx
  - JS CLI: .js, .ts, .jsx, .tsx, and test files
- Batch size for processing (default: 10)
- Number of similar documents to retrieve (default: 5)

## Features by Version

### Python CLI Features
- Python-specific code analysis
- PEP 8 compliance checking
- Type hint suggestions
- Package dependency analysis
- Performance optimization recommendations

### JavaScript CLI Features
- JavaScript/TypeScript analysis
- Framework-specific suggestions
- Modern JS features utilization
- Type system optimization
- Bundle size considerations

## Limitations

- Requires local Ollama installation
- Large codebases might require significant processing time
- Memory usage can be high with large projects
- Currently limited to Python and JavaScript/TypeScript

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.