# ResearchGPT

A powerful Python-based research assistant that leverages Large Language Models (LLMs) to process PDF documents, clean extracted data, generate embeddings, and provide interactive Q&A capabilities for research paper analysis.

## Overview

ResearchGPT is designed to enhance research workflows by providing tools to:
- Extract text and metadata from PDF research papers
- Clean and process extracted data for optimal analysis
- Generate embeddings for semantic search and retrieval
- Provide context-aware interactive Q&A about research content
- Supplement document knowledge with external web search

## Project Structure

- `src/`: Core source code
  - `embeddings_manager.py`: Handles document embeddings and semantic search
  - `pdf_processor.py`: Processes PDF files to extract text and metadata
  - `data_cleaner.py`: Cleans and normalizes extracted data
  - `context_manager.py`: Manages conversation history and context
  - `external_search.py`: Integrates web search capabilities
  - `cli.py`: Command-line interface implementation
  - `utils.py`: Utility functions and helpers
- `data/`: Data storage
  - `raw/`: Place PDF documents here for processing
  - `processed/`: Contains processed document data
  - `embeddings/`: Stores vector embeddings
- `tests/`: Test files
- `config.yaml`: Configuration settings
- `main.py`: Entry point for batch processing
- `update_embeddings.py`: Script to update embeddings

## Features

- **PDF Processing**: Extract text and metadata from research papers
- **Semantic Search**: Find relevant document chunks based on queries
- **Interactive Q&A**: Chat with your documents through a CLI interface
- **Context Awareness**: Follow-up questions with conversation memory
- **External Search**: Supplement answers with web search results
- **Configurable Models**: Support for various embedding and LLM models

## Requirements

See `requirements.txt` for a complete list of dependencies.

Primary dependencies include:
- PyPDF2: For PDF processing
- langchain: For embedding and retrieval pipelines
- transformers: For Hugging Face model integration
- sentence-transformers: For document embedding
- click: For command-line interface

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config.yaml`
4. Place PDF files in the `data/raw/` directory
5. Process documents: `python -m src.cli process`
6. Start interactive mode: `python -m src.cli start`

## Command Line Interface

ResearchGPT provides a comprehensive CLI:

- **Process PDFs**: `python -m src.cli process_pdfs`
- **Create Embeddings**: `python -m src.cli process`
- **Single Query**: `python -m src.cli query "Your question here"`
- **Interactive Chat**: `python -m src.cli start`
- **Toggle Context**: `python -m src.cli context --enable/--disable`
- **Configure Search**: `python -m src.cli external_search --enable/--disable`
- **Direct Web Search**: `python -m src.cli web_search "Your search query"`

## Configuration

Configure the application through `config.yaml`:

```yaml
processing:
  batch_size: 5
  max_workers: 4
models:
  embedding_model: sentence-transformers/all-mpnet-base-v2
  completion_model: microsoft/phi-2
chunking:
  chunk_size: 1000
  chunk_overlap: 200
external_search:
  enabled: true
  search_engine: duckduckgo
```

## Interactive Mode Commands

When using interactive mode:
- Type `help` for assistance
- Type `/clear` to clear conversation context
- Type `/context on` or `/context off` to toggle context awareness
- Type `/search on` or `/search off` to toggle external search
- Type `/ddg query` or `/google query` for direct web search
- Type `exit` or `quit` to end the session

## Contributors

- Romil Shah: shahromil2807@gmail.com
- Keo Corak
- Grant Billings

## License

[Add your license information here]
