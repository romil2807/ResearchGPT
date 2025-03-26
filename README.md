# ResearchGPT

A Python-based CLI tool that leverages Large Language Models (LLMs) to process PDF documents, clean the extracted data, and generate embeddings for advanced analysis and search capabilities.

## Overview

This project provides tools to:
- Extract text and metadata from PDF documents
- Clean and process the extracted data
- Generate embeddings for semantic search and analysis
- Store processed data for future use

## Project Structure

- `src/`: Core source code
  - `embeddings_manager.py`: Handles the creation and management of document embeddings
  - `pdf_processor.py`: Processes PDF files to extract text and metadata
  - `data_cleaner.py`: Cleans and normalizes extracted data
- `data/`: Data storage
  - `processed/`: Contains processed document data
- `tests/`: Test files
- `config.yaml`: Configuration settings
- `main.py`: Entry point for the application

## Requirements

See `requirements.txt` for a list of dependencies.

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config.yaml`
4. Run the application: `python main.py`

## License

[Add your license information here]

## Contributors

[Add contributor information here] 
