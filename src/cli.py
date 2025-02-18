import click
from pathlib import Path
import json
from .embeddings_manager import EmbeddingsManager
from .utils import ConfigManager, Logger

@click.group()
def cli():
    """Research Paper Q&A CLI"""
    pass

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
def process(config):
    """Process PDFs and create embeddings"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize embeddings manager
    manager = EmbeddingsManager(config_data, logger)
    
    # Load processed documents
    with open(Path('data/processed/processed_papers.json'), 'r') as f:
        docs = json.load(f)
    
    # Create embeddings
    manager.create_embeddings(docs)
    logger.info("Embeddings created successfully")

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.argument('question')
def query(config, question):
    """Query the research papers"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize embeddings manager
    manager = EmbeddingsManager(config_data, logger)
    
    # Get answer
    answer = manager.query(question)
    click.echo(answer)

if __name__ == '__main__':
    cli() 