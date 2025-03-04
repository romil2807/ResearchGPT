import click
from pathlib import Path
import json
from .embeddings_manager import EmbeddingsManager
from .utils import ConfigManager, Logger
from .pdf_processor import PDFProcessor

@click.group()
def cli():
    """Research Paper Q&A CLI"""
    pass

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--input-dir', default='data/raw', help='Directory containing PDF files')
def process_pdfs(config, input_dir):
    """Process PDFs and save cleaned text"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize PDF processor
    processor = PDFProcessor(config_data, logger)
    
    # Process PDFs
    processor.process_pdfs(input_dir=input_dir)
    logger.info("PDF processing completed")

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
def process(config):
    """Process PDFs and create embeddings"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize embeddings manager
    manager = EmbeddingsManager(config_data, logger)
    
    try:
        # Check if processed file exists, if not process PDFs first
        processed_file = Path('data/processed/processed_papers.json')
        if not processed_file.exists():
            logger.info("Processed papers file not found. Processing PDFs first...")
            processor = PDFProcessor(config_data, logger)
            processor.process_pdfs()
        
        # Load processed documents
        with open(processed_file, 'r', encoding='utf-8', errors='ignore') as f:
            docs = json.load(f)
        
        if not docs:
            logger.warning("No documents found to embed")
            return
        
        # Create embeddings
        manager.create_embeddings(docs)
        logger.info("Embeddings created successfully")
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

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

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
def start(config):
    """Start an interactive chat session with ResearchGPT"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize embeddings manager
    manager = EmbeddingsManager(config_data, logger)
    
    click.echo(click.style("ResearchGPT Interactive Mode", fg="green", bold=True))
    click.echo(click.style("Type 'exit' or 'quit' to end the session", fg="yellow"))
    click.echo(click.style("Type 'help' for assistance", fg="yellow"))
    click.echo("")
    
    history = []
    
    while True:
        # Get user input
        user_input = click.prompt(click.style("You", fg="green", bold=True))
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit']:
            click.echo(click.style("Goodbye!", fg="green"))
            break
            
        # Check for help command
        if user_input.lower() == 'help':
            click.echo(click.style("ResearchGPT Help:", fg="blue", bold=True))
            click.echo("- Ask questions about the research papers in your database")
            click.echo("- Type 'exit' or 'quit' to end the session")
            click.echo("- Your questions are answered based on the content of the papers")
            continue
            
        # Process the query
        try:
            # Add to history
            history.append({"role": "user", "content": user_input})
            
            # Get response
            click.echo(click.style("ResearchGPT is thinking...", fg="yellow"))
            response = manager.query(user_input)
            
            # Add to history
            history.append({"role": "assistant", "content": response})
            
            # Display response
            click.echo(click.style("ResearchGPT:", fg="blue", bold=True))
            click.echo(response)
            click.echo("")
            
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            click.echo(click.style(f"Error: {str(e)}", fg="red"))
    
    return

if __name__ == '__main__':
    cli() 