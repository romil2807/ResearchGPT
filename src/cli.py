import click
from pathlib import Path
import json
from src.utils import ConfigManager, Logger
from src.embeddings_manager import EmbeddingsManager
from src.pdf_processor import PDFProcessor

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
@click.option('--external-search/--no-external-search', default=True, help='Use external search')
@click.argument('question')
def query(config, external_search, question):
    """Query the research papers with optional external search"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize embeddings manager
    manager = EmbeddingsManager(config_data, logger)
    
    # Get answer with or without external search
    answer = manager.query(question, use_external_search=external_search)
    click.echo(answer)

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
def start(config):
    """Start an interactive chat session with ResearchGPT"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Initialize embeddings manager
    manager = EmbeddingsManager(config_data, logger)
    
    # Check if context awareness is enabled
    context_enabled = config_data.get('context_awareness', {}).get('enabled', False)
    
    # Check if external search is enabled
    search_enabled = config_data.get('external_search', {}).get('enabled', False)
    
    click.echo(click.style("ResearchGPT Interactive Mode", fg="green", bold=True))
    click.echo(click.style("Type 'exit' or 'quit' to end the session", fg="yellow"))
    click.echo(click.style("Type 'help' for assistance", fg="yellow"))
    if context_enabled:
        click.echo(click.style("Context awareness is ENABLED. Type '/clear' to clear context history.", fg="blue"))
    else:
        click.echo(click.style("Context awareness is DISABLED.", fg="yellow"))
    
    if search_enabled:
        click.echo(click.style("External search is ENABLED. Type '/search off' to disable.", fg="blue"))
    else:
        click.echo(click.style("External search is DISABLED. Type '/search on' to enable.", fg="yellow"))
    
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
            click.echo("- Type '/clear' to clear conversation context history")
            click.echo("- Type '/context on' or '/context off' to toggle context awareness")
            click.echo("- Type '/search on' or '/search off' to toggle external search")
            click.echo("- Type '/google [query]' or '/ddg [query]' for direct web search")
            click.echo("- Your questions are answered based on the content of the papers")
            continue
            
        # Check for clear context command
        if user_input.lower() == '/clear':
            if context_enabled:
                manager.clear_context()
                click.echo(click.style("Context history cleared.", fg="blue"))
            else:
                click.echo(click.style("Context awareness is not enabled.", fg="yellow"))
            continue
            
        # Check for context toggle commands
        if user_input.lower() in ['/context on', '/context off']:
            enabled = user_input.lower() == '/context on'
            # Update the config in memory
            if 'context_awareness' not in config_data:
                config_data['context_awareness'] = {}
            config_data['context_awareness']['enabled'] = enabled
            
            # Update the manager's context manager
            manager.context_manager.enabled = enabled
            
            # Also save to config file
            config_file = Path(config)
            ConfigManager.save_config(config_data, config_file)
            
            status = "ENABLED" if enabled else "DISABLED"
            click.echo(click.style(f"Context awareness is now {status}.", fg="blue"))
            continue
            
        # Check for external search toggle commands
        if user_input.lower() in ['/search on', '/search off']:
            enabled = user_input.lower() == '/search on'
            # Update the config in memory
            if 'external_search' not in config_data:
                config_data['external_search'] = {}
            config_data['external_search']['enabled'] = enabled
            
            # Update the manager's search manager
            manager.toggle_external_search(enabled)
            
            # Also save to config file
            config_file = Path(config)
            ConfigManager.save_config(config_data, config_file)
            
            status = "ENABLED" if enabled else "DISABLED"
            click.echo(click.style(f"External search is now {status}.", fg="blue"))
            continue
            
        # Check for direct search commands
        if user_input.lower().startswith('/ddg ') or user_input.lower().startswith('/google '):
            # Extract search query
            search_query = user_input.split(' ', 1)[1]
            
            click.echo(click.style(f"Searching for: {search_query}", fg="yellow"))
            
            # Set search engine based on command
            if 'external_search' not in config_data:
                config_data['external_search'] = {}
                
            if user_input.lower().startswith('/ddg '):
                config_data['external_search']['search_engine'] = 'duckduckgo'
            else:
                config_data['external_search']['search_engine'] = 'google'
                
            # Enable search
            config_data['external_search']['enabled'] = True
            manager.toggle_external_search(True)
            
            # Perform search
            search_results = manager.search_manager.search_and_format(search_query)
            
            if search_results:
                click.echo(search_results)
            else:
                click.echo(click.style("No search results found.", fg="yellow"))
                
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

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--enable/--disable', default=True, help='Enable or disable context awareness')
def context(config, enable):
    """Enable or disable context awareness"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Update config
    if 'context_awareness' not in config_data:
        config_data['context_awareness'] = {}
    
    config_data['context_awareness']['enabled'] = enable
    
    # Save config
    config_file = Path(config)
    ConfigManager.save_config(config_data, config_file)
    
    status = "enabled" if enable else "disabled"
    logger.info(f"Context awareness {status}")
    click.echo(f"Context awareness is now {status}")

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--enable/--disable', default=True, help='Enable or disable external search')
@click.option('--engine', type=click.Choice(['duckduckgo', 'google']), default='duckduckgo',
              help='Search engine to use')
def external_search(config, enable, engine):
    """Configure external search settings"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Update config
    if 'external_search' not in config_data:
        config_data['external_search'] = {}
    
    config_data['external_search']['enabled'] = enable
    config_data['external_search']['search_engine'] = engine
    
    # Save config
    config_file = Path(config)
    ConfigManager.save_config(config_data, config_file)
    
    status = "enabled" if enable else "disabled"
    logger.info(f"External search {status} with engine {engine}")
    click.echo(f"External search is now {status} using {engine}")

@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.argument('query')
def web_search(config, query):
    """Perform a direct web search"""
    config_data = ConfigManager.load_config(Path(config))
    logger = Logger.setup_logger()
    
    # Ensure external search is enabled
    if 'external_search' not in config_data:
        config_data['external_search'] = {}
        config_data['external_search']['enabled'] = True
        config_data['external_search']['search_engine'] = 'duckduckgo'
    
    # Initialize search manager
    from .external_search import ExternalSearchManager
    search_manager = ExternalSearchManager(config_data, logger)
    
    # Perform search
    click.echo(f"Searching for: {query}")
    results = search_manager.search_and_format(query)
    
    if results:
        click.echo(results)
    else:
        click.echo("No search results found.")

if __name__ == '__main__':
    cli() 