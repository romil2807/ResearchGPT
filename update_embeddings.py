import json
from pathlib import Path
import sys
import os

# Add the project root to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.embeddings_manager import EmbeddingsManager
from src.utils import ConfigManager, Logger

def update_embeddings():
    # Setup
    base_dir = Path(__file__).parent
    config = ConfigManager.load_config(base_dir / 'config.yaml')
    logger = Logger.setup_logger()
    
    # Load the summarized papers
    processed_dir = base_dir / 'data' / 'processed'
    results_path = processed_dir / 'results_papers.json'
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            summarized_papers = json.load(f)
        logger.info(f"Loaded {len(summarized_papers)} papers with summaries")
        
        # Initialize embeddings manager
        embeddings_manager = EmbeddingsManager(config, logger)
        
        # Create new embeddings with summaries
        embeddings_manager.create_embeddings(summarized_papers)
        
        logger.info("Successfully updated embeddings with paper summaries")
        
    except FileNotFoundError:
        logger.error(f"Could not find results_papers.json at {results_path}")
    except json.JSONDecodeError:
        logger.error("Error parsing results_papers.json - invalid JSON format")
    except Exception as e:
        logger.error(f"Error updating embeddings: {str(e)}")

if __name__ == "__main__":
    update_embeddings()