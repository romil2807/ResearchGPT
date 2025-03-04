import os
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.pdf_processor import PDFProcessor
from src.data_cleaner import DataCleaner
from src.utils import Logger, ConfigManager

def process_batch(pdf_files: list, processor: PDFProcessor, cleaner: DataCleaner, logger):
    """Process a batch of PDF files."""
    results = []
    for pdf_file in pdf_files:
        try:
            # Extract text from PDF - this returns a string, not a dictionary
            text = processor.extract_text_from_pdf(pdf_file)
            
            if text:
                # Clean the text
                cleaned_text = cleaner.clean_text(text)
                
                # Extract metadata
                metadata = cleaner.extract_metadata(text)
                
                # Extract sections if available
                try:
                    sections = cleaner.extract_sections(cleaned_text)
                except:
                    sections = []
                
                # Create processed document
                processed_doc = {
                    'filename': pdf_file.name,
                    'metadata': {
                        'title': metadata.get('title', ''),
                        'authors': metadata.get('authors', []),
                        'abstract': metadata.get('abstract', ''),
                        'keywords': metadata.get('keywords', []),
                        'year': metadata.get('year')
                    },
                    'sections': sections,
                    'cleaned_text': cleaned_text
                }
                
                results.append(processed_doc)
                
                # Add debug info if needed
                try:
                    debug_info = cleaner.debug_extraction(text)
                    if debug_info.get('contains_abstract'):
                        logger.info(f"Found abstract marker in {pdf_file.name}")
                except Exception as e:
                    logger.warning(f"Debug extraction failed for {pdf_file.name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
    
    return results

def main():
    # Setup
    base_dir = Path(__file__).parent
    config = ConfigManager.load_config(base_dir / 'config.yaml')
    
    # Setup logging
    logger = Logger.setup_logger(
        base_dir / config['logging']['log_dir'] if config['logging']['log_to_file'] else None
    )
    
    # Initialize directories
    data_dir = base_dir / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    processor = PDFProcessor(config, logger)  # Updated constructor parameters
    cleaner = DataCleaner(logger=logger, config=config)
    
    # Get list of PDF files
    pdf_files = list(raw_dir.glob('*.pdf'))
    total_files = len(pdf_files)
    logger.info(f"Found {total_files} PDF files to process")
    
    # Process files in batches
    batch_size = config['processing'].get('batch_size', 5)
    max_workers = config['processing'].get('max_workers', 4)
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i + batch_size]
            results = process_batch(batch, processor, cleaner, logger)
            all_results.extend(results)
            
            logger.info(f"Processed batch {i//batch_size + 1} of {(total_files + batch_size - 1)//batch_size}")
    
    # Save results
    output_file = processed_dir / 'processed_papers.json'
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing complete. Processed {len(all_results)} documents successfully")
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 