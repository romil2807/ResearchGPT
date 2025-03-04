import os
from pathlib import Path
from typing import Dict, List
import PyPDF2
from tqdm import tqdm
import json
from PyPDF2 import PdfReader
from .data_cleaner import DataCleaner
from .utils import ConfigManager, Logger

class PDFProcessor:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.data_cleaner = DataCleaner(logger=logger, config=config)
        
        # Initialize domain-specific models if configured
        self.domain_models = {}
        if 'domain_models' in config:
            self._initialize_domain_models(config['domain_models'])
        
    def _initialize_domain_models(self, domain_config):
        """Initialize domain-specific models for specialized processing"""
        try:
            for domain, model_info in domain_config.items():
                if self.logger:
                    self.logger.info(f"Initializing {domain} domain model: {model_info['name']}")
                
                # Load the appropriate model based on type
                if model_info['type'] == 'huggingface':
                    from transformers import AutoModel, AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
                    model = AutoModel.from_pretrained(model_info['name'])
                    self.domain_models[domain] = {
                        'tokenizer': tokenizer,
                        'model': model,
                        'config': model_info
                    }
                elif model_info['type'] == 'spacy':
                    import spacy
                    model = spacy.load(model_info['name'])
                    self.domain_models[domain] = {
                        'model': model,
                        'config': model_info
                    }
                
                if self.logger:
                    self.logger.info(f"Successfully loaded {domain} model")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing domain models: {str(e)}")
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise
        
    def process_pdfs(self, input_dir='data/raw', output_file='data/processed/processed_papers.json'):
        """Process all PDFs in the input directory and save results to output file"""
        if self.logger:
            self.logger.info(f"Processing PDFs from {input_dir}")
        
        # Create output directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        processed_docs = []
        pdf_files = list(Path(input_dir).glob('*.pdf'))
        
        if not pdf_files:
            if self.logger:
                self.logger.warning(f"No PDF files found in {input_dir}")
            return []
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            if self.logger:
                self.logger.info(f"Processing {pdf_path.name}")
            
            try:
                # Extract text from PDF using the dedicated method
                text = self.extract_text_from_pdf(pdf_path)
                
                # Ensure text is a valid string
                if not text or not isinstance(text, str):
                    if self.logger:
                        self.logger.warning(f"No valid text extracted from {pdf_path.name}")
                    continue
                    
                # Detect document domain
                domain = self.detect_document_domain(text)
                
                # Process with domain-specific model if available
                domain_results = self.process_with_domain_model(text, domain)
                
                # Clean the text
                try:
                    cleaned_text = self.data_cleaner.clean_text(text)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error cleaning text from {pdf_path.name}: {str(e)}")
                    cleaned_text = text  # Use original text if cleaning fails
                
                # Extract metadata
                try:
                    metadata = self.data_cleaner.extract_metadata(text)
                    
                    # Convert any lists in metadata to strings to avoid Chroma errors
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            metadata[key] = ", ".join(str(item) for item in value)
                    
                    # Add domain information to metadata
                    metadata['detected_domain'] = domain
                    if domain_results['domain_processed']:
                        metadata['domain_model_used'] = domain_results['model_used']
                    
                    # Add entities as string if available
                    if 'entities_str' in domain_results:
                        metadata['domain_entities'] = domain_results['entities_str']
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error extracting metadata from {pdf_path.name}: {str(e)}")
                    metadata = {'detected_domain': domain}  # Use minimal metadata if extraction fails
                
                # Create document entry
                doc = {
                    "filename": pdf_path.name,
                    "cleaned_text": cleaned_text,
                    "metadata": metadata,
                    "domain": domain
                }
                
                processed_docs.append(doc)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing {pdf_path.name}: {str(e)}")
        
        # Save processed documents
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_docs, f, ensure_ascii=False, indent=2)
        
        if self.logger:
            self.logger.info(f"Processed {len(processed_docs)} documents and saved to {output_file}")
        
        return processed_docs

    def validate_dataset(self) -> Dict[str, any]:
        """
        Validate the PDF dataset and return statistics.
        """
        # Fix the missing input_dir attribute
        input_dir = Path(self.config.get('input_dir', 'data/raw'))
        
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': [],
            'total_size': 0,
            'avg_pages': 0,
            'file_types': set()
        }
        
        pdf_files = list(input_dir.glob('*.*'))
        stats['total_files'] = len(pdf_files)
        total_pages = 0
        
        for pdf_file in pdf_files:
            try:
                # Check file type
                stats['file_types'].add(pdf_file.suffix.lower())
                
                # Check if it's a valid PDF
                if pdf_file.suffix.lower() != '.pdf':
                    stats['invalid_files'].append(f"{pdf_file.name} (Not a PDF)")
                    continue
                    
                # Check file size
                file_size = pdf_file.stat().st_size
                stats['total_size'] += file_size
                
                # Try to read the PDF
                with open(pdf_file, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    total_pages += len(reader.pages)
                    stats['valid_files'] += 1
                    
            except Exception as e:
                stats['invalid_files'].append(f"{pdf_file.name} ({str(e)})")
        
        stats['avg_pages'] = total_pages / stats['valid_files'] if stats['valid_files'] > 0 else 0
        stats['total_size_mb'] = stats['total_size'] / (1024 * 1024)
        
        return stats 

    def detect_document_domain(self, text):
        """
        Detect the domain of a document based on its content.
        Returns the most likely domain (agriculture, biology, general)
        """
        # Define domain-specific keywords
        domain_keywords = {
            'agriculture': ['crop', 'farm', 'soil', 'plant', 'harvest', 'yield', 'irrigation', 
                           'fertilizer', 'pesticide', 'agriculture', 'agricultural'],
            'biology': ['gene', 'protein', 'cell', 'tissue', 'organism', 'species', 
                       'dna', 'rna', 'enzyme', 'molecular', 'biological'],
            'medicine': ['patient', 'treatment', 'disease', 'clinical', 'medical', 
                        'therapy', 'diagnosis', 'hospital', 'physician', 'drug']
        }
        
        # Count occurrences of domain keywords
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                # Case-insensitive count
                count = text.lower().count(keyword.lower())
                domain_scores[domain] += count
        
        # Get the domain with the highest score
        if max(domain_scores.values()) > 0:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'  # Default domain if no keywords match

    def process_with_domain_model(self, text, domain):
        """
        Process text using domain-specific models if available
        """
        if domain in self.domain_models:
            model_info = self.domain_models[domain]
            
            if self.logger:
                self.logger.info(f"Processing with {domain} domain model")
            
            # Process based on model type
            if model_info['config']['type'] == 'huggingface':
                # Extract entities or perform other domain-specific processing
                tokenizer = model_info['tokenizer']
                model = model_info['model']
                
                # This is a simplified example - actual processing would depend on the model
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                
                # Return enhanced text or extracted information
                return {
                    'domain_processed': True,
                    'domain': domain,
                    'model_used': model_info['config']['name'],
                    'original_text': text
                }
                
            elif model_info['config']['type'] == 'spacy':
                # Use spaCy for domain-specific NER or other processing
                nlp = model_info['model']
                doc = nlp(text)
                
                # Extract domain-specific entities but convert to string to avoid list in metadata
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                entities_str = "; ".join([f"{text}:{label}" for text, label in entities])
                
                return {
                    'domain_processed': True,
                    'domain': domain,
                    'model_used': model_info['config']['name'],
                    'entities_str': entities_str,  # String representation instead of list
                    'original_text': text
                }
        
        # If no domain model is available, return the original text
        return {
            'domain_processed': False,
            'domain': domain,
            'original_text': text
        } 