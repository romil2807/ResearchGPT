import re
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from src.utils import TextProcessingUtils
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataCleaner:
    def __init__(self, logger=None, config=None):
        """
        Initialize the DataCleaner with optional logger and config.
        
        Args:
            logger: Logger instance for tracking processing
            config: Configuration dictionary
        """
        self.logger = logger
        self.config = config or {
            'cleaning': {
                'min_line_length': 20  # Default value if no config provided
            }
        }
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to download NLTK data: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess the text.
        """
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Remove extra whitespace
            cleaned = ' '.join(text.split())
            
            # Remove special characters if specified in config
            if self.config.get('remove_special_chars', False):
                cleaned = re.sub(r'[^\w\s]', '', cleaned)
            
            # Convert to lowercase if specified in config
            if self.config.get('convert_to_lowercase', True):
                cleaned = cleaned.lower()
            
            # Remove specific words or phrases if needed
            stop_words = self.config.get('stop_words', [])
            if stop_words:
                for word in stop_words:
                    cleaned = cleaned.replace(f" {word} ", " ")
            
            return cleaned
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error cleaning text: {str(e)}")
            return ""
    
    def _clean_line(self, line: str) -> str:
        """Enhanced line cleaning with academic paper specific rules."""
        # Skip if line is too short
        if len(line.strip()) < self.config['cleaning']['min_line_length']:
            return ""
        
        # Remove common paper artifacts
        line = re.sub(r'\d+\s*\|\s*Page', '', line)  # Remove page markers
        line = re.sub(r'©\s*\d{4}.*?rights reserved\.?', '', line)  # Remove copyright
        line = re.sub(r'https?://\S+', '', line)  # Remove URLs
        line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', line)  # Remove emails
        
        # Remove figure and table references
        line = re.sub(r'(Figure|Table|Fig\.)\s*\d+[a-zA-Z]?', '', line)
        
        # Remove citation markers
        line = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', line)  # Remove [1] style citations
        line = re.sub(r'\(\w+\s*(?:et al\.?)?,\s*\d{4}\)', '', line)  # Remove (Author, 2023) style
        
        # Clean whitespace
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """
        Extract metadata from the text.
        This is a simple implementation that should be enhanced based on your specific needs.
        """
        try:
            # Initialize metadata dictionary with default values
            metadata = {
                "title": "",
                "authors": [],
                "abstract": "",
                "keywords": [],
                "year": None
            }
            
            # Simple extraction logic - this should be improved for production use
            lines = text.split('\n')
            
            # Only process if we have content
            if not lines or not isinstance(lines, list):
                return metadata
            
            # Try to extract title (assuming it's in the first few lines)
            for i in range(min(5, len(lines))):
                if lines[i] and len(lines[i]) > 10 and lines[i].isupper():
                    metadata["title"] = lines[i]
                    break
            
            # Look for abstract
            abstract_start = -1
            abstract_end = -1
            
            for i in range(len(lines)):
                line = lines[i].strip().lower()
                if line.startswith("abstract"):
                    abstract_start = i
                elif abstract_start > -1 and line.startswith("introduction"):
                    abstract_end = i
                    break
            
            if abstract_start > -1 and abstract_end > -1:
                metadata["abstract"] = " ".join(lines[abstract_start+1:abstract_end])
            
            # Extract year from text (simple regex approach)
            year_matches = re.findall(r'\b(19|20)\d{2}\b', text)
            if year_matches:
                try:
                    metadata["year"] = int(year_matches[0])
                except:
                    pass
            
            return metadata
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error extracting metadata: {str(e)}")
            return {
                "title": "",
                "authors": [],
                "abstract": "",
                "keywords": [],
                "year": None
            }
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract main sections from the paper with improved detection.
        """
        sections = []
        current_section = None
        current_content = []
        
        # Common section headers in research papers
        section_patterns = [
            r'^abstract$',
            r'^introduction$',
            r'^materials?\s+and\s+methods$',
            r'^results?(?:\s+and\s+discussion)?$',
            r'^discussion$',
            r'^conclusion',
            r'^references$',
            r'^acknowledgments?$'
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Check if line matches a section header
            is_section = any(re.match(pattern, line.lower()) for pattern in section_patterns)
            
            if is_section:
                # Save previous section
                if current_section:
                    sections.append({
                        'title': current_section,
                        'content': '\n'.join(current_content).strip()
                    })
                current_section = line
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'title': current_section,
                'content': '\n'.join(current_content).strip()
            })
        
        return sections
    
    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """
        Extract citations from the text.
        """
        citations = []
        
        # Match different citation patterns
        patterns = {
            'numeric': r'\[(\d+(?:,\s*\d+)*)\]',
            'author_year': r'\(([^)]+?,\s*\d{4}[a-z]?)\)',
            'footnote': r'(?:^|\s)\d+\.\s+([^.]+?\(\d{4}\)[^.]+\.)'
        }
        
        for style, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append({
                    'style': style,
                    'text': match.group(0),
                    'content': match.group(1),
                    'position': match.start()
                })
        
        return sorted(citations, key=lambda x: x['position'])
    
    def debug_extraction(self, text: str) -> Dict[str, any]:
        """
        Debug helper to show what patterns are matching and where.
        """
        debug_info = {
            'text_length': len(text),
            'first_100_chars': text[:100],
            'contains_abstract': 'ABSTRACT' in text.upper(),
            'abstract_position': text.upper().find('ABSTRACT'),
            'section_matches': []
        }
        
        # Test section pattern matches
        for line in text.split('\n'):
            line = line.strip()
            if line.upper().startswith('ABSTRACT') or line.upper().startswith('INTRODUCTION'):
                debug_info['section_matches'].append(f"Found section: {line}")
        
        if self.logger:
            self.logger.info(f"Extraction debug info: {debug_info}")
        
        return debug_info
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text by removing common artifacts."""
        if not text:
            return ""
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove hyphenation at line breaks
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        # Remove common PDF artifacts
        text = re.sub(r'\([^)]*\d{4}\s*\)', '', text)  # Remove year citations
        text = re.sub(r'\[\d+\]', '', text)  # Remove reference numbers
        # Clean up whitespace
        text = text.strip()
        
        return text
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Chunk documents into smaller pieces with meaningful overlap
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        return text_splitter.split_documents(documents) 