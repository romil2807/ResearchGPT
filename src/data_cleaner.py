import re
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from src.utils import TextProcessingUtils

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
        Enhanced text cleaning with better pattern matching.
        """
        if not text:
            return ""
            
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        in_references = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if we've reached the references section
            if TextProcessingUtils.is_reference_line(line):
                in_references = True
                continue
                
            # Skip references section
            if in_references:
                continue
                
            # Skip headers and footers
            if TextProcessingUtils.is_header_footer(line):
                continue
                
            # Clean up various artifacts
            line = self._clean_line(line)
            if line:
                cleaned_lines.append(line)
        
        # Join lines and normalize whitespace
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
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
        Enhanced metadata extraction with better pattern matching for research papers.
        """
        metadata = {
            'title': None,
            'author': None,
            'abstract': None,
            'keywords': None
        }
        
        # Split text into lines for processing
        lines = text.split('\n')
        
        # Extract title - look for first substantive line before author names
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not TextProcessingUtils.is_header_footer(line):
                # Skip date/publication lines
                if re.match(r'^Published|^Received|^\d{4}$', line):
                    continue
                metadata['title'] = line
                break
        
        # Extract authors - look for lines between title and abstract
        if metadata['title']:
            title_pos = text.find(metadata['title'])
            abstract_pos = text.find('ABSTRACT')
            if title_pos != -1 and abstract_pos != -1:
                author_text = text[title_pos + len(metadata['title']):abstract_pos].strip()
                # Clean up author text
                author_text = re.sub(r'\d+\s*$', '', author_text)  # Remove trailing numbers
                author_text = re.sub(r'\s+and\s+', ', ', author_text)  # Standardize separators
                # Remove footnote markers
                author_text = re.sub(r'\d+$', '', author_text)
                metadata['author'] = author_text.strip()
        
        # Extract abstract with improved pattern matching
        abstract_patterns = [
            r'ABSTRACT\s*\n+(.*?)(?=\n\s*(?:Additional index words:|MATERIALS AND METHODS|INTRODUCTION|$))',
            r'Abstract:?\s*(.*?)(?=\n\s*(?:Additional index words:|MATERIALS AND METHODS|INTRODUCTION|$))',
            r'(?:ABSTRACT|Abstract).*?\n(.*?)(?=\n\s*(?:Additional index words:|MATERIALS AND METHODS|INTRODUCTION|$))'
        ]
        
        for pattern in abstract_patterns:
            try:
                abstract_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if abstract_match:
                    abstract_text = abstract_match.group(1).strip()
                    abstract_text = self._clean_extracted_text(abstract_text)
                    if len(abstract_text) > 50:  # Minimum length to be considered valid
                        metadata['abstract'] = abstract_text
                        break
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error with abstract pattern {pattern}: {str(e)}")
                continue
        
        # Extract keywords - look for "Additional index words:" or similar
        keywords_patterns = [
            r'Additional index words:?\s*([^.]*?)(?=\.|$)',
            r'Key\s*words?:?\s*([^.]*?)(?=\.|$)',
            r'Keywords?:?\s*([^.]*?)(?=\.|$)'
        ]
        
        for pattern in keywords_patterns:
            keywords_match = re.search(pattern, text, re.IGNORECASE)
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                # Clean up keywords
                keywords_text = re.sub(r'[Ll]\.$', '', keywords_text)  # Remove trailing L.
                metadata['keywords'] = [k.strip() for k in keywords_text.split(',')]
                break
        
        return metadata
    
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