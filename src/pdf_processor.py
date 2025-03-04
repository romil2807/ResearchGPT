import os
from pathlib import Path
from typing import Dict, List
import PyPDF2
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the PDF processor.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory to save processed text files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, str]:
        """
        Extract text from a PDF file with improved extraction.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            Dict containing metadata and extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    'filename': pdf_path.name,
                    'num_pages': len(reader.pages),
                    'title': None,
                    'author': None
                }
                
                # Try to get document info
                if reader.metadata:
                    metadata['title'] = reader.metadata.get('/Title', '').strip()
                    metadata['author'] = reader.metadata.get('/Author', '').strip()
                
                # Extract text from all pages with better handling
                text = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text.strip())
                
                # Join with newlines to preserve structure
                full_text = '\n'.join(text)
                
                return {
                    'metadata': metadata,
                    'text': full_text
                }
                
        except Exception as e:
            raise Exception(f"Error processing {pdf_path}: {str(e)}")
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """
        Process all PDFs in the input directory.
        
        Returns:
            List of dictionaries containing processed text and metadata
        """
        pdf_files = list(self.input_dir.glob('*.pdf'))
        processed_docs = []
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                result = self.extract_text_from_pdf(pdf_file)
                if result:
                    # Save individual text file with UTF-8 encoding
                    output_file = self.output_dir / f"{pdf_file.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(result['text'])
                    
                    processed_docs.append(result)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        return processed_docs

    def validate_dataset(self) -> Dict[str, any]:
        """
        Validate the PDF dataset and return statistics.
        """
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': [],
            'total_size': 0,
            'avg_pages': 0,
            'file_types': set()
        }
        
        pdf_files = list(self.input_dir.glob('*.*'))
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