import logging
from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime
import re

class Logger:
    @staticmethod
    def setup_logger(log_dir: Optional[Path] = None) -> logging.Logger:
        """
        Set up a logger with both file and console handlers.
        
        Args:
            log_dir: Directory to store log files. If None, logs only to console.
        """
        logger = logging.getLogger('research_qa')
        logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log_dir is provided)
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f'research_qa_{timestamp}.log'
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger

class ConfigManager:
    @staticmethod
    def load_config(config_path: Path) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {str(e)}")

class TextProcessingUtils:
    @staticmethod
    def is_reference_line(line: str) -> bool:
        """Check if a line appears to be a reference."""
        reference_patterns = [
            r'^\[\d+\]',  # [1] style
            r'^\d+\.',    # 1. style
            r'^References?:?$',
            r'^Bibliography:?$',
            r'^Works Cited:?$'
        ]
        return any(re.match(pattern, line.strip()) for pattern in reference_patterns)
    
    @staticmethod
    def is_header_footer(line: str) -> bool:
        """Check if a line appears to be a header or footer."""
        header_footer_patterns = [
            r'^\d+$',  # Page numbers
            r'^Page \d+',
            r'^Copyright',
            r'All rights reserved',
            r'^Running head:',
            r'\d{1,2}/\d{1,2}/\d{2,4}'  # Dates
        ]
        return any(re.match(pattern, line.strip()) for pattern in header_footer_patterns)

def setup_directories():
    """Create required project directories if they don't exist."""
    dirs = [
        'data/raw',
        'data/processed',
        'data/embeddings',
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def get_llm(model_name="gpt-3.5-turbo", temperature=0.3):
    """
    Get the language model for generating responses
    
    Args:
        model_name: Name of the model to use
        temperature: Temperature for generation
        
    Returns:
        LLM instance
    """
    # If using Hugging Face models
    if "gpt" not in model_name.lower():
        from langchain_community.llms import HuggingFacePipeline
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,  # Increase token limit
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2,  # Add repetition penalty
            pad_token_id=tokenizer.eos_token_id
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    # If using OpenAI models
    else:
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
