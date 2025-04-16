import numpy as np
import faiss
from typing import List, Dict, Any
import json
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import torch
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
import re
try:
    from .context_manager import ContextManager
    from .external_search import ExternalSearchManager
except ImportError:
    # For direct imports in some environments
    from context_manager import ContextManager
    from external_search import ExternalSearchManager

class EmbeddingsManager:
    def __init__(self, config: dict, logger=None):
        """Initialize the embeddings manager with LangChain components."""
        self.config = config
        self.logger = logger
        
        # Get models config
        models_config = config.get('models', {})
        
        # Initialize embedding model name
        self.embedding_model_name = models_config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        
        # Initialize the actual embedding model
        self.initialize_embeddings()
        
        # Initialize context manager
        self.context_manager = ContextManager(config, logger)
        
        # Initialize external search manager
        self.search_manager = ExternalSearchManager(config, logger)
        
        # Try to load existing vectorstore
        try:
            self.vectorstore = Chroma(
                persist_directory="data/embeddings",
                embedding_function=self.embeddings
            )
            if self.logger:
                self.logger.info("Loaded existing vectorstore from data/embeddings")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not load vectorstore: {str(e)}")
            self.vectorstore = None
        
        # Get token and model name
        hf_token = models_config.get('hf_token')
        completion_model = models_config.get('completion_model')
        
        if not completion_model:
            raise ValueError("completion_model must be specified in config.yaml")
        
        # Initialize LLM with token
        self.tokenizer = AutoTokenizer.from_pretrained(
            completion_model,
            token=hf_token
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            completion_model,
            token=hf_token
        )
        
        # Update pipeline type for GPT-2
        self.pipeline = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=0 if models_config.get('use_gpu', False) else -1,
            max_new_tokens=200,          # Reduced to avoid exceeding limits
            do_sample=True,              # Set to True to match temperature setting
            temperature=0.1,             # Keep temperature
            top_p=0.9,                # Slightly reduced
            repetition_penalty=1.2,      # Keep repetition penalty
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Create LangChain LLM wrapper with updated class
        self.llm_chain = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=models_config.get('max_tokens_per_chunk', 1000),
            chunk_overlap=200
        )
        
        # Try to load context manager state
        self.context_manager.load_state()

    def create_embeddings(self, documents: List[Dict[str, Any]]):
        """Create embeddings for the documents."""
        try:
            # Convert documents to LangChain format
            langchain_docs = []
            for doc in documents:
                text = doc.get('cleaned_text', '')
                metadata = doc.get('metadata', {})
                metadata['filename'] = doc.get('filename', '')
                
                # Manual filtering of complex metadata
                filtered_metadata = {}
                for key, value in metadata.items():
                    # Convert empty lists to empty strings
                    if isinstance(value, list) and len(value) == 0:
                        filtered_metadata[key] = ""
                    # Convert non-empty lists to comma-separated strings
                    elif isinstance(value, list):
                        filtered_metadata[key] = ", ".join(str(item) for item in value)
                    # Convert None to empty string
                    elif value is None:
                        filtered_metadata[key] = ""
                    # Keep simple types as they are
                    elif isinstance(value, (str, int, float, bool)):
                        filtered_metadata[key] = value
                    # Convert any other types to strings
                    else:
                        filtered_metadata[key] = str(value)
                
                langchain_docs.append(Document(page_content=text, metadata=filtered_metadata))
            
            # Create vectorstore with persist_directory to auto-persist
            self.vectorstore = Chroma.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                persist_directory="data/embeddings"
            )
            
            # No need to call persist() explicitly - it's handled automatically
            # when persist_directory is provided
            
            if self.logger:
                self.logger.info("Embeddings created successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def query(self, question: str, k: int = 3, use_external_search: bool = True):
        """Query the vectorstore for relevant documents with context awareness and external search."""
        try:
            # Get relevant documents directly
            docs = self.vectorstore.similarity_search(question, k=k)
            
            # Perform external search if enabled
            external_results = ""
            if use_external_search and self.search_manager.enabled:
                if self.logger:
                    self.logger.info(f"Performing external search for: {question}")
                external_results = self.search_manager.search_and_format(question)
            
            # Get context-aware prompt if enabled
            if self.context_manager.enabled:
                # Add the user's question to history
                self.context_manager.add_to_history("user", question)
                
                # Get combined context
                context = self.context_manager.build_combined_context(docs, question)
                
                # Add external search results if available
                if external_results:
                    context += "\n\n" + external_results
                
                # Create a context-aware prompt
                prompt = f"""You are a helpful research assistant. Answer the following question based on the provided context:

{context}

Question: {question}

Answer:"""
            else:
                # Use simple non-context aware prompt
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Add external search results if available
                if external_results:
                    context += "\n\n" + external_results
                
                prompt = f"""You are a helpful research assistant. Answer the following question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response with error handling
            try:
                # First try with the pipeline
                response = self.pipeline(prompt, max_length=len(prompt.split()) + 300)[0]['generated_text']
                # Extract only the answer part
                answer = response.split("Answer:")[-1].strip()
                result = self.clean_generated_text(answer)
                
                # Add the answer to history if context awareness is enabled
                if self.context_manager.enabled:
                    self.context_manager.add_to_history("assistant", result)
                    # Save context state
                    self.context_manager.save_state()
                
                return result
            except Exception as e:
                # If pipeline fails, fall back to a simpler approach
                if self.logger:
                    self.logger.warning(f"Pipeline generation failed: {str(e)}. Falling back to simple response.")
                
                # Create a simple response from the retrieved documents
                simple_response = "Based on the retrieved documents:\n\n"
                for i, doc in enumerate(docs):
                    simple_response += f"Document {i+1}:\n"
                    simple_response += doc.page_content[:300] + "...\n\n"
                
                # Add external search results to the simple response if available
                if external_results:
                    simple_response += "\n" + external_results
                
                # Add the answer to history if context awareness is enabled
                if self.context_manager.enabled:
                    self.context_manager.add_to_history("assistant", simple_response)
                    # Save context state
                    self.context_manager.save_state()
                
                return simple_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying: {str(e)}")
            return f"Error processing query: {str(e)}"

    def chat_query(self, question: str, history=None, k: int = 3, use_external_search: bool = True):
        """Query with chat history context and external search."""
        try:
            # Get relevant documents
            docs = self.vectorstore.similarity_search(question, k=k)
            
            # Perform external search if enabled
            external_results = ""
            if use_external_search and self.search_manager.enabled:
                if self.logger:
                    self.logger.info(f"Performing external search for: {question}")
                external_results = self.search_manager.search_and_format(question)
            
            # Add the user's question to history if context is enabled
            if self.context_manager.enabled:
                self.context_manager.add_to_history("user", question)
                
                # Get context-aware prompt
                context = self.context_manager.build_combined_context(docs, question)
                
                # Add external search results if available
                if external_results:
                    context += "\n\n" + external_results
                
                # Create a context-aware chat prompt
                prompt = f"""You are a helpful research assistant named ResearchGPT. Answer the following question based on the provided context:

{context}

User question: {question}

ResearchGPT's answer:"""
            else:
                # Format context from documents (non-context-aware)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Add external search results if available
                if external_results:
                    context += "\n\n" + external_results
                
                # Create a regular chat prompt
                prompt = f"""You are a helpful research assistant named ResearchGPT. Answer the following question based on the provided context from research papers. Be concise and informative.

Context from research papers:
{context}

User question: {question}

ResearchGPT's answer:"""
            
            # Generate response
            try:
                response = self.pipeline(prompt, max_length=len(prompt.split()) + 300)[0]['generated_text']
                # Extract only the answer part
                answer = response.split("ResearchGPT's answer:")[-1].strip()
                result = self.clean_generated_text(answer)
                
                # Add the assistant's response to history if context is enabled
                if self.context_manager.enabled:
                    self.context_manager.add_to_history("assistant", result)
                    # Save context state
                    self.context_manager.save_state()
                
                return result
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Pipeline generation failed: {str(e)}. Falling back to simple response.")
                
                # Create a simple response
                simple_response = "Based on the research papers, I found the following information:\n\n"
                for i, doc in enumerate(docs):
                    source = doc.metadata.get('filename', f'Document {i+1}')
                    simple_response += f"From {source}:\n"
                    simple_response += doc.page_content[:250] + "...\n\n"
                
                # Add external search results if available
                if external_results:
                    simple_response += "\n" + external_results
                
                # Add the assistant's response to history if context is enabled
                if self.context_manager.enabled:
                    self.context_manager.add_to_history("assistant", simple_response)
                    # Save context state
                    self.context_manager.save_state()
                    
                return simple_response
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in chat query: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def clear_context(self):
        """Clear the context history."""
        if self.context_manager.enabled:
            self.context_manager.clear_history()
            self.context_manager.save_state()
            if self.logger:
                self.logger.info("Context history cleared")
                
    def toggle_external_search(self, enabled: bool):
        """Toggle external search functionality."""
        if self.search_manager:
            self.search_manager.enabled = enabled
            if self.logger:
                status = "enabled" if enabled else "disabled"
                self.logger.info(f"External search {status}")
            return True
        return False

    def initialize_embeddings(self):
        """Initialize the embeddings model"""
        # Use a more powerful embedding model
        if self.embedding_model_name == "openai":
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        elif self.embedding_model_name == "huggingface":
            # Use a more powerful model like BGE or E5
            model_name = self.config.get('embeddings', {}).get('model_name', "BAAI/bge-large-en-v1.5")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            # Default to using the model name directly
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            ) 

    def clean_generated_text(self, text):
        """Clean up generated text to remove artifacts."""
        # Remove any non-alphanumeric sequences that look like garbage
        # Remove sequences of random characters
        text = re.sub(r'[A-Za-z]{1,2}(\s+[A-Za-z]{1,2}){3,}', '', text)
        
        # Remove sequences that look like code or garbage
        text = re.sub(r'[^\w\s.,;:?!\'"-]{3,}', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text