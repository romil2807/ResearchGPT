from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import numpy as np

class ContextManager:
    """
    Manages context awareness for the research system.
    This includes conversation history, document context, and
    relevance scoring for dynamic context selection.
    """
    
    def __init__(self, config: dict, logger=None):
        """
        Initialize the context manager with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
            logger: Optional logger for logging
        """
        self.config = config
        self.logger = logger
        
        # Get context awareness settings
        self.context_config = config.get('context_awareness', {})
        self.enabled = self.context_config.get('enabled', False)
        self.history_size = self.context_config.get('history_size', 5)
        self.context_window = self.context_config.get('context_window', 3)
        self.memory_type = self.context_config.get('memory_type', 'conversation')
        self.use_metadata = self.context_config.get('use_metadata', True)
        self.cross_document = self.context_config.get('cross_document_context', True)
        self.similarity_threshold = self.context_config.get('similarity_threshold', 0.75)
        self.section_weighting = self.context_config.get('section_relevance_weighting', True)
        self.dynamic_selection = self.context_config.get('dynamic_context_selection', True)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize document context cache
        self.document_context_cache = {}
        
        # Initialize section/entity cache for cross-document connections
        self.entity_cache = {}
        self.section_cache = {}
        
        if self.logger:
            self.logger.info(f"Context manager initialized with settings: enabled={self.enabled}, history_size={self.history_size}")
    
    def add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (user/assistant)
            content: The message content
        """
        if not self.enabled:
            return
            
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Trim history if exceeds history_size
        if len(self.conversation_history) > self.history_size:
            self.conversation_history = self.conversation_history[-self.history_size:]
    
    def get_conversation_context(self) -> str:
        """
        Get the formatted conversation history as context.
        
        Returns:
            Formatted conversation history string
        """
        if not self.enabled or not self.conversation_history:
            return ""
            
        context = "Previous conversation:\n"
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "ResearchGPT"
            context += f"{role}: {msg['content']}\n"
            
        return context
    
    def get_document_context(self, documents: List[Dict], query: str) -> str:
        """
        Get document context with awareness of relevance to the query.
        
        Args:
            documents: List of retrieved documents
            query: The user's query
            
        Returns:
            Formatted document context string
        """
        if not self.enabled or not documents:
            return ""
            
        # If dynamic selection is enabled, sort documents by relevance scores
        if self.dynamic_selection:
            # Here you would use the embedding model to compute relevance
            # For now, we'll use a simple heuristic based on term overlap
            query_terms = set(query.lower().split())
            
            def relevance_score(doc):
                content = doc.get('page_content', '')
                content_terms = set(content.lower().split())
                overlap = len(query_terms.intersection(content_terms))
                return overlap / max(1, len(query_terms))
            
            documents = sorted(documents, key=relevance_score, reverse=True)
            
            # Limit to top context_window documents
            documents = documents[:self.context_window]
        
        # Build context string with metadata if enabled
        context = "Context from research papers:\n\n"
        
        for i, doc in enumerate(documents):
            # Handle both langchain Document objects and dictionary formats
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = doc.metadata
            else:
                content = doc.get('cleaned_text', '')
                metadata = doc.get('metadata', {})
            
            # Add metadata if enabled
            if self.use_metadata:
                title = metadata.get('title', f"Document {i+1}")
                authors = metadata.get('authors', [])
                year = metadata.get('year', '')
                
                context += f"Document: {title}\n"
                
                if authors:
                    if isinstance(authors, list):
                        context += f"Authors: {', '.join(authors)}\n"
                    else:
                        context += f"Authors: {authors}\n"
                        
                if year:
                    context += f"Year: {year}\n"
                    
                context += "\n"
            
            # Add content
            context += content[:1000] + "...\n\n"
            
        return context
    
    def build_combined_context(self, documents: List[Dict], query: str) -> str:
        """
        Build a combined context from conversation history and document context.
        
        Args:
            documents: List of retrieved documents
            query: The user's query
            
        Returns:
            Combined context string for use in prompts
        """
        if not self.enabled:
            return ""
            
        context_parts = []
        
        # Get conversation context if using conversation memory type
        if self.memory_type == "conversation":
            conv_context = self.get_conversation_context()
            if conv_context:
                context_parts.append(conv_context)
        
        # Get document context
        doc_context = self.get_document_context(documents, query)
        if doc_context:
            context_parts.append(doc_context)
            
        # Combine contexts
        if context_parts:
            return "\n\n".join(context_parts)
        
        return ""
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        if self.logger:
            self.logger.info("Conversation history cleared")
            
    def track_entity_mentions(self, entity: str, document_id: str, section: str = None):
        """
        Track entity mentions across documents for cross-document context.
        
        Args:
            entity: The entity being tracked
            document_id: The document ID where entity was found
            section: Optional section where entity was found
        """
        if not self.enabled or not self.cross_document:
            return
            
        if entity not in self.entity_cache:
            self.entity_cache[entity] = []
            
        # Add document reference
        if document_id not in [ref['doc_id'] for ref in self.entity_cache[entity]]:
            self.entity_cache[entity].append({
                'doc_id': document_id,
                'section': section
            })
    
    def save_state(self, filepath: str = "data/context_state.json"):
        """
        Save the current context state to a file.
        
        Args:
            filepath: Path to save the state
        """
        if not self.enabled:
            return
            
        state = {
            'conversation_history': self.conversation_history,
            'entity_cache': self.entity_cache,
            'section_cache': self.section_cache
        }
        
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
                
            if self.logger:
                self.logger.info(f"Context state saved to {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving context state: {str(e)}")
    
    def load_state(self, filepath: str = "data/context_state.json"):
        """
        Load context state from a file.
        
        Args:
            filepath: Path to load the state from
        """
        if not self.enabled:
            return
            
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                self.conversation_history = state.get('conversation_history', [])
                self.entity_cache = state.get('entity_cache', {})
                self.section_cache = state.get('section_cache', {})
                
                if self.logger:
                    self.logger.info(f"Context state loaded from {filepath}")
            else:
                if self.logger:
                    self.logger.info(f"No context state file found at {filepath}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading context state: {str(e)}") 