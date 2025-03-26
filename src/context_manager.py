import json
from pathlib import Path
from typing import List, Dict
import datetime

class ContextManager:
    """Manages conversation context for context-aware responses"""
    
    def __init__(self, config, logger=None):
        """Initialize the context manager"""
        self.config = config
        self.logger = logger
        self.history = []
        self.max_history = config.get('context_awareness', {}).get('max_history', 10)
        self.enabled = config.get('context_awareness', {}).get('enabled', False)
        self.state_file = Path("data/context/context_state.json")
        
    def add_to_history(self, role, content):
        """Add a message to the conversation history"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history if it exceeds the maximum
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
    def clear_history(self):
        """Clear the conversation history"""
        self.history = []
        
    def get_history(self):
        """Get the conversation history"""
        return self.history
    
    def build_combined_context(self, docs, question):
        """
        Build a combined context from retrieved documents and conversation history
        """
        # Format documents
        docs_context = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get('filename', f'Document {i+1}')
            docs_context += f"From {source}:\n{doc.page_content}\n\n"
            
        # Format conversation history
        history_context = ""
        if self.history:
            history_context = "Previous conversation:\n"
            for msg in self.history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_context += f"{role}: {msg['content']}\n"
            
        # Combine contexts
        combined_context = f"Document Context:\n{docs_context}\n\n"
        if history_context:
            combined_context += f"Conversation Context:\n{history_context}\n\n"
            
        return combined_context
    
    def save_state(self):
        """Save the context state to a file"""
        if not self.enabled:
            return
            
        try:
            # Create the directory if it doesn't exist
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the state
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "history": self.history,
                    "timestamp": datetime.datetime.now().isoformat()
                }, f, indent=2)
                
            if self.logger:
                self.logger.debug("Context state saved")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving context state: {str(e)}")
    
    def load_state(self):
        """Load the context state from a file"""
        if not self.enabled:
            return
            
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.history = state.get("history", [])
                    
                if self.logger:
                    self.logger.debug(f"Context state loaded with {len(self.history)} messages")
            else:
                if self.logger:
                    self.logger.debug("No context state file found")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading context state: {str(e)}") 