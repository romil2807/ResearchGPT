import numpy as np
import faiss
from typing import List, Dict, Any
import json
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class EmbeddingsManager:
    def __init__(self, config: dict, logger=None):
        """
        Initialize the embeddings manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.embedding_model = config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        self.completion_model = config.get('completion_model', 'deepseek-ai/deepseek-coder-6.7b-instruct')
        
        # Initialize local models
        self.embedding_model = SentenceTransformer(self.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.completion_model)
        self.llm = AutoModelForCausalLM.from_pretrained(self.completion_model)
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            device=0 if config.get('use_gpu', False) else -1
        )
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
    def create_embeddings(self, processed_docs: List[Dict[str, Any]]) -> None:
        """
        Create embeddings for processed documents.
        """
        try:
            embeddings = []
            for doc in processed_docs:
                # Create chunks from the document
                chunks = self._create_chunks(doc['cleaned_text'])
                
                # Get embeddings for each chunk
                for chunk in chunks:
                    # Local embedding generation
                    embedding = self.embedding_model.encode(chunk['text'])
                    embeddings.append(embedding)
                    
                    # Store document info
                    self.documents.append({
                        'text': chunk['text'],
                        'metadata': {
                            'title': doc['metadata']['title'],
                            'filename': doc['filename'],
                            'chunk_index': chunk['index']
                        }
                    })
            
            # Create FAISS index
            embedding_dim = len(embeddings[0])
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.index.add(np.array(embeddings, dtype=np.float32))
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def _create_chunks(self, text: str, max_tokens: int = 1000) -> List[Dict[str, Any]]:
        """
        Split text into chunks suitable for embedding.
        """
        chunks = []
        tokens = self.tokenizer.encode(text)
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_size += 1
            
            if current_size >= max_tokens:
                chunks.append({
                    'text': self.tokenizer.decode(current_chunk),
                    'index': chunk_index
                })
                current_chunk = []
                current_size = 0
                chunk_index += 1
        
        if current_chunk:
            chunks.append({
                'text': self.tokenizer.decode(current_chunk),
                'index': chunk_index
            })
        
        return chunks
    
    def query(self, question: str, k: int = 3) -> str:
        """
        Query the document collection using semantic search.
        """
        try:
            # Local embedding generation
            question_embedding = self.embedding_model.encode(question)
            
            # Search similar chunks
            D, I = self.index.search(
                np.array([question_embedding], dtype=np.float32), 
                k
            )
            
            # Prepare context from similar chunks
            context = []
            for idx in I[0]:
                doc = self.documents[idx]
                context.append(f"From {doc['metadata']['title']}:\n{doc['text']}")
            
            # Modified prompt for local model
            prompt = f"""### Instruction:
Answer the research question using the provided context.
Include citations in [Author, Year] format when possible.

### Context:
{'\n\n'.join(context)}

### Question:
{question}

### Response:"""
            
            # Local model generation
            response = self.generator(
                prompt,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True
            )
            
            return response[0]['generated_text'].split("### Response:")[-1].strip()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying: {str(e)}")
            return f"Error processing query: {str(e)}" 