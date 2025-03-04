import numpy as np
import faiss
from typing import List, Dict, Any
import json
from pathlib import Path
import tiktoken
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

class EmbeddingsManager:
    def __init__(self, config: dict, logger=None):
        """Initialize the embeddings manager with LangChain components."""
        self.config = config
        self.logger = logger
        
        # Get models config
        models_config = config.get('models', {})
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=models_config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')
        )
        
        # Try to load existing vectorstore
        try:
            self.vectorstore = Chroma(
                persist_directory="data/embeddings",
                embedding_function=self.embedding_model
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
            max_new_tokens=150,      # Control length of new generated tokens
            max_length=1024,         # Maximum total length including input
            num_return_sequences=1,  # Only generate one response
            temperature=0.2,         # Control randomness (0.0 to 1.0)
            do_sample=True,          # Use sampling instead of greedy decoding
            top_k=50,               # Top k sampling
            top_p=0.95,             # Nucleus sampling
            pad_token_id=self.tokenizer.eos_token_id,  # Proper padding token
            eos_token_id=self.tokenizer.eos_token_id   # Proper end of sequence token
        )
        
        # Create LangChain LLM wrapper with updated class
        self.llm_chain = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=models_config.get('max_tokens_per_chunk', 1000),
            chunk_overlap=200
        )

    def create_embeddings(self, processed_docs: List[Dict[str, Any]]) -> None:
        """Create embeddings using LangChain."""
        try:
            documents = []
            for doc in processed_docs:
                # Create LangChain Document object
                documents.append(
                    Document(
                        page_content=doc['cleaned_text'],
                        metadata={
                            "title": doc['metadata']['title'],
                            "filename": doc['filename']
                        }
                    )
                )
            
            # Split documents
            splits = self.text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_model,
                persist_directory="data/embeddings"
            )
            
            # Persist the vectorstore
            self.vectorstore.persist()
            
            if self.logger:
                self.logger.info("Embeddings created and persisted successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating embeddings: {str(e)}")

    def query(self, question: str, k: int = 3) -> str:
        """Query using LangChain retrieval chain."""
        try:
            # Create prompt template
            prompt_template = """Use the following pieces of context to answer the question. Include citations in [Author, Year] format when possible.

            Context: {context}

            Question: {question}

            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create retrieval chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm_chain,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            # Get answer
            result = qa_chain.invoke({"query": question})
            return result["result"]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying: {str(e)}")
            return f"Error processing query: {str(e)}" 