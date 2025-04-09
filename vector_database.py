"""
vector_database.py - Advanced vector database management for LegalMind AI
"""
import os
import json
import time
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Union, Any

# Langchain imports
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

# Import project configuration
from config import VECTORSTORE_DIR, METADATA_DIR, PDFS_DIR, VECTOR_DB_TYPES

class VectorDatabaseManager:
    """Advanced vector database management system"""
    
    def __init__(self, db_type="faiss", embedding_model="all-MiniLM-L6-v2", persist_directory=None):
        """
        Initialize a vector database manager
        
        Args:
            db_type: Type of vector database ("faiss" or "chroma")
            embedding_model: Name of the embedding model to use
            persist_directory: Directory to persist database
        """
        self.db_type = db_type
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory or VECTORSTORE_DIR
        self.embedding_model = self._get_embedding_model(embedding_model)
        self.vector_db = None
        self.collection_name = None
        self.metadata = {}
        
    def _get_embedding_model(self, model_name):
        """
        Get embedding model based on name
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Configured embedding model
        """
        try:
            if model_name.startswith("ollama:"):
                # Use Ollama model if specified with ollama: prefix
                ollama_model = model_name.split(":", 1)[1]
                return OllamaEmbeddings(model=ollama_model)
            else:
                # Use HuggingFace embeddings
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            print(f"Error initializing embedding model {model_name}: {e}")
            print("Falling back to all-MiniLM-L6-v2 model")
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def create_new_db(self, documents: List[Document], collection_name: str) -> bool:
        """
        Create a new vector database from documents
        
        Args:
            documents: List of documents to add to the database
            collection_name: Name of the collection to create
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            print("Error: No documents provided")
            return False
            
        try:
            self.collection_name = collection_name
            start_time = time.time()
            
            # Create the vector database based on type
            if self.db_type == "faiss":
                self.vector_db = FAISS.from_documents(
                    documents,
                    self.embedding_model,
                    normalize_L2=True
                )
                
                # Save the database
                db_path = os.path.join(self.persist_directory, collection_name)
                self.vector_db.save_local(db_path)
                
            elif self.db_type == "chroma":
                db_path = os.path.join(self.persist_directory, collection_name)
                self.vector_db = Chroma.from_documents(
                    documents,
                    self.embedding_model,
                    persist_directory=db_path
                )
                
                # Persist the database
                if hasattr(self.vector_db, 'persist'):
                    self.vector_db.persist()
            else:
                print(f"Error: Unsupported database type {self.db_type}")
                return False
                
            # Save metadata about the collection
            self.metadata = {
                "collection_name": collection_name,
                "db_type": self.db_type,
                "embedding_model": self.embedding_model_name,
                "created_at": datetime.now().isoformat(),
                "document_count": len(documents),
                "processing_time": time.time() - start_time
            }
            
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Error creating vector database: {e}")
            return False
            
    def load_db(self, collection_name: str) -> bool:
        """
        Load an existing vector database
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection_name = collection_name
            db_path = os.path.join(self.persist_directory, collection_name)
            
            # Check if the database exists
            if self.db_type == "faiss":
                if not (os.path.exists(f"{db_path}.faiss") and os.path.exists(f"{db_path}.pkl")):
                    print(f"Error: Vector database {collection_name} not found")
                    return False
                    
                self.vector_db = FAISS.load_local(db_path, self.embedding_model)
                
            elif self.db_type == "chroma":
                if not os.path.exists(db_path):
                    print(f"Error: Vector database {collection_name} not found")
                    return False
                    
                self.vector_db = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embedding_model
                )
            else:
                print(f"Error: Unsupported database type {self.db_type}")
                return False
                
            # Load metadata
            self._load_metadata()
            
            return True
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False
            
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to an existing vector database
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vector_db:
            print("Error: No vector database loaded")
            return False
            
        if not documents:
            print("Warning: No documents provided")
            return True
            
        try:
            # Add documents to the database
            if self.db_type == "faiss":
                self.vector_db.add_documents(documents)
                
                # Save the updated database
                db_path = os.path.join(self.persist_directory, self.collection_name)
                self.vector_db.save_local(db_path)
                
            elif self.db_type == "chroma":
                self.vector_db.add_documents(documents)
                
                # Persist the database
                if hasattr(self.vector_db, 'persist'):
                    self.vector_db.persist()
                    
            # Update metadata
            if "document_count" in self.metadata:
                self.metadata["document_count"] += len(documents)
            else:
                self.metadata["document_count"] = len(documents)
                
            self.metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")
            return False
            
    def search(self, query: str, k: int = 4, search_type: str = "similarity") -> List[Document]:
        """
        Search the vector database
        
        Args:
            query: Query string
            k: Number of results to return
            search_type: Type of search ("similarity" or "mmr")
            
        Returns:
            List of relevant documents
        """
        if not self.vector_db:
            print("Error: No vector database loaded")
            return []
            
        try:
            if search_type == "similarity":
                return self.vector_db.similarity_search(query, k=k)
            elif search_type == "mmr":
                return self.vector_db.max_marginal_relevance_search(query, k=k)
            else:
                print(f"Error: Unsupported search type {search_type}")
                return self.vector_db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []
            
    def delete_collection(self) -> bool:
        """
        Delete the current collection
        
        Returns:
            True if successful, False otherwise
        """
        if not self.collection_name:
            print("Error: No collection loaded")
            return False
            
        try:
            db_path = os.path.join(self.persist_directory, self.collection_name)
            
            if self.db_type == "faiss":
                # Delete FAISS files
                if os.path.exists(f"{db_path}.faiss"):
                    os.remove(f"{db_path}.faiss")
                if os.path.exists(f"{db_path}.pkl"):
                    os.remove(f"{db_path}.pkl")
            elif self.db_type == "chroma":
                # Delete Chroma directory
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)
            
            # Delete metadata file
            metadata_path = os.path.join(METADATA_DIR, f"{self.collection_name}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            self.vector_db = None
            self.collection_name = None
            self.metadata = {}
            
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "collection_name": self.collection_name,
            "db_type": self.db_type,
            "embedding_model": self.embedding_model_name
        }
        
        # Add metadata
        stats.update(self.metadata)
        
        return stats
        
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections
        
        Returns:
            List of collection information
        """
        collections = []
        
        # Get all metadata files
        metadata_dir = METADATA_DIR
        
        for file in os.listdir(metadata_dir):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(metadata_dir, file), "r") as f:
                        metadata = json.load(f)
                        collections.append(metadata)
                except Exception as e:
                    print(f"Error loading metadata for {file}: {e}")
                    
        return collections
        
    def _save_metadata(self) -> None:
        """Save metadata to file"""
        if not self.collection_name:
            return
            
        try:
            metadata_path = os.path.join(METADATA_DIR, f"{self.collection_name}.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")
            
    def _load_metadata(self) -> None:
        """Load metadata from file"""
        if not self.collection_name:
            return
            
        try:
            metadata_path = os.path.join(METADATA_DIR, f"{self.collection_name}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata = {}

# Helper functions for document preparation
def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    
    return text_splitter.split_documents(documents)

def process_pdf(file_path: str, max_pages: int = 0) -> List[Document]:
    """
    Process a PDF file into documents
    
    Args:
        file_path: Path to the PDF file
        max_pages: Maximum number of pages to process (0 for all)
        
    Returns:
        List of documents
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return []
        
    try:
        # Try PDFPlumberLoader first
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
        except Exception as e:
            print(f"PDFPlumberLoader failed: {e}, trying PyPDFLoader")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
        # Apply page limit if specified
        if max_pages > 0 and len(documents) > max_pages:
            documents = documents[:max_pages]
            
        return documents
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

def create_vector_db_from_pdf(
    file_path: str,
    collection_name: str,
    db_type: str = "faiss",
    embedding_model: str = "all-MiniLM-L6-v2",
    max_pages: int = 0,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Optional[VectorDatabaseManager]:
    """
    Create a vector database from a PDF file
    
    Args:
        file_path: Path to the PDF file
        collection_name: Name of the collection to create
        db_type: Type of vector database
        embedding_model: Name of the embedding model
        max_pages: Maximum number of pages to process
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        VectorDatabaseManager if successful, None otherwise
    """
    try:
        # Process the PDF
        documents = process_pdf(file_path, max_pages)
        
        if not documents:
            print(f"Error: No documents extracted from {file_path}")
            return None
            
        # Split documents into chunks
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        
        if not chunks:
            print(f"Error: No chunks created from documents")
            return None
            
        # Create vector database
        manager = VectorDatabaseManager(db_type, embedding_model)
        success = manager.create_new_db(chunks, collection_name)
        
        if not success:
            print("Error creating vector database")
            return None
            
        return manager
    except Exception as e:
        print(f"Error creating vector database from PDF: {e}")
        return None