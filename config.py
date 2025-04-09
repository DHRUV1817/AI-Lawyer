"""
Configuration module for LegalMind AI application.
Centralizes all configuration settings and directory paths.
"""
import os
from dotenv import load_dotenv

# Load environment variables - DON'T modify the .env file content after loading
load_dotenv(override=True)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Create necessary directories
PDFS_DIR = os.path.join(DATA_DIR, "pdfs")
VECTORSTORE_DIR = os.path.join(DATA_DIR, "vectorstore")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")

# Create all directories if they don't exist
for directory in [DATA_DIR, PDFS_DIR, VECTORSTORE_DIR, METADATA_DIR, CACHE_DIR, LOGS_DIR, CONVERSATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model settings
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "deepseek-r1-distill-llama-70b")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Available embedding models
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast, lightweight model good for general use"
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2",
        "description": "More accurate but slightly slower model"
    }
}

# Available LLM models
LLM_MODELS = {
    "deepseek-r1-distill-llama-70b": {
        "name": "deepseek-r1-distill-llama-70b",
        "description": "Balanced performance and speed"
    },
    "llama3-70b-8192": {
        "name": "llama3-70b-8192",
        "description": "High quality with longer context"
    },
    "mixtral-8x7b-32768": {
        "name": "mixtral-8x7b-32768",
        "description": "Best for complex reasoning"
    }
}

# Vector database types
VECTOR_DB_TYPES = {
    "faiss": {
        "description": "Fast, efficient vector database, good for most use cases"
    },
    "chroma": {
        "description": "Persistent vector database with advanced filtering"
    }
}

# Performance settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))