"""
utils.py - Advanced utilities with improved processing and caching for LegalMind AI
"""
import os
import re
import hashlib
import json
import time
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
import shutil

# Langchain imports
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Import project configuration
from config import (
    PDFS_DIR, 
    VECTORSTORE_DIR, 
    CACHE_DIR, 
    LOGS_DIR, 
    EMBEDDING_MODELS
)

def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    """
    Get configured embedding model with appropriate settings
    
    Args:
        model_name: Name of the HuggingFace model to use
        
    Returns:
        Configured embedding model
    """
    if model_name not in EMBEDDING_MODELS:
        print(f"Warning: Unknown embedding model {model_name}. Defaulting to all-MiniLM-L6-v2.")
        model_name = "all-MiniLM-L6-v2"
    
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Improved retrieval with normalized embeddings
        )
    except Exception as e:
        print(f"Error initializing embedding model {model_name}: {e}")
        print("Falling back to all-MiniLM-L6-v2 model")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def get_file_hash(file_path):
    """
    Get a unique hash for a file to use for caching
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash of the file
    """
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as file:
            buf = file.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = file.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error calculating file hash: {e}")
        return None

def preprocess_text(text):
    """
    Clean and simplify text to improve processing quality
    
    Args:
        text: Raw text to process
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)  # Fix hyphenated words
    
    # Clean up formatting artifacts
    text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipses
    text = re.sub(r'_{3,}', '___', text)  # Normalize underscores
    
    # Fix quotation marks
    text = re.sub(r'``|\'\'', '"', text)
    
    # Remove headers/footers (common in legal documents)
    text = re.sub(r'^\s*page \d+\s*of \d+\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text.strip()

def extract_metadata_from_pdf(file_path):
    """
    Extract metadata from PDF file
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary of metadata
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        
        metadata = {
            "filename": os.path.basename(file_path),
            "num_pages": len(reader.pages),
            "pdf_info": {}
        }
        
        # Extract more specific metadata if available
        if reader.metadata:
            for key in reader.metadata:
                if reader.metadata[key]:
                    metadata["pdf_info"][key.lower()] = str(reader.metadata[key])
        
        return metadata
    except Exception as e:
        print(f"Error extracting PDF metadata: {e}")
        return {
            "filename": os.path.basename(file_path),
            "error": str(e)
        }

def get_text_splitter(chunking_method="standard", chunk_size=1500, chunk_overlap=150):
    """
    Get appropriate text splitter based on method
    
    Args:
        chunking_method: Method for splitting text ("standard" or "semantic")
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Configured text splitter
    """
    if chunking_method.lower() == "semantic":
        # Semantic chunking is better for preserving meaning across chunks
        try:
            return SentenceTransformersTokenTextSplitter(
                model_name="all-MiniLM-L6-v2",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except Exception as e:
            print(f"Error initializing semantic text splitter: {e}. Falling back to standard.")
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True
            )
    else:
        # Standard recursive splitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )

def detect_document_structure(text):
    """
    Detect if a document has a specific structure (like legal sections)
    
    Args:
        text: Document text
        
    Returns:
        Dictionary with structure information
    """
    structure_info = {
        "has_sections": False,
        "has_headers": False,
        "section_pattern": None
    }
    
    # Check for section patterns common in legal documents
    section_patterns = [
        r'Section \d+\.',
        r'§ \d+',
        r'Article \d+\.',
        r'Chapter \d+\.'
    ]
    
    for pattern in section_patterns:
        if re.search(pattern, text):
            structure_info["has_sections"] = True
            structure_info["section_pattern"] = pattern
            break
    
    # Check for markdown-style headers
    if re.search(r'#+\s+\w+', text):
        structure_info["has_headers"] = True
    
    return structure_info

def process_pdf(
    file_path, 
    progress_callback=None, 
    max_pages=0, 
    sample_mode=False,
    chunking_method="standard",
    embedding_model="all-MiniLM-L6-v2",
    force_reprocess=False,
    chunk_size=1500,
    chunk_overlap=150
):
    """
    Load and process a PDF file with enhanced features
    
    Args:
        file_path: Path to the PDF file
        progress_callback: Function to call with progress updates (0-100)
        max_pages: Maximum number of pages to process (0 for all)
        sample_mode: Whether to use sample mode (process only sample pages)
        chunking_method: Method for splitting text ("standard" or "semantic")
        embedding_model: Name of the embedding model to use
        force_reprocess: Force reprocessing even if cached version exists
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        FAISS vector database or None if processing failed
    """
    if not os.path.exists(file_path):
        if progress_callback:
            progress_callback(f"File not found: {file_path}", -1)
        return None

    try:
        # Start timing for performance metrics
        start_time = time.time()
        
        # Check for cached version first
        file_hash = get_file_hash(file_path)
        if not file_hash:
            if progress_callback:
                progress_callback("Could not calculate file hash", -1)
            return None
            
        # Build cache path with all parameters
        cache_parts = [file_hash]
        if sample_mode:
            cache_parts.append("sample")
        if max_pages > 0:
            cache_parts.append(f"max{max_pages}")
        cache_parts.append(chunking_method)
        cache_parts.append(embedding_model.replace("-", "_"))
        
        cache_name = "_".join(cache_parts)
        cache_path = os.path.join(VECTORSTORE_DIR, cache_name)
        
        # Check for cached version unless force reprocessing
        if not force_reprocess and os.path.exists(cache_path + ".faiss"):
            try:
                if progress_callback:
                    progress_callback("Loading from cache...", 20)
                    
                # Get the embedding model
                embeddings = get_embedding_model(embedding_model)
                
                # Load from cache
                vector_db = FAISS.load_local(cache_path, embeddings)
                
                if progress_callback:
                    progress_callback("Successfully loaded from cache", 100)
                    
                # Log cache hit
                print(f"Cache hit for {file_path} with parameters: {cache_name}")
                
                return vector_db
            except Exception as e:
                print(f"Error loading from cache: {e}, proceeding with normal processing")
                # If loading fails, proceed with normal processing
        
        # Extract metadata from PDF
        if progress_callback:
            progress_callback("Extracting document metadata...", 5)
            
        pdf_metadata = extract_metadata_from_pdf(file_path)
        
        # Update progress
        if progress_callback:
            progress_callback("Loading document...", 10)
        
        # Try different PDF loaders in case one fails
        documents = []
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
        except Exception as e:
            print(f"PDFPlumberLoader failed, trying PyPDFLoader: {e}")
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            except Exception as e2:
                if progress_callback:
                    progress_callback(f"Error loading PDF with multiple loaders: {e2}", -1)
                return None
        
        if not documents:
            if progress_callback:
                progress_callback("No content found in PDF", -1)
            return None
        
        # Apply page limits and sampling
        total_pages = len(documents)
        if max_pages > 0 and total_pages > max_pages:
            documents = documents[:max_pages]
            processed_pages = max_pages
        elif sample_mode:
            # In sample mode, take a representative sample throughout the document
            sample_size = min(10, total_pages)
            # Take pages from throughout the document, not just the beginning
            if total_pages > sample_size:
                step = total_pages // sample_size
                sample_indices = [i * step for i in range(sample_size)]
                documents = [documents[i] for i in sample_indices if i < total_pages]
            processed_pages = len(documents)
        else:
            processed_pages = total_pages
            
        # Add progress update
        if progress_callback:
            progress_callback(f"Processing {processed_pages} pages...", 20)
            
        # Apply document preprocessing to improve quality
        for i, doc in enumerate(documents):
            # Add page information to metadata
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['page'] = i + 1
            doc.metadata['source'] = os.path.basename(file_path)
            
            # Preprocess the content
            doc.page_content = preprocess_text(doc.page_content)
            
            # Progress update for large documents
            if progress_callback and i % 10 == 0 and total_pages > 20:
                progress_percentage = 20 + int((i / total_pages) * 20)
                progress_callback(f"Preprocessing page {i+1}/{total_pages}...", progress_percentage)
        
        # Update progress
        if progress_callback:
            progress_callback("Analyzing document structure...", 40)
            
        # Join all text for structure analysis
        all_text = "\n".join([doc.page_content for doc in documents])
        structure_info = detect_document_structure(all_text)
        
        # Get appropriate text splitter based on document structure
        if progress_callback:
            progress_callback(f"Splitting into chunks using {chunking_method} method...", 45)
            
        text_splitter = get_text_splitter(
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Split documents into chunks
        text_chunks = text_splitter.split_documents(documents)
        
        if not text_chunks:
            if progress_callback:
                progress_callback("No chunks created from document", -1)
            return None
            
        # Add progress update
        if progress_callback:
            progress_callback(f"Created {len(text_chunks)} chunks. Creating embeddings...", 60)
            
        # Create vector database with the selected embedding model
        try:
            embeddings = get_embedding_model(embedding_model)
            
            vector_db = FAISS.from_documents(
                text_chunks, 
                embeddings,
                normalize_L2=True  # Adds performance optimization
            )
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error creating embeddings: {e}", -1)
            return None
        
        # Add metadata to the vector database
        processing_metadata = {
            "document": pdf_metadata,
            "processing": {
                "timestamp": datetime.now().isoformat(),
                "num_pages": processed_pages,
                "total_pages": total_pages,
                "num_chunks": len(text_chunks),
                "embedding_model": embedding_model,
                "chunking_method": chunking_method,
                "sample_mode": sample_mode,
                "max_pages": max_pages,
                "processing_time": time.time() - start_time,
                "document_structure": structure_info
            }
        }
        
        # Save metadata
        metadata_path = f"{cache_path}_metadata.json"
        try:
            with open(metadata_path, "w") as f:
                json.dump(processing_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
        
        # Update progress
        if progress_callback:
            progress_callback("Saving to cache...", 85)
            
        # Save to cache for future use
        try:
            vector_db.save_local(cache_path)
        except Exception as e:
            print(f"Warning: Could not save to cache: {e}")
            # Continue even if saving to cache fails
        
        # Update progress
        if progress_callback:
            progress_callback("Processing complete!", 100)
            
        return vector_db
    except Exception as e:
        print(f"Error processing PDF: {e}")
        if progress_callback:
            progress_callback(f"Error: {str(e)}", -1)  # -1 indicates error
        return None

def clean_response(response):
    """
    Clean LLM response from any tags or extra formatting
    
    Args:
        response: Raw LLM response
        
    Returns:
        Cleaned text
    """
    if response is None:
        return "No response generated."
        
    if hasattr(response, 'content'):
        clean_text = response.content
    else:
        clean_text = str(response)
    
    # Remove thinking tags and their content
    clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)
    
    # Remove any remaining HTML tags
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    
    # Remove extra whitespace and newlines
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text.strip())
    
    return clean_text

def format_legal_document(text, format_type="markdown"):
    """
    Format a legal document for better readability
    
    Args:
        text: Raw document text
        format_type: Output format ("markdown", "html", or "text")
        
    Returns:
        Formatted document
    """
    # First, clean the text
    text = preprocess_text(text)
    
    # Replace section identifiers with formatted versions
    # Section numbers
    text = re.sub(r'(Section|SECTION)\s+(\d+)', r'## Section \2', text)
    
    # Legal section symbols
    text = re.sub(r'§\s*(\d+)', r'## § \1', text)
    
    # Format case names (italics)
    if format_type == "markdown":
        text = re.sub(r'([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+)', r'*\1*', text)
    elif format_type == "html":
        text = re.sub(r'([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+)', r'<em>\1</em>', text)
    
    # Format citations
    citation_pattern = r'\(\d+\s+[A-Za-z\.]+\s+\d+(?:,\s*\d+)?\)'
    if format_type == "markdown":
        text = re.sub(citation_pattern, r'`\g<0>`', text)
    elif format_type == "html":
        text = re.sub(citation_pattern, r'<code>\g<0></code>', text)
    
    return text

def list_cached_documents():
    """
    List all cached documents
    
    Returns:
        List of dictionaries with document information
    """
    documents = []
    
    if not os.path.exists(VECTORSTORE_DIR):
        return documents
        
    # Get all FAISS files
    for file in os.listdir(VECTORSTORE_DIR):
        if file.endswith(".faiss"):
            base_name = file[:-6]  # Remove .faiss extension
            
            # Look for metadata file
            metadata_file = os.path.join(VECTORSTORE_DIR, f"{base_name}_metadata.json")
            metadata = {}
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                except Exception:
                    pass
            
            document_info = {
                "id": base_name,
                "filename": metadata.get("document", {}).get("filename", base_name),
                "processed_at": metadata.get("processing", {}).get("timestamp", "Unknown"),
                "pages": metadata.get("processing", {}).get("num_pages", "Unknown"),
                "chunks": metadata.get("processing", {}).get("num_chunks", "Unknown"),
                "embedding_model": metadata.get("processing", {}).get("embedding_model", "Unknown")
            }
            
            documents.append(document_info)
    
    return documents

def delete_cached_document(document_id):
    """
    Delete a cached document
    
    Args:
        document_id: ID of the document to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete .faiss file
        faiss_file = os.path.join(VECTORSTORE_DIR, f"{document_id}.faiss")
        if os.path.exists(faiss_file):
            os.remove(faiss_file)
            
        # Delete .pkl file
        pkl_file = os.path.join(VECTORSTORE_DIR, f"{document_id}.pkl")
        if os.path.exists(pkl_file):
            os.remove(pkl_file)
            
        # Delete metadata file
        metadata_file = os.path.join(VECTORSTORE_DIR, f"{document_id}_metadata.json")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            
        return True
    except Exception as e:
        print(f"Error deleting cached document: {e}")
        return False

def clear_cache():
    """
    Clear all cache files
    
    Returns:
        Number of files deleted
    """
    count = 0
    try:
        if os.path.exists(VECTORSTORE_DIR):
            for file in os.listdir(VECTORSTORE_DIR):
                file_path = os.path.join(VECTORSTORE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
                    
        if os.path.exists(CACHE_DIR):
            for file in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
                    
        return count
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return count