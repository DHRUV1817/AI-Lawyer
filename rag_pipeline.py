"""
rag_pipeline.py - Enhanced RAG pipeline with state-of-the-art features for LegalMind AI
"""
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from utils import clean_response
import os
from typing import List, Tuple, Dict, Any, Optional
import time
import json

# Import project configuration
from config import GROQ_API_KEY, MAX_RETRIES, LLM_MODELS, LOGS_DIR

# Enhanced LLM setup with multiple model options
def get_llm(model: str = "deepseek-r1-distill-llama-70b", temperature: float = 0.2):
    """
    Get LLM with proper error handling, multiple model options, and temperature control
    
    Args:
        model: The model to use
        temperature: The temperature setting (0.0 to 1.0)
        
    Returns:
        Configured LLM or None if error
    """
    groq_api_key = GROQ_API_KEY
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        return None
    
    # Validate model name
    valid_models = list(LLM_MODELS.keys())
    
    if model not in valid_models:
        print(f"Warning: Model {model} not in known list. Defaulting to deepseek-r1-distill-llama-70b.")
        model = "deepseek-r1-distill-llama-70b"
    
    # Clamp temperature
    temperature = max(0.0, min(1.0, temperature))
    
    try:
        return ChatGroq(
            model=model, 
            api_key=groq_api_key,
            temperature=temperature
        )
    except Exception as e:
        print(f"Error initializing Groq LLM: {e}")
        return None

# Enhanced legal prompt template with improved context handling and source attribution
LEGAL_PROMPT_TEMPLATE = """
You are LegalMind AI, an advanced AI Legal Assistant specializing in Indian law and legal analysis. 
Use the legal information provided in the context to answer the user's question about Indian law.

Focus only on the legal aspects described in the provided context. Your primary objective is to provide 
accurate, well-structured legal analysis based on the information you have been given.

Important guidelines:
1. Stay focused on the legal information in the provided context - do not invent or assume legal principles not present.
2. For each important point or claim you make, refer to the source document by indicating the document number.
3. Structure your response clearly with appropriate headers and bullet points for complex answers.
4. Use legal terminology correctly and precisely.
5. If the question has multiple parts, address each part systematically.
6. If you don't know the answer, clearly state that you don't have enough information.
7. Always end with a disclaimer that this is not legal advice.

Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the question:
Question: {question} 

Format your answer professionally and ensure you cite the document numbers for key information. Remember, you're a legal assistant helping with research and analysis.
"""

# Advanced retrieval function with hybrid search
def retrieve_docs(vector_db, query, k=5, retrieval_method="hybrid"):
    """
    Retrieve relevant documents from vector database with improved retrieval options
    
    Args:
        vector_db: The vector database
        query: User query
        k: Number of documents to retrieve
        retrieval_method: "similarity", "mmr", or "hybrid"
        
    Returns:
        List of retrieved documents
    """
    if vector_db is None:
        return []
    
    try:
        if retrieval_method == "similarity":
            # Standard similarity search
            return vector_db.similarity_search(query, k=k)
        elif retrieval_method == "mmr":
            # Maximum Marginal Relevance - better diversity
            return vector_db.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
        elif retrieval_method == "hybrid":
            # Hybrid approach: combine similarity and MMR
            similarity_docs = vector_db.similarity_search(query, k=int(k/2))
            mmr_docs = vector_db.max_marginal_relevance_search(query, k=k-len(similarity_docs))
            
            # Combine and deduplicate
            all_docs = similarity_docs + mmr_docs
            unique_docs = []
            content_set = set()
            
            for doc in all_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in content_set:
                    content_set.add(content_hash)
                    unique_docs.append(doc)
                    
                    # Break if we have enough documents
                    if len(unique_docs) >= k:
                        break
                        
            return unique_docs
        else:
            print(f"Unknown retrieval method: {retrieval_method}. Using similarity search.")
            return vector_db.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def get_context(documents):
    """
    Extract content from documents to create context with document identifiers
    
    Args:
        documents: List of documents
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant information found."
    
    context_parts = []
    for i, doc in enumerate(documents):
        # Add document identifier and metadata if available
        doc_id = f"Document {i+1}"
        source = f"(Source: {doc.metadata.get('source', 'Unknown')})" if hasattr(doc, 'metadata') and doc.metadata.get('source') else ""
        page = f", Page {doc.metadata.get('page', 'Unknown')}" if hasattr(doc, 'metadata') and doc.metadata.get('page') else ""
        
        context_parts.append(f"{doc_id} {source}{page}:\n{doc.page_content}\n")
    
    return "\n".join(context_parts)

def answer_query(
    vector_db, 
    query, 
    max_retries=2, 
    retrieval_method="hybrid",
    show_sources=False,
    k=5,
    temperature=0.2,
    model="deepseek-r1-distill-llama-70b"
) -> Tuple[str, Optional[List[str]]]:
    """
    Generate answer using enhanced RAG pipeline with improved features
    
    Args:
        vector_db: Vector database
        query: User query
        max_retries: Number of retries on failure
        retrieval_method: Method for document retrieval
        show_sources: Whether to return source information
        k: Number of documents to retrieve
        temperature: LLM temperature
        model: LLM model to use
        
    Returns:
        Tuple of (response text, source documents if requested)
    """
    if not query or not vector_db:
        return "Either your question or the document is missing. Please check and try again.", None
    
    # Initialize LLM if needed
    llm = get_llm(model=model, temperature=temperature)
    if not llm:
        return "Unable to initialize the language model. Please check your API key and try again.", None
    
    # Track performance metrics
    start_time = time.time()
    
    # Try answering with retries
    for attempt in range(max_retries):
        try:
            # Retrieve relevant documents with timing
            retrieval_start = time.time()
            documents = retrieve_docs(vector_db, query, k=k, retrieval_method=retrieval_method)
            retrieval_time = time.time() - retrieval_start
            
            if not documents:
                return "I couldn't find relevant information in the document to answer your question. Please try a different question or upload a document with the relevant information.", None
            
            # Get context from documents
            context_start = time.time()
            context = get_context(documents)
            context_time = time.time() - context_start
            
            # Create prompt and chain
            llm_start = time.time()
            prompt = ChatPromptTemplate.from_template(LEGAL_PROMPT_TEMPLATE)
            chain = prompt | llm
            
            # Get and clean response
            response = chain.invoke({"question": query, "context": context})
            llm_time = time.time() - llm_start
            
            total_time = time.time() - start_time
            
            # Log performance metrics
            performance_metrics = {
                "retrieval_time": round(retrieval_time, 2),
                "context_time": round(context_time, 2),
                "llm_time": round(llm_time, 2),
                "total_time": round(total_time, 2)
            }
            
            # Log to console
            print(f"Performance: Retrieval: {retrieval_time:.2f}s, Context: {context_time:.2f}s, LLM: {llm_time:.2f}s, Total: {total_time:.2f}s")
            
            # Prepare source information if requested
            sources = None
            if show_sources:
                sources = []
                for i, doc in enumerate(documents):
                    source_info = f"Document {i+1}"
                    if hasattr(doc, 'metadata'):
                        if 'source' in doc.metadata:
                            source_info += f" | Source: {doc.metadata['source']}"
                        if 'page' in doc.metadata:
                            source_info += f" | Page: {doc.metadata['page']}"
                    
                    # Add a snippet of content
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    source_info += f"\nExcerpt: {content_preview}"
                    
                    sources.append(source_info)
            
            # Log the query for improvement
            save_query_log(query, clean_response(response), sources or [], performance_metrics)
                    
            # Return the response and sources if requested
            return clean_response(response), sources
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} failed: {e}. Retrying...")
                continue
            else:
                return f"I encountered an error while processing your question: {str(e)}. Please try again with a simpler query.", None

def save_query_log(query, response, sources, performance_metrics):
    """
    Save query logs for analysis and improvement
    
    Args:
        query: User query
        response: Generated response
        sources: Source documents used
        performance_metrics: Performance metrics
    """
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "response_length": len(response),
        "num_sources": len(sources) if sources else 0,
        "performance_metrics": performance_metrics
    }
    
    log_file = os.path.join(LOGS_DIR, f"query_log_{time.strftime('%Y%m%d')}.jsonl")
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def get_document_summary(vector_db, max_tokens=500):
    """
    Generate a summary of the document for quick overview
    
    Args:
        vector_db: Vector database containing the document
        max_tokens: Maximum tokens for summary
        
    Returns:
        Document summary
    """
    if not vector_db:
        return "No document loaded."
    
    # Get a sample of documents from the database
    try:
        # Get documents that represent the key sections
        docs = vector_db.similarity_search("summarize the main topics and key points of this document", k=5)
        
        if not docs:
            return "Unable to generate summary from this document."
        
        # Create a summary prompt
        summary_prompt = """
        You are a legal document summarization expert. Based on the following excerpts from a legal document,
        provide a concise summary (maximum 3 paragraphs) of what the document appears to be about, its key topics,
        and main legal points. Focus on the factual content only:
        
        {context}
        
        Brief Summary:
        """
        
        # Initialize LLM
        llm = get_llm(temperature=0.1)  # Low temperature for factual summary
        
        if not llm:
            return "Unable to initialize the language model for summary generation."
        
        # Extract context
        context = get_context(docs)
        
        # Generate summary
        prompt = ChatPromptTemplate.from_template(summary_prompt)
        chain = prompt | llm
        
        response = chain.invoke({"context": context})
        
        return clean_response(response)
    except Exception as e:
        print(f"Error generating document summary: {e}")
        return f"Unable to generate summary: {str(e)}"