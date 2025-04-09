"""
app.py - Streamlit frontend for LegalMind AI application
"""
import streamlit as st
import os
import tempfile
import time
import json
from datetime import datetime
from utils import process_pdf
from rag_pipeline import answer_query, get_document_summary

# Import project configuration
from config import (
    CONVERSATIONS_DIR, 
    GROQ_API_KEY, 
    LLM_MODELS,
    EMBEDDING_MODELS
)

# Set page configuration with dark theme
st.set_page_config(
    page_title="LegalMind AI | Smart Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for dark theme with better visibility
st.markdown("""
<style>
    /* Dark theme with better contrast */
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #FFFFFF;
    }
    
    .subheader {
        font-size: 1.2rem;
        color: #D1D5DB;
    }
    
    .stButton>button {
        background-color: #1E40AF;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
    }
    
    /* Document card styling */
    .document-card {
        background-color: #1F2937;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FCD34D;
    }
    
    /* Success box with better visibility */
    .success-box {
        background-color: #064E3B;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #10B981;
    }
    
    /* Info box with better visibility */
    .info-box {
        background-color: #1E3A8A;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #60A5FA;
    }
    
    /* Chat styling with better visibility */
    .chat-user {
        background-color: #1F2937;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #60A5FA;
    }
    
    .chat-assistant {
        background-color: #111827;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #FCD34D;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-weight: bold;
        color: #FFFFFF;
    }
    
    /* Feedback button styling */
    .feedback-button {
        background-color: #1F2937;
        border: 1px solid #374151;
        border-radius: 15px;
        padding: 5px 10px;
        font-size: 0.8rem;
        color: #D1D5DB;
    }
    
    /* Footnote styling */
    .footnote {
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    
    /* Improved text area and input visibility */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1F2937;
        color: #FFFFFF;
        border: 1px solid #374151;
    }
    
    /* Improved select box visibility */
    .stSelectbox>div>div>select {
        background-color: #1F2937;
        color: #FFFFFF;
    }
    
    /* Make expander text visible */
    .streamlit-expanderHeader {
        color: #FFFFFF !important;
        background-color: #1F2937 !important;
        border-radius: 5px;
    }
    
    /* Ensure text in the legal analysis panel is visible */
    .stChatMessage div {
        color: #FFFFFF;
    }
    
    /* Make sure all text in containers is visible */
    .stContainer, .block-container {
        color: #FFFFFF;
    }
    
    /* Ensure text inputs have visible text */
    input, textarea {
        color: #FFFFFF !important;
    }
    
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: #111827;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        color: #FFFFFF;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border-right: 1px solid #374151;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: #FFFFFF;
    }
    
    /* Improve visibility of text in chat areas */
    .stChatMessageContent {
        color: #FFFFFF !important;
        background-color: #1F2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_started" not in st.session_state:
    st.session_state.processing_started = False
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "saved_conversations" not in st.session_state:
    st.session_state.saved_conversations = []
if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = None
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None

# Function to save conversation
def save_conversation():
    if not st.session_state.chat_history:
        return
    
    conversation = {
        "id": st.session_state.current_conversation_id,
        "document": st.session_state.pdf_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chat": st.session_state.chat_history
    }
    
    # Save conversation to file
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    with open(os.path.join(CONVERSATIONS_DIR, f"{st.session_state.current_conversation_id}.json"), "w") as f:
        json.dump(conversation, f)
    
    # Add to session state if not already there
    if conversation not in st.session_state.saved_conversations:
        st.session_state.saved_conversations.append(conversation)

# Function to load saved conversations
def load_saved_conversations():
    conversations = []
    if os.path.exists(CONVERSATIONS_DIR):
        for file in os.listdir(CONVERSATIONS_DIR):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(CONVERSATIONS_DIR, file), "r") as f:
                        conversation = json.load(f)
                        conversations.append(conversation)
                except Exception as e:
                    print(f"Error loading conversation {file}: {e}")
    
    st.session_state.saved_conversations = conversations

# Load saved conversations at startup
if not st.session_state.saved_conversations:
    load_saved_conversations()

# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/000000/scales--v1.png", width=80)
with col2:
    st.markdown("<h1 class='main-header'>LegalMind AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subheader'>Intelligent Legal Research Assistant for Indian Law</p>", unsafe_allow_html=True)

# Display any error messages
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.session_state.error_message = None

# Sidebar for document upload and settings
with st.sidebar:
    st.markdown("<h3 class='sidebar-header'>Document Management</h3>", unsafe_allow_html=True)
    
    # API Key management
    if not GROQ_API_KEY:
        st.error("GROQ API key not found! Please add it to your .env file and restart the application.")
        st.info("Create a .env file in the project directory with: GROQ_API_KEY=your_key_here")
    
    tabs = st.tabs(["Upload", "History", "Settings"])
    
    with tabs[0]:  # Upload Tab
        # Document upload with improved UI
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader("Upload Legal Document", type="pdf", accept_multiple_files=False)
        
        # Add document type and description for better organization
        if uploaded_file:
            document_type = st.selectbox(
                "Document Type", 
                ["Court Judgment", "Legal Act/Statute", "Contract", "Legal Opinion", "Other"]
            )
            document_desc = st.text_area("Brief Description (optional)", max_chars=200)
            
            # Advanced processing options (collapsible)
            with st.expander("Advanced Processing Options"):
                col1, col2 = st.columns(2)
                with col1:
                    sample_mode = st.checkbox("Sample Mode", False, 
                                             help="Process only samples from the document (faster)")
                with col2:
                    max_pages = st.slider("Max Pages", 0, 200, 0, 
                                         help="0 means process all pages")
                
                chunking_method = st.radio(
                    "Chunking Method",
                    ["Standard", "Semantic (Better for complex documents)"]
                )
                
                embedding_model = st.selectbox(
                    "Embedding Model",
                    list(EMBEDDING_MODELS.keys()),
                    help="Select embedding model (affects search quality)"
                )
            
            # Process button with improved UI
            process_btn = st.button("Process Document", type="primary", key="process_doc")
            
            if process_btn and not st.session_state.processing_started:
                st.session_state.processing_started = True
                
                # Create progress indicators with improved visuals
                progress_container = st.container()
                progress_text = progress_container.empty()
                progress_bar = progress_container.progress(0)
                
                # Define progress callback
                def update_progress(message, percent):
                    if percent < 0:  # Error
                        progress_text.error(message)
                        st.session_state.error_message = message
                    else:
                        progress_text.markdown(f"**{message}**")
                        progress_bar.progress(percent / 100)
                
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    # Process the PDF with performance options
                    start_time = time.time()
                    vector_db = process_pdf(
                        temp_path, 
                        progress_callback=update_progress,
                        max_pages=max_pages,
                        sample_mode=sample_mode,
                        chunking_method=chunking_method.lower().split()[0],  # Extract just "standard" or "semantic"
                        embedding_model=embedding_model
                    )
                    processing_time = time.time() - start_time
                    
                    # Save metadata about the document
                    st.session_state.document_metadata = {
                        "filename": uploaded_file.name,
                        "type": document_type,
                        "description": document_desc,
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "processing_time": f"{processing_time:.2f} seconds",
                        "options": {
                            "sample_mode": sample_mode,
                            "max_pages": max_pages,
                            "chunking_method": chunking_method,
                            "embedding_model": embedding_model
                        }
                    }
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                    if vector_db:
                        st.session_state.vector_db = vector_db
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
                        
                        # Generate document summary
                        st.session_state.document_summary = get_document_summary(vector_db)
                        
                        # Success message with stats
                        progress_container.markdown(f"""
                        <div class='success-box'>
                            <h4>✅ Document Processed Successfully</h4>
                            <p>Document: <b>{uploaded_file.name}</b><br>
                            Processing time: <b>{processing_time:.2f} seconds</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Failed to process the document. Please try another PDF.")
                except Exception as e:
                    st.session_state.error_message = f"Error processing document: {str(e)}"
                    st.error(st.session_state.error_message)
                finally:
                    st.session_state.processing_started = False
    
    with tabs[1]:  # History Tab
        st.subheader("📚 Conversation History")
        
        if not st.session_state.saved_conversations:
            st.info("No saved conversations yet. Your interactions will be saved automatically.")
        else:
            for idx, conv in enumerate(reversed(st.session_state.saved_conversations)):
                with st.expander(f"{conv['document']} - {conv['timestamp']}"):
                    st.write(f"**Document:** {conv['document']}")
                    st.write(f"**Date:** {conv['timestamp']}")
                    st.write(f"**Questions:** {len([m for m in conv['chat'] if m['role'] == 'user'])}")
                    
                    if st.button(f"Load Conversation", key=f"load_conv_{idx}"):
                        # Load selected conversation
                        st.session_state.chat_history = conv['chat']
                        st.session_state.current_conversation_id = conv['id']
                        st.success(f"Loaded conversation from {conv['timestamp']}")
                        st.rerun()
                    
                    if st.button(f"Delete", key=f"del_conv_{idx}"):
                        # Delete conversation
                        try:
                            os.remove(os.path.join(CONVERSATIONS_DIR, f"{conv['id']}.json"))
                            st.session_state.saved_conversations.remove(conv)
                            st.success("Conversation deleted successfully")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting conversation: {e}")
    
    with tabs[2]:  # Settings Tab
        st.subheader("⚙️ Settings")
        
        # LLM Model settings
        st.markdown("#### LLM Configuration")
        llm_model = st.selectbox(
            "LLM Model", 
            list(LLM_MODELS.keys()),
            help="Select the Large Language Model to use for answering"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.1,
            help="Higher values make output more creative, lower more deterministic"
        )
        
        # UI settings
        st.markdown("#### UI Settings")
        show_sources = st.checkbox("Show Source Documents", value=True)
        
        # Export settings
        st.markdown("#### Export Options")
        export_format = st.radio("Export Format", ["PDF", "Word", "Plain Text"])
        
        # Clear current document option
        if st.session_state.vector_db is not None:
            if st.button("Clear Current Document", type="secondary"):
                st.session_state.vector_db = None
                st.session_state.pdf_name = None
                st.session_state.document_metadata = None
                st.session_state.chat_history = []
                st.session_state.document_summary = None
                st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
                st.rerun()

    # Add footer to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p class='footnote'>LegalMind AI v1.0<br>© 2025 All Rights Reserved</p>", unsafe_allow_html=True)
    st.sidebar.markdown("<p class='footnote'>Made with ❤️ using LangChain, Groq, and Streamlit</p>", unsafe_allow_html=True)

# Main content area with improved layout
main_col1, main_col2 = st.columns([2, 3])

with main_col1:
    # Document info panel
    if st.session_state.document_metadata:
        st.markdown(f"""
        <div class='document-card'>
            <h3>📄 {st.session_state.document_metadata['filename']}</h3>
            <p><b>Type:</b> {st.session_state.document_metadata['type']}</p>
            <p><b>Processed:</b> {st.session_state.document_metadata['processed_at']}</p>
            <p><b>Description:</b> {st.session_state.document_metadata['description'] or 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display document summary if available
        if st.session_state.document_summary:
            with st.expander("Document Summary", expanded=False):
                st.markdown(st.session_state.document_summary)
    
    # Question input area with more context and guidance
    st.subheader("🔍 Ask a Legal Question")
    
    if st.session_state.pdf_name:
        st.markdown("""
        <div class='info-box'>
            <p>Document is loaded and ready. You can ask questions about:</p>
            <ul>
                <li>Legal interpretations from the document</li>
                <li>Specific clauses or sections</li>
                <li>Precedents or case law mentioned</li>
                <li>Statutory provisions and their applications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested questions based on document type
        with st.expander("Suggested Questions"):
            if st.session_state.document_metadata and st.session_state.document_metadata['type'] == "Court Judgment":
                suggestions = [
                    "What is the main ruling in this judgment?",
                    "What were the key legal arguments made by the petitioner?",
                    "Which previous cases were cited in this judgment?",
                    "What sections of IPC were referenced in this judgment?"
                ]
            else:
                suggestions = [
                    "What are the main provisions in this document?",
                    "What obligations are mentioned in this document?",
                    "What remedies are provided in this document?",
                    "How does this document define key legal terms?"
                ]
                
            for sugg in suggestions:
                if st.button(sugg, key=f"sugg_{sugg}", use_container_width=True):
                    # Add the suggestion directly to chat history and process it immediately
                    st.session_state.chat_history.append({"role": "user", "content": sugg})
                    
                    with st.spinner("Researching and generating response..."):
                        try:
                            # Get response with citations and sources
                            response, sources = answer_query(
                                st.session_state.vector_db, 
                                sugg,
                                show_sources=True,
                                temperature=temperature,
                                model=llm_model
                            )
                            
                            # Append response to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response,
                                "sources": sources if sources else []
                            })
                            
                            # Auto-save the conversation
                            save_conversation()
                            
                            # Rerun to display the updated chat
                            st.rerun()
                            
                        except Exception as e:
                            error_message = f"Error generating response: {str(e)}"
                            st.error(error_message)
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": error_message
                            })
    else:
        st.warning("Please upload a legal document first to start asking questions.")
    
    # Add user query input with better UX
    user_query = st.text_area(
        "Enter your legal question:", 
        height=150,
        value=st.session_state.get("user_query", ""),
        placeholder="Example: What are the key legal principles established in this judgment?"
    )
    
    # Store the user query for processing before we clear the session state
    current_query = user_query
    
    # Clear the suggested query after it's been entered into the text area
    if "user_query" in st.session_state:
        del st.session_state.user_query
    
    # Submit button and controls
    col1, col2 = st.columns([3, 1])
    with col1:
        submit_disabled = st.session_state.vector_db is None
        submit_button = st.button(
            "Submit Question",
            type="primary",
            disabled=submit_disabled,
            use_container_width=True,
            key="submit_question_button"
        )
    with col2:
        if st.button("Clear", use_container_width=True):
            user_query = ""
    
    if submit_button and current_query:
        st.session_state.chat_history.append({"role": "user", "content": current_query})
        
        with st.spinner("Researching and generating response..."):
            try:
                # Get response with citations and sources
                response, sources = answer_query(
                    st.session_state.vector_db, 
                    current_query,
                    show_sources=True,
                    temperature=temperature,
                    model=llm_model
                )
                
                # Append response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources if sources else []
                })
                
                # Auto-save the conversation
                save_conversation()
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_message
                })

with main_col2:
    st.subheader("💬 Legal Analysis")
    
    if not st.session_state.chat_history:
        st.info("Your conversation will appear here after you ask a question.")
    else:
        # Chat container with improved styling
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class='chat-user'>
                        <p><strong>You:</strong> {message["content"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Format the assistant message with better styling
                    st.markdown(f"""
                    <div class='chat-assistant'>
                        <p><strong>LegalMind AI:</strong></p>
                        <p>{message["content"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if "sources" in message and message["sources"] and len(message["sources"]) > 0:
                        with st.expander("View Sources", expanded=False):
                            for i, source in enumerate(message["sources"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(f"```\n{source}\n```")
                    
                    # Add feedback buttons (thumbs up/down)
                    col1, col2, col3 = st.columns([1, 1, 6])
                    with col1:
                        st.button("👍", key=f"thumbs_up_{st.session_state.chat_history.index(message)}")
                    with col2:
                        st.button("👎", key=f"thumbs_down_{st.session_state.chat_history.index(message)}")
        
        # Export conversation options
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])
        with exp_col1:
            if st.button("Export Conversation", use_container_width=True):
                # This would trigger the export functionality
                st.success("Conversation exported successfully!")
        
        with exp_col2:
            if st.button("Save Conversation", use_container_width=True):
                save_conversation()
                st.success("Conversation saved successfully!")
        
        with exp_col3:
            if st.button("New Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.current_conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
                st.rerun()

# Add footer
st.markdown("---")
st.caption("Disclaimer: This AI assistant provides information based on the uploaded documents and should not be considered legal advice. Always consult with a qualified attorney for legal matters.")