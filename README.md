# LegalMind AI

LegalMind AI is an intelligent legal research assistant specialized in Indian law. It uses advanced RAG (Retrieval Augmented Generation) technology to analyze legal documents and answer questions based on their content.

## Features

- **Upload and Process Legal Documents**: Easily upload PDF documents including court judgments, legal acts, contracts, and more
- **Advanced Document Processing**: Smart chunking and analysis of legal text for better understanding
- **Intelligent Question Answering**: Ask natural language questions about the document content
- **Source Citations**: Responses include references to specific parts of the document
- **History Management**: Save and reload previous conversations
- **Document Summaries**: Automatic generation of document summaries
- **Customizable Settings**: Choose different language models and embedding options

## Setup Instructions

### Prerequisites

- Python 3.8+
- A Groq API key (sign up at [groq.com](https://groq.com))

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/legalmind-ai.git
   cd legalmind-ai
   ```

2. Create a virtual environment:
   ```
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root by copying the example:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your Groq API key (without quotes):
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Application

Launch the application with:
```
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Usage

1. **Upload a Document**: Use the upload section to add a legal document (PDF format)
2. **Process the Document**: Configure processing options if needed and click "Process Document"
3. **Ask Questions**: Enter legal questions about the document in the question input field
4. **Save or Export**: Save conversations for future reference or export results

## Project Structure

- `app.py` - Streamlit web interface
- `config.py` - Configuration settings and path management
- `rag_pipeline.py` - RAG implementation for question answering
- `utils.py` - Helper functions for document processing
- `vector_database.py` - Vector database management for document embeddings

## Known Limitations

- Currently only supports PDF files
- Best performance with legal documents in English
- Focused primarily on Indian legal system

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit, LangChain, and Groq LLMs
- Uses FAISS for vector storage and efficient retrieval