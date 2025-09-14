# backend/query_engine.py
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Configure Ollama once here
llm = Ollama(model="llama3")
Settings.llm = llm

def ask_question(question: str, index):
    """Query the vector index (can contain multiple PDFs)"""
    query_engine = index.as_query_engine(similarity_top_k=5)  # fetch top 5 chunks
    response = query_engine.query(question)
    return str(response)
