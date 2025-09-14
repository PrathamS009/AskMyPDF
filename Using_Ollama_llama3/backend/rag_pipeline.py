# backend/rag_pipeline.py
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from .pdf_loader import load_pdf
from llama_index.core import Settings
import os

def build_or_load_index(pdf_path="data/uploaded_pdfs", persist_path="data/vector_store"):
    # Use Ollama for LLM + Embeddings
    llm = Ollama(model="llama3",timeout=300)
    embed_model = OllamaEmbedding(model_name="llama3")
    Settings.llm = llm
    Settings.embed_model = embed_model

    # If index already exists, load it
    if os.path.exists(persist_path):
        print(f"Loading existing index from {persist_path}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_path)
        index = load_index_from_storage(storage_context)
    else:
        print("No index found, building a new one...")
        documents = load_pdf(pdf_path)
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_path)
        print(f"Index persisted at {persist_path}")

    return index


def add_to_index(new_pdf_path, persist_path="data/vector_store"):
    """Add a new PDF to the existing index"""
    storage_context = None
    index = None

    if os.path.exists(persist_path):
        storage_context = StorageContext.from_defaults(persist_dir=persist_path)
        index = load_index_from_storage(storage_context)
    else:
        return build_or_load_index(new_pdf_path, persist_path)

    # Load new document
    new_docs = load_pdf(new_pdf_path)
    for doc in new_docs:
        index.insert(doc)  # add into existing index

    # Save again
    index.storage_context.persist(persist_path)
    print(f"Updated index with {new_pdf_path}")
    return index
