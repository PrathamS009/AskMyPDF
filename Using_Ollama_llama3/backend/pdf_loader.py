# backend/pdf_loader.py
import os
from llama_index.core import SimpleDirectoryReader

def load_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The specified path does not exist: {pdf_path}")

    if os.path.isdir(pdf_path):
        # Load all PDFs in a folder
        documents = SimpleDirectoryReader(pdf_path).load_data()
    else:
        # Load a single PDF file
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

    return documents
