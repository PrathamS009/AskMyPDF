from PyPDF2 import PdfReader
from typing import List

def extract_text_from_pdf(file_path) -> str:
    reader=PdfReader(file_path)
    text=""
    for p in reader.pages:
        text+=p.extract_text()
    return text

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks