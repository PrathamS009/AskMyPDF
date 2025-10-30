from sentence_transformers import SentenceTransformer
import faiss
from typing import List

class EmbeddingIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts: List[str] = []

    def build(self, texts: List[str]):
        self.texts = texts
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, query: str, top_k: int = 4) -> list:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)
        results = [self.texts[i] for i in indices[0] if i != -1]
        return results
