import numpy as np
import faiss
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings

from ..models.schemas import AdvisorDocument

class Indexer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.dimension = 1536 # text-embedding-3-small dim
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product -> cosine similarity for normalized vectors
        self.documents: List[AdvisorDocument] = []
        
    def add_documents(self, docs: List[AdvisorDocument]):
        """Embeds and indexes advisor texts."""
        if not docs:
            return
            
        texts = [doc.full_text for doc in docs]
        vectors = self.embeddings.embed_documents(texts)
        
        # Normalize vectors for cosine similarity (FAISS FlatIP expects normalized)
        vectors_np = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors_np)
        
        self.index.add(vectors_np)
        self.documents.extend(docs)
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[AdvisorDocument, float]]:
        """Searches index. Returns docs and their FAISS distances (cosine similarity)."""
        if not self.documents:
            return []
            
        q_vec = self.embeddings.embed_query(query)
        q_np = np.array([q_vec], dtype=np.float32)
        faiss.normalize_L2(q_np)
        
        # FAISS search
        k = min(top_k, len(self.documents))
        distances, indices = self.index.search(q_np, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))
                
        return results
