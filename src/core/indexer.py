import numpy as np
import faiss
from typing import List, Dict, Tuple
from langchain_openai import OpenAIEmbeddings

from ..models.schemas import AdvisorDocument

class Indexer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.dimension = 1536 # text-embedding-3-small dim
        self.bio_index = faiss.IndexFlatIP(self.dimension)
        self.tags_index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[AdvisorDocument] = []
        
    def add_documents(self, docs: List[AdvisorDocument]):
        """Embeds and indexes advisor texts (bio and tags)."""
        if not docs:
            return
            
        bio_texts = [doc.full_text for doc in docs]
        tags_texts = [doc.get_tags_text() for doc in docs]
        
        bio_vectors = self.embeddings.embed_documents(bio_texts)
        tags_vectors = self.embeddings.embed_documents(tags_texts)
        
        # Normalize and add to respective indices
        bio_np = np.array(bio_vectors, dtype=np.float32)
        tags_np = np.array(tags_vectors, dtype=np.float32)
        faiss.normalize_L2(bio_np)
        faiss.normalize_L2(tags_np)
        
        self.bio_index.add(bio_np)
        self.tags_index.add(tags_np)
        self.documents.extend(docs)
        
    def semantic_search(self, query: str, index_type: str = "bio", top_k: int = 5) -> List[Tuple[AdvisorDocument, float]]:
        """Searches specific index. Returns docs and their FAISS distances (cosine similarity)."""
        if not self.documents:
            return []
            
        # Select index
        target_index = self.bio_index if index_type == "bio" else self.tags_index
        
        # Anchor the query to prevent semantic drift if bio search
        if index_type == "bio":
            anchored_query = query + " (情境錨定：尋找金融理財顧問、財富管理、專業投資諮詢與客戶服務)"
        else:
            anchored_query = query # Tags are already structured/keyword-like
        
        q_vec = self.embeddings.embed_query(anchored_query)
        q_np = np.array([q_vec], dtype=np.float32)
        faiss.normalize_L2(q_np)
        
        # FAISS search
        k = min(top_k, len(self.documents))
        distances, indices = target_index.search(q_np, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))
                
        return results
