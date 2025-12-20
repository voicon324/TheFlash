# Search Module - Vector Similarity Search
import numpy as np
from typing import List, Tuple, Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a query and multiple documents.
    
    Args:
        query: Query vector (1D array)
        documents: Document vectors (2D array, each row is a document)
        
    Returns:
        Array of similarity scores
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(documents))
    query_normalized = query / query_norm
    
    # Normalize documents
    doc_norms = np.linalg.norm(documents, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1  # Avoid division by zero
    docs_normalized = documents / doc_norms
    
    # Calculate similarities
    similarities = np.dot(docs_normalized, query_normalized)
    
    return similarities


class VectorSearcher:
    """Simple in-memory vector search using cosine similarity"""
    
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[dict] = []
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[np.ndarray],
        metadata: Optional[List[dict]] = None
    ):
        """
        Add documents and their embeddings to the index.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional list of metadata dicts
        """
        self.documents.extend(documents)
        
        # Convert embeddings to numpy array
        new_embeddings = np.array(embeddings)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Tuple[int, str, float, dict]]:
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (index, document, score, metadata) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate similarities
        similarities = batch_cosine_similarity(query_embedding, self.embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                self.documents[idx],
                float(similarities[idx]),
                self.metadata[idx] if idx < len(self.metadata) else {}
            ))
        
        return results
    
    def clear(self):
        """Clear all documents and embeddings"""
        self.documents = []
        self.embeddings = None
        self.metadata = []
    
    def __len__(self):
        return len(self.documents)


# Global searcher instance
vector_searcher = VectorSearcher()


if __name__ == "__main__":
    # Test search
    print("Testing vector search...")
    
    # Create some test data
    test_docs = [
        "Python là ngôn ngữ lập trình phổ biến",
        "Machine learning là một lĩnh vực của AI",
        "Việt Nam là một quốc gia Đông Nam Á",
    ]
    
    # Create fake embeddings for testing
    np.random.seed(42)
    test_embeddings = [np.random.randn(768) for _ in test_docs]
    
    searcher = VectorSearcher()
    searcher.add_documents(test_docs, test_embeddings)
    
    # Search with first document's embedding
    results = searcher.search(test_embeddings[0], top_k=3)
    
    print(f"\nSearch results for: '{test_docs[0]}'")
    for idx, doc, score, meta in results:
        print(f"  [{idx}] Score: {score:.4f} - {doc[:50]}...")
