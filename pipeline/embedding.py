# Embedding Module
import pickle
import time
from pathlib import Path
from typing import List, Optional, Dict
import requests
import numpy as np

from config import api_config, pipeline_config


class EmbeddingManager:
    """Manager for creating and caching embeddings using VNPT API"""
    
    def __init__(self):
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.cache_file = Path(pipeline_config.OUTPUT_DIR) / pipeline_config.EMBEDDINGS_FILE
        
    def _call_embedding_api(self, text: str) -> Optional[List[float]]:
        """
        Call VNPT embedding API for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        url = api_config.get_embedding_url()
        headers = api_config.get_headers(model_type='embedding')
        
        payload = {
            "model": api_config.EMBEDDING_MODEL,
            "input": text,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                return result['data'][0]['embedding']
            else:
                print(f"Unexpected response format: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except Exception as e:
            print(f"Error processing response: {e}")
            return None
    
    def get_embedding(self, text: str, cache_key: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get embedding for a text, using cache if available.
        
        Args:
            text: Text to embed
            cache_key: Key for caching (defaults to text hash)
            
        Returns:
            Embedding as numpy array
        """
        if cache_key is None:
            cache_key = str(hash(text))
        
        # Check cache
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        # Call API
        embedding = self._call_embedding_api(text)
        
        if embedding is not None:
            embedding_array = np.array(embedding, dtype=np.float32)
            self.embeddings_cache[cache_key] = embedding_array
            return embedding_array
        
        return None
    
    def get_embeddings_batch(
        self, 
        texts: List[str], 
        cache_keys: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            cache_keys: List of cache keys (defaults to text hashes)
            show_progress: Whether to show progress
            
        Returns:
            List of embeddings (None for failed ones)
        """
        if cache_keys is None:
            cache_keys = [str(hash(t)) for t in texts]
        
        results = []
        total = len(texts)
        
        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if show_progress and (i + 1) % 10 == 0:
                print(f"Processing {i + 1}/{total}...")
            
            embedding = self.get_embedding(text, cache_key=key)
            results.append(embedding)
            
            # Rate limiting
            time.sleep(pipeline_config.EMBEDDING_RATE_LIMIT)
        
        return results
    
    def save_cache(self, file_path: Optional[str] = None):
        """Save embeddings cache to file"""
        if file_path is None:
            file_path = self.cache_file
        else:
            file_path = Path(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        
        print(f"Saved {len(self.embeddings_cache)} embeddings to {file_path}")
    
    def load_cache(self, file_path: Optional[str] = None) -> bool:
        """
        Load embeddings cache from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if file_path is None:
            file_path = self.cache_file
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Cache file not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"Loaded {len(self.embeddings_cache)} embeddings from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def clear_cache(self):
        """Clear embeddings cache"""
        self.embeddings_cache = {}


# Global embedding manager instance
embedding_manager = EmbeddingManager()


def embed_text(text: str) -> Optional[np.ndarray]:
    """Convenience function to embed a single text"""
    return embedding_manager.get_embedding(text)


def embed_texts(texts: List[str]) -> List[Optional[np.ndarray]]:
    """Convenience function to embed multiple texts"""
    return embedding_manager.get_embeddings_batch(texts)


if __name__ == "__main__":
    # Test embedding
    print("Testing embedding API...")
    
    test_text = "Xin chào VNPT AI"
    print(f"Text: {test_text}")
    
    embedding = embed_text(test_text)
    
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")
    else:
        print("Failed to get embedding. Check API credentials.")
