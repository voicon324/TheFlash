from typing import List, Optional, Dict
from langchain_core.embeddings import Embeddings
from pydantic import Field, PrivateAttr
import numpy as np
import requests
import time
import pickle
import os
from pathlib import Path
from config import api_config, pipeline_config

class VNPTEmbeddings(Embeddings):
    """VNPT Embeddings wrapper with checkpointing."""
    
    model_name: str = api_config.EMBEDDING_MODEL
    batch_size: int = 100 # Increased for parallel processing 
    sleep_between_batches: float = 1.0
    cache_path: str = str(Path(pipeline_config.OUTPUT_DIR) / "embedding_checkpoint.pkl")
    
    _cache: Dict[str, List[float]] = PrivateAttr(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_cache()

    def _load_cache(self):
        """Load embeddings from checkpoint file."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"Loaded {len(self._cache)} embeddings from checkpoint.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting fresh.")
                self._cache = {}
        else:
            self._cache = {}

    def _save_cache(self):
        """Save embeddings to checkpoint file."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self._cache, f)
        # print(f"Checkpoint saved. Total embeddings: {len(self._cache)}")

    def _call_api(self, text: str) -> List[float]:
        """Call VNPT Embedding API."""
        url = api_config.get_embedding_url()
        headers = api_config.get_headers(model_type='embedding')
        
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float"
        }
        
        # Simple retry loop for embeddings too
        base_wait = 2
        for attempt in range(5):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        return data['data'][0]['embedding']
                
                if response.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(base_wait * (2 ** attempt))
                    continue
                    
                print(f"Embedding API error: {response.status_code} {response.text}")
                break
            except Exception as e:
                print(f"Embedding API exception: {e}")
                time.sleep(base_wait * (2 ** attempt))
                
        raise ValueError(f"Failed to embed text: {text[:50]}...")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with checkpointing."""
        results = []
        texts_to_process = []
        indices_to_process = []
        
        # 1. Check cache first
        for i, text in enumerate(texts):
            # Normalization/Cleaning could happen here
            if text in self._cache:
                results.append(self._cache[text])
            else:
                results.append(None) # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        if not texts_to_process:
            return results

        print(f"Need to compute embeddings for {len(texts_to_process)} documents.")
        
        # 2. Process in parallel batches
        import concurrent.futures
        
        # Increase batch size for parallel processing check
        # But here we process per-item in parallel threads inside the batch loop?
        # Actually, simpler to just pool.map over all items, but we want to checkpoint regularly.
        # So keep batch loop, but inside batch, run parallel.
        
        total = len(texts_to_process)
        max_workers = 12  # Parallelism factor
        
        for i in range(0, total, self.batch_size):
            batch_texts = texts_to_process[i : i + self.batch_size]
            batch_indices = indices_to_process[i : i + self.batch_size]
            
            print(f"Embedding batch {i//self.batch_size + 1}/{(total + self.batch_size - 1)//self.batch_size} (Parallel)...")
            


            # Refactored Parallel Execution inside batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map future -> (text, original_idx)
                future_map = {
                    executor.submit(self._call_api, text): (text, idx)
                    for text, idx in zip(batch_texts, batch_indices)
                }
                
                for future in concurrent.futures.as_completed(future_map):
                    text, original_idx = future_map[future]
                    try:
                        emb = future.result()
                        self._cache[text] = emb
                        results[original_idx] = emb
                    except Exception as e:
                        print(f"Error embedding document at index {original_idx}: {e}")

            # Save checkpoint after each large batch
            self._save_cache()
            
            # Rate limit? With parallel, we hit rate limits faster.
            # _call_api has retry logic.
            # Maybe sleep a bit less or same.
            # time.sleep(0.1) 
            
        # Post-processing: Handle failed embeddings to prevent crash
        if not results:
             return results

        # Determine dimension from existing results or cache
        dim = 1536 # Default fallback (OpenAI size), but try to detect
        for emb in results:
            if emb is not None:
                dim = len(emb)
                break
        else:
             if self._cache:
                 dim = len(next(iter(self._cache.values())))

        # Fill Nones with zero vectors
        failed_count = 0
        zero_vec = [0.0] * dim
        for i in range(len(results)):
            if results[i] is None:
                results[i] = zero_vec
                failed_count += 1
        
        if failed_count > 0:
            print(f"WARNING: Filled {failed_count} failed embeddings with zero vectors.")
            
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        if text in self._cache:
            return self._cache[text]
        
        emb = self._call_api(text)
        # Optional: Cache queries too? For now, we only typically cache documents.
        # But for consistency, let's cache it if it's identical text.
        self._cache[text] = emb
        return emb
