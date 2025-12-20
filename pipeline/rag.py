#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) Module
Integrates knowledge base with LangChain, FAISS, and VNPT Embeddings.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import LangChain components
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Import from pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent))

from embedding_wrapper import VNPTEmbeddings

KNOWLEDGE_BASE_PATH = Path("/home/hkduy/workplace/VNPT_AI/pipeline/knowledge_base/knowledge_base.json")
FAISS_INDEX_PATH = Path("/home/hkduy/workplace/VNPT_AI/pipeline/faiss_index")

@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with its score"""
    content: str
    title: str
    url: str
    category: str
    score: float


class LangChainRAGEngine:
    """RAG Engine using LangChain and FAISS"""
    
    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.embeddings = VNPTEmbeddings()
        self._loaded = False
    
    def load_knowledge_base(self):
        """
        Load knowledge base.
        If FAISS index exists, load it.
        Otherwise, load JSON, compute embeddings (with checkpointing), and build index.
        """
        if FAISS_INDEX_PATH.exists():
            print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
            try:
                self.vector_store = FAISS.load_local(
                    str(FAISS_INDEX_PATH), 
                    self.embeddings,
                    allow_dangerous_deserialization=True # We trust our own file
                )
                self._loaded = True
                print("FAISS index loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Rebuilding...")
        
        # Build from scratch
        if not KNOWLEDGE_BASE_PATH.exists():
            print(f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}")
            return False
            
        print("Loading knowledge base from JSON...")
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    "title": chunk.get('title', ''),
                    "url": chunk.get('url', ''),
                    "category": chunk.get('category', ''),
                    "id": chunk.get('id', '')
                }
            )
            documents.append(doc)
            
        print(f"Created {len(documents)} documents. Building FAISS index (this may take a while)...")
        # FAISS.from_documents calls embed_documents, which uses our VNPTEmbeddings 
        # that handles checkpointing internally.
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save index
        print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
        self.vector_store.save_local(str(FAISS_INDEX_PATH))
        self._loaded = True
        return True

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks relevant to the query.
        """
        if not self._loaded or not self.vector_store:
            print("RAG Engine not loaded.")
            return []
            
        # Search parameters
        search_kwargs = {"k": top_k}
        if category_filter:
            search_kwargs["filter"] = {"category": category_filter}
            
        # Convert filter to the format FAISS generic backend expects if needed,
        # but LangChain FAISS filter usually works with metadata dict matching if exact match.
        # However, standard FAISS implementation in LangChain might not support complex filtering 
        # out of the box without specific setup. Let's try simple retrieval first and filter manually if needed,
        # or rely on metadata filtering if supported.
        # For simple FAISS, post-filtering is safer.
        
        # Get more candidates for post-filtering
        fetch_k = top_k * 3 if category_filter else top_k
        
        # Search
        # similarity_search_with_score returns L2 distance by default for FAISS (lower is better)
        # But we want cosine similarity.
        # If the vectors are normalized, L2 distance is related to cosine similarity.
        # FAISS in LangChain usually uses L2.
        # Let's use similarity_search_with_relevance_scores which normalizes to 0-1 range.
        
        docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
        
        results = []
        for doc, score in docs_and_scores:
            if category_filter and doc.metadata.get('category') != category_filter:
                continue
                
            results.append(RetrievedChunk(
                content=doc.page_content,
                title=doc.metadata.get('title', ''),
                url=doc.metadata.get('url', ''),
                category=doc.metadata.get('category', ''),
                score=score
            ))
            
            if len(results) >= top_k:
                break
                
        return results

    def format_context(self, chunks: List[RetrievedChunk], max_length: int = 3000) -> str:
        """Format retrieved chunks into a context string."""
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            part = f"[{chunk.title}]\n{chunk.content}"
            if total_length + len(part) > max_length:
                break
            context_parts.append(part)
            total_length += len(part)
        
        return "\n\n".join(context_parts)


# Global RAG engine instance
rag_engine = LangChainRAGEngine()

def init_rag() -> bool:
    """Initialize RAG engine"""
    return rag_engine.load_knowledge_base()

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve context for a query"""
    if not rag_engine._loaded:
        init_rag()
    
    chunks = rag_engine.retrieve(query, top_k=top_k)
    return rag_engine.format_context(chunks)

if __name__ == "__main__":
    # Test RAG
    print("Testing LangChain RAG Engine...")
    
    if init_rag():
        test_queries = [
            "Nhà Trần được thành lập vào năm nào?",
            "Vịnh Hạ Long nằm ở đâu?",
            "Ẩm thực Huế có những món gì nổi tiếng?",
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print("="*60)
            
            chunks = rag_engine.retrieve(query, top_k=3)
            
            for i, chunk in enumerate(chunks):
                print(f"\n[{i+1}] {chunk.title} (score: {chunk.score:.4f})")
                print(f"    {chunk.content[:200]}...")
    else:
        print("Failed to initialize RAG engine")
