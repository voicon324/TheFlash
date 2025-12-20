#!/usr/bin/env python3
"""
Merge và xử lý dữ liệu đã cào để tích hợp vào RAG pipeline
"""

import json
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path("/home/hkduy/workplace/VNPT_AI/scraper/data")
OUTPUT_DIR = Path("/home/hkduy/workplace/VNPT_AI/pipeline/knowledge_base")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_articles() -> List[Dict]:
    """Load tất cả articles từ các file JSON"""
    all_articles = []
    
    json_files = list(DATA_DIR.glob("wikipedia_*.json"))
    
    for file_path in json_files:
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            all_articles.extend(articles)
    
    return all_articles


def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Chia text thành các chunks nhỏ hơn với overlap.
    Thích hợp cho embedding và retrieval.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Tìm điểm ngắt tự nhiên (dấu chấm, xuống dòng)
        if end < len(text):
            # Tìm dấu chấm hoặc xuống dòng gần nhất
            for sep in ['. ', '.\n', '\n\n', '\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def process_articles(articles: List[Dict]) -> List[Dict]:
    """
    Xử lý articles thành chunks cho RAG.
    Mỗi chunk sẽ có metadata về nguồn gốc.
    """
    all_chunks = []
    
    for article in articles:
        title = article.get('title', '')
        content = article.get('content', '')
        url = article.get('url', '')
        category = article.get('category', '')
        
        if not content:
            continue
        
        # Chia thành chunks
        text_chunks = split_into_chunks(content, chunk_size=1000, overlap=200)
        
        for i, chunk_content in enumerate(text_chunks):
            all_chunks.append({
                'id': f"{title}_{i}",
                'title': title,
                'content': chunk_content,
                'url': url,
                'category': category,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
            })
    
    return all_chunks


def save_knowledge_base(chunks: List[Dict]):
    """Lưu knowledge base"""
    # Save as JSON
    output_file = OUTPUT_DIR / "knowledge_base.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_file}")
    
    # Save metadata
    metadata = {
        'total_chunks': len(chunks),
        'categories': list(set(c['category'] for c in chunks)),
        'articles': list(set(c['title'] for c in chunks)),
    }
    
    meta_file = OUTPUT_DIR / "metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {meta_file}")


def main():
    print("=" * 60)
    print("Processing Knowledge Base for RAG")
    print("=" * 60)
    
    # Load all articles
    articles = load_all_articles()
    print(f"\nLoaded {len(articles)} articles")
    
    # Process into chunks
    print("\nProcessing articles into chunks...")
    chunks = process_articles(articles)
    print(f"Created {len(chunks)} chunks")
    
    # Statistics
    total_chars = sum(len(c['content']) for c in chunks)
    avg_chunk_size = total_chars // len(chunks) if chunks else 0
    
    print(f"\nStatistics:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average chunk size: {avg_chunk_size} chars")
    
    # Category breakdown
    print(f"\nChunks by category:")
    categories = {}
    for c in chunks:
        cat = c['category']
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Save
    print("\nSaving knowledge base...")
    save_knowledge_base(chunks)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
