
import json
import os
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
DATASET_NAME = "vietgpt/wikipedia_vi"
KNOWLEDGE_BASE_PATH = Path("pipeline/knowledge_base/knowledge_base.json")
CHUNK_SIZE = 1500
OVERLAP = 200
MAX_ARTICLES = 50000  # Full scale run

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Splits text into chunks of `size` with `overlap`.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= text_len:
            break
            
        start += (size - overlap)
        
    return chunks

def load_existing_kb():
    if KNOWLEDGE_BASE_PATH.exists():
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def main():
    logger.info(f"Loading dataset {DATASET_NAME}...")
    try:
        # Load streaming=True if dataset is huge, but vietgpt/wikipedia_vi is likely manageable or we want to shuffle.
        # Let's use streaming to be safe and take first MAX_ARTICLES
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    existing_data = load_existing_kb()
    logger.info(f"Loaded {len(existing_data)} existing chunks from KB.")
    
    # Create a set of existing IDs/Titles to avoid partial dups if possible, 
    # though scraping source is different.
    # existing_titles = {item['title'] for item in existing_data} 
    
    new_chunks = []
    article_count = 0
    
    logger.info(f"Processing up to {MAX_ARTICLES} articles...")
    
    for article in tqdm(dataset):
        if article_count >= MAX_ARTICLES:
            break
            
        # Inspect structure - usually 'title', 'text' or 'content'
        title = article.get('title', 'Unknown')
        content = article.get('text', '') or article.get('content', '')
        url = article.get('url', f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}")
        
        if not content:
            continue
            
        # Basic cleaning (optional, dataset might be raw)
        
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            chunk_entry = {
                "id": f"hf_{title}_{i}",
                "title": title,
                "content": chunk,
                "url": url,
                "category": "wikipedia_hf",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            new_chunks.append(chunk_entry)
            
        article_count += 1
        
        if article_count % 100 == 0:
            print(f"Processed {article_count} articles...", end='\r')

    logger.info(f"\nCompleted processing {article_count} articles.")
    logger.info(f"Generated {len(new_chunks)} new chunks.")
    
    # Merge and Save
    combined_data = existing_data + new_chunks
    
    logger.info(f"Saving knowledge base with total {len(combined_data)} items...")
    
    # Ensure directory exists
    KNOWLEDGE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(KNOWLEDGE_BASE_PATH, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
    logger.info("Done.")

if __name__ == "__main__":
    main()
