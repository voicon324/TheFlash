
import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = Path("pipeline/knowledge_base/knowledge_base.json")

def clean_text(text):
    # Remove citations [1], [2] etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_wiki(topic):
    # Try generic capitalization
    url = f"https://vi.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {topic}: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get title
        title_span = soup.find('span', {'class': 'mw-page-title-main'}) 
        title = title_span.text if title_span else topic
        
        # Get content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
            
        # Extract paragraphs
        paragraphs = content_div.find_all('p')
        text = "\n\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        
        cleaned_text = clean_text(text)
        if len(cleaned_text) < 200: # Too short
            return None
            
        return {
            "title": title,
            "url": url,
            "content": cleaned_text,
            "category": "Wikipedia"
        }
        
    except Exception as e:
        logger.error(f"Error scraping {topic}: {e}")
        return None

def save_to_kb(new_data):
    if not new_data:
        return
        
    KNOWLEDGE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    current_data = []
    if KNOWLEDGE_BASE_PATH.exists():
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
    
    # Check duplicates by URL
    existing_urls = {item.get('url') for item in current_data}
    
    added_count = 0
    for item in new_data:
        if item['url'] not in existing_urls:
            # Chunking large articles
            # Simple chunking: 1000 chars overlap 100
            content = item['content']
            chunk_size = 1500
            overlap = 200
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk_text = content[i : i + chunk_size]
                if len(chunk_text) < 100:
                    continue
                    
                chunk_item = item.copy()
                chunk_item['content'] = chunk_text
                chunk_item['chunk_id'] = i // (chunk_size - overlap)
                current_data.append(chunk_item)
                added_count += 1
                
    with open(KNOWLEDGE_BASE_PATH, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Added {added_count} new chunks to knowledge base.")

def scrape_category(category_url, max_pages=30):
    logger.info(f"Scanning category: {category_url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    links = []
    try:
        response = requests.get(category_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links in the category page
        # Usually in div id="mw-pages"
        pages_div = soup.find('div', {'id': 'mw-pages'})
        if pages_div:
            # Get all links
            anchors = pages_div.find_all('a')
            for a in anchors:
                href = a.get('href')
                if href and href.startswith('/wiki/') and not ':' in href[6:]: # Exclude special pages like Template:, Talk:
                    links.append(href.replace('/wiki/', ''))
                    
        # Limit 
        links = list(set(links))[:max_pages]
        logger.info(f"Found {len(links)} articles in category.")
        return links
        
    except Exception as e:
        logger.error(f"Error scanning category: {e}")
        return []

def main():
    # Categories to scrape
    categories = [
        "https://vi.wikipedia.org/wiki/Thể_loại:Lịch_sử_Việt_Nam",
        "https://vi.wikipedia.org/wiki/Thể_loại:Địa_lý_Việt_Nam",
        "https://vi.wikipedia.org/wiki/Thể_loại:Văn_hóa_Việt_Nam",
        "https://vi.wikipedia.org/wiki/Thể_loại:Kinh_tế_Việt_Nam", 
        "https://vi.wikipedia.org/wiki/Thể_loại:Pháp_luật_Việt_Nam",
        "https://vi.wikipedia.org/wiki/Thể_loại:Giáo_dục_Việt_Nam",
        "https://vi.wikipedia.org/wiki/Thể_loại:Du_lịch_Việt_Nam",
        "https://vi.wikipedia.org/wiki/Thể_loại:Khoa_học_tự_nhiên" # Attempt to get STEM
    ]
    
    all_topics = set()
    
    # specialized topics from individual analysis
    individual_topics = [
        "Chùa Ba La Mật", "Hồ Chí Minh", "John Forbes Nash Jr.", 
        "Antoine Augustin Cournot", "Julius Weisbach", "Daniel Bernoulli"
    ]
    all_topics.update(individual_topics)
    
    # Gather topics from categories
    for cat_url in categories:
        articles = scrape_category(cat_url, max_pages=50) # Get 50 per category
        all_topics.update(articles)
        time.sleep(1)

    topics_list = list(all_topics)
    logger.info(f"Targeting {len(topics_list)} topics for scraping...")
    
    scraped_data = []
    
    # Load existing to skip
    if KNOWLEDGE_BASE_PATH.exists():
        with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
            existing = json.load(f)
            existing_urls = {item.get('url') for item in existing}
    else:
        existing_urls = set()

    count = 0
    for topic in topics_list:
        # Check if already scraped (heuristically by constructing URL, or just let scrape_wiki handle it)
        # scrape_wiki does cleaner check but we can optimization here if needed.
        # Let's just run it, save_to_kb handles dupes, but scraping costs time.
        
        # Construct likely URL to check dupe
        likely_url = f"https://vi.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        if likely_url in existing_urls:
            print(f"Skipping {topic} (already exists)")
            continue

        print(f"[{count+1}/{len(topics_list)}] Scraping: {topic}")
        data = scrape_wiki(topic)
        if data:
            scraped_data.append(data)
            print(f"  -> Success: {len(data['content'])} chars")
            
            # Save incrementally every 10 items
            if len(scraped_data) >= 10:
                save_to_kb(scraped_data)
                scraped_data = []
                
        count += 1
        time.sleep(0.5) # Fast but polite
        
    # Final save
    save_to_kb(scraped_data)

if __name__ == "__main__":
    main()
