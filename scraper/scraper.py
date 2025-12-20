#!/usr/bin/env python3
"""
Vietnam Knowledge Scraper
Cào dữ liệu kiến thức về văn hoá, lịch sử, địa lý, chính trị Việt Nam
từ các nguồn chính thống.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, quote
import random

# Config
OUTPUT_DIR = Path("/home/hkduy/workplace/VNPT_AI/scraper/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
}

# Delay between requests (seconds)
REQUEST_DELAY = 1.0


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove references like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove edit links
    text = re.sub(r'\[sửa\s*\|\s*sửa mã nguồn\]', '', text)
    return text.strip()


def fetch_page(url: str) -> Optional[BeautifulSoup]:
    """Fetch a web page and return BeautifulSoup object"""
    try:
        time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


class WikipediaScraper:
    """Scraper for Vietnamese Wikipedia"""
    
    BASE_URL = "https://vi.wikipedia.org"
    
    # Categories to scrape
    CATEGORIES = {
        "history": [
            "Lịch_sử_Việt_Nam",
            "Triều_đại_Việt_Nam", 
            "Chiến_tranh_liên_quan_đến_Việt_Nam",
            "Nhân_vật_lịch_sử_Việt_Nam",
        ],
        "geography": [
            "Địa_lý_Việt_Nam",
            "Tỉnh_thành_Việt_Nam",
            "Sông_Việt_Nam",
            "Núi_Việt_Nam",
            "Vịnh_Việt_Nam",
        ],
        "culture": [
            "Văn_hóa_Việt_Nam",
            "Ẩm_thực_Việt_Nam",
            "Nghệ_thuật_Việt_Nam",
            "Lễ_hội_Việt_Nam",
            "Di_sản_thế_giới_tại_Việt_Nam",
        ],
        "politics": [
            "Chính_trị_Việt_Nam",
            "Hiến_pháp_Việt_Nam",
            "Quốc_hội_Việt_Nam",
        ],
    }
    
    def get_articles_in_category(self, category: str, limit: int = 50) -> List[str]:
        """Get list of article titles in a category"""
        url = f"{self.BASE_URL}/wiki/Thể_loại:{category}"
        soup = fetch_page(url)
        if not soup:
            return []
        
        articles = []
        # Find article links in category page
        content = soup.find('div', {'id': 'mw-pages'})
        if content:
            for link in content.find_all('a'):
                href = link.get('href', '')
                if href.startswith('/wiki/') and ':' not in href:
                    title = href.replace('/wiki/', '')
                    articles.append(title)
                    if len(articles) >= limit:
                        break
        
        return articles
    
    def get_article_content(self, title: str) -> Optional[Dict]:
        """Get content of a Wikipedia article"""
        # Don't re-encode if already encoded
        url = f"{self.BASE_URL}/wiki/{title}"
        soup = fetch_page(url)
        if not soup:
            return None
        
        # Get article title
        heading = soup.find('h1', {'id': 'firstHeading'})
        article_title = heading.get_text() if heading else title
        
        # Get main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
        
        # Extract paragraphs
        paragraphs = []
        for p in content_div.find_all('p', recursive=True):
            text = clean_text(p.get_text())
            if len(text) > 50:  # Skip very short paragraphs
                paragraphs.append(text)
        
        if not paragraphs:
            return None
        
        # Extract sections
        sections = []
        for heading in content_div.find_all(['h2', 'h3']):
            section_title = clean_text(heading.get_text())
            # Skip non-content sections
            if any(x in section_title.lower() for x in ['tham khảo', 'liên kết', 'chú thích', 'xem thêm']):
                continue
            sections.append(section_title)
        
        return {
            'title': article_title,
            'url': url,
            'content': '\n\n'.join(paragraphs),
            'sections': sections,
            'source': 'wikipedia_vi'
        }
    
    def scrape_category(self, category_name: str, categories: List[str], limit_per_cat: int = 30) -> List[Dict]:
        """Scrape all articles in given categories"""
        print(f"\n=== Scraping {category_name} ===")
        articles = []
        
        for cat in categories:
            print(f"  Category: {cat}")
            titles = self.get_articles_in_category(cat, limit=limit_per_cat)
            print(f"    Found {len(titles)} articles")
            
            for title in titles:
                article = self.get_article_content(title)
                if article:
                    article['category'] = category_name
                    articles.append(article)
                    print(f"    ✓ {article['title'][:50]}...")
        
        return articles
    
    def scrape_all(self, limit_per_cat: int = 30) -> Dict[str, List[Dict]]:
        """Scrape all categories"""
        all_data = {}
        
        for category_name, categories in self.CATEGORIES.items():
            articles = self.scrape_category(category_name, categories, limit_per_cat)
            all_data[category_name] = articles
            
            # Save incrementally
            output_file = OUTPUT_DIR / f"wikipedia_{category_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"  Saved {len(articles)} articles to {output_file}")
        
        return all_data


class VietnamNetScraper:
    """Scraper for VietnamNet news articles"""
    
    BASE_URL = "https://vietnamnet.vn"
    
    SECTIONS = {
        "culture": "/giai-tri/van-hoa",
        "politics": "/thoi-su/chinh-tri",
        "geography": "/doi-song/du-lich",
    }
    
    def get_article_links(self, section_url: str, limit: int = 20) -> List[str]:
        """Get article links from a section page"""
        url = f"{self.BASE_URL}{section_url}"
        soup = fetch_page(url)
        if not soup:
            return []
        
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/') and '.html' in href:
                full_url = f"{self.BASE_URL}{href}"
                if full_url not in links:
                    links.append(full_url)
                    if len(links) >= limit:
                        break
        
        return links
    
    def get_article_content(self, url: str) -> Optional[Dict]:
        """Get content of an article"""
        soup = fetch_page(url)
        if not soup:
            return None
        
        # Get title
        title_tag = soup.find('h1', class_='content-detail-title')
        if not title_tag:
            title_tag = soup.find('h1')
        title = clean_text(title_tag.get_text()) if title_tag else ""
        
        if not title:
            return None
        
        # Get content
        content_div = soup.find('div', class_='maincontent')
        if not content_div:
            content_div = soup.find('article')
        
        if not content_div:
            return None
        
        paragraphs = []
        for p in content_div.find_all('p'):
            text = clean_text(p.get_text())
            if len(text) > 30:
                paragraphs.append(text)
        
        if not paragraphs:
            return None
        
        return {
            'title': title,
            'url': url,
            'content': '\n\n'.join(paragraphs),
            'source': 'vietnamnet'
        }


def main():
    """Main function to run scrapers"""
    print("=" * 60)
    print("Vietnam Knowledge Scraper")
    print("=" * 60)
    
    # Scrape Wikipedia
    print("\n[1] Scraping Vietnamese Wikipedia...")
    wiki_scraper = WikipediaScraper()
    wiki_data = wiki_scraper.scrape_all(limit_per_cat=20)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = 0
    for category, articles in wiki_data.items():
        print(f"  {category}: {len(articles)} articles")
        total += len(articles)
    
    print(f"\nTotal: {total} articles")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
