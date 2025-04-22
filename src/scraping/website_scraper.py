"""
Website Scraper
--------------
Scrapes websites for academic content, with support for
structured data extraction and rate limiting.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from scrapy.linkextractors import LinkExtractor
from scrapy.http import HtmlResponse

from ..utils.helpers import clean_text, setup_logging

logger = logging.getLogger(__name__)


class WebsiteScraper:
    def __init__(self,
                 base_url: str,
                 output_dir: str = "scraped_data",
                 max_pages: Optional[int] = None,
                 exclude_patterns: Optional[List[str]] = None,
                 delay: float = 0.5):
        """
        Initialize website scraper.

        Args:
            base_url: Base URL to start scraping
            output_dir: Directory to save scraped data
            max_pages: Maximum number of pages to scrape
            exclude_patterns: URL patterns to exclude
            delay: Delay between requests in seconds
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.exclude_patterns = exclude_patterns or []
        self.delay = delay

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize scraping state
        self.visited_urls = set()
        self.link_extractor = LinkExtractor()

        # Get base domain for limiting scope
        parsed_url = urlparse(base_url)
        self.base_domain = parsed_url.netloc

        # Set user agent
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AcademicBot/1.0; +http://example.com/bot)'
        }

        self.all_data = []

        logger.info(f"Initialized scraper for {base_url}")
        logger.info(f"Output directory: {output_dir}")
        if max_pages:
            logger.info(f"Maximum pages: {max_pages}")

    def should_visit(self, url: str) -> bool:
        """Check if URL should be visited"""
        if url in self.visited_urls:
            return False

        parsed_url = urlparse(url)

        # Stay within same domain
        if parsed_url.netloc != self.base_domain:
            return False

        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if pattern in url.lower():
                return False

        return True

    def extract_content(self, url: str, html_content: str) -> Dict:
        """Extract meaningful content from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Get title
        title = soup.title.string if soup.title else "No Title"

        # Get meta description
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag and 'content' in meta_tag.attrs:
            meta_desc = meta_tag['content']

        # Extract main content
        content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        content = ' '.join([elem.get_text() for elem in content_elements])

        # Clean content
        content = clean_text(content)

        return {
            'url': url,
            'title': clean_text(title),
            'meta_description': clean_text(meta_desc),
            'content': content,
            'html': str(soup)
        }

    def extract_links(self, url: str, html_content: str) -> List[str]:
        """Extract links from HTML content"""
        response = HtmlResponse(url=url, body=html_content.encode(), encoding='utf-8')
        links = self.link_extractor.extract_links(response)
        return [link.url for link in links]

    def scrape_url(self, url: str) -> List[str]:
        """Scrape a single URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                return []

            # Add to visited
            self.visited_urls.add(url)

            # Extract content
            data = self.extract_content(url, response.text)
            self.all_data.append(data)

            # Extract links
            links = self.extract_links(url, response.text)

            # Delay
            time.sleep(self.delay)

            return links

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []

    def save_progress(self) -> None:
        """Save current progress"""
        progress_dir = os.path.join(self.output_dir, "progress")
        os.makedirs(progress_dir, exist_ok=True)

        # Save data
        with open(os.path.join(progress_dir, "progress.json"), 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

        # Save visited URLs
        with open(os.path.join(progress_dir, "visited_urls.txt"), 'w', encoding='utf-8') as f:
            for url in self.visited_urls:
                f.write(f"{url}\n")

    def crawl(self) -> None:
        """Crawl the website"""
        queue = [self.base_url]

        with tqdm(desc="Scraping pages", unit="page") as pbar:
            while queue:
                if self.max_pages and len(self.visited_urls) >= self.max_pages:
                    logger.info(f"Reached maximum pages ({self.max_pages})")
                    break

                url = queue.pop(0)

                if not self.should_visit(url):
                    continue

                logger.info(f"Scraping: {url}")
                new_links = self.scrape_url(url)

                # Update progress
                pbar.update(1)

                # Add new links to queue
                for link in new_links:
                    absolute_link = urljoin(self.base_url, link)
                    if self.should_visit(absolute_link):
                        queue.append(absolute_link)

                # Save progress periodically
                if len(self.visited_urls) % 10 == 0:
                    self.save_progress()

        logger.info(f"Crawling complete. Scraped {len(self.visited_urls)} pages")

    def save_data(self) -> None:
        """Save scraped data in multiple formats"""
        # Save JSON
        json_path = os.path.join(self.output_dir, "scraped_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)

        # Save as CSV
        import pandas as pd
        df = pd.DataFrame([{
            'url': item['url'],
            'title': item['title'],
            'meta_description': item['meta_description'],
            'content': item['content']
        } for item in self.all_data])

        csv_path = os.path.join(self.output_dir, "scraped_data.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # Save individual text files
        text_dir = os.path.join(self.output_dir, "text_files")
        os.makedirs(text_dir, exist_ok=True)

        for i, item in enumerate(self.all_data):
            filename = f"doc_{i + 1:04d}.txt"
            with open(os.path.join(text_dir, filename), 'w', encoding='utf-8') as f:
                f.write(f"Title: {item['title']}\n")
                f.write(f"URL: {item['url']}\n")
                f.write(f"Description: {item['meta_description']}\n\n")
                f.write(item['content'])

        logger.info(f"Data saved to {self.output_dir}")

    def run(self) -> None:
        """Run the complete scraping pipeline"""
        try:
            self.crawl()
            self.save_data()
            logger.info("Scraping completed successfully")
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Scrape website for academic content')
    parser.add_argument('--url', required=True,
                        help='Website URL to scrape')
    parser.add_argument('--output', default='scraped_data',
                        help='Output directory')
    parser.add_argument('--max-pages', type=int,
                        help='Maximum pages to scrape')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='URL patterns to exclude')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests (seconds)')

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Run scraper
    scraper = WebsiteScraper(
        args.url,
        args.output,
        args.max_pages,
        args.exclude,
        args.delay
    )
    scraper.run()


if __name__ == '__main__':
    main()
