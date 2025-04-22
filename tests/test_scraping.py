"""
Test web scraping functionality
"""

import pytest
from src.scraping.website_scraper import WebsiteScraper


def test_scraper_initialization(test_data_dir):
    """Test WebsiteScraper initialization"""
    scraper = WebsiteScraper(
        base_url="http://example.com",
        output_dir=str(test_data_dir / "scraped"),
        max_pages=10
    )

    assert scraper.base_domain == "example.com"
    assert hasattr(scraper, 'crawl')
    assert hasattr(scraper, 'save_data')


def test_content_extraction(test_data_dir):
    """Test HTML content extraction"""
    scraper = WebsiteScraper(
        base_url="http://example.com",
        output_dir=str(test_data_dir / "scraped")
    )

    html_content = """
    <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test Description">
        </head>
        <body>
            <h1>Test Header</h1>
            <p>Test paragraph.</p>
            <script>var x = 1;</script>
        </body>
    </html>
    """

    data = scraper.extract_content(
        url="http://example.com",
        html_content=html_content
    )

    assert 'title' in data
    assert 'content' in data
    assert 'meta_description' in data
    assert 'script' not in data['content']


def test_url_filtering():
    """Test URL filtering"""
    scraper = WebsiteScraper(
        base_url="http://example.com",
        exclude_patterns=['login', 'admin']
    )

    assert scraper.should_visit("http://example.com/page")
    assert not scraper.should_visit("http://example.com/login")
    assert not scraper.should_visit("http://other.com/page")

