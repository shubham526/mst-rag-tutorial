"""
Pytest configuration and fixtures
"""

import os
import json
import shutil
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary test data directory"""
    tmp_path = tmp_path_factory.mktemp("test_data")
    return tmp_path


@pytest.fixture(scope="session")
def sample_corpus(test_data_dir):
    """Create a sample corpus for testing"""
    corpus_dir = test_data_dir / "sample_corpus"
    corpus_dir.mkdir(exist_ok=True)

    # Create sample passages
    passages = [
        {
            "id": f"doc{i}",
            "text": f"This is sample document {i} about information retrieval.",
            "title": f"Document {i}",
            "url": f"http://example.com/doc{i}",
            "word_count": 8,
            "num_sentences": 1
        }
        for i in range(10)
    ]

    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(passages)
    df.to_csv(corpus_dir / "passages.csv", index=False)

    # Save metadata
    metadata = {
        f"DOC-{i}": {
            "title": f"Document {i}",
            "url": f"http://example.com/doc{i}",
            "num_passages": 1,
            "passage_ids": [f"doc{i}"]
        }
        for i in range(10)
    }

    with open(corpus_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return corpus_dir


@pytest.fixture(scope="session")
def sample_topics(test_data_dir):
    """Create sample topics file"""
    topics = {
        "topics": [
            {
                "id": "1",
                "title": "Information Retrieval Systems",
                "description": "Find documents about IR systems",
                "narrative": "Relevant documents discuss IR systems"
            },
            {
                "id": "2",
                "title": "Sample Documents",
                "description": "Find sample documents",
                "narrative": "Relevant documents are examples"
            }
        ]
    }

    topics_file = test_data_dir / "topics.json"
    with open(topics_file, "w") as f:
        json.dump(topics, f)

    return topics_file


@pytest.fixture(scope="session")
def sample_website(test_data_dir):
    """Create sample website content"""
    website_dir = test_data_dir / "website"
    website_dir.mkdir(exist_ok=True)

    # Create sample HTML files
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Page</title>
        <meta name="description" content="A sample page for testing">
    </head>
    <body>
        <h1>Sample Content</h1>
        <p>This is a sample paragraph for testing.</p>
        <a href="/page2.html">Link to Page 2</a>
    </body>
    </html>
    """

    with open(website_dir / "index.html", "w") as f:
        f.write(html_content)

    return website_dir

