# Academic IR Pipeline

A comprehensive toolkit for building and evaluating academic information retrieval systems. This pipeline allows you to create, index, and evaluate reusable IR test collections from academic content.

## 📚 Table of Contents

1.  [Features](#features)
2.  [Requirements](#requirements)
3.  [Installation](#installation)
4.  [Quick Start](#quick-start)
5.  [Detailed Usage Guide](#detailed-usage-guide)
    * [Web Scraping](#web-scraping)
    * [Corpus Building](#corpus-building)
    * [Index Creation](#index-creation)
    * [Topics and Qrels](#topics-and-qrels)
    * [Evaluation](#evaluation)
6.  [Directory Structure](#directory-structure)
7.  [Configuration](#configuration)
8.  [Running Tests](#running-tests)
9.  [Common Issues](#common-issues)
10. [Contributing](#contributing)
11. [Citation](#citation)

## ✨ Features

-   **Web Scraping**: Robust academic website scraping with rate limiting and content extraction
-   **Corpus Building**: Create structured corpora using sliding window passages
-   **Multiple Index Types**:
    -   BM25 sparse index
    -   Dense retrieval using transformer models
    -   Hybrid retrieval combining both approaches
-   **Topic Integration**: Convert and validate topics, generate automatic qrels
-   **Comprehensive Evaluation**: Standard IR metrics with visualization
-   **Test Collection Creation**: Tools for creating reusable IR test collections

## 📋 Requirements

### System Requirements

-   Python 3.8 or higher
-   Java 11 or higher (for Pyserini)
-   8GB RAM minimum (16GB recommended for dense indexing)
-   CUDA-compatible GPU (optional, for faster dense indexing)

### Python Dependencies

Main dependencies include:

-   pyserini
-   torch
-   sentence-transformers
-   faiss-cpu (or faiss-gpu)
-   numpy
-   pandas
-   spacy
-   beautifulsoup4
-   tqdm

## 🔧 Installation

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/yourusername/academic-ir-pipeline](https://github.com/yourusername/academic-ir-pipeline)
    cd academic-ir-pipeline
    ```

2.  **Create a Virtual Environment (Recommended)**

    ```bash
    # Using venv
    python -m venv venv
    ```bash
    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On Unix or MacOS:
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    # Install base requirements
    pip install -r requirements.txt

    # Install development requirements (for testing)
    pip install -r requirements-dev.txt

    # Download spaCy model
    python -m spacy download en_core_web_sm
    ```

4.  **Install Java (Required for Pyserini)**

    On Ubuntu:

    ```bash
    sudo apt-get update
    sudo apt-get install openjdk-11-jdk
    ```

    On macOS:

    ```bash
    brew install openjdk@11
    ```

    On Windows:

    Download and install OpenJDK 11 from AdoptOpenJDK

    Add Java to your PATH environment variable

5.  **Verify Installation**

    ```bash
    # Run tests to verify installation
    pytest tests/
    ```

## 🚀 Quick Start

Here's a minimal example to get started:

```bash
# 1. Scrape a website
python -m src.scraping.website_scraper \
    --url "https://example.edu/research" \
    --output data/scraped \
    --max-pages 100
```

# 2. Build corpus
```bash
python -m src.corpus.builder \
    --input data/scraped/scraped_data.json \
    --output academic_corpus
```

# 3. Create index
```bash
python -m src.indexing.create_index \
    --corpus academic_corpus \
    --output index \
    --dense \
    --encoder sentence-transformers/msmarco-distilbert-base-v3
```
# 4. Run evaluation
```bash
python -m src.evaluation.ir_tools evaluate \
    --qrels academic_corpus/qrels.txt \
    --run runs/bm25_run.txt \
    --output evaluation_results.json
```

## 📁 Directory Structure
```aiignore
academic-ir-pipeline/
├── src/                    # Source code
│   ├── scraping/          # Web scraping tools
│   ├── corpus/            # Corpus building
│   ├── indexing/          # Index creation
│   ├── evaluation/        # Evaluation tools
│   └── topics/            # Topic handling
├── tests/                 # Test suite
├── data/                  # Data directory
├── docs/                  # Documentation
└── examples/              # Example notebooks


```
---
## 📂 Data & Usage Documentation

This repository includes two key supporting guides to help you work with the pipeline:

- 📦 **[Data Directory Guide](docs/data.md)**:  
  Explains how the `data/` directory is organized, including scraped inputs, processed corpus files, and prebuilt indexes.  
  → Learn about file formats, naming conventions, and how to manage large files.

- 🛠️ **[Usage Guide](docs/usage.md)**:  
  A step-by-step tutorial covering scraping, corpus creation, indexing, topic integration, evaluation, and troubleshooting.  
  → Follow this guide if you're running the pipeline end-to-end or exploring individual components.

---
## 🧪 Running Tests


1. Install test dependencies
 
You can either install them directly:
```bash
pip install pytest pytest-cov pytest-asyncio pytest-timeout
```
Or install all dev dependencies at once:
```bash
pip install -r requirements-dev.txt

```

2. Run the tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_corpus_builder.py

# Run excluding slow tests
pytest -m "not slow"
```

## 🤝 Contributing
- Fork the repository
- Create a feature branch
- Make your changes
- Run tests
- Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Pyserini for IR tools
- BEIR for evaluation methodology
- Sentence Transformers for dense retrieval

## 📧 Contact
For questions and support