# Academic IR Pipeline

A comprehensive toolkit for building and evaluating academic information retrieval systems. This pipeline allows you to create, index, and evaluate reusable IR test collections from academic content. 

🚀 Run the Tutorial in Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/academic-ir-pipeline/blob/main/tutorial/academic_rag_demo.ipynb)

## 📚 Table of Contents

1.  [Features](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-features)
2.  [Requirements](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-requirements)
3.  [Installation](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-installation)
4.  [Quick Start](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-quick-start)
5.  [Directory Structure](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-directory-structure)
6.  [Data and Usage Guide](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-data--usage-documentation)
7.  [Running Tests](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-running-tests)
8.  [Contributing](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-contributing)
11. [LICENCE](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-license)
12. [Acknowledgements](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-acknowledgments)
13. [Contact](https://github.com/shubham526/mst-rag-tutorial?tab=readme-ov-file#-acknowledgments)


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
-   Java 21 or higher (for Pyserini)
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

1. Scrape a website
```bash
python -m src.scraping.website_scraper \
    --url "https://example.edu/research" \
    --output data/scraped \
    --max-pages 100
```

2. Build corpus
```bash
python -m src.corpus.builder \
    --input data/scraped/scraped_data.json \
    --output academic_corpus
```

3. Create index
```bash
python -m src.indexing.create_index \
    --corpus academic_corpus \
    --output index \
    --dense \
    --encoder sentence-transformers/msmarco-distilbert-base-v3
```
4. Run evaluation
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
- [Pyserini](https://github.com/castorini/pyserini) for IR tools
- [BEIR](https://github.com/beir-cellar/beir) for datasets and evaluation methodology
- [Sentence Transformers](https://sbert.net/) for dense retrieval

## 📧 Contact
For questions and support, email shubham.chatterjee@mst.edu.
