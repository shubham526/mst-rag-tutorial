# Usage Guide

This guide provides detailed instructions for using the **RAGKit**.

---

## Table of Contents

1. Web Scraping  
2. Corpus Building  
3. Index Creation  
4. Topic Integration  
5. Evaluation  
6. End-to-End Example

---

## Web Scraping

### Basic Usage

```
python -m src.scraping.website_scraper \
    --url "https://example.edu" \
    --output data/scraped
```

### Advanced Options

```
python -m src.scraping.website_scraper \
    --url "https://example.edu" \
    --output data/scraped \
    --max-pages 100 \
    --delay 1.0 \
    --exclude login admin calendar \
    --timeout 30
```

### Output Structure

```
data/scraped/
├── scraped_data.json     # Main output file
├── scraped_data.csv      # CSV format
└── text_files/           # Individual text files
```

---

## Corpus Building

### Single Input File

```
python -m src.corpus.builder \
    --input data/scraped/scraped_data.json \
    --output academic_corpus \
    --window-size 10 \
    --stride 5
```

### Multiple Input Files

```
python -m src.corpus.multi_builder \
    --input-dir data/multiple_sources \
    --output academic_corpus
```

### Corpus Structure

```
academic_corpus/
├── docs/              # Individual passage files
├── metadata.json      # Document metadata
├── passages.csv       # All passages
└── corpus_info.json   # Corpus statistics
```

---

## Index Creation

### BM25 Index

```
python -m src.indexing.create_index \
    --corpus academic_corpus \
    --output index \
    --threads 8
```

### Dense Index

```
python -m src.indexing.create_index \
    --corpus academic_corpus \
    --output index \
    --dense \
    --encoder sentence-transformers/msmarco-distilbert-base-v3 \
    --batch-size 32 \
    --device cuda
```

### Hybrid Index

```
python -m src.indexing.create_index \
    --corpus academic_corpus \
    --output index \
    --dense \
    --hybrid \
    --encoder sentence-transformers/msmarco-distilbert-base-v3
```

---

## Topic Integration

### Converting Topics

```
python -m src.topics.converter \
    --input topics.json \
    --output topics.xml \
    --format json
```

### Creating Automatic Qrels

```
python -m src.topics.integrator \
    --corpus academic_corpus \
    --topics topics.xml \
    --method hybrid
```

### Available Methods

- `keyword`: Simple keyword matching  
- `bm25`: BM25 scoring  
- `tfidf`: TF-IDF similarity  
- `semantic`: Dense embedding similarity  
- `hybrid`: Combination of methods

---

## Evaluation

### Basic Evaluation

```
python -m src.evaluation.ir_tools evaluate \
    --qrels qrels.txt \
    --run run.txt \
    --metrics map ndcg_cut_10 P_10 recall_1000 \
    --output results.json
```

### Generating Statistics

```
python -m src.evaluation.ir_tools stats \
    --corpus academic_corpus \
    --output stats
```

### Visualization

The evaluation tools automatically generate:

- Precision-Recall curves  
- Metric comparison plots  
- Performance statistics

---

## End-to-End Example

Here's a complete example workflow:

```
# 1. Scrape website
python -m src.scraping.website_scraper \
    --url "https://example.edu" \
    --output data/scraped \
    --max-pages 100

# 2. Build corpus
python -m src.corpus.builder \
    --input data/scraped/scraped_data.json \
    --output academic_corpus

# 3. Create topics and qrels
python -m src.topics.integrator \
    --corpus academic_corpus \
    --topics topics.xml \
    --method hybrid

# 4. Create indexes
python -m src.indexing.create_index \
    --corpus academic_corpus \
    --output index \
    --dense \
    --hybrid

# 5. Run evaluation
python -m src.evaluation.ir_tools evaluate \
    --qrels academic_corpus/qrels.txt \
    --run runs/hybrid_run.txt \
    --output results.json
```

---

## Best Practices

### Web Scraping

- Use appropriate delays between requests  
- Respect `robots.txt`  
- Handle rate limiting and errors gracefully

### Corpus Building

- Choose appropriate window size and stride  
- Clean and normalize text properly  
- Validate corpus quality

### Indexing

- Use suitable batch sizes depending on GPU memory  
- Enable multi-threading for CPU-based indexing  
- Inspect and validate index statistics

### Evaluation

- Use multiple evaluation metrics  
- Generate visualizations for analysis  
- Compare BM25, dense, and hybrid performance

---

## Troubleshooting

### 1. Out of Memory

```
# Reduce batch size
python -m src.indexing.create_index --batch-size 16

# Use CPU if GPU memory is insufficient
python -m src.indexing.create_index --device cpu
```

### 2. Slow Processing

```
# Increase number of threads
python -m src.indexing.create_index --threads 16

# Reduce corpus size for testing
python -m src.corpus.builder --max-docs 1000
```

### 3. Quality Issues

```
# Adjust passage creation parameters
python -m src.corpus.builder --window-size 15 --stride 7

# Use more sophisticated qrel creation
python -m src.topics.integrator --method hybrid
```

---

📁 For more examples and detailed explanations, see the Jupyter notebooks in the `examples/` directory.
