
# 🏫 Corpus Building Process for Missouri S&T Data
This document describes how the academic corpus was built from web sources.


We collected academic web content using the `website_scraper.py` tool on the following domains of Missouri S&T:

- **Main University Website**: `mst.edu`
- **Admissions and Future Students**: `futurestudents.mst.edu`
- **Distance and Online Education**: `distance.mst.edu`
- **Athletics**: `minerathletics.com`
- **Alumni Association**: `mineralumni.com`
- **Student Diversity Initiatives**: `lgbtqrolla.org/resources/campus/`
- **S&T Sites (Educational Blogs and Projects)**: `sites.mst.edu`

These domains encompass a wide array of content, including:
- Academic programs
- Student life
- Research initiatives
- Athletic events
- Alumni activities
- Diversity resources
- Various educational projects

By aggregating data from these sources, we aim to build a rich and representative corpus for information retrieval tasks.

*Note: All data collection was performed in compliance with the respective websites' terms of service and robots.txt directives.*

## Processing Pipeline

Raw HTML was parsed and cleaned, then transformed into structured text using:

1. **Data Collection**:
   - `src.scraping.website_scraper.py` for web scraping
   
2. **Corpus Creation**:
   - `src.corpus.builder.py` for turning documents into passages
   
3. **Text Processing** (if applicable):
   - `src.corpus.cleaning.py` for preprocessing

## Data Organization

The processed data is organized as follows:

1. **Raw Scraped Data**: 
   - Stored in `data/scraped_data/{domain}/`
   - Contains raw JSON files with scraped content

2. **Processed Corpus**:
   - Stored in `data/academic_corpus/`
   - Contains:
     - `passages.csv`: Main corpus file
     - `metadata.json`: Document metadata
     - `docs/`: Individual passage files

## Usage

To build a new corpus:
1. Run the scraper:
   
   ```bash
   python -m src.scraping.website_scraper
   ```
2. Process the scraped data:
   
   ```bash
   python -m src.corpus.builder
   ```

## Dataset Access

### Downloading from Hugging Face

The Missouri S&T dataset is available on Hugging Face. To download and use it:

1. **Install Required Packages**:
   ```bash
   pip install huggingface_hub pandas
   ```

2. **Download Dataset Files**:
   ```python
   from huggingface_hub import hf_hub_download
   import os
   
   # Create directory for the dataset
   os.makedirs("prebuilt_indexes/custom_mst_site", exist_ok=True)
   
   # Download key files
   for filename in ["topics.json", "auto_qrels.txt", "passages.csv"]:
       filepath = hf_hub_download(
           repo_id="your-hf-repo-id",  # Replace with actual repository ID
           filename=f"custom_mst_site/{filename}",
           repo_type="dataset"
       )
       
       # Copy to your working directory
       os.system(f"cp {filepath} prebuilt_indexes/custom_mst_site/")
   ```

3. **Loading and Using the Corpus**:
   ```python
   import pandas as pd
   
   # Load passages
   passages_df = pd.read_csv("prebuilt_indexes/custom_mst_site/passages.csv")
   
   # Example: View the first few passages
   print(passages_df.head())
   
   # Example: Access passage content
   for idx, row in passages_df.iterrows():
       print(f"Passage ID: {row['id']}")
       print(f"Content: {row['content'][:100]}...")  # Show first 100 chars
       print(f"Source: {row['url']}")
       print("-" * 50)
       if idx >= 2:  # Just show the first 3 examples
           break
   ```

### Working with the Dataset

The MST dataset contains three key files:

1. **passages.csv**: Contains the corpus text data
   - Format: CSV with columns for passage ID, content, URL, and metadata
   - Usage: Use for building search indexes or retrieving relevant passages

2. **topics.json**: Contains the search queries/topics
   - Format: JSON with topic IDs, titles, descriptions, and narratives
   - Usage: Use as queries for testing retrieval systems

3. **auto_qrels.txt**: Contains relevance judgments
   - Format: TREC-style qrels format with topic ID, 0, document ID, and relevance score
   - Usage: Use for evaluating retrieval effectiveness

Example workflow for information retrieval experiments:
1. Index the passages from `passages.csv`
2. Run queries from `topics.json`
3. Evaluate results using `auto_qrels.txt`


