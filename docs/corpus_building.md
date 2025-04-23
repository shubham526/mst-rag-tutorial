
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

1. Configure scraping sources in `config/scraping_config.yaml`
2. Run the scraper:
   
   ```bash
   python -m src.scraping.website_scraper
   ```
4. Process the scraped data:
   
   ```bash
   python -m src.corpus.builder
   ```
