# Data Directory

This directory contains data files used by the Academic IR Pipeline.

## Directory Structure
```
data/
├── scraped_data/         # Raw scraped website data
│   ├── example.edu/      # Data from each domain
│   └── research.edu/
├── academic_corpus/      # Processed corpus
│   ├── docs/             # Individual passage files
│   ├── metadata.json     # Corpus metadata
│   └── passages.csv      # All passages in CSV format
└── prebuilt_indexes/     # Pre-built indexes
    ├── sparse/           # BM25 indexes
    └── dense/            # Dense retrieval indexes
```

**Note:** Large files in this directory should not be committed to Git. The `.gitignore` file is configured to ignore most data files while keeping the directory structure.

## Contents

For detailed information, see the following documentation files:

- [File Formats](file_formats.md) - Explanation of all data file formats
- [Prebuilt Indexes](prebuilt_indexes.md) - Information about index creation and downloading
- [Corpus Building](corpus_building.md) - Details on the corpus building process
- [Topics and Evaluation](topics_and_evaluation.md) - Topic generation and evaluation methodology
