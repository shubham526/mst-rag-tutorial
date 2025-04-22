# Data Directory

This directory contains data files used by the Academic IR Pipeline.

## Directory Structure
```aiignore
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

## File Formats

### Scraped Data

-   `scraped_data.json`: Contains raw scraped content

    ```json
    [
      {
        "url": "[https://example.edu/page](https://example.edu/page)",
        "title": "Page Title",
        "meta_description": "Page description",
        "content": "Main content..."
      }
    ]
    ```

### Corpus Files

-   `passages.csv`: Main corpus file with all passages
    -   Columns: id, text, title, url, word_count, num_sentences
-   `metadata.json`: Document-level metadata
-   Individual passage files in `docs/` directory

### Topics and Qrels

-   `topics.json`: NIST-style topic definitions
-   `qrels.txt`: Relevance judgments

## Usage

This directory is used automatically by the pipeline components. Files will be created and organized here during:

-   Web scraping
-   Corpus building
-   Index creation
-   Evaluation runs

## Note

Large files in this directory should not be committed to Git. The `.gitignore` file is configured to ignore most data files while keeping the directory structure.

## Prebuilt Indexes
The `prebuilt_indexes/` directory may contain large files not suitable for version control. To avoid bloating the repository, we host all prebuilt sparse and dense indexes on Hugging Face Datasets. 

Great — this script is very helpful for downloading prebuilt indexes from Hugging Face! To complement it, you should add a **code snippet and usage explanation** in your documentation (e.g., in your `README.md` or `docs/advanced_rag.md`) that shows **how to run this script** to fetch prebuilt indexes for a selected dataset.

Here’s a suggested markdown block to include under a section like:

---

### 🔽 How to Download Prebuilt Indexes from Hugging Face

We host all prebuilt RAG indexes on the [Hugging Face Hub](https://huggingface.co/datasets/ShubhamC/rag-tutorial-prebuilt-indexes). These include dense vectors, FAISS indexes, and corpus files for popular academic datasets.

To automatically download the correct files, use the helper script below:

<details>
<summary>▶️ <strong>Python Script: Download Indexes</strong> (click to expand)</summary>

```python
from huggingface_hub import hf_hub_download
import os, pickle, numpy as np
import faiss

HUB_REPO_ID = "ShubhamC/rag-tutorial-prebuilt-indexes"
selected_dataset = "beir/trec-covid"  # Replace with your dataset
repo_folder_name = selected_dataset.replace('/', '_')
base_path = f"prebuilt_indexes/{repo_folder_name}"
os.makedirs(base_path, exist_ok=True)

files_to_download = ["corpus.pkl", "embeddings.npy", "faiss_index.bin", "doc_ids.pkl"]

print(f"Downloading pre-built indexes for {selected_dataset} from HF Hub...")

for file_name in files_to_download:
    local_path = os.path.join(base_path, file_name)
    if not os.path.exists(local_path):
        print(f"Downloading {file_name}...")
        try:
            path_in_repo = f"{repo_folder_name}/{file_name}"
            downloaded_path = hf_hub_download(
                repo_id=HUB_REPO_ID,
                filename=path_in_repo,
                repo_type="dataset",
                local_dir=base_path,
                local_dir_use_symlinks=False
            )
            # Move if needed
            if not os.path.exists(local_path) and os.path.exists(downloaded_path):
                os.rename(downloaded_path, local_path)
            print(f"✅ {file_name} saved to {local_path}")
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")
```
</details>

📌 **Tip:** If you want to generate sample data instead (for demos or testing), the script will automatically create dummy files when download fails.

---
