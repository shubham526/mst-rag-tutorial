# Prebuilt Indexes

The `prebuilt_indexes/` directory contains retrieval indexes that are used for efficient information retrieval. Due to their size, these files are not stored in the Git repository but are hosted on Hugging Face Datasets.

## ⚙️ How the Pre-built Indexes Were Created

To ensure smooth performance, we've prepared **pre-built retrieval indexes** for all datasets in advance. This saves time and avoids requiring participants to download large corpora or compute document embeddings live.

We used the following approach to generate these indexes using the `ir_datasets`, `sentence-transformers`, and `faiss` libraries.

### 🛠️ Index Construction Pipeline
For each dataset, we followed this process:

1. **Load the dataset** using `ir_datasets`, including documents, queries, and relevance judgments (if available).
2. **Preprocess each document** by combining its title and text (if both are available).
3. **Generate dense embeddings** using the [`msmarco-distilbert-base-v3`](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3) SentenceTransformer model.
4. **Normalize embeddings** and index them using **FAISS** with an inner product search (cosine similarity on normalized vectors).
5. **Store metadata**, including:
   - A pickled dictionary of document texts and titles
   - The FAISS index (`faiss_index.bin`)
   - The NumPy matrix of document embeddings
   - Sample queries and relevance judgments for evaluation

This was implemented via a Python script (`create_prebuilt_indexes.py`) using the following function:

```python
create_prebuilt_index(dataset_name="beir/trec-covid",
                      output_dir="prebuilt_indexes/beir_trec-covid",
                      model_name="sentence-transformers/msmarco-distilbert-base-v3")
```

You can modify and rerun this script to generate new indexes using your own datasets or preferred embedding models.

### 📁 Files in Each Index
Each prebuilt index contains:
- `corpus.pkl`: A dictionary mapping document IDs to `{text, title}`
- `embeddings.npy`: Dense vectors for each document
- `faiss_index.bin`: FAISS index for fast retrieval
- `doc_ids.pkl`: Mapping of embedding rows to document IDs
- `sample_queries.pkl`: Example queries for demonstration
- `qrels.pkl`: Relevance judgments (if available)

> ✅ **Note:** In this tutorial, we only load these prebuilt files and use them directly, skipping embedding and indexing steps to keep things interactive and lightweight.

## 🔽 How to Download Prebuilt Indexes from Hugging Face

We host all prebuilt RAG indexes on the [Hugging Face Hub](https://huggingface.co/datasets/ShubhamC/rag-tutorial-prebuilt-indexes). These include dense vectors, FAISS indexes, and corpus files for popular academic datasets.

To automatically download the correct files, use the helper script below:

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

📌 **Tip:** If you want to generate sample data instead (for demos or testing), the script will automatically create dummy files when download fails.
