"""
Pre-built Index Creator
----------------------
Create pre-built indexes for common IR datasets using
Pyserini and BEIR datasets.
"""

import os
import logging
from typing import Dict, Tuple, Optional
import torch
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

from ..utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def create_prebuilt_index(dataset_name: str,
                          output_dir: str,
                          model_name: str = "sentence-transformers/msmarco-distilbert-base-v3",
                          batch_size: int = 32,
                          device: Optional[str] = None) -> Dict:
    """
    Create a pre-built index for a specific dataset.

    Args:
        dataset_name: Name of the dataset in BEIR format
        output_dir: Directory to save the index
        model_name: Name of the encoder model
        batch_size: Batch size for encoding
        device: Device to use (defaults to CUDA if available)

    Returns:
        Dictionary containing index information
    """
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Creating pre-built index for {dataset_name}")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Download and load dataset
    try:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, output_dir)

        # Load corpus, queries, and qrels
        corpus, queries, qrels = load_beir_data(data_path)
        logger.info(f"Loaded dataset with {len(corpus)} documents")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Create dense index
    try:
        # Load encoder model
        encoder = SentenceTransformer(model_name)
        encoder.to(device)

        # Encode corpus
        logger.info("Encoding corpus...")
        corpus_embeddings = encode_corpus(
            corpus,
            encoder,
            batch_size,
            device
        )

        # Build FAISS index
        logger.info("Building FAISS index...")
        faiss_index = build_faiss_index(corpus_embeddings)

        # Save components
        save_index_components(
            output_dir,
            corpus,
            queries,
            qrels,
            corpus_embeddings,
            faiss_index
        )

        index_info = {
            "dataset": dataset_name,
            "model": model_name,
            "num_documents": len(corpus),
            "num_queries": len(queries),
            "embedding_dim": corpus_embeddings.shape[1]
        }

        logger.info("Index creation completed successfully")
        return index_info

    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def load_beir_data(data_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load data from BEIR dataset"""
    data_loader = GenericDataLoader(data_path)
    corpus, queries, qrels = data_loader.load(split="test")
    return corpus, queries, qrels


def encode_corpus(corpus: Dict,
                  encoder: SentenceTransformer,
                  batch_size: int,
                  device: str) -> np.ndarray:
    """Encode corpus documents"""
    # Prepare texts
    texts = [
        f"{doc.get('title', '')}. {doc.get('text', '')}"
        for doc in corpus.values()
    ]

    # Encode in batches
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size),
                  desc="Encoding documents"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_embeddings = encoder.encode(
                batch,
                convert_to_tensor=True,
                device=device
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index from embeddings"""
    dimension = embeddings.shape[1]

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create index
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def save_index_components(output_dir: str,
                          corpus: Dict,
                          queries: Dict,
                          qrels: Dict,
                          embeddings: np.ndarray,
                          faiss_index: faiss.Index) -> None:
    """Save all index components"""
    import pickle

    # Save corpus
    with open(os.path.join(output_dir, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    # Save queries
    with open(os.path.join(output_dir, "queries.pkl"), "wb") as f:
        pickle.dump(queries, f)

    # Save qrels
    with open(os.path.join(output_dir, "qrels.pkl"), "wb") as f:
        pickle.dump(qrels, f)

    # Save embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)

    # Save FAISS index
    faiss.write_index(faiss_index,
                      os.path.join(output_dir, "faiss_index.bin"))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create pre-built indexes for IR datasets')
    parser.add_argument('--dataset', required=True,
                        help='Name of BEIR dataset')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--model',
                        default="sentence-transformers/msmarco-distilbert-base-v3",
                        help='Encoder model name')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--device',
                        help='Device to use (cpu/cuda)')

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Create index
    create_prebuilt_index(
        args.dataset,
        args.output,
        args.model,
        args.batch_size,
        args.device
    )


if __name__ == '__main__':
    main()
