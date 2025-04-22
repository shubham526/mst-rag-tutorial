"""
Index Builder
------------
Create Pyserini indexes for academic corpus, supporting both
sparse (BM25) and dense retrieval indexes.
"""

import os
import json
import logging
import subprocess
from typing import Dict, Optional, List
import pandas as pd
from tqdm import tqdm
from pyserini.index.lucene import IndexReader
from ..utils.helpers import setup_logging

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build Pyserini indexes from academic corpus."""

    def __init__(self, corpus_dir: str, output_dir: str = "pyserini_index"):
        """
        Initialize the index builder.

        Args:
            corpus_dir: Path to the academic corpus directory
            output_dir: Directory to save the index
        """
        self.corpus_dir = corpus_dir
        self.output_dir = output_dir
        self.passages_csv = os.path.join(corpus_dir, "passages.csv")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_dir = os.path.join(output_dir, "jsonl")
        os.makedirs(self.jsonl_dir, exist_ok=True)

        # Initialize paths
        self.index_dir = os.path.join(output_dir, "lucene-index")

        # Verify corpus exists
        if not os.path.exists(self.passages_csv):
            raise FileNotFoundError(f"Passages file not found at {self.passages_csv}")

        # Check for Java
        self._check_java()

    def _check_java(self) -> None:
        """Verify Java installation"""
        try:
            java_version = subprocess.check_output(
                ['java', '-version'],
                stderr=subprocess.STDOUT,
                text=True
            )
            logger.info(f"Found Java: {java_version.split('\n')[0]}")
        except Exception as e:
            logger.error("Java not found. Please install Java 11 or newer.")
            raise RuntimeError("Java is required for indexing")

    def convert_to_jsonl(self) -> int:
        """
        Convert passages to Pyserini-compatible JSONL format.

        Returns:
            Number of documents converted
        """
        logger.info("Converting passages to JSONL format")

        try:
            passages_df = pd.read_csv(self.passages_csv)
            count = 0
            jsonl_path = os.path.join(self.jsonl_dir, "passages.jsonl")

            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for _, row in tqdm(passages_df.iterrows(),
                                   total=len(passages_df),
                                   desc="Converting passages"):
                    doc = {
                        "id": str(row['id']),
                        "contents": row['text'],
                        "title": row.get('title', ''),
                        "url": row.get('url', ''),
                        "word_count": int(row.get('word_count', 0))
                    }

                    # Add optional fields
                    for field in ['sentence_start_idx', 'sentence_end_idx', 'num_sentences']:
                        if field in row and pd.notna(row[field]):
                            doc[field] = int(row[field])

                    f.write(json.dumps(doc) + '\n')
                    count += 1

            logger.info(f"Converted {count} passages to JSONL format")
            return count

        except Exception as e:
            logger.error(f"Error converting to JSONL: {e}")
            raise

    def build_sparse_index(self,
                           stemming: bool = True,
                           stopwords: bool = True,
                           threads: int = 8) -> Optional[str]:
        """
        Build sparse BM25 index using Pyserini.

        Args:
            stemming: Whether to apply Porter stemming
            stopwords: Whether to remove stopwords
            threads: Number of indexing threads

        Returns:
            Path to created index
        """
        logger.info("Building sparse BM25 index")

        # Construct indexing command
        cmd = [
            'java', '-cp', 'anserini.jar',
            'io.anserini.index.IndexCollection',
            '-collection', 'JsonCollection',
            '-generator', 'DefaultLuceneDocumentGenerator',
            '-threads', str(threads),
            '-input', self.jsonl_dir,
            '-index', self.index_dir,
            '-storePositions',
            '-storeDocvectors',
            '-storeRaw'
        ]

        # Add stemming option
        if stemming:
            cmd.extend(['-stemmer', 'porter'])
        else:
            cmd.extend(['-stemmer', 'none'])

        # Add stopwords option
        if not stopwords:
            cmd.append('-keepStopwords')

        try:
            logger.info(f"Running indexing command: {' '.join(cmd)}")
            result = subprocess.run(cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)

            logger.info("Sparse index built successfully")
            logger.debug(f"Indexing output:\n{result.stdout}")

            if result.stderr:
                logger.warning(f"Indexing warnings:\n{result.stderr}")

            return self.index_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Error building index: {e}")
            logger.error(f"Command output:\n{e.output}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error building index: {e}")
            return None

    def build_dense_index(self,
                          encoder_name: str,
                          batch_size: int = 32,
                          device: str = 'cpu',
                          use_fp16: bool = False) -> Optional[str]:
        """
        Build dense index using Sentence Transformers.

        Args:
            encoder_name: Name of the encoder model
            batch_size: Batch size for encoding
            device: Device to use (cpu/cuda)
            use_fp16: Whether to use FP16 precision

        Returns:
            Path to created index
        """
        dense_dir = os.path.join(self.output_dir, "dense-index")
        os.makedirs(dense_dir, exist_ok=True)

        logger.info(f"Building dense index with encoder {encoder_name}")

        cmd = [
            'python', '-m', 'pyserini.encode',
            'input',
            '--corpus', os.path.join(self.jsonl_dir, "passages.jsonl"),
            '--fields', 'text',
            'output',
            '--embeddings', dense_dir,
            '--to-faiss',
            'encoder',
            '--encoder', encoder_name,
            '--batch', str(batch_size)
        ]

        # Add device and precision options
        if device.startswith('cuda'):
            cmd.extend(['--device', device])
            if use_fp16:
                cmd.append('--fp16')

        try:
            logger.info(f"Running encoding command: {' '.join(cmd)}")
            result = subprocess.run(cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)

            logger.info("Dense index built successfully")
            logger.debug(f"Encoding output:\n{result.stdout}")

            if result.stderr:
                logger.warning(f"Encoding warnings:\n{result.stderr}")

            return dense_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Error building dense index: {e}")
            logger.error(f"Command output:\n{e.output}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error building dense index: {e}")
            return None

    def verify_index(self, index_path: str) -> Optional[Dict]:
        """
        Verify index by checking statistics.

        Args:
            index_path: Path to index directory

        Returns:
            Dictionary of index statistics
        """
        try:
            reader = IndexReader(index_path)
            stats = reader.stats()

            logger.info("Index statistics:")
            logger.info(f"  Documents: {stats['documents']}")
            logger.info(f"  Unique terms: {stats['unique_terms']}")
            logger.info(f"  Total terms: {stats['total_terms']}")

            return stats

        except Exception as e:
            logger.error(f"Error verifying index: {e}")
            return None

    def build(self,
              sparse: bool = True,
              dense: bool = False,
              encoder_name: Optional[str] = None,
              **kwargs) -> Dict[str, Optional[str]]:
        """
        Build all requested indexes.

        Args:
            sparse: Whether to build sparse index
            dense: Whether to build dense index
            encoder_name: Encoder for dense index
            **kwargs: Additional arguments for index building

        Returns:
            Dictionary of index types to paths
        """
        results = {}

        # Convert to JSONL format first
        doc_count = self.convert_to_jsonl()
        if doc_count == 0:
            logger.error("No documents converted to JSONL")
            return results

        # Build sparse index
        if sparse:
            sparse_path = self.build_sparse_index(
                stemming=kwargs.get('stemming', True),
                stopwords=kwargs.get('stopwords', True),
                threads=kwargs.get('threads', 8)
            )
            results['sparse'] = sparse_path

            if sparse_path:
                self.verify_index(sparse_path)

        # Build dense index
        if dense:
            if not encoder_name:
                logger.error("Encoder name required for dense index")
                return results

            dense_path = self.build_dense_index(
                encoder_name=encoder_name,
                batch_size=kwargs.get('batch_size', 32),
                device=kwargs.get('device', 'cpu'),
                use_fp16=kwargs.get('use_fp16', False)
            )
            results['dense'] = dense_path

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build Pyserini indexes')
    parser.add_argument('--corpus', required=True,
                        help='Path to corpus directory')
    parser.add_argument('--output', default='pyserini_index',
                        help='Output directory for indexes')

    # Sparse index options
    parser.add_argument('--no-stemming', action='store_true',
                        help='Disable Porter stemming')
    parser.add_argument('--keep-stopwords', action='store_true',
                        help='Keep stopwords')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of indexing threads')

    # Dense index options
    parser.add_argument('--dense', action='store_true',
                        help='Build dense index')
    parser.add_argument('--encoder',
                        help='Encoder model for dense index')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--device', default='cpu',
                        help='Device for encoding')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Build indexes
    builder = IndexBuilder(args.corpus, args.output)
    results = builder.build(
        sparse=True,
        dense=args.dense,
        encoder_name=args.encoder,
        stemming=not args.no_stemming,
        stopwords=not args.keep_stopwords,
        threads=args.threads,
        batch_size=args.batch_size,
        device=args.device,
        use_fp16=args.fp16
    )

    # Report results
    for index_type, path in results.items():
        if path:
            logger.info(f"{index_type.capitalize()} index built at: {path}")
        else:
            logger.error(f"Failed to build {index_type} index")


if __name__ == '__main__':
    main()
