"""
Multi-File Corpus Builder
------------------------
Process multiple scraped data files into a unified corpus.
"""

import os
import glob
import logging
from typing import List, Optional
from tqdm import tqdm

from .builder import CorpusBuilder
from ..utils.helpers import setup_logging

logger = logging.getLogger(__name__)


class MultiCorpusBuilder(CorpusBuilder):
    def __init__(self,
                 input_dir: str,
                 output_dir: str = "academic_corpus",
                 window_size: int = 10,
                 stride: int = 5):
        """
        Initialize the multi-file corpus builder.

        Args:
            input_dir: Directory containing scraped_data.json files
            output_dir: Directory to save the processed corpus
            window_size: Number of sentences per window
            stride: Number of sentences to stride
        """
        super().__init__(None, output_dir, window_size, stride)
        self.input_dir = input_dir
        self.input_files = []

    def find_input_files(self) -> List[str]:
        """Find all scraped_data.json files in input directory"""
        pattern = os.path.join(self.input_dir, "**", "scraped_data.json")
        files = glob.glob(pattern, recursive=True)
        logger.info(f"Found {len(files)} scraped_data.json files")
        self.input_files = files
        return files

    def load_data(self) -> None:
        """Load and combine data from all input files"""
        if not self.input_files:
            self.find_input_files()

        if not self.input_files:
            raise ValueError(f"No scraped_data.json files found in {self.input_dir}")

        self.raw_data = []

        for file_path in tqdm(self.input_files, desc="Loading files"):
            try:
                super().__init__(file_path, self.output_dir,
                                 self.window_size, self.stride)
                super().load_data()
                self.raw_data.extend(self.raw_data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(self.raw_data)} total documents")

    def process_documents(self) -> None:
        """Process documents with source tracking"""
        logger.info("Processing documents from multiple sources...")

        all_passages = []
        doc_count = 0

        for doc in tqdm(self.raw_data, desc="Processing documents"):
            if len(doc.get('content', '')) < 100:
                continue

            clean_content = self.clean_text(doc['content'])
            passages = self.segment_into_passages(
                clean_content,
                doc.get('title', 'No Title'),
                doc.get('url', '')
            )

            # Add source file information
            source_file = doc.get('source_file', 'unknown')
            for passage in passages:
                passage['source_file'] = source_file

            all_passages.extend(passages)

            # Create metadata with source information
            doc_id = f"DOC-{doc_count}"
            self.metadata[doc_id] = {
                "title": doc.get('title', 'No Title'),
                "url": doc.get('url', ''),
                "meta_description": doc.get('meta_description', ''),
                "num_passages": len(passages),
                "passage_ids": [p['id'] for p in passages],
                "source_file": source_file
            }

            doc_count += 1

        self.documents = all_passages
        logger.info(f"Created {len(all_passages)} passages from {doc_count} documents")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Build corpus from multiple scraped data files')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing scraped_data.json files')
    parser.add_argument('--output', default='academic_corpus',
                        help='Output directory')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Sentences per window')
    parser.add_argument('--stride', type=int, default=5,
                        help='Sentences to stride')

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Build corpus
    builder = MultiCorpusBuilder(
        args.input_dir,
        args.output,
        args.window_size,
        args.stride
    )
    builder.build()


if __name__ == '__main__':
    main()
