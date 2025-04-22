"""
Corpus Builder
-------------
Process scraped website data into a structured corpus suitable
for academic information retrieval and QA tasks using a sliding
window approach.
"""

import os
import json
import logging
from typing import Dict, List, Optional
import pandas as pd
import spacy
from tqdm import tqdm

from ..utils.helpers import clean_text, create_unique_id

logger = logging.getLogger(__name__)


class CorpusBuilder:
    def __init__(self,
                 input_data_path: str,
                 output_dir: str = "academic_corpus",
                 window_size: int = 10,
                 stride: int = 5):
        """
        Initialize the corpus builder.

        Args:
            input_data_path: Path to the scraped data JSON file
            output_dir: Directory to save the processed corpus
            window_size: Number of sentences per window
            stride: Number of sentences to stride when creating new window
        """
        self.input_data_path = input_data_path
        self.output_dir = output_dir
        self.window_size = window_size
        self.stride = stride

        self.documents = []
        self.metadata = {}

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "docs"), exist_ok=True)

        # Load spaCy model for sentence segmentation
        logger.info("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm",
                              disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
        self.nlp.enable_pipe("senter")
        logger.info("spaCy model loaded successfully")

    def load_data(self) -> None:
        """Load the scraped data"""
        logger.info(f"Loading data from {self.input_data_path}")
        try:
            with open(self.input_data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            logger.info(f"Loaded {len(self.raw_data)} documents")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def segment_into_passages(self,
                              content: str,
                              title: str,
                              url: str) -> List[Dict]:
        """
        Segment content into passages using sliding window.

        Args:
            content: The text content to segment
            title: The document title
            url: The document URL

        Returns:
            List of passage dictionaries
        """
        # Process the content with spaCy
        doc = self.nlp(content)
        sentences = list(doc.sents)
        passages = []

        # Handle short documents
        if len(sentences) <= self.window_size:
            passage_text = " ".join([sent.text for sent in sentences])
            passage_id = create_unique_id(passage_text)

            passages.append({
                "id": passage_id,
                "text": passage_text.strip(),
                "title": title,
                "url": url,
                "sentence_start_idx": 0,
                "sentence_end_idx": len(sentences) - 1,
                "num_sentences": len(sentences),
                "word_count": len(passage_text.split())
            })
            return passages

        # Process longer documents with sliding window
        for i in range(0, len(sentences), self.stride):
            window = sentences[i:i + self.window_size]
            if not window:
                break

            passage_text = " ".join([sent.text for sent in window])
            passage_id = create_unique_id(passage_text)

            passages.append({
                "id": passage_id,
                "text": passage_text.strip(),
                "title": title,
                "url": url,
                "sentence_start_idx": i,
                "sentence_end_idx": i + len(window) - 1,
                "num_sentences": len(window),
                "word_count": len(passage_text.split())
            })

        return passages

    def process_documents(self) -> None:
        """Process all documents into a suitable format for IR"""
        logger.info("Processing documents...")

        all_passages = []
        doc_count = 0

        for doc in tqdm(self.raw_data, desc="Processing documents"):
            # Skip documents with very little content
            if len(doc.get('content', '')) < 100:
                continue

            # Clean the content
            clean_content = clean_text(doc['content'])

            # Segment into passages
            passages = self.segment_into_passages(
                clean_content,
                doc.get('title', 'No Title'),
                doc.get('url', '')
            )

            # Add to all passages
            all_passages.extend(passages)

            # Create document-level metadata
            doc_id = f"DOC-{doc_count}"
            self.metadata[doc_id] = {
                "title": doc.get('title', 'No Title'),
                "url": doc.get('url', ''),
                "meta_description": doc.get('meta_description', ''),
                "num_passages": len(passages),
                "passage_ids": [p['id'] for p in passages]
            }

            doc_count += 1

        self.documents = all_passages
        logger.info(f"Created {len(all_passages)} passages from {doc_count} documents")

    def save_corpus(self) -> None:
        """Save the processed corpus in multiple formats"""
        # Save individual passage files
        docs_dir = os.path.join(self.output_dir, "docs")
        logger.info(f"Saving {len(self.documents)} passages to {docs_dir}")

        for passage in tqdm(self.documents, desc="Saving passages"):
            filename = f"{passage['id']}.txt"
            with open(os.path.join(docs_dir, filename), 'w', encoding='utf-8') as f:
                f.write(passage['text'])

        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

        # Save corpus info
        corpus_info = {
            "total_documents": len(self.metadata),
            "total_passages": len(self.documents),
            "average_passage_length": sum(p['word_count'] for p in self.documents) / len(
                self.documents) if self.documents else 0,
            "average_sentences_per_passage": sum(p['num_sentences'] for p in self.documents) / len(
                self.documents) if self.documents else 0,
            "window_size": self.window_size,
            "stride": self.stride
        }

        info_path = os.path.join(self.output_dir, "corpus_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_info, f, indent=2)

        # Save all passages as CSV
        df = pd.DataFrame(self.documents)
        csv_path = os.path.join(self.output_dir, "passages.csv")
        df.to_csv(csv_path, index=False)

        # Save in TREC format
        self._save_trec_format()

        logger.info(f"Corpus saved to {self.output_dir}")

    def _save_trec_format(self) -> None:
        """Save corpus in TREC format"""
        trec_path = os.path.join(self.output_dir, "trec_corpus.txt")

        with open(trec_path, 'w', encoding='utf-8') as f:
            for passage in self.documents:
                f.write("<DOC>\n")
                f.write(f"<DOCNO>{passage['id']}</DOCNO>\n")
                f.write(f"<TITLE>{passage['title']}</TITLE>\n")
                f.write(f"<URL>{passage['url']}</URL>\n")
                f.write(f"<TEXT>\n{passage['text']}\n</TEXT>\n")
                f.write("</DOC>\n\n")

    def create_templates(self) -> None:
        """Create templates for NIST-style qrels and topics"""
        # Create qrels template
        qrels_path = os.path.join(self.output_dir, "qrels_template.txt")
        with open(qrels_path, 'w', encoding='utf-8') as f:
            f.write("# Format: query_id 0 passage_id relevance_score\n")
            f.write("# Example: 1 0 md5hash 2\n")
            f.write("# Relevance: 0=Not relevant, 1=Partially relevant, 2=Highly relevant\n")

        # Create topics template
        topics_path = os.path.join(self.output_dir, "topics_template.xml")
        with open(topics_path, 'w', encoding='utf-8') as f:
            f.write("<topics>\n")
            f.write("  <topic number=\"1\">\n")
            f.write("    <title>Example topic title</title>\n")
            f.write("    <description>Brief description of information need</description>\n")
            f.write("    <narrative>Detailed explanation of relevance criteria</narrative>\n")
            f.write("  </topic>\n")
            f.write("</topics>\n")

    def build(self) -> None:
        """Main method to build the corpus"""
        try:
            self.load_data()
            self.process_documents()
            self.save_corpus()
            self.create_templates()
            logger.info("Corpus building completed successfully!")
        except Exception as e:
            logger.error(f"Error building corpus: {e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build an Academic IR Corpus')
    parser.add_argument('--input', required=True, help='Path to scraped_data.json')
    parser.add_argument('--output', default='academic_corpus', help='Output directory')
    parser.add_argument('--window-size', type=int, default=10, help='Sentences per window')
    parser.add_argument('--stride', type=int, default=5, help='Sentences to stride')

    args = parser.parse_args()

    builder = CorpusBuilder(
        args.input,
        args.output,
        args.window_size,
        args.stride
    )
    builder.build()


if __name__ == '__main__':
    main()
