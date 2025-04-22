"""
Topic Integrator
--------------
Integrate topics with corpus, validate coverage,
and create automatic relevance judgments.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.helpers import clean_text, setup_logging

logger = logging.getLogger(__name__)


class TopicIntegrator:
    def __init__(self,
                 corpus_dir: str,
                 topics_file: str):
        """
        Initialize topic integrator.

        Args:
            corpus_dir: Path to corpus directory
            topics_file: Path to topics file
        """
        self.corpus_dir = corpus_dir
        self.topics_file = topics_file
        self.topics = {}

        # Load corpus data
        self.load_corpus()

        # Load topics
        self.load_topics()

        logger.info(f"Loaded {len(self.topics)} topics")

    def load_corpus(self) -> None:
        """Load corpus data"""
        passages_path = os.path.join(self.corpus_dir, "passages.csv")
        if not os.path.exists(passages_path):
            raise FileNotFoundError(f"Passages file not found: {passages_path}")

        self.passages = pd.read_csv(passages_path)
        logger.info(f"Loaded {len(self.passages)} passages")

    def load_topics(self) -> None:
        """Load topics from file"""
        ext = os.path.splitext(self.topics_file)[1].lower()

        try:
            if ext == '.xml':
                self._load_topics_xml()
            elif ext == '.json':
                self._load_topics_json()
            else:
                raise ValueError(f"Unsupported topic format: {ext}")
        except Exception as e:
            logger.error(f"Error loading topics: {e}")
            raise

    def _load_topics_xml(self) -> None:
        """Load topics from XML file"""
        tree = ET.parse(self.topics_file)
        root = tree.getroot()

        for topic in root.findall('topic'):
            topic_num = topic.get('number')
            title = topic.find('title')
            desc = topic.find('description')
            narr = topic.find('narrative')

            self.topics[topic_num] = {
                'title': title.text.strip() if title is not None else "",
                'description': desc.text.strip() if desc is not None else "",
                'narrative': narr.text.strip() if narr is not None else ""
            }

    def _load_topics_json(self) -> None:
        """Load topics from JSON file"""
        with open(self.topics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            topics_list = data
        elif 'topics' in data:
            topics_list = data['topics']
        else:
            topics_list = [data]

        for topic in topics_list:
            topic_id = str(topic.get('id', topic.get('number', '')))
            self.topics[topic_id] = {
                'title': topic.get('title', ''),
                'description': topic.get('description', ''),
                'narrative': topic.get('narrative', '')
            }

    def validate_topics(self) -> Dict:
        """Validate topics against corpus content"""
        logger.info("Validating topics...")

        results = {}
        for topic_id, topic in self.topics.items():
            # Create topic terms
            text = f"{topic['title']} {topic['description']}"
            terms = set(re.findall(r'\w+', text.lower()))

            # Count matching documents
            matches = 0
            sample_matches = []

            for _, passage in self.passages.iterrows():
                passage_terms = set(re.findall(r'\w+',
                                               passage['text'].lower()))
                if len(terms & passage_terms) >= 2:  # At least 2 matching terms
                    matches += 1
                    if len(sample_matches) < 5:
                        sample_matches.append(passage['id'])

            results[topic_id] = {
                'matching_passages': matches,
                'coverage': (matches / len(self.passages)) * 100,
                'sample_matches': sample_matches
            }

        # Save results
        output_path = os.path.join(self.corpus_dir, "topic_validation.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Validation results saved to {output_path}")
        return results

    def create_automatic_qrels(self,
                               method: str = 'hybrid',
                               output_path: Optional[str] = None) -> str:
        """
        Create automatic relevance judgments.

        Args:
            method: Method to use ('bm25', 'tfidf', 'semantic', 'hybrid')
            output_path: Path to save qrels file

        Returns:
            Path to created qrels file
        """
        if output_path is None:
            output_path = os.path.join(self.corpus_dir,
                                       f"auto_qrels_{method}.txt")

        logger.info(f"Creating automatic qrels using {method} method")

        qrels = []

        if method == 'hybrid':
            # Combine multiple methods
            bm25_qrels = self._create_bm25_qrels()
            tfidf_qrels = self._create_tfidf_qrels()
            semantic_qrels = self._create_semantic_qrels()

            # Combine judgments (taking maximum relevance score)
            combined = {}
            for qrels_list in [bm25_qrels, tfidf_qrels, semantic_qrels]:
                for qrel in qrels_list:
                    key = (qrel[0], qrel[2])  # (topic_id, doc_id)
                    rel = int(qrel[3])
                    if key not in combined or rel > combined[key]:
                        combined[key] = rel

            # Convert back to qrels format
            qrels = [
                f"{topic_id} 0 {doc_id} {rel}"
                for (topic_id, doc_id), rel in combined.items()
            ]

        else:
            if method == 'bm25':
                qrels = self._create_bm25_qrels()
            elif method == 'tfidf':
                qrels = self._create_tfidf_qrels()
            elif method == 'semantic':
                qrels = self._create_semantic_qrels()
            else:
                raise ValueError(f"Unknown method: {method}")

        # Write qrels file
        with open(output_path, 'w', encoding='utf-8') as f:
            for qrel in qrels:
                f.write(f"{qrel}\n")

        logger.info(f"Created {len(qrels)} qrels in {output_path}")
        return output_path

    def _create_bm25_qrels(self) -> List[str]:
        """Create qrels using BM25"""
        qrels = []

        # Prepare corpus
        corpus = self.passages['text'].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        for topic_id, topic in tqdm(self.topics.items(),
                                    desc="Creating BM25 qrels"):
            query = f"{topic['title']} {topic['description']}"
            tokenized_query = query.split()

            # Get BM25 scores
            scores = bm25.get_scores(tokenized_query)

            # Create qrels for top matches
            for i, score in enumerate(scores):
                if score > 0:
                    rel_score = 2 if score > 10 else 1
                    qrels.append(f"{topic_id} 0 {self.passages.iloc[i]['id']} {rel_score}")

        return qrels

    def _create_tfidf_qrels(self) -> List[str]:
        """Create qrels using TF-IDF"""
        qrels = []

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        corpus_vectors = vectorizer.fit_transform(self.passages['text'])

        for topic_id, topic in tqdm(self.topics.items(),
                                    desc="Creating TF-IDF qrels"):
            query = f"{topic['title']} {topic['description']}"
            query_vector = vectorizer.transform([query])

            # Calculate similarities
            similarities = corpus_vectors.dot(query_vector.T).toarray().flatten()

            # Create qrels for top matches
            for i, sim in enumerate(similarities):
                if sim > 0.1:
                    rel_score = 2 if sim > 0.3 else 1
                    qrels.append(f"{topic_id} 0 {self.passages.iloc[i]['id']} {rel_score}")

        return qrels

    def _create_semantic_qrels(self) -> List[str]:
        """Create qrels using semantic similarity"""
        qrels = []

        # Load model
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')

        # Encode passages
        passage_embeddings = model.encode(
            self.passages['text'].tolist(),
            show_progress_bar=True,
            convert_to_tensor=True
        )

        for topic_id, topic in tqdm(self.topics.items(),
                                    desc="Creating semantic qrels"):
            query = f"{topic['title']} {topic['description']}"
            query_embedding = model.encode([query], convert_to_tensor=True)

            # Calculate similarities
            similarities = util.pytorch_cos_sim(
                query_embedding,
                passage_embeddings
            )[0].cpu().numpy()

            # Create qrels for top matches
            for i, sim in enumerate(similarities):
                if sim > 0.4:
                    rel_score = 2 if sim > 0.6 else 1
                    qrels.append(f"{topic_id} 0 {self.passages.iloc[i]['id']} {rel_score}")

        return qrels

    def generate_statistics(self) -> Dict:
        """Generate statistics about topics and coverage"""
        # Validate topics first
        validation = self.validate_topics()

        # Calculate statistics
        stats = {
            'total_topics': len(self.topics),
            'topics_with_matches': sum(1 for res in validation.values()
                                       if res['matching_passages'] > 0),
            'average_coverage': np.mean([res['coverage']
                                         for res in validation.values()]),
            'topics_by_coverage': sorted(
                [(tid, res['coverage'])
                 for tid, res in validation.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }

        # Save statistics
        output_path = os.path.join(self.corpus_dir, "topic_stats.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to {output_path}")
        return stats

    def integrate(self) -> None:
        """Run complete topic integration pipeline"""
        try:
            # 1. Validate topics
            self.validate_topics()

            # 2. Generate statistics
            self.generate_statistics()

            # 3. Create automatic qrels
            for method in ['bm25', 'tfidf', 'semantic', 'hybrid']:
                self.create_automatic_qrels(method)

            # 4. Copy topics file to corpus directory
            import shutil
            dest = os.path.join(self.corpus_dir, "topics.xml")
            shutil.copy(self.topics_file, dest)

            logger.info("Topic integration completed successfully")

        except Exception as e:
            logger.error(f"Error during topic integration: {e}")
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Integrate topics with corpus')
    parser.add_argument('--corpus', required=True,
                        help='Corpus directory')
    parser.add_argument('--topics', required=True,
                        help='Topics file (XML or JSON)')

    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Run integration
    integrator = TopicIntegrator(args.corpus, args.topics)
    integrator.integrate()


if __name__ == '__main__':
    main()
