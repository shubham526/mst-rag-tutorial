"""
IR Evaluation Tools
------------------
Tools for evaluating IR systems including metrics calculation,
assessment tools, and visualization.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import xml.etree.ElementTree as ET
from tqdm import tqdm

from .metrics import compute_metrics
from ..utils.helpers import setup_logging

logger = logging.getLogger(__name__)


class IRStats:
    """Calculate and display IR statistics for the corpus"""

    def __init__(self, corpus_dir: str):
        """
        Initialize IR statistics calculator.

        Args:
            corpus_dir: Path to the corpus directory
        """
        self.corpus_dir = corpus_dir
        self.passages_path = os.path.join(corpus_dir, "passages.csv")
        self.metadata_path = os.path.join(corpus_dir, "metadata.json")
        self.corpus_info_path = os.path.join(corpus_dir, "corpus_info.json")

        # Load corpus data
        self.load_corpus_data()

    def load_corpus_data(self) -> None:
        """Load all corpus data files"""
        try:
            self.passages = pd.read_csv(self.passages_path)

            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            if os.path.exists(self.corpus_info_path):
                with open(self.corpus_info_path, 'r', encoding='utf-8') as f:
                    self.corpus_info = json.load(f)
            else:
                self.corpus_info = {}

            logger.info(f"Loaded corpus with {len(self.passages)} passages")
        except Exception as e:
            logger.error(f"Error loading corpus data: {e}")
            raise

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive corpus statistics"""
        stats = {
            "document_stats": self.document_stats(),
            "passage_stats": self.passage_stats(),
            "vocabulary_stats": self.vocabulary_stats(),
            "domain_stats": self.domain_stats()
        }
        return stats

    def document_stats(self) -> Dict:
        """Calculate document-level statistics"""
        return {
            "total_documents": len(self.metadata),
            "avg_passages_per_doc": len(self.passages) / len(self.metadata),
            "docs_with_metadata": sum(1 for doc in self.metadata.values()
                                      if doc.get('meta_description', '')),
        }

    def passage_stats(self) -> Dict:
        """Calculate passage-level statistics"""
        word_counts = self.passages['word_count'].tolist()
        sentence_counts = self.passages['num_sentences'].tolist()

        return {
            "total_passages": len(self.passages),
            "avg_passage_length": np.mean(word_counts),
            "std_passage_length": np.std(word_counts),
            "min_passage_length": min(word_counts),
            "max_passage_length": max(word_counts),
            "avg_sentences": np.mean(sentence_counts),
            "std_sentences": np.std(sentence_counts),
        }

    def vocabulary_stats(self) -> Dict:
        """Calculate vocabulary statistics"""
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit_transform(self.passages['text'])

        return {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "top_terms": sorted(
                [(term, count)
                 for term, count in vectorizer.vocabulary_.items()],
                key=lambda x: x[1],
                reverse=True
            )[:100]
        }

    def domain_stats(self) -> Dict:
        """Calculate domain-level statistics"""
        domains = defaultdict(int)
        for doc in self.metadata.values():
            url = doc.get('url', '')
            if url:
                domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                domains[domain] += 1

        return {
            "total_domains": len(domains),
            "domain_distribution": dict(domains)
        }

    def plot_statistics(self, output_dir: Optional[str] = None) -> None:
        """Generate visualization plots for statistics"""
        if output_dir is None:
            output_dir = os.path.join(self.corpus_dir, "stats")
        os.makedirs(output_dir, exist_ok=True)

        # Plot passage length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.passages['word_count'], bins=50, alpha=0.75)
        plt.title('Passage Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'passage_lengths.png'))
        plt.close()

        # Plot sentences per passage
        plt.figure(figsize=(10, 6))
        plt.hist(self.passages['num_sentences'], bins=30, alpha=0.75)
        plt.title('Sentences per Passage')
        plt.xlabel('Number of Sentences')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'sentences_per_passage.png'))
        plt.close()

        logger.info(f"Statistics plots saved to {output_dir}")


class TRECEvaluator:
    """Evaluate IR runs using TREC-style qrels"""

    def __init__(self, qrels_path: str):
        """
        Initialize TREC evaluator.

        Args:
            qrels_path: Path to qrels file
        """
        self.qrels_path = qrels_path
        self.qrels = self.load_qrels()

    def load_qrels(self) -> Dict:
        """Load relevance judgments"""
        qrels = defaultdict(dict)

        with open(self.qrels_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                parts = line.strip().split()
                if len(parts) == 4:
                    qid, _, docid, rel = parts
                    qrels[qid][docid] = int(rel)

        return qrels

    def evaluate_run(self,
                     run_file: str,
                     metrics: Optional[List[str]] = None) -> Dict:
        """
        Evaluate a run file.

        Args:
            run_file: Path to run file
            metrics: List of metrics to compute

        Returns:
            Dictionary of evaluation results
        """
        if metrics is None:
            metrics = ['map', 'ndcg_cut_10', 'P_10', 'recall_1000']

        # Load run
        run = self.load_run(run_file)

        # Calculate metrics
        results = {}
        for qid in self.qrels:
            if qid in run:
                query_results = compute_metrics(
                    self.qrels[qid],
                    run[qid],
                    metrics
                )
                results[qid] = query_results

        # Calculate averages
        averages = {}
        for metric in metrics:
            values = [res[metric] for res in results.values() if metric in res]
            if values:
                averages[metric] = np.mean(values)

        return {
            'query_results': results,
            'averages': averages
        }

    def load_run(self, run_file: str) -> Dict:
        """Load run file"""
        run = defaultdict(dict)

        with open(run_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid, _, docid, rank, score, _ = parts[:6]
                    run[qid][docid] = float(score)

        return run


class IRVisualizer:
    """Visualization tools for IR evaluation"""

    @staticmethod
    def plot_precision_recall_curve(results: Dict,
                                    output_path: str) -> None:
        """Plot precision-recall curves"""
        plt.figure(figsize=(10, 6))

        # Plot curve for each query
        for qid, qresults in results['query_results'].items():
            recalls = [qresults.get(f'recall_{k}', 0)
                       for k in [5, 10, 20, 50, 100, 200, 500, 1000]]
            precisions = [qresults.get(f'P_{k}', 0)
                          for k in [5, 10, 20, 50, 100, 200, 500, 1000]]

            plt.plot(recalls, precisions, marker='o', label=f'Query {qid}')

        plt.title('Precision-Recall Curves')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_metric_comparison(results: Dict,
                               output_path: str) -> None:
        """Plot comparison of different metrics"""
        metrics = ['P_10', 'recall_100', 'ndcg_cut_10', 'map']
        query_ids = sorted(results['query_results'].keys())

        plt.figure(figsize=(12, 6))
        x = np.arange(len(query_ids))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [results['query_results'][qid].get(metric, 0)
                      for qid in query_ids]
            plt.bar(x + i * width, values, width, label=metric)

        plt.xlabel('Query ID')
        plt.ylabel('Score')
        plt.title('IR Metrics Comparison')
        plt.xticks(x + width * 1.5, query_ids, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='IR Evaluation Tools')
    subparsers = parser.add_subparsers(dest='command')

    # Stats command
    stats_parser = subparsers.add_parser('stats')
    stats_parser.add_argument('--corpus', required=True,
                              help='Corpus directory')
    stats_parser.add_argument('--output', help='Output directory for plots')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--qrels', required=True,
                             help='Path to qrels file')
    eval_parser.add_argument('--run', required=True,
                             help='Path to run file')
    eval_parser.add_argument('--metrics', nargs='+',
                             help='Metrics to compute')
    eval_parser.add_argument('--output', required=True,
                             help='Output file for results')

    args = parser.parse_args()

    if args.command == 'stats':
        stats = IRStats(args.corpus)
        results = stats.calculate_statistics()
        stats.plot_statistics(args.output)

        if args.output:
            with open(os.path.join(args.output, 'stats.json'), 'w') as f:
                json.dump(results, f, indent=2)

    elif args.command == 'evaluate':
        evaluator = TRECEvaluator(args.qrels)
        results = evaluator.evaluate_run(args.run, args.metrics)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate plots
        output_dir = os.path.dirname(args.output)
        IRVisualizer.plot_precision_recall_curve(
            results,
            os.path.join(output_dir, 'precision_recall.png')
        )
        IRVisualizer.plot_metric_comparison(
            results,
            os.path.join(output_dir, 'metrics_comparison.png')
        )


if __name__ == '__main__':
    main()
