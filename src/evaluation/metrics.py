"""
IR Evaluation Metrics
-------------------
Implementation of standard IR evaluation metrics.
"""

from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict


def compute_metrics(qrels: Dict[str, int],
                    run: Dict[str, float],
                    metrics: List[str]) -> Dict[str, float]:
    """
    Compute IR evaluation metrics.

    Args:
        qrels: Dictionary of document IDs to relevance scores
        run: Dictionary of document IDs to retrieval scores
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric names to values
    """
    results = {}

    # Sort documents by score
    ranked_docs = sorted(run.items(), key=lambda x: x[1], reverse=True)

    for metric in metrics:
        if metric.startswith('P_'):  # Precision at k
            k = int(metric.split('_')[1])
            results[metric] = precision_at_k(qrels, ranked_docs, k)

        elif metric.startswith('recall_'):  # Recall at k
            k = int(metric.split('_')[1])
            results[metric] = recall_at_k(qrels, ranked_docs, k)

        elif metric.startswith('ndcg_cut_'):  # NDCG at k
            k = int(metric.split('_')[2])
            results[metric] = ndcg_at_k(qrels, ranked_docs, k)

        elif metric == 'map':  # Mean Average Precision
            results[metric] = mean_average_precision(qrels, ranked_docs)

        elif metric == 'mrr':  # Mean Reciprocal Rank
            results[metric] = mean_reciprocal_rank(qrels, ranked_docs)

    return results


def precision_at_k(qrels: Dict[str, int],
                   ranked_docs: List[tuple],
                   k: int) -> float:
    """Calculate precision at k"""
    if not ranked_docs or k <= 0:
        return 0.0

    relevant = 0
    for i, (doc_id, _) in enumerate(ranked_docs[:k]):
        if doc_id in qrels and qrels[doc_id] > 0:
            relevant += 1

    return relevant / k


def recall_at_k(qrels: Dict[str, int],
                ranked_docs: List[tuple],
                k: int) -> float:
    """Calculate recall at k"""
    if not ranked_docs or k <= 0:
        return 0.0

    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0

    relevant = 0
    for doc_id, _ in ranked_docs[:k]:
        if doc_id in qrels and qrels[doc_id] > 0:
            relevant += 1

    return relevant / total_relevant


def ndcg_at_k(qrels: Dict[str, int],
              ranked_docs: List[tuple],
              k: int) -> float:
    """Calculate NDCG at k"""
    if not ranked_docs or k <= 0:
        return 0.0

    dcg = 0.0
    idcg = 0.0

    # Calculate DCG
    for i, (doc_id, _) in enumerate(ranked_docs[:k]):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / np.log2(i + 2)

    # Calculate IDCG
    ideal_rels = sorted([rel for rel in qrels.values() if rel > 0], reverse=True)
    for i, rel in enumerate(ideal_rels[:k]):
        idcg += (2 ** rel - 1) / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def mean_average_precision(qrels: Dict[str, int],
                           ranked_docs: List[tuple]) -> float:
    """Calculate Mean Average Precision"""
    if not ranked_docs:
        return 0.0

    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0

    relevant_so_far = 0
    sum_precision = 0.0

    for i, (doc_id, _) in enumerate(ranked_docs):
        if doc_id in qrels and qrels[doc_id] > 0:
            relevant_so_far += 1
            sum_precision += relevant_so_far / (i + 1)

    return sum_precision / total_relevant


def mean_reciprocal_rank(qrels: Dict[str, int],
                         ranked_docs: List[tuple]) -> float:
    """Calculate Mean Reciprocal Rank"""
    if not ranked_docs:
        return 0.0

    for i, (doc_id, _) in enumerate(ranked_docs):
        if doc_id in qrels and qrels[doc_id] > 0:
            return 1.0 / (i + 1)

    return 0.0
