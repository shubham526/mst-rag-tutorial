"""
Test evaluation tools
"""

import pytest
from src.evaluation.ir_tools import IRStats, TRECEvaluator, IRVisualizer
from src.evaluation.metrics import compute_metrics


def test_ir_stats(sample_corpus):
    """Test IR statistics calculation"""
    stats = IRStats(str(sample_corpus))
    results = stats.calculate_statistics()

    assert "document_stats" in results
    assert "passage_stats" in results
    assert "vocabulary_stats" in results
    assert results["document_stats"]["total_documents"] > 0


def test_trec_evaluator(test_data_dir):
    """Test TREC evaluation"""
    # Create sample qrels
    qrels_content = """
    1 0 doc1 2
    1 0 doc2 1
    2 0 doc1 0
    2 0 doc3 1
    """
    qrels_file = test_data_dir / "qrels.txt"
    qrels_file.write_text(qrels_content)

    # Create sample run
    run_content = """
    1 Q0 doc1 1 0.9 test
    1 Q0 doc2 2 0.8 test
    2 Q0 doc3 1 0.7 test
    2 Q0 doc1 2 0.6 test
    """
    run_file = test_data_dir / "run.txt"
    run_file.write_text(run_content)

    evaluator = TRECEvaluator(str(qrels_file))
    results = evaluator.evaluate_run(
        str(run_file),
        metrics=['map', 'ndcg_cut_10', 'P_10']
    )

    assert 'query_results' in results
    assert 'averages' in results
    assert len(results['query_results']) > 0


def test_evaluation_metrics():
    """Test individual evaluation metrics"""
    qrels = {
        'doc1': 2,
        'doc2': 1,
        'doc3': 0
    }

    run = {
        'doc1': 0.9,
        'doc2': 0.8,
        'doc3': 0.7
    }

    metrics = ['P_5', 'recall_10', 'ndcg_cut_10']
    results = compute_metrics(qrels, run, metrics)

    assert all(metric in results for metric in metrics)
    assert all(0 <= results[metric] <= 1 for metric in results)


def test_ir_visualizer(test_data_dir):
    """Test visualization tools"""
    results = {
        'query_results': {
            '1': {
                'P_10': 0.8,
                'recall_100': 0.9,
                'ndcg_cut_10': 0.85,
                'map': 0.75
            },
            '2': {
                'P_10': 0.7,
                'recall_100': 0.8,
                'ndcg_cut_10': 0.75,
                'map': 0.65
            }
        }
    }

    # Test precision-recall curve
    output_path = test_data_dir / "pr_curve.png"
    IRVisualizer.plot_precision_recall_curve(results, str(output_path))
    assert output_path.exists()

    # Test metric comparison
    output_path = test_data_dir / "metrics.png"
    IRVisualizer.plot_metric_comparison(results, str(output_path))
    assert output_path.exists()

