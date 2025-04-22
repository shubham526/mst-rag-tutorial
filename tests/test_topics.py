"""
Test topic handling functionality
"""

import pytest
from src.topics.converter import convert_to_nist_format
from src.topics.integrator import TopicIntegrator


def test_topic_conversion(test_data_dir):
    """Test topic format conversion"""
    # Test JSON to NIST conversion
    topics = {
        "topics": [
            {
                "id": "1",
                "title": "Test Topic",
                "description": "Test Description",
                "narrative": "Test Narrative"
            }
        ]
    }

    input_file = test_data_dir / "test_topics.json"
    output_file = test_data_dir / "topics.xml"

    with open(input_file, "w") as f:
        import json
        json.dump(topics, f)

    num_topics = convert_to_nist_format(
        str(input_file),
        str(output_file),
        "json"
    )

    assert num_topics == 1
    assert output_file.exists()


def test_topic_integrator(sample_corpus, sample_topics):
    """Test topic integration"""
    integrator = TopicIntegrator(
        corpus_dir=str(sample_corpus),
        topics_file=str(sample_topics)
    )

    # Test topic validation
    validation = integrator.validate_topics()
    assert len(validation) > 0
    assert all('matching_passages' in v for v in validation.values())

    # Test automatic qrels creation
    for method in ['bm25', 'tfidf', 'semantic', 'hybrid']:
        qrels_path = integrator.create_automatic_qrels(method=method)
        assert qrels_path is not None
        assert Path(qrels_path).exists()

    # Test statistics generation
    stats = integrator.generate_statistics()
    assert 'total_topics' in stats
    assert 'average_coverage' in stats
    assert 'topics_by_coverage' in stats

