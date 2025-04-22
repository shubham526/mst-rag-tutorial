"""
Test corpus building functionality
"""

import pytest
import pandas as pd
from pathlib import Path
from src.corpus.builder import CorpusBuilder
from src.corpus.multi_builder import MultiCorpusBuilder


def test_corpus_builder_initialization(test_data_dir):
    """Test CorpusBuilder initialization"""
    builder = CorpusBuilder(
        input_data_path=str(test_data_dir / "sample.json"),
        output_dir=str(test_data_dir / "output")
    )

    assert builder.window_size == 10
    assert builder.stride == 5
    assert Path(builder.output_dir).exists()


def test_segment_into_passages(test_data_dir):
    """Test passage segmentation"""
    builder = CorpusBuilder(
        input_data_path=str(test_data_dir / "sample.json"),
        output_dir=str(test_data_dir / "output")
    )

    content = "This is sentence one. This is sentence two. " * 15
    passages = builder.segment_into_passages(
        content=content,
        title="Test Document",
        url="http://example.com"
    )

    assert len(passages) > 0
    assert all(isinstance(p, dict) for p in passages)
    assert all(len(p['text'].split()) > 0 for p in passages)


def test_multi_corpus_builder(test_data_dir):
    """Test MultiCorpusBuilder"""
    builder = MultiCorpusBuilder(
        input_dir=str(test_data_dir),
        output_dir=str(test_data_dir / "output")
    )

    assert hasattr(builder, 'find_input_files')
    assert callable(builder.find_input_files)


def test_corpus_saving(test_data_dir, sample_corpus):
    """Test corpus saving functionality"""
    output_dir = test_data_dir / "test_output"
    builder = CorpusBuilder(
        input_data_path=str(sample_corpus / "sample.json"),
        output_dir=str(output_dir)
    )

    # Create some test documents
    builder.documents = [
        {
            "id": "doc1",
            "text": "Sample text",
            "title": "Sample Title",
            "url": "http://example.com",
            "word_count": 2,
            "num_sentences": 1
        }
    ]

    builder.save_corpus()

    assert (output_dir / "passages.csv").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "docs").is_dir()

