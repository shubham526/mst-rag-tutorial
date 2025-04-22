"""
Test indexing functionality
"""

import pytest
from pathlib import Path
from src.indexing.create_index import IndexBuilder
from src.indexing.prebuilt_indexes import create_prebuilt_index


def test_index_builder_initialization(sample_corpus):
    """Test IndexBuilder initialization"""
    builder = IndexBuilder(
        corpus_dir=str(sample_corpus),
        output_dir=str(sample_corpus / "index")
    )

    assert Path(builder.jsonl_dir).exists()
    assert hasattr(builder, 'build_sparse_index')
    assert hasattr(builder, 'build_dense_index')


def test_jsonl_conversion(sample_corpus):
    """Test conversion to JSONL format"""
    builder = IndexBuilder(
        corpus_dir=str(sample_corpus),
        output_dir=str(sample_corpus / "index")
    )

    count = builder.convert_to_jsonl()
    assert count > 0
    assert (Path(builder.jsonl_dir) / "passages.jsonl").exists()


@pytest.mark.slow
def test_sparse_index_building(sample_corpus):
    """Test building sparse BM25 index"""
    builder = IndexBuilder(
        corpus_dir=str(sample_corpus),
        output_dir=str(sample_corpus / "index")
    )

    index_path = builder.build_sparse_index(
        stemming=True,
        stopwords=True,
        threads=1
    )

    assert index_path is not None
    assert Path(index_path).exists()
    assert builder.verify_index(index_path) is not None


@pytest.mark.slow
def test_dense_index_building(sample_corpus):
    """Test building dense index"""
    builder = IndexBuilder(
        corpus_dir=str(sample_corpus),
        output_dir=str(sample_corpus / "index")
    )

    index_path = builder.build_dense_index(
        encoder_name="sentence-transformers/msmarco-distilbert-base-v3",
        batch_size=2,
        device="cpu"
    )

    assert index_path is not None
    assert Path(index_path).exists()

