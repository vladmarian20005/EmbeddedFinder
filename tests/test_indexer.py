"""Tests for the indexing pipeline."""

from unittest.mock import MagicMock
import pytest

from embedded_finder.indexer import Indexer, IndexStats
from embedded_finder.store import VectorStore


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_batch.return_value = [[0.1] * 10]
    return embedder


@pytest.fixture
def store(tmp_dir):
    return VectorStore(persist_dir=str(tmp_dir / "db"))


@pytest.fixture
def indexer(mock_embedder, store):
    return Indexer(embedder=mock_embedder, store=store)


def test_index_directory(indexer, sample_files, mock_embedder):
    # Make embed_batch return one embedding per input text
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 10] * len(texts)

    stats = indexer.index_directory(sample_files)
    assert stats.total_files > 0
    assert stats.indexed > 0
    assert stats.errors == 0


def test_index_skips_already_indexed(indexer, sample_files, mock_embedder):
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 10] * len(texts)

    stats1 = indexer.index_directory(sample_files)
    assert stats1.indexed > 0

    # Second run should skip all
    stats2 = indexer.index_directory(sample_files)
    assert stats2.skipped == stats2.total_files
    assert stats2.indexed == 0


def test_index_tracks_errors(indexer, sample_files, mock_embedder):
    mock_embedder.embed_batch.side_effect = Exception("API error")

    stats = indexer.index_directory(sample_files)
    assert stats.errors > 0
    assert len(stats.error_files) > 0


def test_index_single_file(indexer, sample_files, mock_embedder):
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 10] * len(texts)

    txt_file = sample_files / "hello.txt"
    stats = indexer.index_file(txt_file)
    assert stats.indexed == 1
    assert stats.total_files == 1


def test_index_stats_defaults():
    stats = IndexStats()
    assert stats.total_files == 0
    assert stats.indexed == 0
    assert stats.skipped == 0
    assert stats.errors == 0
    assert stats.chunks_created == 0
    assert stats.error_files == []


def test_on_progress_callback(indexer, sample_files, mock_embedder):
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 10] * len(texts)
    progress_calls = []

    def on_progress(file_info, stats):
        progress_calls.append(file_info.name)

    indexer.index_directory(sample_files, on_progress=on_progress)
    assert len(progress_calls) > 0
