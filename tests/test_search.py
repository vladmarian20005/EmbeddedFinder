"""Tests for semantic search engine."""

from unittest.mock import MagicMock
import pytest

from embedded_finder.search import SearchEngine, SearchResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 10
    return embedder


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "documents": [["Hello world content", "Python code here"]],
        "metadatas": [[
            {"file_path": "/a.txt", "file_name": "a.txt", "file_extension": ".txt",
             "file_size": 100, "chunk_index": 0, "total_chunks": 1},
            {"file_path": "/b.py", "file_name": "b.py", "file_extension": ".py",
             "file_size": 200, "chunk_index": 0, "total_chunks": 1},
        ]],
        "distances": [[0.2, 0.5]],
    }
    return store


@pytest.fixture
def engine(mock_embedder, mock_store):
    return SearchEngine(mock_embedder, mock_store)


def test_search_returns_results(engine):
    results = engine.search("hello")
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_results_sorted_by_score(engine):
    results = engine.search("hello")
    assert results[0].score >= results[1].score


def test_search_results_have_correct_fields(engine):
    results = engine.search("hello")
    r = results[0]
    assert r.file_path == "/a.txt"
    assert r.file_name == "a.txt"
    assert r.file_extension == ".txt"
    assert r.file_size == 100
    assert r.snippet == "Hello world content"
    assert 0 <= r.score <= 1


def test_search_empty_query(engine):
    results = engine.search("")
    assert results == []


def test_search_whitespace_query(engine):
    results = engine.search("   ")
    assert results == []


def test_search_min_score_filter(engine):
    # With distance 0.2, score = 1 - 0.1 = 0.9
    # With distance 0.5, score = 1 - 0.25 = 0.75
    results = engine.search("hello", min_score=0.8)
    assert len(results) == 1
    assert results[0].file_path == "/a.txt"


def test_search_n_results_controls_retrieval_window(engine):
    """search() returns all deduped candidates; callers truncate after ranking."""
    results = engine.search("hello", n_results=1)
    # n_results controls the ChromaDB retrieval window (n_results * 5),
    # but all deduped results are returned for the ranker to re-rank.
    assert len(results) >= 1


def test_search_deduplicates_files(mock_embedder):
    store = MagicMock()
    store.query.return_value = {
        "ids": [["f1_c0", "f1_c1", "f2_c0"]],
        "documents": [["chunk 1", "chunk 2", "other file"]],
        "metadatas": [[
            {"file_path": "/a.txt", "file_name": "a.txt", "file_extension": ".txt",
             "file_size": 100, "chunk_index": 0, "total_chunks": 2},
            {"file_path": "/a.txt", "file_name": "a.txt", "file_extension": ".txt",
             "file_size": 100, "chunk_index": 1, "total_chunks": 2},
            {"file_path": "/b.py", "file_name": "b.py", "file_extension": ".py",
             "file_size": 200, "chunk_index": 0, "total_chunks": 1},
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    engine = SearchEngine(mock_embedder, store)
    results = engine.search("hello")
    file_paths = [r.file_path for r in results]
    assert file_paths == ["/a.txt", "/b.py"]


def test_search_empty_store(mock_embedder):
    store = MagicMock()
    store.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    engine = SearchEngine(mock_embedder, store)
    results = engine.search("hello")
    assert results == []


def test_search_score_calculation(engine):
    # distance=0.2 -> score = 1 - 0.1 = 0.9
    results = engine.search("hello")
    assert results[0].score == 0.9
