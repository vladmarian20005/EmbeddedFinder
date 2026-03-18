"""Tests for ChromaDB vector store."""

import pytest

from embedded_finder.store import VectorStore


@pytest.fixture
def store(tmp_dir):
    """Create a VectorStore in a temp directory."""
    return VectorStore(persist_dir=str(tmp_dir / "db"))


def test_add_and_count(store):
    store.add_documents(
        ids=["doc1", "doc2"],
        texts=["hello world", "foo bar"],
        embeddings=[[0.1] * 10, [0.2] * 10],
        metadatas=[{"file_path": "/a.txt"}, {"file_path": "/b.txt"}],
    )
    assert store.count() == 2


def test_query_returns_results(store):
    store.add_documents(
        ids=["doc1"],
        texts=["hello world"],
        embeddings=[[0.1] * 10],
        metadatas=[{"file_path": "/a.txt", "content_hash": "abc"}],
    )
    results = store.query(embedding=[0.1] * 10, n_results=5)
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "doc1"


def test_query_empty_store(store):
    results = store.query(embedding=[0.1] * 10)
    assert results["ids"] == [[]]


def test_delete_document(store):
    store.add_documents(
        ids=["doc1", "doc2"],
        texts=["hello", "world"],
        embeddings=[[0.1] * 10, [0.2] * 10],
    )
    store.delete_document("doc1")
    assert store.count() == 1


def test_delete_by_file_path(store):
    store.add_documents(
        ids=["f1_c0", "f1_c1", "f2_c0"],
        texts=["chunk1", "chunk2", "other"],
        embeddings=[[0.1] * 10, [0.2] * 10, [0.3] * 10],
        metadatas=[
            {"file_path": "/a.txt"},
            {"file_path": "/a.txt"},
            {"file_path": "/b.txt"},
        ],
    )
    store.delete_by_file_path("/a.txt")
    assert store.count() == 1


def test_get_indexed_files(store):
    store.add_documents(
        ids=["doc1", "doc2"],
        texts=["hello", "world"],
        embeddings=[[0.1] * 10, [0.2] * 10],
        metadatas=[
            {"file_path": "/a.txt", "content_hash": "hash1"},
            {"file_path": "/b.txt", "content_hash": "hash2"},
        ],
    )
    indexed = store.get_indexed_files()
    assert indexed["/a.txt"] == "hash1"
    assert indexed["/b.txt"] == "hash2"


def test_clear(store):
    store.add_documents(
        ids=["doc1"],
        texts=["hello"],
        embeddings=[[0.1] * 10],
    )
    assert store.count() == 1
    store.clear()
    assert store.count() == 0


def test_upsert_overwrites(store):
    store.add_documents(
        ids=["doc1"],
        texts=["version 1"],
        embeddings=[[0.1] * 10],
        metadatas=[{"file_path": "/a.txt", "content_hash": "v1"}],
    )
    store.add_documents(
        ids=["doc1"],
        texts=["version 2"],
        embeddings=[[0.2] * 10],
        metadatas=[{"file_path": "/a.txt", "content_hash": "v2"}],
    )
    assert store.count() == 1
    indexed = store.get_indexed_files()
    assert indexed["/a.txt"] == "v2"
