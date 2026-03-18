"""Tests for Flask web interface."""

import json
from unittest.mock import MagicMock
import pytest

from embedded_finder.web.app import create_app
from embedded_finder.search import SearchResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 10
    return embedder


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.count.return_value = 5
    store.get_indexed_files.return_value = {"/a.txt": "h1", "/b.py": "h2"}
    store.query.return_value = {
        "ids": [["doc1"]],
        "documents": [["Hello world content"]],
        "metadatas": [[{
            "file_path": "/test/hello.py",
            "file_name": "hello.py",
            "file_extension": ".py",
            "file_size": 1024,
            "chunk_index": 0,
            "total_chunks": 1,
        }]],
        "distances": [[0.2]],
    }
    return store


@pytest.fixture
def client(mock_embedder, mock_store):
    app = create_app(embedder=mock_embedder, store=mock_store)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"EmbeddedFinder" in response.data
    assert b"search" in response.data.lower()


def test_index_has_search_input(client):
    response = client.get("/")
    assert b"<input" in response.data
    assert b"query" in response.data


def test_api_search_returns_results(client):
    response = client.post(
        "/api/search",
        data=json.dumps({"query": "hello world"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "results" in data
    assert data["count"] >= 1
    assert data["results"][0]["file_name"] == "hello.py"
    assert data["results"][0]["score"] > 0


def test_api_search_empty_query(client):
    response = client.post(
        "/api/search",
        data=json.dumps({"query": ""}),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_api_search_no_body(client):
    response = client.post(
        "/api/search",
        data="",
        content_type="application/json",
    )
    assert response.status_code == 400


def test_api_status(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["documents"] == 5
    assert data["unique_files"] == 2


def test_api_search_with_options(client):
    response = client.post(
        "/api/search",
        data=json.dumps({"query": "test", "n_results": 5, "min_score": 0.5}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "results" in data
