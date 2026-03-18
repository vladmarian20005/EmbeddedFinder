"""Tests for Google embedding integration."""

from unittest.mock import MagicMock
import pytest

from embedded_finder.embedder import Embedder, EmbeddingError


class FakeEmbedding:
    def __init__(self, values):
        self.values = values


class FakeEmbedResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


@pytest.fixture
def mock_client():
    """Create a mock GenAI client."""
    client = MagicMock()
    fake_embedding = FakeEmbedding([0.1] * 3072)
    client.models.embed_content.return_value = FakeEmbedResult([fake_embedding])
    return client


def test_embed_text_returns_correct_dimension(mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    result = embedder.embed_text("Hello world")
    assert len(result) == 3072
    assert all(isinstance(v, float) for v in result)


def test_embed_text_calls_api_with_correct_model(mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    embedder.embed_text("Hello world")
    mock_client.models.embed_content.assert_called_once_with(
        model="gemini-embedding-001",
        contents="Hello world",
    )


def test_embed_text_raises_on_empty(mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="empty text"):
        embedder.embed_text("")


def test_embed_text_raises_on_whitespace(mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="empty text"):
        embedder.embed_text("   ")


def test_embed_batch_returns_multiple(mock_client):
    fake1 = FakeEmbedding([0.1] * 3072)
    fake2 = FakeEmbedding([0.2] * 3072)
    mock_client.models.embed_content.return_value = FakeEmbedResult([fake1, fake2])

    embedder = Embedder(api_key="test-key", client=mock_client)
    results = embedder.embed_batch(["text one", "text two"])
    assert len(results) == 2
    assert results[0][0] == 0.1
    assert results[1][0] == 0.2


def test_embed_batch_empty_list(mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    assert embedder.embed_batch([]) == []


def test_no_api_key_raises():
    with pytest.raises(EmbeddingError, match="No API key"):
        Embedder()


def test_embed_batch_handles_rate_limit(mock_client):
    fake = FakeEmbedding([0.1] * 3072)
    mock_client.models.embed_content.side_effect = [
        Exception("429 rate limit exceeded"),
        FakeEmbedResult([fake]),
    ]

    embedder = Embedder(api_key="test-key", client=mock_client)
    results = embedder.embed_batch(["test text"])
    assert len(results) == 1


def test_embed_batch_raises_on_non_rate_error(mock_client):
    mock_client.models.embed_content.side_effect = Exception("Invalid request")

    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="Embedding failed"):
        embedder.embed_batch(["test text"])


def test_custom_model(mock_client):
    embedder = Embedder(api_key="test-key", model="custom-model", client=mock_client)
    embedder.embed_text("Hello")
    mock_client.models.embed_content.assert_called_once_with(
        model="custom-model",
        contents="Hello",
    )
