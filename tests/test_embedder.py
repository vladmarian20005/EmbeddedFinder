"""Tests for Google embedding integration."""

from unittest.mock import MagicMock, patch
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
        model="gemini-embedding-2-preview",
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


# --- embed_file tests (multimodal) ---

@pytest.fixture
def mock_types():
    """Mock the _get_types function to return a fake types module."""
    fake_types = MagicMock()
    fake_types.Part.from_bytes.return_value = MagicMock()
    fake_types.Content.return_value = MagicMock()
    with patch("embedded_finder.embedder._get_types", return_value=fake_types):
        yield fake_types


def test_embed_file_returns_embedding(mock_types, mock_client, tmp_dir):
    img = tmp_dir / "test.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    embedder = Embedder(api_key="test-key", client=mock_client)
    result = embedder.embed_file(img, "image/png")
    assert len(result) == 3072

    mock_types.Part.from_bytes.assert_called_once()
    call_kwargs = mock_types.Part.from_bytes.call_args
    assert call_kwargs[1]["mime_type"] == "image/png"


def test_embed_file_raises_for_missing_file(mock_types, mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="File not found"):
        embedder.embed_file("/nonexistent/file.png", "image/png")


def test_embed_file_raises_for_empty_file(mock_types, mock_client, tmp_dir):
    empty = tmp_dir / "empty.png"
    empty.write_bytes(b"")

    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="empty"):
        embedder.embed_file(empty, "image/png")


def test_embed_file_handles_rate_limit(mock_types, mock_client, tmp_dir):
    fake = FakeEmbedding([0.1] * 3072)
    mock_client.models.embed_content.side_effect = [
        Exception("429 rate limit exceeded"),
        FakeEmbedResult([fake]),
    ]

    img = tmp_dir / "test.jpg"
    img.write_bytes(b"\xff\xd8" + b"\x00" * 50)

    embedder = Embedder(api_key="test-key", client=mock_client)
    result = embedder.embed_file(img, "image/jpeg")
    assert len(result) == 3072


def test_embed_file_raises_on_non_rate_error(mock_types, mock_client, tmp_dir):
    mock_client.models.embed_content.side_effect = Exception("Invalid content")

    img = tmp_dir / "bad.png"
    img.write_bytes(b"\x00" * 50)

    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="Embedding file failed"):
        embedder.embed_file(img, "image/png")


# --- embed_bytes tests ---

def test_embed_bytes_returns_embedding(mock_types, mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    result = embedder.embed_bytes(b"\x89PNG" + b"\x00" * 100, "image/png")
    assert len(result) == 3072

    mock_types.Part.from_bytes.assert_called_once()
    call_kwargs = mock_types.Part.from_bytes.call_args
    assert call_kwargs[1]["mime_type"] == "image/png"
    assert call_kwargs[1]["data"] == b"\x89PNG" + b"\x00" * 100


def test_embed_bytes_raises_on_empty_data(mock_types, mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="empty data"):
        embedder.embed_bytes(b"", "image/png")


# --- embed_file_via_api tests ---

def test_embed_file_via_api(mock_client, tmp_dir):
    fake = FakeEmbedding([0.5] * 3072)
    mock_client.models.embed_content.return_value = FakeEmbedResult([fake])

    uploaded = MagicMock()
    uploaded.state = "ACTIVE"
    uploaded.name = "files/abc123"
    mock_client.files.upload.return_value = uploaded

    big_file = tmp_dir / "big.mp4"
    big_file.write_bytes(b"\x00" * 1000)

    embedder = Embedder(api_key="test-key", client=mock_client)
    result = embedder.embed_file_via_api(big_file, "video/mp4")
    assert len(result) == 3072

    mock_client.files.upload.assert_called_once()
    mock_client.files.delete.assert_called_once_with(name="files/abc123")


def test_embed_file_via_api_cleans_up_on_error(mock_client, tmp_dir):
    uploaded = MagicMock()
    uploaded.state = "ACTIVE"
    uploaded.name = "files/abc123"
    mock_client.files.upload.return_value = uploaded
    mock_client.models.embed_content.side_effect = Exception("API error")

    big_file = tmp_dir / "big.mp4"
    big_file.write_bytes(b"\x00" * 1000)

    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError):
        embedder.embed_file_via_api(big_file, "video/mp4")

    # Should still clean up
    mock_client.files.delete.assert_called_once_with(name="files/abc123")


def test_embed_file_via_api_missing_file(mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="File not found"):
        embedder.embed_file_via_api("/nonexistent/file.mp4", "video/mp4")


# --- embed_multipart tests ---

def test_embed_multipart(mock_types, mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    parts = [MagicMock(), MagicMock()]
    result = embedder.embed_multipart(parts)
    assert len(result) == 3072

    mock_types.Content.assert_called_once_with(parts=parts)


def test_embed_multipart_empty_raises(mock_types, mock_client):
    embedder = Embedder(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingError, match="empty parts"):
        embedder.embed_multipart([])


# --- average_embeddings tests ---

def test_average_embeddings():
    emb1 = [1.0, 2.0, 3.0]
    emb2 = [3.0, 4.0, 5.0]
    avg = Embedder.average_embeddings([emb1, emb2])
    assert avg == [2.0, 3.0, 4.0]


def test_average_embeddings_single():
    emb = [1.0, 2.0, 3.0]
    avg = Embedder.average_embeddings([emb])
    assert avg == [1.0, 2.0, 3.0]


def test_average_embeddings_empty_raises():
    with pytest.raises(EmbeddingError, match="empty embeddings"):
        Embedder.average_embeddings([])
