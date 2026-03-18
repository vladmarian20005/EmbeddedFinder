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


# --- Multimodal indexing tests ---

def test_index_image_uses_native_embed(mock_embedder, store, tmp_dir):
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 10] * len(texts)
    mock_embedder.embed_file.return_value = [0.2] * 10

    # Create an image file
    img = tmp_dir / "photo.png"
    img.write_bytes(b"\x89PNG" + b"\x00" * 100)

    indexer = Indexer(embedder=mock_embedder, store=store)
    stats = indexer.index_file(img)

    assert stats.indexed == 1
    # Should have called embed_file, not embed_batch
    mock_embedder.embed_file.assert_called_once()
    mock_embedder.embed_batch.assert_not_called()


def test_index_text_file_uses_text_embed(mock_embedder, store, tmp_dir):
    mock_embedder.embed_batch.side_effect = lambda texts: [[0.1] * 10] * len(texts)

    txt = tmp_dir / "code.py"
    txt.write_text("def hello(): return 42")

    indexer = Indexer(embedder=mock_embedder, store=store)
    stats = indexer.index_file(txt)

    assert stats.indexed == 1
    mock_embedder.embed_batch.assert_called_once()
    mock_embedder.embed_file.assert_not_called()


def test_index_small_pdf_uses_native_embed(mock_embedder, store, tmp_dir):
    from PyPDF2 import PdfWriter
    mock_embedder.embed_file.return_value = [0.3] * 10

    pdf = tmp_dir / "small.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf, "wb") as f:
        writer.write(f)

    indexer = Indexer(embedder=mock_embedder, store=store)
    stats = indexer.index_file(pdf)

    assert stats.indexed == 1
    mock_embedder.embed_file.assert_called_once()


def test_index_audio_uses_native_embed(mock_embedder, store, tmp_dir):
    mock_embedder.embed_file.return_value = [0.4] * 10

    audio = tmp_dir / "song.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 200)

    indexer = Indexer(embedder=mock_embedder, store=store)
    stats = indexer.index_file(audio)

    assert stats.indexed == 1
    mock_embedder.embed_file.assert_called_once()


def test_index_native_file_skips_when_unchanged(mock_embedder, store, tmp_dir):
    mock_embedder.embed_file.return_value = [0.2] * 10

    img = tmp_dir / "photo.jpg"
    img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

    indexer = Indexer(embedder=mock_embedder, store=store)

    stats1 = indexer.index_file(img)
    assert stats1.indexed == 1

    stats2 = indexer.index_file(img)
    assert stats2.skipped == 1
    assert stats2.indexed == 0
