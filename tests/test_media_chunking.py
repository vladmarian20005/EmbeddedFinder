"""Tests for media file chunking (ffmpeg-based splitting)."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from embedded_finder.extractor import chunk_media_file


@pytest.fixture
def audio_file(tmp_dir):
    """Create a fake audio file."""
    f = tmp_dir / "long_audio.mp3"
    f.write_bytes(b"\xff\xfb" + b"\x00" * 200)
    return f


@pytest.fixture
def video_file(tmp_dir):
    """Create a fake video file."""
    f = tmp_dir / "long_video.mp4"
    f.write_bytes(b"\x00\x00\x00\x1c" + b"\x00" * 200)
    return f


def _mock_ffmpeg_success(chunk_path):
    """Create a fake chunk file to simulate ffmpeg output."""
    Path(chunk_path).write_bytes(b"\x00" * 50)


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_audio_creates_correct_number_of_chunks(mock_run, mock_duration, audio_file):
    mock_duration.return_value = 180.0  # 3 minutes
    # chunk_seconds=60, overlap=10 → chunks at 0, 50, 100, 150
    mock_run.side_effect = lambda cmd, **kw: _mock_ffmpeg_success(cmd[-1])

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert len(chunks) == 4
    assert all(c.exists() for c in chunks)

    # Clean up
    if chunks:
        import shutil
        shutil.rmtree(chunks[0].parent, ignore_errors=True)


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_video_creates_chunks(mock_run, mock_duration, video_file):
    mock_duration.return_value = 240.0  # 4 minutes
    # chunk_seconds=90, overlap=15 → chunks at 0, 75, 150, 225
    mock_run.side_effect = lambda cmd, **kw: _mock_ffmpeg_success(cmd[-1])

    chunks = chunk_media_file(video_file, chunk_seconds=90, overlap_seconds=15)
    assert len(chunks) == 4

    if chunks:
        import shutil
        shutil.rmtree(chunks[0].parent, ignore_errors=True)


@patch("embedded_finder.extractor.get_media_duration")
def test_chunk_returns_empty_when_duration_unknown(mock_duration, audio_file):
    mock_duration.return_value = None
    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert chunks == []


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_handles_ffmpeg_not_installed(mock_run, mock_duration, audio_file):
    mock_duration.return_value = 120.0
    mock_run.side_effect = FileNotFoundError("ffmpeg not found")

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert chunks == []


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_handles_ffmpeg_timeout(mock_run, mock_duration, audio_file):
    import subprocess
    mock_duration.return_value = 120.0
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert chunks == []


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_handles_ffmpeg_error(mock_run, mock_duration, audio_file):
    import subprocess
    mock_duration.return_value = 120.0
    mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert chunks == []


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_single_short_file(mock_run, mock_duration, audio_file):
    mock_duration.return_value = 30.0  # Shorter than chunk_seconds
    mock_run.side_effect = lambda cmd, **kw: _mock_ffmpeg_success(cmd[-1])

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert len(chunks) == 1

    if chunks:
        import shutil
        shutil.rmtree(chunks[0].parent, ignore_errors=True)


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_ffmpeg_called_with_correct_args(mock_run, mock_duration, audio_file):
    mock_duration.return_value = 120.0
    mock_run.side_effect = lambda cmd, **kw: _mock_ffmpeg_success(cmd[-1])

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)

    # First call should start at 0
    first_call = mock_run.call_args_list[0]
    cmd = first_call[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd
    ss_idx = cmd.index("-ss")
    assert cmd[ss_idx + 1] == "0.0"
    assert "-t" in cmd
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "60"

    if chunks:
        import shutil
        shutil.rmtree(chunks[0].parent, ignore_errors=True)


@patch("embedded_finder.extractor.get_media_duration")
@patch("embedded_finder.extractor.subprocess.run")
def test_chunk_creates_temp_directory(mock_run, mock_duration, audio_file):
    mock_duration.return_value = 60.0
    mock_run.side_effect = lambda cmd, **kw: _mock_ffmpeg_success(cmd[-1])

    chunks = chunk_media_file(audio_file, chunk_seconds=60, overlap_seconds=10)
    assert len(chunks) >= 1

    # Chunks should be in a temp directory
    chunk_dir = chunks[0].parent
    assert "efind_chunks_" in chunk_dir.name

    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)
