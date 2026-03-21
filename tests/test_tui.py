"""Tests for the interactive TUI."""

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO

from embedded_finder.tui import (
    _parse_command,
    _score_style,
    _format_size,
    _truncate_path,
    print_banner,
    print_help,
    print_results,
    _handle_status,
)
from embedded_finder.search import SearchResult


# ── Command parsing ──────────────────────────────────────────────────────────

def test_parse_empty():
    assert _parse_command("") == ("empty", "")
    assert _parse_command("   ") == ("empty", "")


def test_parse_query():
    cmd, args = _parse_command("find python files")
    assert cmd == "query"
    assert args == "find python files"


def test_parse_slash_command():
    assert _parse_command("/help") == ("help", "")
    assert _parse_command("/quit") == ("quit", "")


def test_parse_slash_command_with_args():
    cmd, args = _parse_command("/index /home/user/project")
    assert cmd == "index"
    assert args == "/home/user/project"


def test_parse_slash_command_case_insensitive():
    cmd, _ = _parse_command("/HELP")
    assert cmd == "help"



# ── Formatting helpers ───────────────────────────────────────────────────────

def test_score_style_high():
    assert _score_style(0.95) == "bold green"


def test_score_style_good():
    assert _score_style(0.80) == "green"


def test_score_style_low():
    assert _score_style(0.30) == "dim"


def test_format_size_bytes():
    assert _format_size(500) == "500B"


def test_format_size_kb():
    assert _format_size(2048) == "2K"


def test_format_size_mb():
    assert _format_size(1500000) == "1.4M"


def test_truncate_path_short():
    assert _truncate_path("/a/b.py") == "/a/b.py"


def test_truncate_path_long():
    long_path = "/very/long/deeply/nested/directory/structure/somewhere/file.py"
    result = _truncate_path(long_path, max_len=30)
    assert "file.py" in result
    assert "…" in result


# ── Banner and help ──────────────────────────────────────────────────────────

def test_print_banner_no_error(capsys):
    """Banner prints without errors."""
    print_banner(doc_count=42, file_count=10)
    # Just verify it doesn't crash


def test_print_banner_empty_index(capsys):
    """Banner with no files indexed doesn't crash."""
    print_banner(doc_count=0, file_count=0)


def test_print_help_no_error(capsys):
    """Help prints without errors."""
    print_help()


# ── Result display ───────────────────────────────────────────────────────────

def _make_result(**kwargs):
    defaults = dict(
        file_path="/test/hello.py",
        file_name="hello.py",
        score=0.95,
        snippet="def hello(): return 42",
        file_extension=".py",
        file_size=1024,
    )
    defaults.update(kwargs)
    return SearchResult(**defaults)


def test_print_results_no_error():
    results = [_make_result(), _make_result(file_path="/b.py", file_name="b.py", score=0.8)]
    print_results(results, "test query", 0.5)


def test_print_results_empty():
    print_results([], "nothing", 0.1)


def test_print_results_with_multimodal():
    results = [
        _make_result(file_extension=".png", file_name="photo.png"),
        _make_result(file_extension=".mp3", file_name="song.mp3"),
    ]
    print_results(results, "media files", 0.3)


# ── Status display ───────────────────────────────────────────────────────────

def test_handle_status_with_data():
    store = MagicMock()
    store.count.return_value = 42
    store.get_indexed_files.return_value = {
        "/a.py": "h1", "/b.txt": "h2", "/c.png": "h3",
    }
    _handle_status(store)


def test_handle_status_empty():
    store = MagicMock()
    store.count.side_effect = Exception("no db")
    _handle_status(store)
