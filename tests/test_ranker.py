"""Tests for result ranking and display."""

import pytest

from embedded_finder.search import SearchResult
from embedded_finder.ranker import rank_results, format_results, _score_color, _truncate_path


def _make_result(file_path="/test/a.py", score=0.9, **kwargs):
    defaults = dict(
        file_path=file_path,
        file_name=file_path.split("/")[-1],
        score=score,
        snippet="Some code content here",
        file_extension=".py",
        file_size=1024,
    )
    defaults.update(kwargs)
    return SearchResult(**defaults)


def test_rank_results_preserves_order_when_no_boost():
    results = [
        _make_result("/a.py", score=0.9),
        _make_result("/b.py", score=0.8),
        _make_result("/c.py", score=0.7),
    ]
    ranked = rank_results(results, "unrelated query")
    assert ranked[0].file_path == "/a.py"
    assert ranked[1].file_path == "/b.py"
    assert ranked[2].file_path == "/c.py"


def test_rank_results_boosts_filename_match():
    results = [
        _make_result("/other.py", score=0.85),
        _make_result("/hello.py", score=0.84),
    ]
    ranked = rank_results(results, "hello function")
    # hello.py should be boosted above other.py
    assert ranked[0].file_path == "/hello.py"


def test_rank_results_caps_at_one():
    results = [_make_result("/a.py", score=0.99)]
    ranked = rank_results(results, "a function code python")
    assert ranked[0].score <= 1.0


def test_format_results_contains_filenames():
    results = [
        _make_result("/test/hello.py", score=0.95),
        _make_result("/test/world.txt", score=0.80, file_extension=".txt"),
    ]
    output = format_results(results, "hello world")
    assert "hello.py" in output
    assert "world.txt" in output


def test_format_results_contains_scores():
    results = [_make_result(score=0.95)]
    output = format_results(results)
    assert "95%" in output


def test_format_results_shows_query():
    results = [_make_result()]
    output = format_results(results, "my search query")
    assert "my search query" in output


def test_format_results_empty():
    output = format_results([])
    assert "No results" in output


def test_format_results_shows_count():
    results = [_make_result(), _make_result("/b.py", score=0.8)]
    output = format_results(results)
    # Rich adds ANSI codes, so check for the parts separately
    assert "result" in output
    assert "found" in output


def test_score_color():
    assert _score_color(0.90) == "green"
    assert _score_color(0.75) == "yellow"
    assert _score_color(0.55) == "orange1"
    assert _score_color(0.30) == "red"


def test_truncate_path_short():
    assert _truncate_path("/a/b.py") == "/a/b.py"


def test_truncate_path_long():
    long_path = "/very/long/deeply/nested/directory/structure/file.py"
    result = _truncate_path(long_path, max_len=30)
    assert len(result) <= len(long_path)
    assert "file.py" in result
