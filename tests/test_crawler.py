"""Tests for file crawler module."""

import pytest
from pathlib import Path

from embedded_finder.crawler import crawl, FileInfo


def test_crawl_finds_supported_files(sample_files):
    results = crawl(sample_files)
    names = {r.name for r in results}
    assert "hello.txt" in names
    assert "code.py" in names
    assert "notes.md" in names
    assert "data.json" in names
    assert "nested.txt" in names


def test_crawl_returns_file_info_objects(sample_files):
    results = crawl(sample_files)
    for r in results:
        assert isinstance(r, FileInfo)
        assert r.path.exists()
        assert r.size_bytes > 0
        assert r.name == r.path.name
        assert r.extension == r.path.suffix.lower()


def test_crawl_filters_by_extension(sample_files):
    results = crawl(sample_files, extensions={".py"})
    assert len(results) == 1
    assert results[0].name == "code.py"


def test_crawl_respects_max_size(sample_files):
    # Create a file that's larger than 1 byte limit
    big_file = sample_files / "big.txt"
    big_file.write_text("x" * 1000)

    results = crawl(sample_files, max_size_mb=0.0001)  # ~100 bytes
    names = {r.name for r in results}
    assert "big.txt" not in names


def test_crawl_skips_ignored_dirs(tmp_dir):
    # Create files in normal and ignored dirs
    (tmp_dir / "good.py").write_text("print('hello')")
    git_dir = tmp_dir / ".git"
    git_dir.mkdir()
    (git_dir / "config.py").write_text("git config")
    node_dir = tmp_dir / "node_modules"
    node_dir.mkdir()
    (node_dir / "index.js").write_text("module.exports = {}")

    results = crawl(tmp_dir)
    paths = {str(r.path) for r in results}
    assert any("good.py" in p for p in paths)
    assert not any(".git" in p for p in paths)
    assert not any("node_modules" in p for p in paths)


def test_crawl_skips_hidden_dirs(tmp_dir):
    (tmp_dir / "visible.txt").write_text("visible")
    hidden = tmp_dir / ".hidden"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("secret")

    results = crawl(tmp_dir)
    names = {r.name for r in results}
    assert "visible.txt" in names
    assert "secret.txt" not in names


def test_crawl_finds_nested_files(sample_files):
    results = crawl(sample_files)
    names = {r.name for r in results}
    assert "nested.txt" in names


def test_crawl_raises_for_nonexistent_dir():
    with pytest.raises(ValueError, match="Not a directory"):
        crawl("/nonexistent/path")


def test_crawl_handles_empty_dir(tmp_dir):
    results = crawl(tmp_dir)
    assert results == []


def test_crawl_custom_ignore_patterns(tmp_dir):
    (tmp_dir / "keep.txt").write_text("keep")
    skip_dir = tmp_dir / "skipme"
    skip_dir.mkdir()
    (skip_dir / "skipped.txt").write_text("skip")

    results = crawl(tmp_dir, ignore_patterns={"skipme"})
    names = {r.name for r in results}
    assert "keep.txt" in names
    assert "skipped.txt" not in names
