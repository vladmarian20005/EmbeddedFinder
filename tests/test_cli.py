"""Tests for CLI interface."""

from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from embedded_finder.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "EmbeddedFinder" in result.output
    assert "index" in result.output
    assert "search" in result.output
    assert "status" in result.output
    assert "clear" in result.output


def test_index_requires_api_key(runner, tmp_dir):
    result = runner.invoke(cli, ["index", str(tmp_dir)])
    assert result.exit_code != 0
    assert "GOOGLE_API_KEY" in result.output


def test_search_requires_api_key(runner):
    result = runner.invoke(cli, ["search", "test query"])
    assert result.exit_code != 0
    assert "GOOGLE_API_KEY" in result.output


@patch("embedded_finder.store.VectorStore.__init__", return_value=None)
@patch("embedded_finder.store.VectorStore.count", return_value=42)
@patch("embedded_finder.store.VectorStore.get_indexed_files", return_value={"/a.txt": "h1", "/b.py": "h2"})
def test_status_shows_info(mock_files, mock_count, mock_init, runner):
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "42" in result.output


@patch("embedded_finder.store.VectorStore.__init__", return_value=None)
@patch("embedded_finder.store.VectorStore.clear")
def test_clear_with_confirmation(mock_clear, mock_init, runner):
    result = runner.invoke(cli, ["clear"], input="y\n")
    assert result.exit_code == 0
    assert "cleared" in result.output.lower()
    mock_clear.assert_called_once()


@patch("embedded_finder.store.VectorStore.__init__", return_value=None)
@patch("embedded_finder.store.VectorStore.clear")
def test_clear_abort(mock_clear, mock_init, runner):
    result = runner.invoke(cli, ["clear"], input="n\n")
    mock_clear.assert_not_called()


@patch("embedded_finder.cli.get_api_key", return_value="test-key")
def test_search_displays_results(mock_key, runner):
    from embedded_finder.search import SearchResult

    mock_engine = MagicMock()
    mock_engine.search.return_value = [
        SearchResult(
            file_path="/test/hello.py",
            file_name="hello.py",
            score=0.95,
            snippet="def hello(): print('world')",
            file_extension=".py",
            file_size=1024,
        ),
    ]

    with patch("embedded_finder.embedder.Embedder.__init__", return_value=None), \
         patch("embedded_finder.store.VectorStore.__init__", return_value=None), \
         patch("embedded_finder.search.SearchEngine.__init__", return_value=None), \
         patch("embedded_finder.search.SearchEngine.search", return_value=mock_engine.search.return_value):
        result = runner.invoke(cli, ["search", "python hello function"])

    assert result.exit_code == 0
    assert "hello.py" in result.output
    assert "95.0%" in result.output


@patch("embedded_finder.cli.get_api_key", return_value="test-key")
def test_search_no_results(mock_key, runner):
    with patch("embedded_finder.embedder.Embedder.__init__", return_value=None), \
         patch("embedded_finder.store.VectorStore.__init__", return_value=None), \
         patch("embedded_finder.search.SearchEngine.__init__", return_value=None), \
         patch("embedded_finder.search.SearchEngine.search", return_value=[]):
        result = runner.invoke(cli, ["search", "nonexistent stuff"])

    assert result.exit_code == 0
    assert "No results" in result.output
