"""Tests for configuration module."""

from embedded_finder.config import (
    DEFAULT_DB_DIR,
    DEFAULT_EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
    IGNORE_DIRS,
    get_api_key,
    get_db_dir,
)


def test_default_embedding_model():
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-001"


def test_supported_extensions_include_common_types():
    assert ".py" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".json" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS


def test_ignore_dirs_include_common_patterns():
    assert ".git" in IGNORE_DIRS
    assert "node_modules" in IGNORE_DIRS
    assert "__pycache__" in IGNORE_DIRS


def test_get_api_key_returns_none_when_unset():
    assert get_api_key() is None


def test_get_api_key_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-123")
    assert get_api_key() == "test-key-123"


def test_get_db_dir_returns_default():
    assert str(get_db_dir()) == DEFAULT_DB_DIR


def test_get_db_dir_respects_env(monkeypatch):
    monkeypatch.setenv("EMBEDDEDFINDER_DB_DIR", "/custom/db/path")
    assert str(get_db_dir()) == "/custom/db/path"
