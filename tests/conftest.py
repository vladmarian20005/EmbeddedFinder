"""Shared test fixtures for EmbeddedFinder."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_files(tmp_dir):
    """Create sample files in a temporary directory."""
    files = {
        "hello.txt": "Hello, this is a plain text file about greetings.",
        "code.py": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "notes.md": "# Meeting Notes\n\nDiscussed project timeline and deliverables.\n",
        "data.json": '{"name": "test", "values": [1, 2, 3]}',
        "sub/nested.txt": "This is a nested file in a subdirectory.",
    }
    for rel_path, content in files.items():
        path = tmp_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_dir


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure tests don't leak environment variables."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDEDFINDER_DB_DIR", raising=False)
