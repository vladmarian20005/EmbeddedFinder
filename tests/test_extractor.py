"""Tests for file content extraction module."""

import json
import pytest
from pathlib import Path

from embedded_finder.extractor import extract_text, chunk_text


def test_extract_plain_text(tmp_dir):
    f = tmp_dir / "hello.txt"
    f.write_text("Hello, world!")
    assert extract_text(f) == "Hello, world!"


def test_extract_python_file(tmp_dir):
    f = tmp_dir / "code.py"
    f.write_text("def foo():\n    return 42\n")
    text = extract_text(f)
    assert "def foo():" in text
    assert "return 42" in text


def test_extract_markdown(tmp_dir):
    f = tmp_dir / "notes.md"
    f.write_text("# Title\n\nSome content here.\n")
    text = extract_text(f)
    assert "# Title" in text
    assert "Some content here." in text


def test_extract_json(tmp_dir):
    f = tmp_dir / "data.json"
    data = {"name": "test", "values": [1, 2, 3]}
    f.write_text(json.dumps(data))
    text = extract_text(f)
    assert "name" in text
    assert "test" in text
    assert "values" in text


def test_extract_json_pretty_prints(tmp_dir):
    f = tmp_dir / "data.json"
    f.write_text('{"a":1}')
    text = extract_text(f)
    # Should be pretty-printed
    assert "  " in text or "\n" in text


def test_extract_handles_encoding_errors(tmp_dir):
    f = tmp_dir / "binary.txt"
    f.write_bytes(b"\x80\x81\x82\x83")
    # Should not raise, returns content via latin-1 fallback
    text = extract_text(f)
    assert isinstance(text, str)


def test_extract_returns_empty_for_missing_file():
    text = extract_text("/nonexistent/file.txt")
    assert text == ""


def test_extract_docx(tmp_dir):
    from docx import Document
    f = tmp_dir / "doc.docx"
    doc = Document()
    doc.add_paragraph("First paragraph about testing.")
    doc.add_paragraph("Second paragraph about code.")
    doc.save(str(f))

    text = extract_text(f)
    assert "First paragraph about testing." in text
    assert "Second paragraph about code." in text


# --- chunk_text tests ---

def test_chunk_text_short_text():
    chunks = chunk_text("Short text.")
    assert chunks == ["Short text."]


def test_chunk_text_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_text():
    # Create text longer than 100 tokens (~400 chars)
    paragraphs = [f"Paragraph {i} " * 20 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_tokens=100)
    assert len(chunks) > 1
    # All content should be preserved
    rejoined = " ".join(chunks)
    for i in range(10):
        assert f"Paragraph {i}" in rejoined


def test_chunk_text_respects_paragraph_boundaries():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, max_tokens=1000)
    # Short enough to fit in one chunk
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0]


def test_chunk_text_handles_single_long_paragraph():
    # One paragraph that's very long
    text = "word " * 5000  # ~25000 chars
    chunks = chunk_text(text, max_tokens=100)
    assert len(chunks) > 1
