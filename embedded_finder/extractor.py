"""File content extraction for EmbeddedFinder."""

import json
from pathlib import Path

from embedded_finder.config import DEFAULT_CHUNK_MAX_TOKENS

# Extensions treated as plain text (read directly)
PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb",
    ".php", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".css", ".scss", ".less", ".xml", ".svg",
    ".sql", ".r", ".m", ".lua", ".pl", ".ex", ".exs",
    ".csv",
}


def extract_text(file_path: str | Path) -> str:
    """Extract text content from a file.

    Args:
        file_path: Path to the file.

    Returns:
        Extracted text content, or empty string on failure.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            return _extract_pdf(path)
        elif ext == ".docx":
            return _extract_docx(path)
        elif ext == ".json":
            return _extract_json(path)
        elif ext in PLAIN_TEXT_EXTENSIONS:
            return _extract_plain_text(path)
        else:
            # Try reading as plain text as fallback
            return _extract_plain_text(path)
    except Exception:
        return ""


def _extract_plain_text(path: Path) -> str:
    """Read a file as plain text with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_json(path: Path) -> str:
    """Extract readable text from a JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _extract_plain_text(path)


def chunk_text(text: str, max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS) -> list[str]:
    """Split text into chunks suitable for embedding.

    Uses a simple word-based approximation where 1 token ~ 0.75 words.
    Splits on paragraph boundaries when possible.

    Args:
        text: The text to chunk.
        max_tokens: Maximum tokens per chunk.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    # Approximate: 1 token ≈ 4 characters (conservative estimate)
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        # If a single paragraph is too large, split it by lines
        if len(paragraph) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            line_chunks = _split_long_text(paragraph, max_chars)
            chunks.extend(line_chunks)
            continue

        if len(current_chunk) + len(paragraph) + 2 > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split a long text block by lines, then by words if needed."""
    # First try splitting by lines
    lines = text.split("\n")
    if len(lines) > 1:
        chunks = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > max_chars:
                if current:
                    chunks.append(current.strip())
                current = line
            else:
                current = current + "\n" + line if current else line
        if current.strip():
            chunks.append(current.strip())
        return chunks

    # No line breaks — split by words
    words = text.split()
    chunks = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = word
        else:
            current = current + " " + word if current else word
    if current.strip():
        chunks.append(current.strip())
    return chunks
