"""File content extraction for EmbeddedFinder."""

import io
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from embedded_finder.config import (
    DEFAULT_CHUNK_MAX_TOKENS,
    EXTENSION_MIME_MAP,
    NATIVE_IMAGE_EXTENSIONS,
    CONVERTIBLE_IMAGE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    NATIVE_EMBED_EXTENSIONS,
    MAX_NATIVE_PDF_PAGES,
)

logger = logging.getLogger(__name__)

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


def get_mime_type(file_path: str | Path) -> str:
    """Get the MIME type for a file based on its extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_MIME_MAP.get(ext, "application/octet-stream")


def is_natively_embeddable(file_path: str | Path) -> bool:
    """Check if a file can be embedded natively (without text extraction).

    Returns True for images (including convertible formats), audio, video,
    and small PDFs (<=6 pages).
    """
    ext = Path(file_path).suffix.lower()
    if ext in NATIVE_EMBED_EXTENSIONS:
        return True
    if ext in CONVERTIBLE_IMAGE_EXTENSIONS:
        return True
    if ext == ".pdf":
        return get_pdf_page_count(file_path) <= MAX_NATIVE_PDF_PAGES
    return False


def convert_image_to_png(file_path: str | Path) -> tuple[bytes, str]:
    """Convert an image file to PNG format using Pillow.

    Args:
        file_path: Path to the image file.

    Returns:
        Tuple of (png_bytes, "image/png").
    """
    from PIL import Image

    with Image.open(str(file_path)) as img:
        # Convert to RGB if necessary (e.g., RGBA, P mode)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"


def prepare_native_file(file_path: str | Path) -> tuple[bytes, str]:
    """Prepare a file for native embedding.

    Converts unsupported image formats to PNG; reads raw bytes otherwise.

    Args:
        file_path: Path to the file.

    Returns:
        Tuple of (file_bytes, mime_type).
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in CONVERTIBLE_IMAGE_EXTENSIONS:
        return convert_image_to_png(path)

    mime_type = get_mime_type(path)
    return path.read_bytes(), mime_type


def get_media_duration(file_path: str | Path) -> float | None:
    """Get the duration of an audio or video file in seconds using mutagen.

    Args:
        file_path: Path to the media file.

    Returns:
        Duration in seconds, or None if duration cannot be determined.
    """
    try:
        import mutagen
        media = mutagen.File(str(file_path))
        if media is not None and media.info is not None:
            return media.info.length
    except Exception:
        pass
    return None


def chunk_media_file(
    file_path: str | Path,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[Path]:
    """Split a media file into overlapping chunks using ffmpeg.

    Args:
        file_path: Path to the media file.
        chunk_seconds: Duration of each chunk in seconds.
        overlap_seconds: Overlap between consecutive chunks in seconds.

    Returns:
        List of paths to temporary chunk files. Caller is responsible for cleanup.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    duration = get_media_duration(path)
    if duration is None:
        logger.warning("Cannot determine duration for chunking: %s", path)
        return []

    chunk_dir = tempfile.mkdtemp(prefix="efind_chunks_")
    chunks = []
    start = 0.0
    i = 0

    while start < duration:
        chunk_path = Path(chunk_dir) / f"chunk_{i}{ext}"
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-c", "copy",
            str(chunk_path),
        ]
        try:
            subprocess.run(
                cmd, capture_output=True, check=True, timeout=120,
            )
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunks.append(chunk_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("ffmpeg chunking failed for %s chunk %d: %s", path, i, e)
            break

        start += chunk_seconds - overlap_seconds
        i += 1

    return chunks


def get_pdf_page_count(file_path: str | Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        return len(reader.pages)
    except Exception:
        return 999  # Assume large on error, fallback to text extraction


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
    from pypdf import PdfReader

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
