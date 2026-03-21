"""Configuration management for EmbeddedFinder."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_DB_DIR = ".embeddedfinder/db"
DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_CHUNK_MAX_TOKENS = 2000
DEFAULT_SEARCH_RESULTS = 10
MAX_NATIVE_PDF_PAGES = 6

# Multimodal extensions — these are embedded natively (raw bytes sent to model)
# API only supports PNG and JPEG natively; other image formats are converted to PNG
NATIVE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CONVERTIBLE_IMAGE_EXTENSIONS = {".gif", ".webp", ".bmp"}
IMAGE_EXTENSIONS = NATIVE_IMAGE_EXTENSIONS | CONVERTIBLE_IMAGE_EXTENSIONS
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}  # Only officially supported containers
NATIVE_EMBED_EXTENSIONS = NATIVE_IMAGE_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
# PDFs are conditionally native (≤6 pages) or text-extracted (>6 pages)

# Media duration limits (seconds) — enforced by the Gemini API
MAX_AUDIO_DURATION_SECONDS = 80
MAX_VIDEO_DURATION_SECONDS = 120

# Media chunking constants for long audio/video
AUDIO_CHUNK_SECONDS = 60
AUDIO_OVERLAP_SECONDS = 10
VIDEO_CHUNK_SECONDS = 90
VIDEO_OVERLAP_SECONDS = 15

# Files API threshold — files larger than this are uploaded via the Files API
FILES_API_THRESHOLD_MB = 20

# MIME type mapping for multimodal embedding
EXTENSION_MIME_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".flac": "audio/flac", ".m4a": "audio/mp4",
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".pdf": "application/pdf",
}

SUPPORTED_EXTENSIONS = {
    # Plain text / code
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs", ".rb",
    ".php", ".swift", ".kt", ".scala", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".css", ".scss", ".less", ".xml", ".svg",
    ".sql", ".r", ".m", ".lua", ".pl", ".ex", ".exs",
    # Data
    ".json", ".csv",
    # Documents
    ".pdf", ".docx",
    # Images (native multimodal)
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    # Audio (native multimodal)
    ".mp3", ".wav", ".ogg", ".flac", ".m4a",
    # Video (native multimodal)
    ".mp4", ".mov",
}

IGNORE_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".eggs", "*.egg-info",
    ".embeddedfinder",
}


def get_api_key() -> str | None:
    """Get the Google API key from environment."""
    return os.environ.get("GOOGLE_API_KEY")


def get_db_dir() -> Path:
    """Get the database directory path."""
    return Path(os.environ.get("EMBEDDEDFINDER_DB_DIR", DEFAULT_DB_DIR))
