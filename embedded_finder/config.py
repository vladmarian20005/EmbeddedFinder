"""Configuration management for EmbeddedFinder."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_DB_DIR = ".embeddedfinder/db"
DEFAULT_MAX_FILE_SIZE_MB = 50
DEFAULT_CHUNK_MAX_TOKENS = 2000
DEFAULT_SEARCH_RESULTS = 10

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
