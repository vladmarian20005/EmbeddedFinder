"""File discovery and crawling for EmbeddedFinder."""

import fnmatch
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from embedded_finder.config import IGNORE_DIRS, SUPPORTED_EXTENSIONS, DEFAULT_MAX_FILE_SIZE_MB


@dataclass
class FileInfo:
    """Metadata about a discovered file."""
    path: Path
    name: str
    extension: str
    size_bytes: int
    modified_at: datetime


def crawl(
    root_path: str | Path,
    extensions: set[str] | None = None,
    max_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
    ignore_patterns: set[str] | None = None,
) -> list[FileInfo]:
    """Recursively discover files in a directory.

    Args:
        root_path: Directory to crawl.
        extensions: Set of file extensions to include (e.g. {".py", ".txt"}).
                   Defaults to SUPPORTED_EXTENSIONS.
        max_size_mb: Skip files larger than this (in MB).
        ignore_patterns: Directory name patterns to skip.
                        Defaults to IGNORE_DIRS.

    Returns:
        List of FileInfo for each discovered file.
    """
    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    allowed_ext = extensions if extensions is not None else SUPPORTED_EXTENSIONS
    ignore = ignore_patterns if ignore_patterns is not None else IGNORE_DIRS
    max_size_bytes = int(max_size_mb * 1024 * 1024)

    results: list[FileInfo] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out ignored directories in-place to prevent os.walk from descending
        dirnames[:] = [
            d for d in dirnames
            if not _should_ignore(d, ignore)
        ]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            ext = filepath.suffix.lower()

            if ext not in allowed_ext:
                continue

            try:
                stat = filepath.stat()
            except (OSError, PermissionError):
                continue

            if stat.st_size > max_size_bytes:
                continue

            results.append(FileInfo(
                path=filepath,
                name=filename,
                extension=ext,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime),
            ))

    return results


def _should_ignore(dirname: str, patterns: set[str]) -> bool:
    """Check if a directory name matches any ignore pattern."""
    if dirname.startswith("."):
        # Check if the dotfile/dir is explicitly in patterns
        if dirname in patterns:
            return True
        # Skip hidden dirs by default
        return True
    for pattern in patterns:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    return False
