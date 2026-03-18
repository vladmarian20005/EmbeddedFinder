"""Indexing pipeline for EmbeddedFinder."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from embedded_finder.crawler import crawl, FileInfo
from embedded_finder.extractor import extract_text, chunk_text
from embedded_finder.embedder import Embedder
from embedded_finder.store import VectorStore


@dataclass
class IndexStats:
    """Statistics from an indexing run."""
    total_files: int = 0
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    chunks_created: int = 0
    error_files: list[str] = field(default_factory=list)


class Indexer:
    """Ties together crawler, extractor, embedder, and store."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
    ):
        self._embedder = embedder
        self._store = store

    def index_directory(
        self,
        path: str | Path,
        extensions: set[str] | None = None,
        on_progress: callable = None,
    ) -> IndexStats:
        """Index all supported files in a directory.

        Args:
            path: Directory to index.
            extensions: Optional extension filter.
            on_progress: Optional callback(file_info, stats) for progress updates.

        Returns:
            IndexStats with counts.
        """
        files = crawl(path, extensions=extensions)
        stats = IndexStats(total_files=len(files))
        existing = self._store.get_indexed_files()

        for file_info in files:
            try:
                self._index_file(file_info, existing, stats)
            except Exception as e:
                stats.errors += 1
                stats.error_files.append(f"{file_info.path}: {e}")

            if on_progress:
                on_progress(file_info, stats)

        return stats

    def index_file(self, file_path: str | Path) -> IndexStats:
        """Index a single file."""
        path = Path(file_path).resolve()
        stat = path.stat()
        file_info = FileInfo(
            path=path,
            name=path.name,
            extension=path.suffix.lower(),
            size_bytes=stat.st_size,
            modified_at=__import__("datetime").datetime.fromtimestamp(stat.st_mtime),
        )
        stats = IndexStats(total_files=1)
        existing = self._store.get_indexed_files()
        try:
            self._index_file(file_info, existing, stats)
        except Exception as e:
            stats.errors += 1
            stats.error_files.append(f"{file_info.path}: {e}")
        return stats

    def _index_file(
        self,
        file_info: FileInfo,
        existing: dict[str, str],
        stats: IndexStats,
    ) -> None:
        """Index a single file, skipping if content hash matches."""
        text = extract_text(file_info.path)
        if not text.strip():
            stats.skipped += 1
            return

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        file_path_str = str(file_info.path)

        # Skip if already indexed with same hash
        if existing.get(file_path_str) == content_hash:
            stats.skipped += 1
            return

        # Remove old entries for this file
        self._store.delete_by_file_path(file_path_str)

        # Chunk the text
        chunks = chunk_text(text)
        if not chunks:
            stats.skipped += 1
            return

        # Generate embeddings
        embeddings = self._embedder.embed_batch(chunks)

        # Store with metadata
        ids = [f"{file_path_str}::chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "file_path": file_path_str,
                "file_name": file_info.name,
                "file_extension": file_info.extension,
                "file_size": file_info.size_bytes,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_hash": content_hash,
            }
            for i in range(len(chunks))
        ]

        self._store.add_documents(
            ids=ids,
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        stats.indexed += 1
        stats.chunks_created += len(chunks)
