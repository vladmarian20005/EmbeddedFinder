"""Indexing pipeline for EmbeddedFinder."""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from embedded_finder.crawler import crawl, FileInfo
from embedded_finder.extractor import (
    extract_text, chunk_text, is_natively_embeddable, get_mime_type,
    prepare_native_file, get_media_duration, chunk_media_file,
)
from embedded_finder.embedder import Embedder
from embedded_finder.store import VectorStore
from embedded_finder.config import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    CONVERTIBLE_IMAGE_EXTENSIONS,
    MAX_AUDIO_DURATION_SECONDS,
    MAX_VIDEO_DURATION_SECONDS,
    AUDIO_CHUNK_SECONDS,
    AUDIO_OVERLAP_SECONDS,
    VIDEO_CHUNK_SECONDS,
    VIDEO_OVERLAP_SECONDS,
    FILES_API_THRESHOLD_MB,
    DEFAULT_INDEXING_WORKERS,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics from an indexing run."""
    total_files: int = 0
    indexed: int = 0
    skipped: int = 0
    errors: int = 0
    chunks_created: int = 0
    error_files: list[str] = field(default_factory=list)


@dataclass
class _PreparedTextFile:
    """Pre-processed text file ready for batched embedding."""
    file_info: FileInfo
    content_hash: str
    chunks: list[str]


class Indexer:
    """Ties together crawler, extractor, embedder, and store."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        max_workers: int = DEFAULT_INDEXING_WORKERS,
    ):
        self._embedder = embedder
        self._store = store
        self._max_workers = max_workers
        self._db_lock = threading.Lock()

    def index_directory(
        self,
        path: str | Path,
        extensions: set[str] | None = None,
        on_progress: callable = None,
    ) -> IndexStats:
        """Index all supported files in a directory.

        Uses a two-phase approach for text files:
        1. Extract and chunk text from all files in parallel (CPU-bound)
        2. Batch all chunks into minimal API calls (IO-bound)

        Native files (images/audio/video) are processed concurrently.

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

        # Split into text files and native (multimodal) files
        text_files = []
        native_files = []
        for fi in files:
            if is_natively_embeddable(fi.path):
                native_files.append(fi)
            else:
                text_files.append(fi)

        # Phase 1: Prepare text files in parallel (extract + chunk, no API calls)
        prepared: list[_PreparedTextFile] = []
        for fi in text_files:
            try:
                result = self._prepare_text_file(fi, existing, stats)
                if result is not None:
                    prepared.append(result)
            except Exception as e:
                stats.errors += 1
                stats.error_files.append(f"{fi.path}: {e}")
            if on_progress:
                on_progress(fi, stats)

        # Phase 2: Batch-embed all text chunks across files in minimal API calls
        if prepared:
            try:
                self._batch_embed_and_store(prepared, stats)
            except Exception as e:
                # If the batch embed fails, record errors for all prepared files
                for pf in prepared:
                    stats.errors += 1
                    stats.error_files.append(f"{pf.file_info.path}: {e}")

        # Phase 3: Process native files concurrently
        if native_files:
            stats_lock = threading.Lock()

            def _process_native(file_info):
                try:
                    self._index_native(file_info, existing, stats, stats_lock)
                except Exception as e:
                    with stats_lock:
                        stats.errors += 1
                        stats.error_files.append(f"{file_info.path}: {e}")
                if on_progress:
                    with stats_lock:
                        on_progress(file_info, stats)

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    executor.submit(_process_native, fi): fi for fi in native_files
                }
                for future in as_completed(futures):
                    exc = future.exception()
                    if exc:
                        fi = futures[future]
                        with stats_lock:
                            stats.errors += 1
                            stats.error_files.append(f"{fi.path}: {exc}")
                        if on_progress:
                            with stats_lock:
                                on_progress(fi, stats)

        return stats

    def _prepare_text_file(
        self,
        file_info: FileInfo,
        existing: dict[str, str],
        stats: IndexStats,
    ) -> _PreparedTextFile | None:
        """Extract and chunk a text file without making API calls.

        Returns a _PreparedTextFile if the file needs embedding, None if skipped.
        """
        file_path_str = str(file_info.path)

        text = extract_text(file_info.path)
        if not text.strip():
            stats.skipped += 1
            return None

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        if existing.get(file_path_str) == content_hash:
            stats.skipped += 1
            return None

        chunks = chunk_text(text)
        if not chunks:
            stats.skipped += 1
            return None

        return _PreparedTextFile(
            file_info=file_info,
            content_hash=content_hash,
            chunks=chunks,
        )

    def _batch_embed_and_store(
        self,
        prepared: list[_PreparedTextFile],
        stats: IndexStats,
    ) -> None:
        """Embed all prepared text chunks in minimal API calls, then store results."""
        # Flatten all chunks into a single list for cross-file batching
        all_chunks: list[str] = []
        # Track where each file's chunks start in the flat list
        file_offsets: list[tuple[int, int]] = []  # (start_index, num_chunks)
        for pf in prepared:
            start = len(all_chunks)
            all_chunks.extend(pf.chunks)
            file_offsets.append((start, len(pf.chunks)))

        # Batch API call across all files (embed_batch handles internal batching)
        all_embeddings = self._embedder.embed_batch(all_chunks, task_type="RETRIEVAL_DOCUMENT")

        # Distribute embeddings back to files and store
        for pi, pf in enumerate(prepared):
            file_path_str = str(pf.file_info.path)
            start, num_chunks = file_offsets[pi]
            file_embeddings = all_embeddings[start : start + num_chunks]

            ids = [f"{file_path_str}::chunk_{i}" for i in range(num_chunks)]
            metadatas = [
                {
                    "file_path": file_path_str,
                    "file_name": pf.file_info.name,
                    "file_extension": pf.file_info.extension,
                    "file_size": pf.file_info.size_bytes,
                    "chunk_index": i,
                    "total_chunks": num_chunks,
                    "content_hash": pf.content_hash,
                    "embed_mode": "text",
                }
                for i in range(num_chunks)
            ]

            self._store.delete_by_file_path(file_path_str)
            self._store.add_documents(
                ids=ids,
                texts=pf.chunks,
                embeddings=file_embeddings,
                metadatas=metadatas,
            )

            stats.indexed += 1
            stats.chunks_created += num_chunks

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
        stats_lock: threading.Lock | None = None,
    ) -> None:
        """Index a single file, skipping if content hash matches.

        Uses native multimodal embedding for images, audio, video, and small PDFs.
        Falls back to text extraction + chunking for text/code files and large PDFs.
        """
        if is_natively_embeddable(file_info.path):
            self._index_native(file_info, existing, stats, stats_lock)
        else:
            self._index_text(file_info, existing, stats, stats_lock)

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of a file using chunked reads (8KB blocks)."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(8192)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    def _update_stats(self, stats, stats_lock, **kwargs):
        """Thread-safe stats update helper."""
        if stats_lock:
            with stats_lock:
                for key, val in kwargs.items():
                    setattr(stats, key, getattr(stats, key) + val)
        else:
            for key, val in kwargs.items():
                setattr(stats, key, getattr(stats, key) + val)

    def _index_native(
        self,
        file_info: FileInfo,
        existing: dict[str, str],
        stats: IndexStats,
        stats_lock: threading.Lock | None = None,
    ) -> None:
        """Index a file using native multimodal embedding (images, audio, video, small PDFs)."""
        file_path_str = str(file_info.path)
        ext = file_info.extension

        # Streaming hash for dedup (avoids loading entire file into RAM)
        content_hash = self._compute_file_hash(file_info.path)

        if existing.get(file_path_str) == content_hash:
            self._update_stats(stats, stats_lock, skipped=1)
            return

        with self._db_lock:
            self._store.delete_by_file_path(file_path_str)

        # Check media duration limits and handle long media via chunking
        if ext in AUDIO_EXTENSIONS or ext in VIDEO_EXTENSIONS:
            duration = get_media_duration(file_info.path)
            max_duration = (
                MAX_AUDIO_DURATION_SECONDS if ext in AUDIO_EXTENSIONS
                else MAX_VIDEO_DURATION_SECONDS
            )

            if duration is not None and duration > max_duration:
                self._index_media_chunks(file_info, content_hash, stats, stats_lock)
                return

        # Prepare file bytes (converts unsupported image formats to PNG)
        file_bytes, mime_type = prepare_native_file(file_info.path)

        if not file_bytes:
            self._update_stats(stats, stats_lock, skipped=1)
            return

        # Route large files through the Files API
        if file_info.size_bytes > FILES_API_THRESHOLD_MB * 1024 * 1024:
            # For convertible images, we already have PNG bytes — use embed_bytes
            if ext in CONVERTIBLE_IMAGE_EXTENSIONS:
                embedding = self._embedder.embed_bytes(file_bytes, mime_type, task_type="RETRIEVAL_DOCUMENT")
            else:
                embedding = self._embedder.embed_file_via_api(file_info.path, mime_type, task_type="RETRIEVAL_DOCUMENT")
        elif ext in CONVERTIBLE_IMAGE_EXTENSIONS:
            # Converted images must use embed_bytes (bytes are PNG, not original format)
            embedding = self._embedder.embed_bytes(file_bytes, mime_type, task_type="RETRIEVAL_DOCUMENT")
        else:
            embedding = self._embedder.embed_file(file_info.path, mime_type, task_type="RETRIEVAL_DOCUMENT")

        # For native embeds, store file name as the document text (for search snippet)
        display_text = f"[{file_info.extension.upper().lstrip('.')}] {file_info.name}"
        doc_id = f"{file_path_str}::native_0"

        with self._db_lock:
            self._store.add_documents(
                ids=[doc_id],
                texts=[display_text],
                embeddings=[embedding],
                metadatas=[{
                    "file_path": file_path_str,
                    "file_name": file_info.name,
                    "file_extension": file_info.extension,
                    "file_size": file_info.size_bytes,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "content_hash": content_hash,
                    "embed_mode": "native",
                }],
            )

        self._update_stats(stats, stats_lock, indexed=1, chunks_created=1)

    def _index_media_chunks(
        self,
        file_info: FileInfo,
        content_hash: str,
        stats: IndexStats,
        stats_lock: threading.Lock | None = None,
    ) -> None:
        """Index a long audio/video file by splitting into overlapping chunks."""
        ext = file_info.extension
        file_path_str = str(file_info.path)

        if ext in AUDIO_EXTENSIONS:
            chunk_s, overlap_s = AUDIO_CHUNK_SECONDS, AUDIO_OVERLAP_SECONDS
        else:
            chunk_s, overlap_s = VIDEO_CHUNK_SECONDS, VIDEO_OVERLAP_SECONDS

        chunks = chunk_media_file(file_info.path, chunk_s, overlap_s)
        if not chunks:
            logger.warning(
                "Could not chunk media file (ffmpeg may not be available): %s",
                file_info.path,
            )
            self._update_stats(stats, stats_lock, skipped=1)
            return

        try:
            embeddings = []
            for chunk_path in chunks:
                mime_type = get_mime_type(chunk_path)
                emb = self._embedder.embed_file(chunk_path, mime_type, task_type="RETRIEVAL_DOCUMENT")
                embeddings.append(emb)

            ids = []
            texts = []
            metadatas = []
            for i, chunk_path in enumerate(chunks):
                start_time = i * (chunk_s - overlap_s)
                end_time = start_time + chunk_s

                doc_id = f"{file_path_str}::media_chunk_{i}"
                display = f"[{ext.upper().lstrip('.')}] {file_info.name} (chunk {i})"

                ids.append(doc_id)
                texts.append(display)
                metadatas.append({
                    "file_path": file_path_str,
                    "file_name": file_info.name,
                    "file_extension": ext,
                    "file_size": file_info.size_bytes,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_hash": content_hash,
                    "embed_mode": "native_chunked",
                    "start_time": start_time,
                    "end_time": end_time,
                })

            with self._db_lock:
                self._store.add_documents(
                    ids=ids,
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

            self._update_stats(stats, stats_lock, indexed=1, chunks_created=len(chunks))

        finally:
            # Clean up temporary chunk files
            if chunks:
                chunk_dir = chunks[0].parent
                shutil.rmtree(chunk_dir, ignore_errors=True)

    def _index_text(
        self,
        file_info: FileInfo,
        existing: dict[str, str],
        stats: IndexStats,
        stats_lock: threading.Lock | None = None,
    ) -> None:
        """Index a file using text extraction + chunking + text embedding.

        Used by index_file() for single-file indexing (e.g., watcher).
        For batch indexing, index_directory() uses _prepare_text_file + _batch_embed_and_store.
        """
        file_path_str = str(file_info.path)

        text = extract_text(file_info.path)
        if not text.strip():
            self._update_stats(stats, stats_lock, skipped=1)
            return

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        if existing.get(file_path_str) == content_hash:
            self._update_stats(stats, stats_lock, skipped=1)
            return

        chunks = chunk_text(text)
        if not chunks:
            self._update_stats(stats, stats_lock, skipped=1)
            return

        # Embedding call runs concurrently — rate limiter handles throttling
        embeddings = self._embedder.embed_batch(chunks, task_type="RETRIEVAL_DOCUMENT")

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
                "embed_mode": "text",
            }
            for i in range(len(chunks))
        ]

        with self._db_lock:
            self._store.delete_by_file_path(file_path_str)
            self._store.add_documents(
                ids=ids,
                texts=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        self._update_stats(stats, stats_lock, indexed=1, chunks_created=len(chunks))
