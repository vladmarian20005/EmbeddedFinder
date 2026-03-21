"""Indexing pipeline for EmbeddedFinder."""

import hashlib
import logging
import shutil
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
        """Index a single file, skipping if content hash matches.

        Uses native multimodal embedding for images, audio, video, and small PDFs.
        Falls back to text extraction + chunking for text/code files and large PDFs.
        """
        if is_natively_embeddable(file_info.path):
            self._index_native(file_info, existing, stats)
        else:
            self._index_text(file_info, existing, stats)

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

    def _index_native(
        self,
        file_info: FileInfo,
        existing: dict[str, str],
        stats: IndexStats,
    ) -> None:
        """Index a file using native multimodal embedding (images, audio, video, small PDFs)."""
        file_path_str = str(file_info.path)
        ext = file_info.extension

        # Streaming hash for dedup (avoids loading entire file into RAM)
        content_hash = self._compute_file_hash(file_info.path)

        if existing.get(file_path_str) == content_hash:
            stats.skipped += 1
            return

        self._store.delete_by_file_path(file_path_str)

        # Check media duration limits and handle long media via chunking
        if ext in AUDIO_EXTENSIONS or ext in VIDEO_EXTENSIONS:
            duration = get_media_duration(file_info.path)
            max_duration = (
                MAX_AUDIO_DURATION_SECONDS if ext in AUDIO_EXTENSIONS
                else MAX_VIDEO_DURATION_SECONDS
            )

            if duration is not None and duration > max_duration:
                self._index_media_chunks(file_info, content_hash, stats)
                return

        # Prepare file bytes (converts unsupported image formats to PNG)
        file_bytes, mime_type = prepare_native_file(file_info.path)

        if not file_bytes:
            stats.skipped += 1
            return

        # Route large files through the Files API
        if file_info.size_bytes > FILES_API_THRESHOLD_MB * 1024 * 1024:
            # For convertible images, we already have PNG bytes — use embed_bytes
            if ext in CONVERTIBLE_IMAGE_EXTENSIONS:
                embedding = self._embedder.embed_bytes(file_bytes, mime_type)
            else:
                embedding = self._embedder.embed_file_via_api(file_info.path, mime_type)
        elif ext in CONVERTIBLE_IMAGE_EXTENSIONS:
            # Converted images must use embed_bytes (bytes are PNG, not original format)
            embedding = self._embedder.embed_bytes(file_bytes, mime_type)
        else:
            embedding = self._embedder.embed_file(file_info.path, mime_type)

        # For native embeds, store file name as the document text (for search snippet)
        display_text = f"[{file_info.extension.upper().lstrip('.')}] {file_info.name}"
        doc_id = f"{file_path_str}::native_0"

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

        stats.indexed += 1
        stats.chunks_created += 1

    def _index_media_chunks(
        self,
        file_info: FileInfo,
        content_hash: str,
        stats: IndexStats,
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
            stats.skipped += 1
            return

        try:
            embeddings = []
            for chunk_path in chunks:
                mime_type = get_mime_type(chunk_path)
                emb = self._embedder.embed_file(chunk_path, mime_type)
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

            self._store.add_documents(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            stats.indexed += 1
            stats.chunks_created += len(chunks)

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
    ) -> None:
        """Index a file using text extraction + chunking + text embedding."""
        file_path_str = str(file_info.path)

        text = extract_text(file_info.path)
        if not text.strip():
            stats.skipped += 1
            return

        content_hash = hashlib.sha256(text.encode()).hexdigest()

        if existing.get(file_path_str) == content_hash:
            stats.skipped += 1
            return

        self._store.delete_by_file_path(file_path_str)

        chunks = chunk_text(text)
        if not chunks:
            stats.skipped += 1
            return

        embeddings = self._embedder.embed_batch(chunks)

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

        self._store.add_documents(
            ids=ids,
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        stats.indexed += 1
        stats.chunks_created += len(chunks)
