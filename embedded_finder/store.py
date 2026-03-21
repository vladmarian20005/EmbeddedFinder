"""ChromaDB vector store for EmbeddedFinder."""

import json
import logging
from pathlib import Path

import chromadb

from embedded_finder.config import get_db_dir

logger = logging.getLogger(__name__)


class _HashIndex:
    """Lightweight file_path → content_hash index backed by a JSON file.

    Avoids scanning all ChromaDB metadata just to check which files changed.
    """

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, str] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt hash index at %s, rebuilding", self._path)
                self._data = {}

    def _save(self):
        self._path.write_text(
            json.dumps(self._data, separators=(",", ":")),
            encoding="utf-8",
        )

    def get_all(self) -> dict[str, str]:
        return dict(self._data)

    def set(self, file_path: str, content_hash: str):
        self._data[file_path] = content_hash
        self._save()

    def delete(self, file_path: str):
        if file_path in self._data:
            del self._data[file_path]
            self._save()

    def clear(self):
        self._data = {}
        if self._path.exists():
            self._path.unlink()


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    COLLECTION_NAME = "embedded_finder"

    def __init__(self, persist_dir: str | Path | None = None):
        db_path = Path(persist_dir or get_db_dir())
        db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._hash_index = _HashIndex(db_path / "_hash_index.dat")

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents with their embeddings to the store."""
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        # Update hash index from metadata
        if metadatas:
            for meta in metadatas:
                fp = meta.get("file_path")
                ch = meta.get("content_hash")
                if fp and ch:
                    self._hash_index.set(fp, ch)

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> dict:
        """Query the store for similar documents.

        Args:
            embedding: Query embedding vector.
            n_results: Max results to return.
            where: Optional ChromaDB metadata filter (e.g. {"embed_mode": "native"}).

        Returns:
            ChromaDB query result dict with keys: ids, documents, metadatas, distances
        """
        count = self._collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        n = min(n_results, count)
        kwargs = {"query_embeddings": [embedding], "n_results": n}
        if where is not None:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete_document(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self._collection.delete(ids=[doc_id])

    def delete_by_file_path(self, file_path: str) -> None:
        """Delete all chunks for a given file path."""
        results = self._collection.get(
            where={"file_path": file_path},
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
        self._hash_index.delete(file_path)

    def get_indexed_files(self) -> dict[str, str]:
        """Get a mapping of file_path -> content_hash for all indexed files.

        Uses a fast JSON hash index. Falls back to a ChromaDB metadata scan
        (and rebuilds the index) on first call after upgrading from an older version.
        """
        cached = self._hash_index.get_all()
        if cached:
            return cached

        # Fallback: rebuild from ChromaDB metadata (migration path)
        if self._collection.count() > 0:
            results = self._collection.get(include=["metadatas"])
            for meta in (results.get("metadatas") or []):
                if meta and "file_path" in meta and "content_hash" in meta:
                    self._hash_index.set(meta["file_path"], meta["content_hash"])
            return self._hash_index.get_all()

        return {}

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the store."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._hash_index.clear()
