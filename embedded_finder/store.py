"""ChromaDB vector store for EmbeddedFinder."""

from pathlib import Path

import chromadb

from embedded_finder.config import get_db_dir


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    COLLECTION_NAME = "embedded_finder"

    def __init__(self, persist_dir: str | Path | None = None):
        db_path = str(persist_dir or get_db_dir())
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

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

    def query(
        self,
        embedding: list[float],
        n_results: int = 10,
    ) -> dict:
        """Query the store for similar documents.

        Returns:
            ChromaDB query result dict with keys: ids, documents, metadatas, distances
        """
        count = self._collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        n = min(n_results, count)
        return self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
        )

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

    def get_indexed_files(self) -> dict[str, str]:
        """Get a mapping of file_path -> content_hash for all indexed files."""
        results = self._collection.get(include=["metadatas"])
        file_hashes: dict[str, str] = {}
        for meta in (results.get("metadatas") or []):
            if meta and "file_path" in meta and "content_hash" in meta:
                file_hashes[meta["file_path"]] = meta["content_hash"]
        return file_hashes

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
