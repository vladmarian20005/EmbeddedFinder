"""Semantic search engine for EmbeddedFinder."""

from dataclasses import dataclass

from embedded_finder.embedder import Embedder
from embedded_finder.store import VectorStore


@dataclass
class SearchResult:
    """A single search result."""
    file_path: str
    file_name: str
    score: float
    snippet: str
    file_extension: str
    file_size: int
    chunk_index: int = 0
    total_chunks: int = 1


class SearchEngine:
    """Semantic search over indexed files."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    def search(
        self,
        query: str,
        n_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for files matching a natural language query.

        Args:
            query: Natural language description of what to find.
            n_results: Maximum number of results.
            min_score: Minimum similarity score (0-1, where 1 is identical).

        Returns:
            List of SearchResult sorted by relevance (highest first).
        """
        if not query.strip():
            return []

        # Embed the query
        query_embedding = self._embedder.embed_text(query)

        # Query the store (request extra results to allow dedup)
        raw = self._store.query(embedding=query_embedding, n_results=n_results * 3)

        if not raw["ids"][0]:
            return []

        # Convert to SearchResult, deduplicating by file path
        results = []
        seen_files: set[str] = set()

        ids = raw["ids"][0]
        documents = raw["documents"][0] if raw.get("documents") else [""] * len(ids)
        metadatas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(ids)
        distances = raw["distances"][0] if raw.get("distances") else [0.0] * len(ids)

        for i, doc_id in enumerate(ids):
            meta = metadatas[i] or {}
            file_path = meta.get("file_path", doc_id)

            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = 1.0 - (distances[i] / 2.0)

            if score < min_score:
                continue

            # Keep only the best-scoring chunk per file
            if file_path in seen_files:
                continue
            seen_files.add(file_path)

            snippet = (documents[i] or "")[:300]

            results.append(SearchResult(
                file_path=file_path,
                file_name=meta.get("file_name", ""),
                score=round(score, 4),
                snippet=snippet,
                file_extension=meta.get("file_extension", ""),
                file_size=meta.get("file_size", 0),
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 1),
            ))

            if len(results) >= n_results:
                break

        return results
