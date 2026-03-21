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
    embed_mode: str = "text"  # "text", "native", or "native_chunked"


class SearchEngine:
    """Semantic search over indexed files."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self._embedder = embedder
        self._store = store

    @staticmethod
    def _merge_raw_results(
        *raw_results: dict,
    ) -> tuple[list, list, list, list]:
        """Merge multiple ChromaDB query results, deduplicating by ID."""
        seen_ids: set[str] = set()
        ids, documents, metadatas, distances = [], [], [], []

        for raw in raw_results:
            if not raw["ids"][0]:
                continue
            r_ids = raw["ids"][0]
            r_docs = raw["documents"][0] if raw.get("documents") else [""] * len(r_ids)
            r_metas = raw["metadatas"][0] if raw.get("metadatas") else [{}] * len(r_ids)
            r_dists = raw["distances"][0] if raw.get("distances") else [0.0] * len(r_ids)

            for i, doc_id in enumerate(r_ids):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    ids.append(doc_id)
                    documents.append(r_docs[i])
                    metadatas.append(r_metas[i])
                    distances.append(r_dists[i])

        return ids, documents, metadatas, distances

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
        query_embedding = self._embedder.embed_text(query, task_type="RETRIEVAL_QUERY")

        # Two-pass retrieval: general query + targeted native-embed query.
        # Native embeddings (images/audio/video) score systematically lower in
        # raw cosine similarity against text queries. Without the second pass,
        # they can be entirely absent from the candidate pool, and the ranker
        # can't boost results it never sees.
        raw = self._store.query(embedding=query_embedding, n_results=n_results * 5)

        # Second pass: retrieve ALL native-embed results. Native files (images,
        # audio, video) are typically a small fraction of the index, but their
        # raw cosine scores are systematically lower than text. Retrieving all
        # ensures the ranker can evaluate every media file.
        native_raw = self._store.query(
            embedding=query_embedding,
            n_results=self._store.count(),
            where={"embed_mode": {"$in": ["native", "native_chunked"]}},
        )

        # Merge both result sets (dedup by ID happens below via seen_files)
        ids, documents, metadatas, distances = self._merge_raw_results(raw, native_raw)

        if not ids:
            return []

        # Convert to SearchResult, deduplicating by file path
        results = []
        seen_files: set[str] = set()

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
                embed_mode=meta.get("embed_mode", "text"),
            ))

        return results
