"""Google Gemini embedding integration for EmbeddedFinder."""

import time

from embedded_finder.config import DEFAULT_EMBEDDING_MODEL, get_api_key


class EmbeddingError(Exception):
    """Raised when embedding fails."""
    pass


def _create_client(api_key: str):
    """Create a Google GenAI client. Isolated for testability."""
    from google import genai
    return genai.Client(api_key=api_key)


class Embedder:
    """Wrapper around Google's Gemini embedding API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        client=None,
    ):
        key = api_key or get_api_key()
        if not key and client is None:
            raise EmbeddingError(
                "No API key provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = client if client is not None else _create_client(key)
        self._model = model

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Generate embeddings for multiple texts with batching."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t if t.strip() else " " for t in batch]

            retries = 0
            max_retries = 3
            while retries <= max_retries:
                try:
                    result = self._client.models.embed_content(
                        model=self._model,
                        contents=batch,
                    )
                    for embedding in result.embeddings:
                        all_embeddings.append(list(embedding.values))
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate" in error_str or "429" in error_str or "quota" in error_str:
                        retries += 1
                        if retries > max_retries:
                            raise EmbeddingError(
                                f"Rate limit exceeded after {max_retries} retries: {e}"
                            )
                        time.sleep(2 ** retries)
                    else:
                        raise EmbeddingError(f"Embedding failed: {e}")

        return all_embeddings
