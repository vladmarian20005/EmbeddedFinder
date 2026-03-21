"""Google Gemini embedding integration for EmbeddedFinder."""

import logging
import random
import time
from pathlib import Path

from embedded_finder.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_MAX_RETRIES,
    RETRY_BASE_SECONDS,
    RETRY_MAX_JITTER_SECONDS,
    get_api_key,
)

logger = logging.getLogger(__name__)

# Lazy-loaded; patched in tests
types = None


def _get_types():
    global types
    if types is None:
        from google.genai import types as _types
        types = _types
    return types


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
        rate_limiter=None,
    ):
        key = api_key or get_api_key()
        if not key and client is None:
            raise EmbeddingError(
                "No API key provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = client if client is not None else _create_client(key)
        self._model = model
        self._rate_limiter = rate_limiter

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: list[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> list[list[float]]:
        """Generate embeddings for multiple texts with batching."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t if t.strip() else " " for t in batch]

            if self._rate_limiter:
                estimated_tokens = sum(len(t) // 4 for t in batch)
                self._rate_limiter.acquire(estimated_tokens)

            retries = 0
            while retries <= DEFAULT_MAX_RETRIES:
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
                        if retries > DEFAULT_MAX_RETRIES:
                            raise EmbeddingError(
                                f"Rate limit exceeded after {DEFAULT_MAX_RETRIES} retries: {e}"
                            )
                        time.sleep(
                            RETRY_BASE_SECONDS * (2 ** retries)
                            + random.uniform(0, RETRY_MAX_JITTER_SECONDS)
                        )
                    else:
                        raise EmbeddingError(f"Embedding failed: {e}")

        return all_embeddings

    def embed_file(self, file_path: str | Path, mime_type: str) -> list[float]:
        """Embed a file natively using multimodal embedding (images, PDFs, audio, video).

        Sends raw file bytes to the Gemini Embedding 2 model via types.Part.from_bytes().

        Args:
            file_path: Path to the file.
            mime_type: MIME type of the file (e.g. 'image/png', 'application/pdf').

        Returns:
            Embedding vector as a list of floats.
        """
        path = Path(file_path)
        if not path.exists():
            raise EmbeddingError(f"File not found: {file_path}")

        file_bytes = path.read_bytes()
        if not file_bytes:
            raise EmbeddingError(f"File is empty: {file_path}")

        return self.embed_bytes(file_bytes, mime_type)

    def embed_bytes(self, data: bytes, mime_type: str) -> list[float]:
        """Embed raw bytes using multimodal embedding.

        Same as embed_file but accepts bytes directly — needed for converted images
        and other pre-processed content.

        Args:
            data: Raw bytes of the content.
            mime_type: MIME type of the content.

        Returns:
            Embedding vector as a list of floats.
        """
        if not data:
            raise EmbeddingError("Cannot embed empty data.")

        if self._rate_limiter:
            # Binary files (images/audio/video) don't tokenize by byte count.
            # Use a fixed conservative estimate instead of len(data)//4 which
            # wildly overestimates (a 10MB MP3 is NOT 2.5M tokens).
            self._rate_limiter.acquire(2000)

        retries = 0
        while retries <= DEFAULT_MAX_RETRIES:
            try:
                _types = _get_types()
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=_types.Part.from_bytes(
                        data=data,
                        mime_type=mime_type,
                    ),
                )
                return list(result.embeddings[0].values)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "quota" in error_str:
                    retries += 1
                    if retries > DEFAULT_MAX_RETRIES:
                        raise EmbeddingError(
                            f"Rate limit exceeded after {DEFAULT_MAX_RETRIES} retries: {e}"
                        )
                    time.sleep(
                        RETRY_BASE_SECONDS * (2 ** retries)
                        + random.uniform(0, RETRY_MAX_JITTER_SECONDS)
                    )
                else:
                    raise EmbeddingError(f"Embedding file failed: {e}")

    def embed_file_via_api(self, file_path: str | Path, mime_type: str) -> list[float]:
        """Embed a file using the Google Files API for large files.

        Uploads the file, waits for processing, embeds via file reference,
        then cleans up.

        Args:
            file_path: Path to the file.
            mime_type: MIME type of the file.

        Returns:
            Embedding vector as a list of floats.
        """
        path = Path(file_path)
        if not path.exists():
            raise EmbeddingError(f"File not found: {file_path}")

        uploaded_file = None
        try:
            uploaded_file = self._client.files.upload(
                file=str(path),
                config={"mime_type": mime_type},
            )

            # Poll until the file is active
            max_polls = 30
            for _ in range(max_polls):
                if uploaded_file.state == "ACTIVE":
                    break
                time.sleep(2)
                uploaded_file = self._client.files.get(name=uploaded_file.name)
            else:
                raise EmbeddingError(
                    f"File upload did not become ACTIVE after polling: {file_path}"
                )

            result = self._client.models.embed_content(
                model=self._model,
                contents=uploaded_file,
            )
            return list(result.embeddings[0].values)
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Files API embedding failed: {e}")
        finally:
            if uploaded_file is not None:
                try:
                    self._client.files.delete(name=uploaded_file.name)
                except Exception:
                    logger.warning("Failed to clean up uploaded file: %s", uploaded_file.name)

    def embed_multipart(self, parts: list) -> list[float]:
        """Embed multi-part content (e.g., text + image combined).

        Args:
            parts: List of content parts (strings, Part objects, etc.).

        Returns:
            Embedding vector as a list of floats.
        """
        if not parts:
            raise EmbeddingError("Cannot embed empty parts list.")

        _types = _get_types()
        content = _types.Content(parts=parts)

        if self._rate_limiter:
            self._rate_limiter.acquire(2000)

        retries = 0
        while retries <= DEFAULT_MAX_RETRIES:
            try:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=content,
                )
                return list(result.embeddings[0].values)
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "quota" in error_str:
                    retries += 1
                    if retries > DEFAULT_MAX_RETRIES:
                        raise EmbeddingError(
                            f"Rate limit exceeded after {DEFAULT_MAX_RETRIES} retries: {e}"
                        )
                    time.sleep(
                        RETRY_BASE_SECONDS * (2 ** retries)
                        + random.uniform(0, RETRY_MAX_JITTER_SECONDS)
                    )
                else:
                    raise EmbeddingError(f"Multipart embedding failed: {e}")

    @staticmethod
    def average_embeddings(embeddings: list[list[float]]) -> list[float]:
        """Compute element-wise average of multiple embedding vectors.

        Args:
            embeddings: List of embedding vectors (all same dimension).

        Returns:
            Averaged embedding vector.
        """
        if not embeddings:
            raise EmbeddingError("Cannot average empty embeddings list.")

        dim = len(embeddings[0])
        n = len(embeddings)
        avg = [0.0] * dim
        for emb in embeddings:
            for j in range(dim):
                avg[j] += emb[j]
        for j in range(dim):
            avg[j] /= n
        return avg
