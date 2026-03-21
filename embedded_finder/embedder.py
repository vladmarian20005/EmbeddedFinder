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


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate-limit / quota error."""
    error_str = str(error).lower()
    return "rate" in error_str or "429" in error_str or "quota" in error_str


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

    def _retry_embed(self, fn, error_prefix: str):
        """Execute an embedding call with retry on rate-limit errors.

        Args:
            fn: Callable that performs the actual embedding API call.
            error_prefix: Prefix for error messages (e.g., "Embedding failed").

        Returns:
            The result of fn().
        """
        retries = 0
        while retries <= DEFAULT_MAX_RETRIES:
            try:
                return fn()
            except Exception as e:
                if _is_rate_limit_error(e):
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
                    raise EmbeddingError(f"{error_prefix}: {e}")

    def _embed_config(self, task_type: str | None):
        """Build an EmbedContentConfig if task_type is set, else None."""
        if not task_type:
            return None
        return _get_types().EmbedContentConfig(task_type=task_type)

    def embed_text(self, text: str, task_type: str | None = None) -> list[float]:
        """Generate an embedding for a single text."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        config = self._embed_config(task_type)

        def _do_embed():
            kwargs = {"model": self._model, "contents": text}
            if config is not None:
                kwargs["config"] = config
            result = self._client.models.embed_content(**kwargs)
            return list(result.embeddings[0].values)

        return self._retry_embed(_do_embed, "Embedding failed")

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        task_type: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with batching."""
        if not texts:
            return []

        config = self._embed_config(task_type)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch = [t if t.strip() else " " for t in batch]

            if self._rate_limiter:
                estimated_tokens = sum(len(t) // 4 for t in batch)
                self._rate_limiter.acquire(estimated_tokens)

            def _do_batch(b=batch):
                kwargs = {"model": self._model, "contents": b}
                if config is not None:
                    kwargs["config"] = config
                result = self._client.models.embed_content(**kwargs)
                return [list(emb.values) for emb in result.embeddings]

            batch_embeddings = self._retry_embed(_do_batch, "Embedding failed")
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_file(self, file_path: str | Path, mime_type: str, task_type: str | None = None) -> list[float]:
        """Embed a file natively using multimodal embedding (images, PDFs, audio, video).

        Sends raw file bytes to the Gemini Embedding 2 model via types.Part.from_bytes().

        Args:
            file_path: Path to the file.
            mime_type: MIME type of the file (e.g. 'image/png', 'application/pdf').
            task_type: Optional task type (e.g. "RETRIEVAL_DOCUMENT").

        Returns:
            Embedding vector as a list of floats.
        """
        path = Path(file_path)
        if not path.exists():
            raise EmbeddingError(f"File not found: {file_path}")

        file_bytes = path.read_bytes()
        if not file_bytes:
            raise EmbeddingError(f"File is empty: {file_path}")

        return self.embed_bytes(file_bytes, mime_type, task_type=task_type)

    def embed_bytes(self, data: bytes, mime_type: str, task_type: str | None = None) -> list[float]:
        """Embed raw bytes using multimodal embedding.

        Same as embed_file but accepts bytes directly — needed for converted images
        and other pre-processed content.

        Args:
            data: Raw bytes of the content.
            mime_type: MIME type of the content.
            task_type: Optional task type (e.g. "RETRIEVAL_DOCUMENT").

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

        config = self._embed_config(task_type)

        def _do_embed():
            _types = _get_types()
            kwargs = {
                "model": self._model,
                "contents": [_types.Part.from_bytes(data=data, mime_type=mime_type)],
            }
            if config is not None:
                kwargs["config"] = config
            result = self._client.models.embed_content(**kwargs)
            return list(result.embeddings[0].values)

        return self._retry_embed(_do_embed, "Embedding file failed")

    def embed_file_via_api(self, file_path: str | Path, mime_type: str, task_type: str | None = None) -> list[float]:
        """Embed a file using the Google Files API for large files.

        Uploads the file, waits for processing, embeds via file reference,
        then cleans up.

        Args:
            file_path: Path to the file.
            mime_type: MIME type of the file.
            task_type: Optional task type (e.g. "RETRIEVAL_DOCUMENT").

        Returns:
            Embedding vector as a list of floats.
        """
        path = Path(file_path)
        if not path.exists():
            raise EmbeddingError(f"File not found: {file_path}")

        config = self._embed_config(task_type)
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

            kwargs = {"model": self._model, "contents": uploaded_file}
            if config is not None:
                kwargs["config"] = config
            result = self._client.models.embed_content(**kwargs)
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

    def embed_multipart(self, parts: list, task_type: str | None = None) -> list[float]:
        """Embed multi-part content (e.g., text + image combined).

        Args:
            parts: List of content parts (strings, Part objects, etc.).
                   Note: Gemini API allows max 6 images per request.
            task_type: Optional task type (e.g. "RETRIEVAL_DOCUMENT").

        Returns:
            Embedding vector as a list of floats.
        """
        if not parts:
            raise EmbeddingError("Cannot embed empty parts list.")

        _types = _get_types()
        content = _types.Content(parts=parts)
        config = self._embed_config(task_type)

        if self._rate_limiter:
            self._rate_limiter.acquire(2000)

        def _do_embed():
            kwargs = {"model": self._model, "contents": content}
            if config is not None:
                kwargs["config"] = config
            result = self._client.models.embed_content(**kwargs)
            return list(result.embeddings[0].values)

        return self._retry_embed(_do_embed, "Multipart embedding failed")

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
