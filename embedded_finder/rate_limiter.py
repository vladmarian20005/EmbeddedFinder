"""Proactive token-bucket rate limiter for Gemini API calls."""

import threading
import time


class TokenBucketRateLimiter:
    """Dual token-bucket rate limiter tracking both TPM and RPM.

    Callers call acquire(estimated_tokens) before making an API request.
    If the bucket is depleted, the caller blocks until capacity is available.
    """

    def __init__(self, tpm: int, rpm: int):
        self._tpm = tpm
        self._rpm = rpm
        self._token_bucket = float(tpm)
        self._request_bucket = float(rpm)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int = 0) -> None:
        """Block until capacity is available, then consume tokens."""
        while True:
            with self._lock:
                self._refill()
                if self._token_bucket >= estimated_tokens and self._request_bucket >= 1.0:
                    self._token_bucket -= estimated_tokens
                    self._request_bucket -= 1.0
                    return
                # Calculate how long to wait
                token_wait = (
                    max(0, (estimated_tokens - self._token_bucket) / (self._tpm / 60.0))
                    if estimated_tokens > 0
                    else 0
                )
                request_wait = max(0, (1.0 - self._request_bucket) / (self._rpm / 60.0))
                wait = max(token_wait, request_wait)
            time.sleep(max(wait, 0.01))

    def _refill(self) -> None:
        """Refill buckets based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._token_bucket = min(
            float(self._tpm),
            self._token_bucket + elapsed * (self._tpm / 60.0),
        )
        self._request_bucket = min(
            float(self._rpm),
            self._request_bucket + elapsed * (self._rpm / 60.0),
        )
