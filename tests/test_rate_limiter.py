"""Tests for token-bucket rate limiter."""

import threading
import time

import pytest

from embedded_finder.rate_limiter import TokenBucketRateLimiter


def test_acquire_succeeds_when_bucket_full():
    limiter = TokenBucketRateLimiter(tpm=60_000, rpm=60)
    # Should return immediately — bucket is full
    start = time.monotonic()
    limiter.acquire(100)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1


def test_acquire_zero_tokens():
    limiter = TokenBucketRateLimiter(tpm=60_000, rpm=60)
    # acquire(0) should still consume 1 request
    limiter.acquire(0)
    limiter.acquire(0)  # Should still work — plenty of request budget


def test_acquire_consumes_tokens():
    limiter = TokenBucketRateLimiter(tpm=100, rpm=100)
    # Consume all tokens
    limiter.acquire(100)
    # Next acquire should block briefly (needs refill)
    start = time.monotonic()
    limiter.acquire(1)
    elapsed = time.monotonic() - start
    assert elapsed > 0.01  # Had to wait for refill


def test_acquire_consumes_requests():
    limiter = TokenBucketRateLimiter(tpm=1_000_000, rpm=2)
    # Consume both request slots
    limiter.acquire(0)
    limiter.acquire(0)
    # Next acquire should block (needs request bucket refill)
    start = time.monotonic()
    limiter.acquire(0)
    elapsed = time.monotonic() - start
    assert elapsed > 0.01


def test_refill_restores_capacity():
    limiter = TokenBucketRateLimiter(tpm=6000, rpm=60)
    # tpm=6000 → 100 tokens/sec refill
    # Drain tokens
    limiter.acquire(6000)
    # Wait for partial refill (~100 tokens after 1 second)
    time.sleep(0.5)
    # Should be able to acquire a small amount now
    start = time.monotonic()
    limiter.acquire(10)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5  # Shouldn't block long — already waited


def test_bucket_does_not_overfill():
    limiter = TokenBucketRateLimiter(tpm=100, rpm=10)
    # Wait a long time — bucket should cap at tpm
    time.sleep(0.1)
    limiter.acquire(0)  # Trigger refill
    # Access internal state to verify cap
    assert limiter._token_bucket <= 100
    assert limiter._request_bucket <= 10


def test_concurrent_acquire_is_threadsafe():
    limiter = TokenBucketRateLimiter(tpm=1_000_000, rpm=1000)
    results = []
    errors = []

    def worker():
        try:
            for _ in range(10):
                limiter.acquire(10)
            results.append(True)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(errors) == 0
    assert len(results) == 4


def test_large_token_request_blocks_until_available():
    limiter = TokenBucketRateLimiter(tpm=600, rpm=60)
    # tpm=600 → 10 tokens/sec
    # Drain all tokens
    limiter.acquire(600)
    # Request 5 tokens — need 0.5 sec to refill
    start = time.monotonic()
    limiter.acquire(5)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3  # Should have blocked waiting for refill
    assert elapsed < 3.0   # But not forever
