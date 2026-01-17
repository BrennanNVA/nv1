"""Unit tests for retry and rate limiting."""

from datetime import datetime

import pytest

from src.nova.core.retry import RateLimiter, retry_with_backoff


class TestRateLimiter:
    """Test rate limiter."""

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiter enforces limits."""
        limiter = RateLimiter(max_requests=2, time_window=1)

        # First two requests should go through immediately
        start = datetime.now()
        await limiter.acquire()
        await limiter.acquire()

        # Third request should wait
        await limiter.acquire()
        elapsed = (datetime.now() - start).total_seconds()

        # Should have waited at least some time
        assert elapsed > 0.9  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_can_make_request(self):
        """Test can_make_request check."""
        limiter = RateLimiter(max_requests=1, time_window=60)

        assert limiter.can_make_request() is True
        await limiter.acquire()
        assert limiter.can_make_request() is False


class TestRetryWithBackoff:
    """Test retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call."""

        async def func():
            return "success"

        result = await retry_with_backoff(func, max_retries=3)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await retry_with_backoff(func, max_retries=3, initial_delay=0.1)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that max retries are respected."""

        async def func():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await retry_with_backoff(func, max_retries=2, initial_delay=0.1)
