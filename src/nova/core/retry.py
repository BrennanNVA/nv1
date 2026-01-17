"""Retry logic with exponential backoff and rate limiting."""

import asyncio
import logging
import random
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter to enforce API rate limits."""

    def __init__(
        self,
        max_requests: int,
        time_window: int = 60,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = timedelta(seconds=time_window)
        self.requests: deque[datetime] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        async with self._lock:
            now = datetime.now()

            # Remove requests outside time window
            cutoff = now - self.time_window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # If at limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                oldest_request = self.requests[0]
                wait_until = oldest_request + self.time_window
                wait_seconds = (wait_until - now).total_seconds()

                if wait_seconds > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_seconds:.2f}s")
                    await asyncio.sleep(wait_seconds)
                    # Clean up again after waiting
                    now = datetime.now()
                    cutoff = now - self.time_window
                    while self.requests and self.requests[0] < cutoff:
                        self.requests.popleft()

            # Record this request
            self.requests.append(now)

    def can_make_request(self) -> bool:
        """Check if a request can be made without waiting."""
        now = datetime.now()
        cutoff = now - self.time_window

        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        return len(self.requests) < self.max_requests


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(initial_delay * (exponential_base**attempt), max_delay)

                # Add jitter (±20%)
                if jitter:
                    jitter_amount = delay * 0.2 * (random.random() * 2 - 1)
                    delay = delay + jitter_amount

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                raise

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed with no exception")


class RetryableService:
    """Base class for services that need retry logic and rate limiting."""

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Initialize retryable service.

        Args:
            rate_limiter: Optional rate limiter instance
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
        """
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def _execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> T:
        """
        Execute a function with rate limiting and retry logic.

        Args:
            func: Async function to execute
            retryable_exceptions: Exceptions that should trigger retry

        Returns:
            Result of the function
        """

        async def rate_limited_func() -> T:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            return await func()

        return await retry_with_backoff(
            rate_limited_func,
            max_retries=self.max_retries,
            initial_delay=self.retry_delay,
            retryable_exceptions=retryable_exceptions,
        )


# Fix type hint for Awaitable
from collections.abc import Awaitable
