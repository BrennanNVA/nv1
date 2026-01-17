"""Structured logging with file and console outputs, including circuit breaker integration."""

import logging
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class CircuitBreaker:
    """Circuit breaker to halt system on excessive errors."""

    def __init__(
        self,
        max_errors: int = 5,
        time_window: int = 60,
        halt_on_breach: bool = True,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            max_errors: Maximum number of errors allowed in time window
            time_window: Time window in seconds
            halt_on_breach: Whether to halt system on breach
        """
        self.max_errors = max_errors
        self.time_window = timedelta(seconds=time_window)
        self.halt_on_breach = halt_on_breach
        self.errors: deque[datetime] = deque()
        self.is_open: bool = False

    def record_error(self) -> None:
        """Record an error occurrence."""
        now = datetime.now()
        self.errors.append(now)

        # Remove errors outside time window
        cutoff = now - self.time_window
        while self.errors and self.errors[0] < cutoff:
            self.errors.popleft()

        # Check if circuit should open
        if len(self.errors) >= self.max_errors:
            self.is_open = True
            if self.halt_on_breach:
                raise SystemExit(
                    f"Circuit breaker opened: {len(self.errors)} errors in {self.time_window.total_seconds()}s"
                )

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.errors.clear()
        self.is_open = False

    def can_proceed(self) -> bool:
        """Check if system can proceed (circuit is closed)."""
        # Clean old errors
        now = datetime.now()
        cutoff = now - self.time_window
        while self.errors and self.errors[0] < cutoff:
            self.errors.popleft()

        if len(self.errors) >= self.max_errors:
            self.is_open = True
            return False

        self.is_open = False
        return True


# Global circuit breaker instance
_circuit_breaker: Optional[CircuitBreaker] = None


def setup_logger(
    name: str = "nova_aetus",
    log_dir: Path = Path("logs"),
    level: str = "INFO",
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> logging.Logger:
    """
    Set up structured logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        circuit_breaker: Optional circuit breaker instance

    Returns:
        Configured logger instance
    """
    global _circuit_breaker
    _circuit_breaker = circuit_breaker

    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "nova_aetus") -> logging.Logger:
    """
    Get logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class CircuitBreakerHandler(logging.Handler):
    """Custom logging handler that integrates with circuit breaker."""

    def __init__(self, circuit_breaker: CircuitBreaker, level: int = logging.ERROR) -> None:
        """
        Initialize handler.

        Args:
            circuit_breaker: Circuit breaker instance
            level: Minimum level to trigger circuit breaker
        """
        super().__init__(level)
        self.circuit_breaker = circuit_breaker

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record and trigger circuit breaker for errors."""
        if record.levelno >= logging.ERROR:
            try:
                self.circuit_breaker.record_error()
            except SystemExit:
                # Circuit breaker halted system, propagate
                raise
