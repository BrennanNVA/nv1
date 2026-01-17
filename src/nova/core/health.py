"""Health check system for monitoring system status."""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status of a component."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    last_check: Optional[datetime] = None
    response_time_ms: Optional[float] = None


class SystemHealth(BaseModel):
    """Overall system health."""

    status: HealthStatus
    timestamp: datetime
    components: dict[str, ComponentHealth]
    uptime_seconds: Optional[float] = None

    @property
    def all_healthy(self) -> bool:
        """Check if all components are healthy."""
        return all(comp.status == HealthStatus.HEALTHY for comp in self.components.values())

    @property
    def any_unhealthy(self) -> bool:
        """Check if any component is unhealthy."""
        return any(comp.status == HealthStatus.UNHEALTHY for comp in self.components.values())


class HealthChecker:
    """Health check service."""

    def __init__(self, start_time: Optional[datetime] = None) -> None:
        """
        Initialize health checker.

        Args:
            start_time: System start time for uptime calculation
        """
        self.start_time = start_time or datetime.now()
        self.components: dict[str, ComponentHealth] = {}

    async def check_component(
        self,
        name: str,
        check_func: callable,
        timeout: float = 5.0,
    ) -> ComponentHealth:
        """
        Check health of a component.

        Args:
            name: Component name
            check_func: Async function that returns (bool, Optional[str])
            timeout: Timeout in seconds

        Returns:
            ComponentHealth object
        """
        start = datetime.now()

        try:
            result = await asyncio.wait_for(check_func(), timeout=timeout)
            if isinstance(result, tuple):
                is_healthy, message = result
            else:
                is_healthy = bool(result)
                message = None

            elapsed_ms = (datetime.now() - start).total_seconds() * 1000

            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY

            health = ComponentHealth(
                name=name,
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {timeout}s",
                last_check=datetime.now(),
            )
        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                last_check=datetime.now(),
            )

        self.components[name] = health
        return health

    async def check_database(self, storage_service) -> tuple[bool, Optional[str]]:
        """Check database health."""
        if not storage_service or not storage_service.pool:
            return False, "Database not connected"

        try:
            async with storage_service.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True, None
        except Exception as e:
            return False, str(e)

    async def check_sentiment(self, sentiment_analyzer) -> tuple[bool, Optional[str]]:
        """Check sentiment analyzer health (Ollama)."""
        if not sentiment_analyzer:
            return False, "Sentiment analyzer not initialized"

        # Simple check: try to list models (lightweight operation)
        try:
            import ollama

            models = ollama.list()
            if models.get("models"):
                return True, None
            return False, "No Ollama models available"
        except Exception as e:
            return False, f"Ollama not accessible: {e}"

    async def get_system_health(
        self,
        storage_service=None,
        sentiment_analyzer=None,
    ) -> SystemHealth:
        """
        Get overall system health.

        Args:
            storage_service: Optional storage service for DB check
            sentiment_analyzer: Optional sentiment analyzer for Ollama check

        Returns:
            SystemHealth object
        """
        # Check all components
        if storage_service:
            await self.check_component(
                "database",
                lambda: self.check_database(storage_service),
            )

        if sentiment_analyzer:
            await self.check_component(
                "sentiment_analyzer",
                lambda: self.check_sentiment(sentiment_analyzer),
            )

        # Determine overall status
        if self.any_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif any(comp.status == HealthStatus.DEGRADED for comp in self.components.values()):
            overall_status = HealthStatus.DEGRADED
        elif self.all_healthy and self.components:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        uptime = (datetime.now() - self.start_time).total_seconds()

        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            components=self.components.copy(),
            uptime_seconds=uptime,
        )


def get_health_endpoint(health_checker: HealthChecker) -> dict:
    """
    Get health check data for HTTP endpoint.

    Args:
        health_checker: HealthChecker instance

    Returns:
        Dictionary with health status (for JSON response)
    """
    # This will be called from an HTTP endpoint
    # For now, return a simple status
    return {
        "status": "ok" if health_checker.all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": {
            name: {
                "status": comp.status.value,
                "message": comp.message,
            }
            for name, comp in health_checker.components.items()
        },
    }
