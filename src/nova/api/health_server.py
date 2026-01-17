"""Simple HTTP server for health check endpoints."""

import logging
from typing import Optional

from aiohttp import web

from ..core.health import HealthChecker
from ..core.metrics import get_metrics

logger = logging.getLogger(__name__)


class HealthServer:
    """Institutional-grade HTTP server for health checks and Prometheus metrics."""

    def __init__(
        self,
        health_checker: HealthChecker,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """
        Initialize health check server.

        Args:
            health_checker: HealthChecker instance
            host: Host to bind to
            port: Port to listen on
        """
        self.health_checker = health_checker
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.metrics = get_metrics()

        # Setup routes
        self.app.router.add_get("/health", self.health_handler)
        self.app.router.add_get("/health/live", self.liveness_handler)
        self.app.router.add_get("/health/ready", self.readiness_handler)
        self.app.router.add_get("/metrics", self.metrics_handler)

    async def metrics_handler(self, request: web.Request) -> web.Response:
        """Handle /metrics endpoint for Prometheus scraping."""
        content = self.metrics.get_prometheus_format()
        return web.Response(text=content, content_type="text/plain")

    async def health_handler(self, request: web.Request) -> web.Response:
        """Handle /health endpoint."""
        system_health = await self.health_checker.get_system_health()

        status_code = 200
        if system_health.status.value == "unhealthy":
            status_code = 503
        elif system_health.status.value == "degraded":
            status_code = 200  # Still return 200 for degraded

        return web.json_response(
            {
                "status": system_health.status.value,
                "timestamp": system_health.timestamp.isoformat(),
                "uptime_seconds": system_health.uptime_seconds,
                "components": {
                    name: {
                        "status": comp.status.value,
                        "message": comp.message,
                        "response_time_ms": comp.response_time_ms,
                    }
                    for name, comp in system_health.components.items()
                },
            },
            status=status_code,
        )

    async def liveness_handler(self, request: web.Request) -> web.Response:
        """Handle /health/live endpoint (Kubernetes liveness probe)."""
        # Always return healthy if server is responding
        return web.json_response({"status": "alive"}, status=200)

    async def readiness_handler(self, request: web.Request) -> web.Response:
        """Handle /health/ready endpoint (Kubernetes readiness probe)."""
        system_health = await self.health_checker.get_system_health()

        if system_health.any_unhealthy:
            return web.json_response(
                {"status": "not_ready", "reason": "Some components unhealthy"},
                status=503,
            )

        return web.json_response({"status": "ready"}, status=200)

    async def start(self) -> None:
        """Start the health check server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(f"Health check server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the health check server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Health check server stopped")
