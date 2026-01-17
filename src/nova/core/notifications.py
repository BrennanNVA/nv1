"""Async Discord webhook notifications."""

import asyncio
import logging
from typing import Optional

from discord_webhook import AsyncDiscordWebhook, DiscordEmbed

from .config import NotificationConfig

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending notifications via Discord webhooks."""

    def __init__(self, config: NotificationConfig) -> None:
        """
        Initialize notification service.

        Args:
            config: Notification configuration
        """
        self.config = config
        self.enabled = config.discord_enabled and bool(config.discord_webhook_url)

        if self.enabled:
            logger.info("Discord notifications enabled")
        else:
            logger.warning("Discord notifications disabled or webhook URL not configured")

    async def send_message(
        self,
        message: str,
        title: Optional[str] = None,
        color: Optional[int] = None,
        level: str = "INFO",
    ) -> bool:
        """
        Send a message to Discord.

        Args:
            message: Message content
            title: Optional title
            color: Optional color (0xRRGGBB format)
            level: Message level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        # Check if level should be sent
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        config_level = level_map.get(self.config.notification_level, logging.INFO)
        message_level = level_map.get(level, logging.INFO)

        if message_level < config_level:
            return False

        try:
            webhook = AsyncDiscordWebhook(url=self.config.discord_webhook_url)

            if title:
                embed = DiscordEmbed(
                    title=title, description=message, color=color or self._get_color(level)
                )
                webhook.add_embed(embed)
            else:
                webhook.content = message

            await webhook.execute()
            logger.debug(f"Discord notification sent: {title or message[:50]}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    async def send_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[dict] = None,
    ) -> bool:
        """
        Send an alert notification.

        Args:
            alert_type: Type of alert (TRADE, ERROR, CIRCUIT_BREAKER, etc.)
            message: Alert message
            details: Optional additional details

        Returns:
            True if sent successfully, False otherwise
        """
        title = f"🚨 {alert_type} Alert"
        color_map = {
            "TRADE": 0x00FF00,  # Green
            "ERROR": 0xFF0000,  # Red
            "CIRCUIT_BREAKER": 0xFF4500,  # Orange Red
            "WARNING": 0xFFFF00,  # Yellow
            "INFO": 0x0099FF,  # Blue
        }
        color = color_map.get(alert_type, 0xFFFFFF)

        if details:
            message += "\n\n**Details:**\n"
            for key, value in details.items():
                message += f"- {key}: {value}\n"

        return await self.send_message(message, title=title, color=color, level="WARNING")

    def _get_color(self, level: str) -> int:
        """Get color code for log level."""
        color_map = {
            "DEBUG": 0x808080,  # Gray
            "INFO": 0x0099FF,  # Blue
            "WARNING": 0xFFFF00,  # Yellow
            "ERROR": 0xFF0000,  # Red
            "CRITICAL": 0xFF4500,  # Orange Red
        }
        return color_map.get(level, 0xFFFFFF)

    def send_message_sync(
        self,
        message: str,
        title: Optional[str] = None,
        color: Optional[int] = None,
        level: str = "INFO",
    ) -> bool:
        """
        Synchronous wrapper for send_message.

        Args:
            message: Message content
            title: Optional title
            color: Optional color
            level: Message level

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.send_message(message, title=title, color=color, level=level)
        )
