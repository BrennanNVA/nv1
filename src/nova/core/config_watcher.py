"""Configuration hot-reload capability."""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object  # Dummy base class

from .config import Config, load_config

logger = logging.getLogger(__name__)


if WATCHDOG_AVAILABLE:

    class ConfigReloadHandler(FileSystemEventHandler):
        """File system event handler for config file changes."""

        def __init__(self, reload_callback: Callable[[Config], None]) -> None:
            """
            Initialize handler.

            Args:
                reload_callback: Function to call when config changes
            """
            self.reload_callback = reload_callback
            self.last_reload_time = 0.0
            self.debounce_seconds = 1.0  # Debounce rapid file changes

        def on_modified(self, event) -> None:
            """Handle file modification events."""
            if event.is_directory:
                return

            if event.src_path.endswith("config.toml"):
                current_time = time.time()

                # Debounce: ignore changes within debounce period
                if current_time - self.last_reload_time < self.debounce_seconds:
                    return

                self.last_reload_time = current_time
                logger.info(f"Config file changed: {event.src_path}, reloading...")

                try:
                    new_config = load_config(Path(event.src_path))
                    self.reload_callback(new_config)
                    logger.info("Configuration reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload configuration: {e}")

else:
    # Dummy class if watchdog not available
    class ConfigReloadHandler:
        """Dummy handler when watchdog is not available."""

        def __init__(self, reload_callback):
            pass


class ConfigWatcher:
    """Watch for configuration file changes and reload."""

    def __init__(
        self,
        config_path: Path,
        reload_callback: Callable[[Config], None],
        enabled: bool = True,
    ) -> None:
        """
        Initialize config watcher.

        Args:
            config_path: Path to config.toml file
            reload_callback: Function to call when config changes
            enabled: Whether to enable file watching
        """
        self.config_path = config_path
        self.reload_callback = reload_callback
        self.enabled = enabled and WATCHDOG_AVAILABLE
        self.observer: Optional[Observer] = None

    def start(self) -> None:
        """Start watching for config file changes."""
        if not self.enabled:
            if not WATCHDOG_AVAILABLE:
                logger.info("Config watching disabled (watchdog not available)")
            else:
                logger.info("Config watching disabled")
            return

        try:
            self.observer = Observer()
            handler = ConfigReloadHandler(self.reload_callback)
            watch_dir = self.config_path.parent
            self.observer.schedule(handler, str(watch_dir), recursive=False)
            self.observer.start()
            logger.info(f"Config watcher started for {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to start config watcher: {e}")

    def stop(self) -> None:
        """Stop watching for config file changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            logger.info("Config watcher stopped")

    def reload_manual(self) -> Optional[Config]:
        """
        Manually reload configuration.

        Returns:
            New Config object, or None if reload failed
        """
        try:
            new_config = load_config(self.config_path)
            self.reload_callback(new_config)
            logger.info("Configuration manually reloaded")
            return new_config
        except Exception as e:
            logger.error(f"Manual config reload failed: {e}")
            return None
