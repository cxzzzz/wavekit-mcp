"""ViewerRegistry - manages Viewer worker processes.

This module provides the ViewerRegistry class which manages all active
viewer worker processes. Each viewer runs in its own subprocess for isolation.

Architecture:
    Main Process (ViewerRegistry)
        - Holds viewer_id → (pipe_endpoint, process) mapping
        - Routes session requests to appropriate viewer worker

    Viewer Worker Process
        - Runs ViewerInstance with Surfer subprocess
        - Handles WCP communication

    Session Worker Process
        - Gets viewer pipe from main process
        - Communicates directly with viewer worker via pipe
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .instance import ViewerConfig

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class ViewerWorkerProxy:
    """
    Proxy to a viewer worker process running in isolation.

    This class manages the lifecycle of a viewer worker subprocess
    and provides communication via multiprocessing.Pipe.
    """

    def __init__(self, viewer_id: str, config: Config):
        self.viewer_id = viewer_id
        self.config = config
        self.created_at = datetime.now().isoformat()

        # Create stderr log file for crash diagnostics
        self._stderr_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        )
        self._stderr_path = self._stderr_file.name
        self._stderr_file.close()

        # Create pipe for bidirectional communication
        self._parent_conn, child_conn = get_context("spawn").Pipe()

        # Spawn worker process
        from .worker import viewer_worker_main

        self._process = get_context("spawn").Process(
            target=viewer_worker_main,
            args=(child_conn, viewer_id, config, self._stderr_path),
            name=f"viewer-{viewer_id}",
            daemon=True,
        )
        self._process.start()

        # Wait for startup
        self._url: str | None = None
        self._wait_for_startup()

    def _wait_for_startup(self, timeout: float = 30.0) -> None:
        """Wait for the viewer worker to start and report its URL."""
        import time
        start = time.monotonic()

        while time.monotonic() - start < timeout:
            if self._parent_conn.poll(0.1):
                msg = self._parent_conn.recv()
                if msg.get("type") == "started":
                    self._url = msg.get("url")
                    return
                elif msg.get("type") == "error":
                    raise RuntimeError(f"Viewer failed to start: {msg.get('message')}")

            if not self._process.is_alive():
                raise RuntimeError("Viewer worker process died during startup")

        raise TimeoutError(f"Viewer startup timed out after {timeout}s")

    @property
    def url(self) -> str:
        """Get the viewer HTTP URL."""
        if self._url is None:
            raise RuntimeError("Viewer not started")
        return self._url

    @property
    def is_running(self) -> bool:
        """Check if the viewer worker is still running."""
        return self._process.is_alive()

    def close(self) -> None:
        """Close the viewer worker."""
        try:
            self._parent_conn.send({"type": "close"})
            self._parent_conn.poll(timeout=5)
            self._process.join(timeout=5)
        except Exception:
            pass
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
                if self._process.is_alive():
                    self._process.kill()

            try:
                self._parent_conn.close()
            except Exception:
                pass

            try:
                Path(self._stderr_path).unlink(missing_ok=True)
            except Exception:
                pass

    def get_pipe(self) -> Any:
        """Get the pipe connection for this viewer worker."""
        return self._parent_conn


class ViewerRegistry:
    """
    Registry for viewer worker processes.

    This class manages all active viewer workers and provides
    access to their pipe connections for session workers.
    """

    _instance: ViewerRegistry | None = None

    def __init__(self, config: Config):
        """
        Initialize the registry.

        Args:
            config: Server configuration
        """
        self._config = config
        self._viewers: dict[str, ViewerWorkerProxy] = {}
        self._lock = None  # Will be set when needed for async operations

    @classmethod
    def get_instance(cls) -> ViewerRegistry | None:
        """Get the global registry instance."""
        return cls._instance

    @classmethod
    def set_instance(cls, registry: ViewerRegistry) -> None:
        """Set the global registry instance."""
        cls._instance = registry

    async def create_viewer(
        self,
        viewer_id: str | None = None,
        config: ViewerConfig | None = None,
    ) -> tuple[str, ViewerWorkerProxy]:
        """
        Create a new viewer worker.

        Args:
            viewer_id: Optional viewer ID. If None, generates one.
            config: Optional viewer configuration (unused for now)

        Returns:
            Tuple of (viewer_id, ViewerWorkerProxy)
        """
        if viewer_id is None:
            viewer_id = uuid.uuid4().hex[:8]

        if viewer_id in self._viewers:
            raise ValueError(f"Viewer {viewer_id} already exists")

        proxy = ViewerWorkerProxy(viewer_id, self._config)
        self._viewers[viewer_id] = proxy
        logger.info(f"Created viewer worker {viewer_id}: {proxy.url}")

        return viewer_id, proxy

    def get_viewer(self, viewer_id: str) -> ViewerWorkerProxy | None:
        """Get a viewer worker proxy by ID."""
        return self._viewers.get(viewer_id)

    def get_viewer_pipe(self, viewer_id: str) -> Any:
        """
        Get the pipe connection for a viewer worker.

        This is used by session workers to communicate directly
        with viewer workers.
        """
        viewer = self._viewers.get(viewer_id)
        if viewer is None:
            raise KeyError(f"Viewer {viewer_id} not found")
        return viewer.get_pipe()

    def list_viewers(self) -> list[str]:
        """List all viewer IDs."""
        return list(self._viewers.keys())

    async def close_viewer(self, viewer_id: str) -> None:
        """Close and remove a viewer worker."""
        viewer = self._viewers.pop(viewer_id, None)
        if viewer:
            viewer.close()
            logger.info(f"Closed viewer {viewer_id}")

    async def close_all(self) -> None:
        """Close all viewer workers."""
        for viewer_id, viewer in list(self._viewers.items()):
            try:
                viewer.close()
            except Exception as e:
                logger.error(f"Error stopping viewer {viewer_id}: {e}")
        self._viewers.clear()
        logger.info("Closed all viewers")

    async def get_or_create(self, viewer_id: str | None = None) -> tuple[str, ViewerWorkerProxy]:
        """
        Get an existing viewer or create a new one.

        Args:
            viewer_id: Optional viewer ID. If None, gets or creates a default viewer.

        Returns:
            Tuple of (viewer_id, ViewerWorkerProxy)
        """
        if viewer_id:
            viewer = self._viewers.get(viewer_id)
            if viewer:
                return viewer_id, viewer
            return await self.create_viewer(viewer_id)

        # No ID specified - get or create a single default viewer
        if self._viewers:
            viewer_id = next(iter(self._viewers))
            return viewer_id, self._viewers[viewer_id]

        return await self.create_viewer()
