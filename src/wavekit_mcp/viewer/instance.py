"""ViewerInstance - manages a Surfer process and WCP connection.

This module provides the ViewerInstance class which is responsible for:
- Starting and stopping the Surfer process
- Managing the WCP connection
- Syncing state between Proxy and Surfer
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .items import DividerItem, DisplayItem, GroupItem, MarkerItem, WaveformItem
from .wcp_client import WcpClient, WcpError
from .vcd_writer import generate_merged_vcd

logger = logging.getLogger(__name__)


@dataclass
class ViewerConfig:
    """Configuration for a ViewerInstance."""
    surfer_path: str | None = None  # Path to surfer binary, None = find in PATH
    wcp_port: int = 0  # 0 = auto-assign
    http_port: int = 0  # 0 = auto-assign
    headless: bool = True  # Run without GUI


class ViewerInstance:
    """
    Manages a single Surfer viewer instance.

    Each instance:
    - Runs a Surfer subprocess
    - Maintains a WCP connection
    - Handles state sync with ViewerProxy
    """

    def __init__(self, viewer_id: str, config: ViewerConfig | None = None):
        self.viewer_id = viewer_id
        self.config = config or ViewerConfig()

        # Process management
        self._process: subprocess.Popen | None = None
        self._wcp: WcpClient | None = None
        self._url: str = ""

        # State
        self._vcd_path: str | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> str:
        """
        Start the Surfer process and establish WCP connection.

        Returns:
            The HTTP URL for accessing Surfer
        """
        if self._process is not None:
            return self._url

        # Find surfer binary
        surfer_path = self.config.surfer_path
        if surfer_path is None:
            surfer_path = shutil.which("surfer")
            if surfer_path is None:
                raise RuntimeError(
                    "Surfer binary not found in PATH. "
                    "Install surfer or set ViewerConfig.surfer_path"
                )

        # Generate random ports if auto-assign
        import random
        wcp_port = self.config.wcp_port or random.randint(10000, 65000)
        http_port = self.config.http_port or random.randint(10000, 65000)

        # Build command
        cmd = [
            surfer_path,
            "--wcp-port", str(wcp_port),
            "--port", str(http_port),
        ]
        if self.config.headless:
            cmd.append("--headless")

        logger.info(f"Starting Surfer: {' '.join(cmd)}")

        # Start process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )

        # Wait for Surfer to start and listen
        await asyncio.sleep(1.0)  # Give it time to start

        # Check if process is still running
        if self._process.poll() is not None:
            stdout, stderr = self._process.communicate()
            raise RuntimeError(
                f"Surfer process exited immediately. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

        # Connect via WCP
        self._wcp = WcpClient("localhost", wcp_port)

        # Retry connection a few times
        max_retries = 10
        for i in range(max_retries):
            try:
                await self._wcp.connect()
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise RuntimeError(f"Failed to connect to Surfer WCP: {e}")
                await asyncio.sleep(0.5)

        self._url = f"http://localhost:{http_port}"
        logger.info(f"Surfer started: {self._url}")

        return self._url

    async def stop(self) -> None:
        """Stop the Surfer process and close WCP connection."""
        # Close WCP connection
        if self._wcp:
            try:
                await self._wcp.shutdown()
                await self._wcp.close()
            except Exception:
                pass
            self._wcp = None

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
            self._process = None

        self._url = ""
        logger.info(f"Viewer {self.viewer_id} stopped")

    @property
    def url(self) -> str:
        return self._url

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # =========================================================================
    # State sync
    # =========================================================================

    async def pull_state(self) -> dict:
        """
        Pull current state from Surfer.

        Returns:
            {
                "top_group": GroupItem,
                "markers": list[MarkerItem],
                "time_range": (start, end),
            }
        """
        if not self._wcp:
            raise RuntimeError("Viewer not started")

        # Get displayed items
        item_ids = await self._wcp.get_item_list()
        items_info = await self._wcp.get_item_info(item_ids) if item_ids else []

        # Build GroupItem tree from items
        top_group = self._build_group_tree(items_info)

        # Get time range from VCD if loaded
        time_range = (0, 0)
        # Note: WCP doesn't have a direct way to get time range,
        # we'd need to parse the VCD or track it from load()

        return {
            "top_group": top_group,
            "markers": [],  # WCP doesn't support querying markers
            "time_range": time_range,
        }

    async def push_state(
        self,
        top_group: GroupItem,
        markers: list[MarkerItem],
    ) -> None:
        """
        Push state to Surfer.

        This will:
        1. Generate a merged VCD from all visible WaveformItems
        2. Load the VCD into Surfer
        3. Add visible signals
        4. Set markers
        """
        if not self._wcp:
            raise RuntimeError("Viewer not started")

        # Get all visible WaveformItems
        visible_waveforms = list(top_group.walk_waveforms(include_hidden=False))

        if not visible_waveforms:
            # Just clear the display
            await self._wcp.clear()
            return

        # Generate merged VCD
        self._vcd_path = generate_merged_vcd(visible_waveforms)
        logger.info(f"Generated VCD: {self._vcd_path}")

        # Clear and load new VCD
        await self._wcp.clear()
        await self._wcp.load(self._vcd_path)

        # Add visible variables (flatten group structure)
        for item in top_group.walk(include_hidden=False):
            if isinstance(item, WaveformItem) and not item.hidden:
                if item.signal_name:
                    ids = await self._wcp.add_variables([item.signal_name])
                    if ids:
                        item.item_id = ids[0]

                        # Set color if specified
                        if item.color:
                            await self._wcp.set_item_color(item.item_id, item.color)

        # Add markers
        if markers:
            marker_infos = [
                {"time": m.time, "name": m.name or ""}
                for m in markers
            ]
            marker_ids = await self._wcp.add_markers(marker_infos)
            for m, mid in zip(markers, marker_ids):
                m.item_id = mid

    # =========================================================================
    # View control
    # =========================================================================

    async def set_cursor(self, timestamp: int) -> None:
        """Set cursor position."""
        if self._wcp:
            await self._wcp.set_cursor(timestamp)

    async def set_viewport_to(self, timestamp: int) -> None:
        """Move viewport center."""
        if self._wcp:
            await self._wcp.set_viewport_to(timestamp)

    async def set_viewport_range(self, start: int, end: int) -> None:
        """Set viewport range."""
        if self._wcp:
            await self._wcp.set_viewport_range(start, end)

    async def zoom_to_fit(self) -> None:
        """Auto-zoom to fit all signals."""
        if self._wcp:
            await self._wcp.zoom_to_fit()

    async def reload(self) -> None:
        """Reload the current VCD file."""
        if self._wcp:
            await self._wcp.reload()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_group_tree(self, items_info: list[dict]) -> GroupItem:
        """
        Build a GroupItem tree from WCP item info.

        This reconstructs the hierarchical structure from flat item list
        by parsing full_name paths.
        """
        top_group = GroupItem(name="top")

        for info in items_info:
            item_type = info.get("type", "Variable")
            name = info.get("name", "")
            item_id = info.get("id")

            if item_type != "Variable":
                # Skip non-Variable items for now
                continue

            # Parse scope from full name
            if '.' in name:
                parts = name.split('.')
                signal_name = parts[-1]
                scope_parts = parts[:-1]
            else:
                signal_name = name
                scope_parts = []

            # Navigate/create scope path
            current_group = top_group
            for scope_part in scope_parts:
                # Find existing child group
                found = None
                for child in current_group.children:
                    if isinstance(child, GroupItem) and child.name == scope_part:
                        found = child
                        break

                if found:
                    current_group = found
                else:
                    # Create new group
                    new_group = GroupItem(name=scope_part)
                    current_group.children.append(new_group)
                    current_group = new_group

            # Create placeholder WaveformItem (no actual data)
            # The actual data would need to be loaded from VCD
            item = WaveformItem(
                item_id=item_id,
                _waveform=None,  # No data loaded yet
            )
            # Store the full name somewhere accessible
            item.display_name = name
            current_group.children.append(item)

        return top_group
