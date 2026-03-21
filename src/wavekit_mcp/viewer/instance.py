"""Viewer - manages a Surfer process.

This module provides the Viewer class for waveform visualization:
- Starting and stopping the Surfer process (GUI or server mode)
- Managing the WCP connection (GUI mode only)
- Syncing state between session and Surfer
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .items import DividerItem, DisplayItem, GroupItem, MarkerItem, MarkerList, WaveformItem
from .wcp_client import WcpClient, WcpError
from .vcd_writer import generate_merged_vcd

logger = logging.getLogger(__name__)

# Error message patterns that indicate no display is available
_NO_DISPLAY_PATTERNS = [
    "neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set",
    "cannot open display",
    "couldn't open display",
    "No protocol specified",
]


@dataclass
class ViewerConfig:
    """Configuration for a Viewer."""
    surfer_path: str | None = None  # Path to surfer binary, None = find in PATH
    surver_path: str | None = None  # Path to surver binary, None = find in PATH


class Viewer:
    """
    Waveform viewer using Surfer.

    Each instance:
    - Runs a Surfer subprocess (GUI mode with WCP, or server mode)
    - In GUI mode: maintains WCP connection for programmatic control
    - In server mode: provides HTTP URL for browser access
    - Provides synchronous API for session code

    Mode selection:
    - Tries GUI mode first (requires display)
    - Falls back to server mode if no display available

    Usage:
        viewer = Viewer()
        url = viewer.start()  # Returns URL
        viewer.top_group.append(waveform)
        viewer.push_state()
        print(viewer.url)  # URL to access viewer
        viewer.close()
    """

    def __init__(self, config: ViewerConfig | None = None):
        self.config = config or ViewerConfig()

        # Process management
        self._process: subprocess.Popen | None = None
        self._wcp: WcpClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mode: Literal["gui", "server"] | None = None
        self._http_port: int = 0

        # State - user modifies these, then calls push_state()
        self.top_group: GroupItem = GroupItem(name="top")
        self.markers: MarkerList = MarkerList()

        # Temporary VCD file (fixed path, overwritten on each push_state)
        self._vcd_path: str | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _find_binary(self, config_path: str | None, binary_name: str) -> str:
        """Find a binary path from config or PATH."""
        if config_path is not None:
            return config_path
        path = shutil.which(binary_name)
        if path is None:
            raise RuntimeError(
                f"{binary_name} binary not found in PATH. "
                f"Install {binary_name} or set ViewerConfig.{binary_name}_path"
            )
        return path

    def start(self) -> str:
        """
        Start the Surfer process.

        Tries GUI mode first (with WCP control), falls back to server mode
        if no display is available.

        Returns:
            The URL to access the viewer:
            - "gui://surfer" for GUI mode (local window)
            - "http://localhost:PORT" for server mode (browser access)
        """
        if self._process is not None:
            return self.url

        # Try GUI mode first
        try:
            return self._start_gui_mode()
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a "no display" error
            if any(pattern in error_msg for pattern in _NO_DISPLAY_PATTERNS):
                logger.info("No display available, falling back to server mode")
                return self._start_server_mode()
            else:
                raise

    def _start_gui_mode(self) -> str:
        """Start Surfer in GUI mode with WCP connection."""
        surfer_path = self._find_binary(self.config.surfer_path, "surfer")

        # Generate random port for WCP
        wcp_port = random.randint(10000, 65000)

        # Build command
        cmd = [surfer_path]

        # Prepare environment with WCP autostart config
        env = os.environ.copy()
        env["SURFER_WCP_AUTOSTART"] = "true"
        env["SURFER_WCP_ADDRESS"] = f"127.0.0.1:{wcp_port}"

        logger.info("Starting Surfer GUI with WCP on port %d", wcp_port)

        # Start process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=env,
        )

        # Create event loop for async operations
        self._loop = asyncio.new_event_loop()

        # Wait for Surfer to start
        time.sleep(1.0)

        # Check if process is still running
        if self._process.poll() is not None:
            stdout, stderr = self._process.communicate()
            self._loop.close()
            self._loop = None
            raise RuntimeError(
                f"Surfer process exited immediately. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

        # Connect via WCP
        self._wcp = WcpClient("localhost", wcp_port)

        # Retry connection
        max_retries = 10
        for i in range(max_retries):
            try:
                self._loop.run_until_complete(self._wcp.connect())
                break
            except Exception as e:
                if i == max_retries - 1:
                    self._loop.close()
                    self._loop = None
                    raise RuntimeError(f"Failed to connect to Surfer WCP: {e}")
                time.sleep(0.5)

        self._mode = "gui"
        logger.info("Surfer started in GUI mode with WCP on port %d", wcp_port)

        # Create temp VCD file path
        fd, self._vcd_path = tempfile.mkstemp(suffix=".vcd", prefix="viewer_")
        os.close(fd)

        return self.url

    def _start_server_mode(self) -> str:
        """Start Surfer in server mode (surver) for browser access."""
        surver_path = self._find_binary(self.config.surver_path, "surver")

        # Create temp VCD file first (surver needs a file to load)
        fd, self._vcd_path = tempfile.mkstemp(suffix=".vcd", prefix="viewer_")
        os.close(fd)

        # Generate a minimal valid VCD (surver can't load empty file)
        self._write_minimal_vcd(self._vcd_path)

        # Try multiple ports in case of collision
        max_port_tries = 10
        for port_try in range(max_port_tries):
            self._http_port = random.randint(10000, 65000)

            # Build command - surver loads VCD at startup
            cmd = [
                surver_path,
                "--port", str(self._http_port),
                "--bind-address", "127.0.0.1",
                self._vcd_path,
            ]

            logger.info("Starting Surver on port %d (attempt %d/%d)",
                       self._http_port, port_try + 1, max_port_tries)

            # Start process
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )

            # Wait for server to start
            time.sleep(0.5)

            # Check if process is still running
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                stderr_text = stderr.decode()

                # If port in use, try another port
                if "Address already in use" in stderr_text:
                    logger.debug("Port %d in use, trying another", self._http_port)
                    continue

                raise RuntimeError(
                    f"Surver process exited immediately. "
                    f"stdout: {stdout.decode()}, stderr: {stderr_text}"
                )

            # Success!
            break
        else:
            raise RuntimeError("Failed to find available port for surver")

        self._mode = "server"
        logger.info("Surver started at %s", self.url)

        return self.url

    def close(self) -> None:
        """Close the viewer and release resources."""
        # Close WCP connection (GUI mode only)
        if self._wcp and self._loop:
            try:
                self._loop.run_until_complete(self._wcp.shutdown())
                self._loop.run_until_complete(self._wcp.close())
            except Exception as e:
                logger.warning("Error closing WCP connection: %s", e)
            self._wcp = None

        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.warning("Error terminating Surfer process: %s", e)
            self._process = None

        # Close event loop
        if self._loop:
            self._loop.close()
            self._loop = None

        # Clean up temp VCD file
        if self._vcd_path:
            try:
                os.unlink(self._vcd_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Error removing temp VCD: %s", e)
            self._vcd_path = None

        self._mode = None
        logger.info("Viewer closed")

    @property
    def url(self) -> str:
        """The viewer URL (computed from mode and port)."""
        if self._mode == "gui":
            return "gui://surfer"
        elif self._mode == "server":
            return f"http://localhost:{self._http_port}"
        return ""

    @property
    def mode(self) -> str | None:
        """Current mode: 'gui', 'server', or None if not started."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """Check if the viewer is running."""
        return self._process is not None and self._process.poll() is None

    # =========================================================================
    # State sync
    # =========================================================================

    def pull_state(self) -> None:
        """
        Pull current state from Surfer into top_group and markers.

        Note: Only supported in GUI mode.
        """
        if self._mode == "server":
            raise RuntimeError("pull_state() is not supported in server mode")

        if not self._wcp:
            raise RuntimeError("Viewer not started")

        # Get displayed items
        item_ids = self._loop.run_until_complete(self._wcp.get_item_list())
        items_info = self._loop.run_until_complete(
            self._wcp.get_item_info(item_ids)
        ) if item_ids else []

        # Build GroupItem tree from items
        self.top_group = self._build_group_tree(items_info)
        self.markers = MarkerList()  # WCP doesn't support querying markers

    def push_state(self) -> None:
        """
        Push current state to Surfer.

        This sends top_group and markers to Surfer for display.
        Call this after modifying top_group or markers.
        """
        if not self._vcd_path:
            raise RuntimeError("Viewer not started")

        # Get all visible WaveformItems
        visible_waveforms = list(self.top_group.walk_waveforms(include_hidden=False))

        if not visible_waveforms:
            # Just clear the display (GUI mode) or do nothing (server mode)
            if self._wcp:
                self._loop.run_until_complete(self._wcp.clear())
            return

        # Generate merged VCD (overwrites same file)
        generate_merged_vcd(visible_waveforms, self._vcd_path)
        logger.debug("Generated VCD: %s", self._vcd_path)

        if self._mode == "gui" and self._wcp:
            # GUI mode: use WCP to load and add variables
            self._loop.run_until_complete(self._wcp.clear())
            self._loop.run_until_complete(self._wcp.load(self._vcd_path))

            # Add visible variables
            for item in self.top_group.walk(include_hidden=False):
                if isinstance(item, WaveformItem) and not item.hidden:
                    if item.signal_name:
                        ids = self._loop.run_until_complete(
                            self._wcp.add_variables([item.signal_name])
                        )
                        if ids:
                            item.item_id = ids[0]

                            # Set color if specified
                            if item.color:
                                self._loop.run_until_complete(
                                    self._wcp.set_item_color(item.item_id, item.color)
                                )

            # Add markers
            markers_list = self.markers.to_list()
            if markers_list:
                marker_infos = [
                    {"time": m.time, "name": m.name or ""}
                    for m in markers_list
                ]
                marker_ids = self._loop.run_until_complete(
                    self._wcp.add_markers(marker_infos)
                )
                for m, mid in zip(markers_list, marker_ids):
                    m.item_id = mid

        elif self._mode == "server":
            # Server mode: trigger reload via HTTP API
            self._reload_server()

    def _write_minimal_vcd(self, path: str) -> None:
        """Write a minimal valid VCD file (placeholder until push_state)."""
        vcd_content = """$timescale 1ns $end
$var wire 1 ! dummy $end
$enddefinitions $end
#0
$dumpvars
0!
$end
"""
        with open(path, 'w') as f:
            f.write(vcd_content)

    def _reload_server(self) -> None:
        """Trigger VCD reload in server mode via HTTP API."""
        if not self._http_port:
            return

        reload_url = f"http://localhost:{self._http_port}/0/reload"
        try:
            req = urllib.request.Request(reload_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.debug("Surver reloaded VCD")
                elif response.status == 304:
                    logger.debug("Surver: VCD unchanged")
                else:
                    logger.warning("Surver reload returned status %d", response.status)
        except urllib.error.URLError as e:
            logger.warning("Failed to reload Surver: %s", e)

    # =========================================================================
    # View control (GUI mode only)
    # =========================================================================

    def set_cursor(self, timestamp: int) -> None:
        """Set cursor position (GUI mode only)."""
        if self._wcp:
            self._loop.run_until_complete(self._wcp.set_cursor(timestamp))

    def set_viewport_to(self, timestamp: int) -> None:
        """Move viewport center (GUI mode only)."""
        if self._wcp:
            self._loop.run_until_complete(self._wcp.set_viewport_to(timestamp))

    def set_viewport_range(self, start: int, end: int) -> None:
        """Set viewport range (GUI mode only)."""
        if self._wcp:
            self._loop.run_until_complete(self._wcp.set_viewport_range(start, end))

    def zoom_to_fit(self) -> None:
        """Auto-zoom to fit all signals (GUI mode only)."""
        if self._wcp:
            self._loop.run_until_complete(self._wcp.zoom_to_fit())

    def reload(self) -> None:
        """Reload the current VCD file."""
        if self._mode == "server":
            self._reload_server()
        elif self._wcp:
            self._loop.run_until_complete(self._wcp.reload())

    def focus(self, item: DisplayItem) -> None:
        """Focus (scroll to) an item in the viewer (GUI mode only)."""
        if self._mode == "server":
            raise RuntimeError("focus() is not supported in server mode")
        if not self._wcp:
            raise RuntimeError("Viewer not started")
        if item.item_id is None:
            raise ValueError("Item not yet added to Surfer (call push_state first)")
        self._loop.run_until_complete(self._wcp.focus_item(item.item_id))

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_group_tree(self, items_info: list[dict]) -> GroupItem:
        """Build a GroupItem tree from WCP item info."""
        top_group = GroupItem(name="top")

        for info in items_info:
            item_type = info.get("type", "Variable")
            name = info.get("name", "")
            item_id = info.get("id")

            if item_type != "Variable":
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
                found = None
                for child in current_group.children:
                    if isinstance(child, GroupItem) and child.name == scope_part:
                        found = child
                        break

                if found:
                    current_group = found
                else:
                    new_group = GroupItem(name=scope_part)
                    current_group.children.append(new_group)
                    current_group = new_group

            # Create placeholder WaveformItem
            item = WaveformItem(item_id=item_id, _waveform=None)
            item.display_name = name
            current_group.children.append(item)

        return top_group

    def __repr__(self) -> str:
        if self._mode == "gui":
            return "<Viewer (GUI mode)>"
        elif self._mode == "server":
            return f"<Viewer (server mode, {self.url})>"
        return "<Viewer (not started)>"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


# Backward compatibility alias
ViewerInstance = Viewer
