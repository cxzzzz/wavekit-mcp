"""Viewer - manages a Surfer process with WCP control.

This module provides the Viewer class for waveform visualization:
- Starting and stopping the Surfer process (GUI mode)
- Managing the WCP connection for programmatic control
- Syncing state between session and Surfer
- Fallback to VCD file export when no display is available
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .items import DividerItem, DisplayItem, GroupItem, MarkerItem, MarkerList, WaveformItem
from .wcp_client import WcpClient, WcpError
from .vcd_writer import generate_merged_vcd, get_wcp_signal_name

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
    fallback_dir: str | None = None  # Directory for VCD files in fallback mode


class Viewer:
    """
    Waveform viewer using Surfer.

    Modes:
    - GUI mode: Starts Surfer with WCP for programmatic control (requires display)
    - Fallback mode: Generates VCD files for user to open with any viewer

    Usage:
        viewer = Viewer()
        url = viewer.start()  # Returns "gui://surfer" or "file:///path/to/vcd"
        viewer.top_group.append(waveform)
        viewer.push_state()
        print(viewer.url)
        viewer.close()
    """

    def __init__(self, config: ViewerConfig | None = None):
        self.config = config or ViewerConfig()

        # Process management (GUI mode only)
        self._process: subprocess.Popen | None = None
        self._wcp: WcpClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mode: Literal["gui", "fallback"] | None = None

        # State - user modifies these, then calls push_state()
        self.top_group: GroupItem = GroupItem(name="top")
        self.markers: MarkerList = MarkerList()

        # VCD file path
        self._vcd_path: str | None = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _find_binary(self, binary_name: str) -> str:
        """Find a binary path from config or PATH."""
        config_path = getattr(self.config, f"{binary_name}_path", None)
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
        Start the viewer.

        Tries GUI mode first (with WCP control), falls back to VCD file export
        if no display is available.

        Returns:
            - "gui://surfer" for GUI mode
            - "file:///path/to/viewer.vcd" for fallback mode
        """
        if self._mode is not None:
            return self.url

        # Try GUI mode first
        try:
            return self._start_gui_mode()
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a "no display" error
            if any(pattern in error_msg for pattern in _NO_DISPLAY_PATTERNS):
                logger.info("No display available, falling back to VCD file mode")
                return self._start_fallback_mode()
            else:
                raise

    def _start_gui_mode(self) -> str:
        """Start Surfer in GUI mode with WCP connection."""
        surfer_path = self._find_binary("surfer")

        # Generate random port for WCP
        wcp_port = random.randint(10000, 65000)

        # Create temporary config directory for surfer
        # Surfer reads config from $XDG_CONFIG_HOME/surfer/config.toml
        self._config_dir = tempfile.mkdtemp(prefix="surfer_config_")
        config_content = f'''[wcp]
autostart = true
address = "127.0.0.1:{wcp_port}"
'''
        config_path = Path(self._config_dir) / "surfer" / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content)

        logger.info("Starting Surfer GUI with WCP on port %d", wcp_port)

        # Prepare environment to use temp config directory
        env = os.environ.copy()
        env["XDG_CONFIG_HOME"] = self._config_dir

        # Start process
        self._process = subprocess.Popen(
            [surfer_path],
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
                f"stderr: {stderr.decode()}"
            )

        # Connect via WCP
        self._wcp = WcpClient("localhost", wcp_port)

        # Retry connection with timeout
        max_retries = 10
        connect_timeout = 5.0  # seconds per attempt
        for i in range(max_retries):
            try:
                # Add timeout to prevent hanging
                connect_task = asyncio.ensure_future(self._wcp.connect(), loop=self._loop)
                self._loop.run_until_complete(
                    asyncio.wait_for(connect_task, timeout=connect_timeout)
                )
                break
            except asyncio.TimeoutError:
                if i == max_retries - 1:
                    self._process.terminate()
                    self._process.wait()
                    self._loop.close()
                    self._loop = None
                    raise RuntimeError(f"WCP connection timed out after {max_retries} retries")
                time.sleep(0.5)
            except Exception as e:
                if i == max_retries - 1:
                    self._process.terminate()
                    self._process.wait()
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

    def _start_fallback_mode(self) -> str:
        """Setup for VCD file export mode (no GUI)."""
        # Determine output directory
        if self.config.fallback_dir:
            fallback_dir = Path(self.config.fallback_dir)
            fallback_dir.mkdir(parents=True, exist_ok=True)
        else:
            fallback_dir = Path(tempfile.gettempdir())

        # Create VCD file path
        self._vcd_path = str(fallback_dir / "viewer.vcd")

        self._mode = "fallback"
        logger.info("Viewer in fallback mode, VCD file: %s", self._vcd_path)

        return self.url

    def close(self) -> None:
        """Close the viewer and release resources."""
        # Close WCP connection
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

        # Clean up temp VCD file (only in GUI mode)
        if self._mode == "gui" and self._vcd_path:
            try:
                os.unlink(self._vcd_path)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Error removing temp VCD: %s", e)

        # Clean up temp config directory (only in GUI mode)
        if self._mode == "gui" and hasattr(self, "_config_dir") and self._config_dir:
            import shutil
            try:
                shutil.rmtree(self._config_dir)
            except Exception as e:
                logger.warning("Error removing temp config dir: %s", e)
            self._config_dir = None

        self._mode = None
        self._vcd_path = None
        logger.info("Viewer closed")

    @property
    def url(self) -> str:
        """The viewer URL or VCD file path."""
        if self._mode == "gui":
            return "gui://surfer"
        elif self._mode == "fallback":
            return f"file://{self._vcd_path}"
        return ""

    @property
    def mode(self) -> str | None:
        """Current mode: 'gui', 'fallback', or None if not started."""
        return self._mode

    @property
    def is_running(self) -> bool:
        """Check if the viewer is running (always True for fallback mode)."""
        if self._mode == "fallback":
            return True
        return self._process is not None and self._process.poll() is None

    # =========================================================================
    # State sync
    # =========================================================================

    def pull_state(self) -> None:
        """
        Pull current state from Surfer into top_group and markers.

        Note: Only supported in GUI mode.
        """
        if self._mode == "fallback":
            raise RuntimeError("pull_state() is not supported in fallback mode")

        if not self._wcp:
            raise RuntimeError("Viewer not started")

        # Get displayed items
        item_ids = self._loop.run_until_complete(self._wcp.get_item_list())
        items_info = self._loop.run_until_complete(
            self._wcp.get_item_info(item_ids)
        ) if item_ids else []

        # Build GroupItem tree from items
        self.top_group = self._build_group_tree(items_info)
        self.markers = MarkerList()

    def push_state(self) -> None:
        """
        Push current state to the viewer.

        In GUI mode: Sends to Surfer via WCP.
        In fallback mode: Generates VCD file for user to open.
        """
        if self._vcd_path is None:
            raise RuntimeError("Viewer not started")

        # Get all visible WaveformItems
        visible_waveforms = list(self.top_group.walk_waveforms(include_hidden=False))

        if not visible_waveforms:
            if self._mode == "gui" and self._wcp:
                self._loop.run_until_complete(self._wcp.clear())
            return

        # Generate merged VCD
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
                        # Use transformed name for WCP (avoids bit selector interpretation)
                        wcp_name = get_wcp_signal_name(item.signal_name)

                        ids = self._loop.run_until_complete(
                            self._wcp.add_variables([wcp_name])
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
                    {"time": m.time, "name": m.name or "", "move_focus": False}
                    for m in markers_list
                ]
                marker_ids = self._loop.run_until_complete(
                    self._wcp.add_markers(marker_infos)
                )
                for m, mid in zip(markers_list, marker_ids):
                    m.item_id = mid

        elif self._mode == "fallback":
            logger.info("VCD file updated: %s", self._vcd_path)

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
        """Reload the current VCD file (GUI mode only)."""
        if self._wcp:
            self._loop.run_until_complete(self._wcp.reload())

    def focus(self, item: DisplayItem) -> None:
        """Focus (scroll to) an item in the viewer (GUI mode only)."""
        if self._mode == "fallback":
            raise RuntimeError("focus() is not supported in fallback mode")
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
        elif self._mode == "fallback":
            return f"<Viewer (fallback mode, {self._vcd_path})>"
        return "<Viewer (not started)>"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()


# Backward compatibility alias
ViewerInstance = Viewer
