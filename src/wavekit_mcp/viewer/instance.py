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
from typing import TYPE_CHECKING, Any, Literal

from .items import MarkerItem, MarkerList
from .wcp_client import WcpClient, WcpError
from .vcd_writer import generate_merged_vcd

if TYPE_CHECKING:
    from wavekit import Waveform

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
    surfer_path: str | None = None  # Path to surfer binary, None = find in PATH
    fallback_dir: str | None = None  # Directory for VCD files in fallback mode


class Viewer:
    """
    Waveform viewer using Surfer.

    Modes:
    - GUI mode: Starts Surfer with WCP for programmatic control (requires display)
    - Fallback mode: Generates VCD files for user to open with any viewer

    Usage:
        viewer = Viewer()  # Auto-starts on creation
        viewer.waveforms.append(waveform1)
        viewer.waveforms.append(waveform2)
        viewer.markers.append(time=1000, name="event")
        viewer.focus(waveform1)  # Scroll to waveform1
        viewer.zoom_to_fit()     # Auto-fit viewport
        viewer.push_state()      # Apply all changes
        print(viewer.url)
        viewer.close()

    Note: All view changes (waveforms, markers, focus, cursor, viewport) are
    accumulated and applied during push_state().
    """

    def __init__(self, config: ViewerConfig | None = None):
        self.config = config or ViewerConfig()

        # Process management (GUI mode only)
        self._process: subprocess.Popen | None = None
        self._wcp: WcpClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mode: Literal["gui", "fallback"] | None = None

        # State - user modifies these, then calls push_state()
        self.waveforms: list[Waveform] = []
        self.markers: MarkerList = MarkerList()

        # Internal state
        self._signal_ids: dict[str, int] = {}  # signal_name -> Surfer item_id
        self._vcd_path: str | None = None
        self._name_mapping: dict[str, str] = {}  # original_name -> WCP name

        # Pending state (applied during push_state)
        self._pending_cursor: int | None = None
        self._pending_viewport: tuple[str, int | tuple[int, int]] | None = None  # ("to", ts) | ("range", (start, end)) | ("fit",)
        self._pending_focus: Waveform | None = None

        # Start the viewer (try GUI mode, fall back to VCD mode)
        try:
            self._start_gui_mode()
        except RuntimeError as e:
            error_msg = str(e)
            if any(pattern in error_msg for pattern in _NO_DISPLAY_PATTERNS):
                logger.info("No display available, falling back to VCD file mode")
                self._start_fallback_mode()
            else:
                raise

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def _find_binary(self, binary_name: str) -> str:
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

    def _start_gui_mode(self) -> None:
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

    def _start_fallback_mode(self) -> None:
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

    def close(self) -> None:
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
            try:
                shutil.rmtree(self._config_dir)
            except Exception as e:
                logger.warning("Error removing temp config dir: %s", e)
            self._config_dir = None

        self._mode = None
        self._vcd_path = None
        self._signal_ids.clear()
        logger.info("Viewer closed")

    @property
    def url(self) -> str:
        if self._mode == "gui":
            return "gui://surfer"
        elif self._mode == "fallback":
            return f"file://{self._vcd_path}"
        return ""

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def is_running(self) -> bool:
        if self._mode == "fallback":
            return True
        return self._process is not None and self._process.poll() is None

    # =========================================================================
    # State sync
    # =========================================================================

    def push_state(self) -> None:
        if self._vcd_path is None:
            raise RuntimeError("Viewer not started")

        if not self.waveforms:
            if self._mode == "gui" and self._wcp:
                self._loop.run_until_complete(self._wcp.clear())
            return

        # Generate merged VCD and get name mapping
        self._vcd_path, self._name_mapping = generate_merged_vcd(self.waveforms, self._vcd_path)
        logger.debug("Generated VCD: %s", self._vcd_path)

        if self._mode == "gui" and self._wcp:
            # GUI mode: use WCP to load and add variables
            self._loop.run_until_complete(self._wcp.clear())
            self._loop.run_until_complete(self._wcp.load(self._vcd_path))
            self._signal_ids.clear()

            # Add visible variables
            for wf in self.waveforms:
                signal_name = wf.signal.full_name
                if signal_name and signal_name in self._name_mapping:
                    # Use transformed name for WCP (avoids bit selector interpretation)
                    wcp_name = self._name_mapping[signal_name]

                    ids = self._loop.run_until_complete(
                        self._wcp.add_variables([wcp_name])
                    )
                    if ids:
                        self._signal_ids[signal_name] = ids[0]

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

            # Focus pending waveform
            if self._pending_focus is not None:
                signal_name = self._pending_focus.signal.full_name
                if signal_name in self._signal_ids:
                    self._loop.run_until_complete(
                        self._wcp.focus_item(self._signal_ids[signal_name])
                    )
                self._pending_focus = None

            # Apply pending cursor
            if self._pending_cursor is not None:
                self._loop.run_until_complete(self._wcp.set_cursor(self._pending_cursor))
                self._pending_cursor = None

            # Apply pending viewport
            if self._pending_viewport is not None:
                vp_type, vp_data = self._pending_viewport
                if vp_type == "to":
                    self._loop.run_until_complete(self._wcp.set_viewport_to(vp_data))
                elif vp_type == "range":
                    start, end = vp_data
                    self._loop.run_until_complete(self._wcp.set_viewport_range(start, end))
                elif vp_type == "fit":
                    self._loop.run_until_complete(self._wcp.zoom_to_fit())
                self._pending_viewport = None

        elif self._mode == "fallback":
            logger.info("VCD file updated: %s", self._vcd_path)

    # =========================================================================
    # View control (applied during push_state)
    # =========================================================================

    def set_cursor(self, timestamp: int) -> None:
        """Set cursor position. Applied during next push_state()."""
        self._pending_cursor = timestamp

    def set_viewport_to(self, timestamp: int) -> None:
        """Move viewport center to timestamp. Applied during next push_state()."""
        self._pending_viewport = ("to", timestamp)

    def set_viewport_range(self, start: int, end: int) -> None:
        """Set viewport range. Applied during next push_state()."""
        self._pending_viewport = ("range", (start, end))

    def zoom_to_fit(self) -> None:
        """Auto-zoom to fit all signals. Applied during next push_state()."""
        self._pending_viewport = ("fit", None)

    def reload(self) -> None:
        """Reload the current VCD file. Applied immediately."""
        if self._wcp:
            self._loop.run_until_complete(self._wcp.reload())

    def focus(self, waveform: Waveform) -> None:
        """Focus (scroll to) a waveform. Applied during next push_state()."""
        self._pending_focus = waveform

    def __repr__(self) -> str:
        if self._mode == "gui":
            return "<Viewer (GUI mode)>"
        elif self._mode == "fallback":
            return f"<Viewer (fallback mode, {self._vcd_path})>"
        return "<Viewer (not started)>"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Backward compatibility alias
ViewerInstance = Viewer
