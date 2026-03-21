"""ViewerProxy - session-level interface to ViewerInstance.

This module provides the ViewerProxy class which runs in the worker process
and communicates with ViewerInstance in the main process via IPC.
"""

from __future__ import annotations

import logging
from typing import Any

from .items import DisplayItem, DividerItem, GroupItem, MarkerItem, MarkerList

logger = logging.getLogger(__name__)


class ViewerProxy:
    """
    Proxy for a ViewerInstance running in the main process.

    This class provides the session-level interface for controlling
    a Surfer viewer. It communicates with ViewerInstance via IPC
    (multiprocessing.Pipe).

    Usage:
        viewer = get_viewer(viewer_id)
        viewer.pull_state()
        viewer.top_group.append(waveform)
        viewer.push_state()
    """

    def __init__(self, viewer_id: str, conn: Any):
        """
        Initialize the proxy.

        Args:
            viewer_id: The viewer instance ID
            conn: Pipe connection to the main process
        """
        self._viewer_id = viewer_id
        self._conn = conn

        # Local state
        self._top_group: GroupItem = GroupItem(name="top")
        self._markers: MarkerList = MarkerList()

    # =========================================================================
    # State sync
    # =========================================================================

    def pull_state(self) -> None:
        """
        Pull current state from Surfer via main process.

        This updates top_group and markers with the latest state.
        """
        response = self._send("pull_state")
        self._top_group = response.get("top_group", GroupItem(name="top"))
        markers_list = response.get("markers", [])
        self._markers._markers = markers_list

    def push_state(self) -> None:
        """
        Push current state to Surfer via main process.

        This sends top_group and markers to the main process, which
        generates a VCD, loads it into Surfer, and configures the display.
        """
        self._send("push_state", {
            "top_group": self._top_group,
            "markers": self._markers.to_list(),
        })

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def viewer_id(self) -> str:
        """The viewer instance ID."""
        return self._viewer_id

    @property
    def top_group(self) -> GroupItem:
        """
        Top-level group containing all waveform items.

        This can be modified locally. Call push_state() to sync to Surfer.
        """
        return self._top_group

    @property
    def markers(self) -> MarkerList:
        """
        List of time markers.

        This can be modified locally. Call push_state() to sync to Surfer.
        """
        return self._markers

    @property
    def url(self) -> str:
        """Get the Surfer HTTP URL."""
        return self._send("get_url")

    @property
    def time_range(self) -> tuple[int, int]:
        """Get the waveform time range (start, end)."""
        return self._send("get_time_range")

    # =========================================================================
    # View control (write-only properties)
    # =========================================================================

    @property
    def cursor(self) -> int:
        """Cursor position (write-only)."""
        raise AttributeError("cursor is write-only, assign a value to set it")

    @cursor.setter
    def cursor(self, timestamp: int) -> None:
        """Set cursor position."""
        self._send("set_cursor", {"timestamp": timestamp})

    @property
    def viewport_center(self) -> int:
        """Viewport center (write-only)."""
        raise AttributeError("viewport_center is write-only")

    @viewport_center.setter
    def viewport_center(self, timestamp: int) -> None:
        """Move viewport center without changing zoom."""
        self._send("set_viewport_to", {"timestamp": timestamp})

    @property
    def viewport(self) -> tuple[int, int]:
        """Viewport range (write-only)."""
        raise AttributeError("viewport is write-only, assign (start, end) to set it")

    @viewport.setter
    def viewport(self, range: tuple[int, int]) -> None:
        """Set viewport range (changes zoom)."""
        self._send("set_viewport_range", {"start": range[0], "end": range[1]})

    # =========================================================================
    # View control methods
    # =========================================================================

    def zoom_to_fit(self) -> None:
        """Auto-zoom to fit all signals."""
        self._send("zoom_to_fit")

    def reload(self) -> None:
        """Reload the current waveform file."""
        self._send("reload")

    def focus(self, item: DisplayItem) -> None:
        """
        Focus (scroll to) an item in the viewer.

        Args:
            item: DisplayItem with an item_id
        """
        if item.item_id is None:
            raise ValueError("Item not yet added to Surfer (call push_state first)")
        self._send("focus_item", {"id": item.item_id})

    # =========================================================================
    # IPC communication
    # =========================================================================

    def _send(self, op: str, args: dict | None = None) -> Any:
        """
        Send a message to the main process and wait for response.

        Args:
            op: Operation name
            args: Optional arguments

        Returns:
            Response data
        """
        msg = {
            "type": "viewer_op",
            "viewer_id": self._viewer_id,
            "op": op,
        }
        if args:
            msg["args"] = args

        self._conn.send(msg)
        response = self._conn.recv()

        if response.get("type") == "error":
            raise RuntimeError(response.get("message", "Unknown error"))

        return response.get("result")

    def __repr__(self) -> str:
        return f"ViewerProxy({self._viewer_id})"
