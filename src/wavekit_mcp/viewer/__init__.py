"""Viewer module for Surfer waveform visualization.

This module provides the components needed to control a Surfer waveform
viewer through the WCP (Waveform Control Protocol).

Key components:
- DisplayItem types (WaveformItem, GroupItem, DividerItem, MarkerItem)
- WcpClient for low-level WCP communication
- ViewerInstance for managing a Surfer process (viewer worker)
- ViewerRegistry for managing multiple viewers (main process)
- ViewerProxy for session-level access (session worker)

Usage in session:
    viewer = get_viewer(viewer_id)
    viewer.pull_state()
    viewer.top_group.append(waveform)
    viewer.push_state()
"""

from .items import (
    DisplayItem,
    DividerItem,
    GroupItem,
    ItemType,
    MarkerItem,
    MarkerList,
    WaveformItem,
)
from .proxy import ViewerProxy
from .registry import ViewerRegistry
from .instance import ViewerInstance, ViewerConfig
from .wcp_client import WcpClient, WcpError

__all__ = [
    "DisplayItem",
    "DividerItem",
    "GroupItem",
    "ItemType",
    "MarkerItem",
    "MarkerList",
    "WaveformItem",
    "ViewerProxy",
    "ViewerRegistry",
    "ViewerInstance",
    "ViewerConfig",
    "WcpClient",
    "WcpError",
    "get_viewer",
]


def get_viewer(viewer_id: str | None = None) -> ViewerProxy:
    """
    Get a ViewerProxy for the specified viewer.

    This function is injected into session namespace and allows
    session code to interact with viewer workers via IPC through
    the main process router.

    Args:
        viewer_id: The viewer ID. If None, uses the default viewer.

    Returns:
        ViewerProxy instance for communicating with the viewer.

    Note:
        This is a placeholder that raises NotImplementedError.
        The actual function is injected by the session worker
        with access to the IPC connection.
    """
    raise NotImplementedError(
        "get_viewer() is not available in this context. "
        "It should be called from within a session after open_viewer()."
    )
