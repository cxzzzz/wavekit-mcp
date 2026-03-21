"""Viewer module for Surfer waveform visualization.

This module provides the components needed to control a Surfer waveform
viewer through the WCP (Waveform Control Protocol).

Usage in session:
    viewer = Viewer()
    viewer.start()
    viewer.top_group.append(waveform)
    viewer.push_state()
    print(viewer.url)
    viewer.close()
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
from .instance import Viewer, ViewerConfig, ViewerInstance
from .wcp_client import WcpClient, WcpError

__all__ = [
    # Item types for building display hierarchy
    "DisplayItem",
    "DividerItem",
    "GroupItem",
    "ItemType",
    "MarkerItem",
    "MarkerList",
    "WaveformItem",
    # Viewer
    "Viewer",
    "ViewerConfig",
    "ViewerInstance",  # Backward compatibility alias
    # WCP client (for testing)
    "WcpClient",
    "WcpError",
]
