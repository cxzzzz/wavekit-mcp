"""Viewer module for Surfer waveform visualization.

This module provides the components needed to control a Surfer waveform
viewer through the WCP (Waveform Control Protocol).

Usage in session:
    viewer = Viewer()
    viewer.waveforms.append(waveform1)
    viewer.waveforms.append(waveform2)
    viewer.markers.append(time=1000, name="event")
    viewer.push_state()
    print(viewer.url)
    viewer.close()
"""

from .items import MarkerItem, MarkerList
from .instance import Viewer, ViewerConfig, ViewerInstance
from .wcp_client import WcpClient, WcpError

__all__ = [
    "MarkerItem",
    "MarkerList",
    "Viewer",
    "ViewerConfig",
    "ViewerInstance",
    "WcpClient",
    "WcpError",
]
