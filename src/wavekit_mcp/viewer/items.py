"""Display item types for the Viewer module.

This module defines the data structures for items that can be displayed
in the Surfer waveform viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    from wavekit import Waveform


class ItemType(Enum):
    """Type of display item."""
    VARIABLE = "Variable"
    GROUP = "Group"
    DIVIDER = "Divider"
    MARKER = "Marker"


@dataclass
class DisplayItem:
    """
    Base class for all display items.

    All display items share these common attributes.
    """
    item_id: int | None = None          # Surfer-assigned ID (None if not yet displayed)
    hidden: bool = False                 # If True, don't sync to Surfer
    color: str | None = None             # Foreground color (WCP supports via set_item_color)


@dataclass
class WaveformItem(DisplayItem):
    """
    Waveform display item.

    This class wraps a wavekit.Waveform object and adds display-related
    attributes. It uses composition rather than inheritance to avoid
    issues with dataclass inheritance from non-dataclass Waveform.

    The wrapped Waveform is accessible via the .waveform property,
    and common attributes (value, time, clock) are forwarded for convenience.

    WCP supports creating Variable items via add_variables.
    """
    # The underlying Waveform data
    _waveform: Any = field(default=None, repr=False)  # wavekit.Waveform

    # WCP-supported attributes
    # color is inherited from DisplayItem

    # WCP-unsupported attributes (reserved for future extension)
    background_color: str | None = None  # Not supported by WCP
    display_name: str | None = None      # Not supported by WCP
    format: str | None = None            # Not supported by WCP (hex/bin/oct/dec)

    @property
    def item_type(self) -> ItemType:
        return ItemType.VARIABLE

    @property
    def waveform(self) -> Waveform | None:
        """Get the underlying Waveform object."""
        return self._waveform

    @waveform.setter
    def waveform(self, value: Waveform) -> None:
        self._waveform = value

    # Forward common Waveform attributes for convenience
    @property
    def value(self) -> np.ndarray | None:
        """Signal values (forwarded from Waveform)."""
        return self._waveform.value if self._waveform else None

    @property
    def time(self) -> np.ndarray | None:
        """Simulation timestamps (forwarded from Waveform)."""
        return self._waveform.time if self._waveform else None

    @property
    def clock(self) -> np.ndarray | None:
        """Clock cycle numbers (forwarded from Waveform)."""
        return self._waveform.clock if self._waveform else None

    @property
    def signal_name(self) -> str | None:
        """Full signal name (forwarded from Waveform.signal)."""
        return self._waveform.signal.full_name if self._waveform else None

    @property
    def width(self) -> int | None:
        """Signal bit width (forwarded from Waveform)."""
        return self._waveform.width if self._waveform else None

    def compress(self):
        """Compress the waveform (forwarded)."""
        if self._waveform:
            return self._waveform.compress()
        return None

    def __repr__(self) -> str:
        status = "hidden" if self.hidden else "visible"
        name = self.signal_name or "<no signal>"
        return f"WaveformItem({name}, {status})"


@dataclass
class GroupItem(DisplayItem):
    """
    Group item (can be nested).

    Note: WCP does NOT support creating Group items. GroupItem is used
    only for logical organization in the Proxy. When push_state() is called,
    the Group structure is flattened and only Variable items are sent to Surfer.
    """
    name: str = ""
    children: list[DisplayItem] = field(default_factory=list)
    expanded: bool = True  # Whether the group is expanded in the UI

    @property
    def item_type(self) -> ItemType:
        return ItemType.GROUP

    def append(self, item: DisplayItem | Waveform, **kwargs) -> DisplayItem:
        """
        Add a child item.

        If a Waveform is passed, it's automatically converted to WaveformItem.

        Args:
            item: DisplayItem or Waveform object
            **kwargs: Additional arguments passed to WaveformItem constructor

        Returns:
            The added item (converted if necessary)
        """
        # Lazy import to avoid circular dependency
        from wavekit import Waveform as WavekitWaveform

        if isinstance(item, WavekitWaveform):
            item = WaveformItem(_waveform=item, **kwargs)
        self.children.append(item)
        return item

    def insert(self, index: int, item: DisplayItem | Waveform, **kwargs) -> DisplayItem:
        """Insert a child item at the specified position."""
        from wavekit import Waveform as WavekitWaveform

        if isinstance(item, WavekitWaveform):
            item = WaveformItem(_waveform=item, **kwargs)
        self.children.insert(index, item)
        return item

    def remove(self, item: DisplayItem) -> None:
        """Remove a child item."""
        self.children.remove(item)

    def clear(self) -> None:
        """Clear all children."""
        self.children.clear()

    def walk(self, include_hidden: bool = False) -> Iterator[DisplayItem]:
        """
        Walk all descendant items (depth-first, including nested groups).

        Args:
            include_hidden: If True, include hidden items

        Yields:
            All descendant DisplayItem objects
        """
        for child in self.children:
            if not include_hidden and child.hidden:
                continue
            yield child
            if isinstance(child, GroupItem):
                yield from child.walk(include_hidden)

    def walk_waveforms(self, include_hidden: bool = False) -> Iterator[WaveformItem]:
        """
        Walk only WaveformItem descendants.

        Args:
            include_hidden: If True, include hidden items

        Yields:
            All WaveformItem descendants
        """
        for item in self.walk(include_hidden):
            if isinstance(item, WaveformItem):
                yield item

    def __iter__(self):
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __getitem__(self, index: int) -> DisplayItem:
        return self.children[index]

    def __repr__(self) -> str:
        return f"GroupItem({self.name}, {len(self.children)} items)"


@dataclass
class DividerItem(DisplayItem):
    """
    Divider (horizontal separator line).

    Note: WCP does NOT support creating Divider items. DividerItem is used
    only for logical organization in the Proxy. It's skipped during push_state().
    """
    name: str = ""

    @property
    def item_type(self) -> ItemType:
        return ItemType.DIVIDER

    def __repr__(self) -> str:
        return f"DividerItem({self.name})"


@dataclass
class MarkerItem:
    """
    Time marker.

    WCP supports creating Marker items via add_markers.
    """
    time: int
    name: str | None = None
    item_id: int | None = None
    color: str | None = None

    def __repr__(self) -> str:
        return f"Marker({self.time}, {self.name})"


class MarkerList:
    """
    List of time markers with convenient methods.
    """

    def __init__(self):
        self._markers: list[MarkerItem] = []

    def append(
        self,
        time: int,
        name: str | None = None,
        color: str | None = None
    ) -> MarkerItem:
        """
        Add a new marker.

        Args:
            time: Timestamp for the marker
            name: Optional name/label
            color: Optional color

        Returns:
            The created MarkerItem
        """
        marker = MarkerItem(time=time, name=name, color=color)
        self._markers.append(marker)
        return marker

    def remove(self, marker: MarkerItem) -> None:
        """Remove a marker."""
        self._markers.remove(marker)

    def clear(self) -> None:
        """Clear all markers."""
        self._markers.clear()

    def __iter__(self):
        return iter(self._markers)

    def __len__(self) -> int:
        return len(self._markers)

    def __getitem__(self, index: int) -> MarkerItem:
        return self._markers[index]

    def __repr__(self) -> str:
        return repr(self._markers)

    def to_list(self) -> list[MarkerItem]:
        """Get a copy of the markers list."""
        return list(self._markers)
