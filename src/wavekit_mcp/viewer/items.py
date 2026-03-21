"""Display items for the Viewer module.

This module defines markers - the only display item type that WCP supports.
"""

from __future__ import annotations

from dataclasses import dataclass


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
