"""Display items for the Viewer module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarkerItem:
    """Time marker."""
    time: int
    name: str | None = None
    item_id: int | None = None
    color: str | None = None

    def __repr__(self) -> str:
        return f"Marker({self.time}, {self.name})"


class MarkerList(list):
    """
    List of time markers with convenient append method.
    """

    def append(
        self,
        marker_or_time: int | MarkerItem | None = None,
        *,
        time: int | None = None,
        name: str | None = None,
        color: str | None = None
    ) -> MarkerItem:
        """
        Add a new marker.

        Two usage patterns:
            markers.append(time=1000, name="event")  # create and add
            markers.append(marker_item)               # add existing MarkerItem
        """
        # Determine which form was used
        if isinstance(marker_or_time, MarkerItem):
            marker = marker_or_time
        elif marker_or_time is not None:
            # Positional timestamp: append(100, name="x")
            marker = MarkerItem(time=marker_or_time, name=name, color=color)
        elif time is not None:
            # Keyword form: append(time=100, name="x")
            marker = MarkerItem(time=time, name=name, color=color)
        else:
            raise ValueError("append() requires either a timestamp or a MarkerItem")

        super().append(marker)
        return marker

    def to_list(self) -> list[MarkerItem]:
        return list(self)
