"""Unit tests for MarkerItem and MarkerList classes."""

import pytest


class TestMarkerList:
    """Tests for MarkerList class."""

    def test_empty(self):
        """Create empty marker list."""
        from wavekit_mcp.viewer import MarkerList

        markers = MarkerList()
        assert len(markers) == 0

    def test_append(self):
        """Add markers."""
        from wavekit_mcp.viewer import MarkerList, MarkerItem

        markers = MarkerList()
        m1 = markers.append(time=100, name="start")
        m2 = markers.append(time=500, name="end", color="#FF0000")

        assert len(markers) == 2
        assert isinstance(m1, MarkerItem)
        assert m1.time == 100
        assert m1.name == "start"
        assert m2.color == "#FF0000"

    def test_remove(self):
        """Remove marker."""
        from wavekit_mcp.viewer import MarkerList

        markers = MarkerList()
        m = markers.append(time=100)
        markers.remove(m)

        assert len(markers) == 0

    def test_iter(self):
        """Iterate markers."""
        from wavekit_mcp.viewer import MarkerList

        markers = MarkerList()
        markers.append(time=100)
        markers.append(time=200)

        times = [m.time for m in markers]
        assert times == [100, 200]

    def test_to_list(self):
        """Get list copy."""
        from wavekit_mcp.viewer import MarkerList

        markers = MarkerList()
        markers.append(time=100)

        lst = markers.to_list()
        assert len(lst) == 1
        assert lst[0].time == 100
