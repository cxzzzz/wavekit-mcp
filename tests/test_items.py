"""Unit tests for DisplayItem classes."""

import pytest
import numpy as np


class TestGroupItem:
    """Tests for GroupItem class."""

    def test_create_empty(self):
        """Create an empty GroupItem."""
        from wavekit_mcp.viewer import GroupItem

        group = GroupItem(name="test")
        assert group.name == "test"
        assert len(group) == 0
        assert list(group) == []

    def test_append_display_item(self):
        """Append DisplayItem to group."""
        from wavekit_mcp.viewer import GroupItem, DividerItem

        group = GroupItem(name="test")
        divider = DividerItem(name="sep")
        group.append(divider)

        assert len(group) == 1
        assert group[0] is divider

    def test_nested_groups(self):
        """Create nested group structure."""
        from wavekit_mcp.viewer import GroupItem

        top = GroupItem(name="top")
        sub1 = GroupItem(name="sub1")
        sub2 = GroupItem(name="sub2")

        top.append(sub1)
        top.append(sub2)

        assert len(top) == 2
        assert top[0].name == "sub1"
        assert top[1].name == "sub2"

    def test_walk_flat(self):
        """Walk flat item list."""
        from wavekit_mcp.viewer import GroupItem, DividerItem

        group = GroupItem(name="test")
        group.append(DividerItem(name="d1"))
        group.append(DividerItem(name="d2"))
        group.append(DividerItem(name="d3"))

        items = list(group.walk())
        assert len(items) == 3
        assert [i.name for i in items] == ["d1", "d2", "d3"]

    def test_walk_nested(self):
        """Walk nested item list."""
        from wavekit_mcp.viewer import GroupItem, DividerItem

        top = GroupItem(name="top")
        sub = GroupItem(name="sub")
        sub.append(DividerItem(name="d1"))
        sub.append(DividerItem(name="d2"))
        top.append(sub)
        top.append(DividerItem(name="d3"))

        items = list(top.walk())
        assert len(items) == 4
        # Order: sub, d1, d2, d3
        assert items[0].name == "sub"
        assert items[1].name == "d1"
        assert items[2].name == "d2"
        assert items[3].name == "d3"

    def test_walk_hidden(self):
        """Walk with hidden items."""
        from wavekit_mcp.viewer import GroupItem, DividerItem

        group = GroupItem(name="test")
        d1 = DividerItem(name="d1")
        d2 = DividerItem(name="d2", hidden=True)
        d3 = DividerItem(name="d3")

        group.append(d1)
        group.append(d2)
        group.append(d3)

        # Default: skip hidden
        items = list(group.walk())
        assert len(items) == 2
        assert [i.name for i in items] == ["d1", "d3"]

        # Include hidden
        items = list(group.walk(include_hidden=True))
        assert len(items) == 3


class TestWaveformItem:
    """Tests for WaveformItem class."""

    def test_create_with_waveform(self, sample_waveform_data):
        """Create WaveformItem with waveform data."""
        from wavekit_mcp.viewer import WaveformItem

        wf = sample_waveform_data("top.clk", [0, 1, 0, 1], [0, 10, 20, 30])
        item = WaveformItem(_waveform=wf, color="#FF0000")

        assert item.waveform is wf
        assert item.color == "#FF0000"
        assert item.signal_name == "top.clk"
        assert len(item.value) == 4

    def test_item_type(self, sample_waveform_data):
        """Check item_type property."""
        from wavekit_mcp.viewer import WaveformItem, ItemType

        wf = sample_waveform_data("test", [0, 1], [0, 10])
        item = WaveformItem(_waveform=wf)

        assert item.item_type == ItemType.VARIABLE

    def test_hidden_item(self, sample_waveform_data):
        """Create hidden WaveformItem."""
        from wavekit_mcp.viewer import WaveformItem

        wf = sample_waveform_data("test", [0, 1], [0, 10])
        item = WaveformItem(_waveform=wf, hidden=True)

        assert item.hidden is True

    def test_repr(self, sample_waveform_data):
        """Test string representation."""
        from wavekit_mcp.viewer import WaveformItem

        wf = sample_waveform_data("top.clk", [0, 1], [0, 10])
        item = WaveformItem(_waveform=wf)
        assert "top.clk" in repr(item)
        assert "visible" in repr(item)


class TestGroupItemWithWaveforms:
    """Tests for GroupItem with WaveformItem objects."""

    def test_append_waveform_auto_convert(self, sample_waveform_data):
        """Auto-convert Waveform to WaveformItem when appending."""
        from wavekit_mcp.viewer import GroupItem, WaveformItem

        group = GroupItem(name="test")
        wf = sample_waveform_data("top.clk", [0, 1], [0, 10])

        item = group.append(wf, color="#00FF00")

        assert isinstance(item, WaveformItem)
        assert item.waveform is wf
        assert item.color == "#00FF00"
        assert len(group) == 1

    def test_walk_waveforms(self, sample_waveform_data):
        """Walk only WaveformItems."""
        from wavekit_mcp.viewer import GroupItem, DividerItem

        group = GroupItem(name="test")
        wf1 = sample_waveform_data("top.clk", [0, 1], [0, 10])
        wf2 = sample_waveform_data("top.data", [0, 255], [0, 10])

        group.append(wf1)
        group.append(DividerItem(name="sep"))
        group.append(wf2)

        waveforms = list(group.walk_waveforms())
        assert len(waveforms) == 2
        assert waveforms[0].signal_name == "top.clk"
        assert waveforms[1].signal_name == "top.data"

    def test_walk_waveforms_nested(self, sample_waveform_data):
        """Walk WaveformItems in nested groups."""
        from wavekit_mcp.viewer import GroupItem

        top = GroupItem(name="top")
        sub1 = GroupItem(name="sub1")
        sub2 = GroupItem(name="sub2")

        wf1 = sample_waveform_data("top.clk", [0, 1], [0, 10])
        wf2 = sample_waveform_data("top.data", [0, 255], [0, 10])

        sub1.append(wf1)
        sub2.append(wf2)
        top.append(sub1)
        top.append(sub2)

        waveforms = list(top.walk_waveforms())
        assert len(waveforms) == 2


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
