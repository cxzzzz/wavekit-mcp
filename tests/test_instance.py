"""Integration tests for Viewer (requires Surfer binary).

Run with: pytest -m integration tests/test_instance.py

These tests require the 'surfer' binary to be installed and available in PATH.
"""

import pytest
import shutil
import tempfile
from pathlib import Path


# Check if Surfer is available
SURFER_AVAILABLE = shutil.which("surfer") is not None


@pytest.fixture
def viewer_config():
    """Create test viewer configuration."""
    from wavekit_mcp.viewer import ViewerConfig

    return ViewerConfig()


@pytest.fixture
def viewer(viewer_config):
    """Create a Viewer for testing."""
    from wavekit_mcp.viewer import Viewer

    v = Viewer(viewer_config)
    try:
        yield v
    finally:
        v.close()


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found")
class TestViewer:
    """Integration tests for Viewer."""

    def test_start_stop(self, viewer_config):
        """Test starting and stopping Surfer."""
        from wavekit_mcp.viewer import Viewer

        viewer = Viewer(viewer_config)

        # URL can be gui://surfer (GUI mode) or file:// (fallback mode)
        assert viewer.url.startswith("gui://") or viewer.url.startswith("file://")
        assert viewer.is_running
        assert viewer.mode in ("gui", "fallback")

        viewer.close()
        assert not viewer.is_running
        assert viewer.mode is None

    def test_url_property(self, viewer):
        """Test URL property."""
        url = viewer.url
        assert url.startswith("gui://") or url.startswith("file://")

    def test_mode_property(self, viewer):
        """Test mode property."""
        assert viewer.mode in ("gui", "fallback")


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found (GUI mode required)")
class TestViewerGUI:
    """Integration tests for GUI mode (requires display)."""

    @pytest.fixture
    def gui_viewer(self, viewer_config):
        """Create a GUI mode viewer (skip if not available)."""
        from wavekit_mcp.viewer import Viewer

        v = Viewer(viewer_config)
        try:
            if v.mode != "gui":
                pytest.skip("GUI mode not available (no display)")
            yield v
        finally:
            v.close()

    def test_get_item_list_empty(self, gui_viewer):
        """Test get_item_list on empty viewer."""
        ids = gui_viewer._loop.run_until_complete(gui_viewer._wcp.get_item_list())
        assert ids == []

    def test_clear(self, gui_viewer):
        """Test clear command."""
        gui_viewer._loop.run_until_complete(gui_viewer._wcp.clear())  # Should not raise

    def test_set_cursor(self, gui_viewer):
        """Test set_cursor command."""
        gui_viewer.set_cursor(1000)  # Should not raise

    def test_zoom_to_fit(self, gui_viewer):
        """Test zoom_to_fit command."""
        gui_viewer.zoom_to_fit()  # Should not raise


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found (GUI mode required)")
class TestViewerWithWaveforms:
    """Integration tests requiring waveform data (GUI mode only)."""

    @pytest.fixture
    def gui_viewer_with_waveforms(self, viewer_config):
        """Create a GUI mode viewer."""
        from wavekit_mcp.viewer import Viewer

        v = Viewer(viewer_config)
        try:
            if v.mode != "gui":
                pytest.skip("GUI mode not available (no display)")
            yield v
        finally:
            v.close()

    @pytest.fixture
    def sample_vcd(self, tmp_path):
        """Create a sample VCD file."""
        vcd_path = tmp_path / "test.vcd"
        vcd_path.write_text("""$timescale 1ns $end
$scope module top $end
$var wire 1 a clk $end
$var wire 8 b data[7:0] $end
$upscope $end
$enddefinitions $end
#0
$var wire 1 a 0 $end
$var wire 8 b 00000000 $end
#10
$var wire 1 a 1 $end
#20
$var wire 1 a 0 $end
$var wire 8 b 11111111 $end
#30
$var wire 1 a 1 $end
#40
$var wire 1 a 0 $end
$var wire 8 b 01010101 $end
#50
$var wire 1 a 1 $end
""")
        return str(vcd_path)

    def test_load_vcd(self, gui_viewer_with_waveforms, sample_vcd):
        """Test loading a VCD file."""
        result = gui_viewer_with_waveforms._loop.run_until_complete(gui_viewer_with_waveforms._wcp.load(sample_vcd))
        # Should have some response
        assert result is not None

    def test_add_variables(self, gui_viewer_with_waveforms, sample_vcd):
        """Test adding variables after loading VCD."""
        gui_viewer_with_waveforms._loop.run_until_complete(gui_viewer_with_waveforms._wcp.load(sample_vcd))

        ids = gui_viewer_with_waveforms._loop.run_until_complete(
            gui_viewer_with_waveforms._wcp.add_variables(["top.clk", "top.data_7_0_"])
        )
        assert len(ids) == 2

        # Verify items are added
        item_list = gui_viewer_with_waveforms._loop.run_until_complete(gui_viewer_with_waveforms._wcp.get_item_list())
        assert len(item_list) == 2


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found")
class TestViewerFallback:
    """Integration tests for fallback mode."""

    @pytest.fixture
    def fallback_viewer(self, viewer_config):
        """Create a fallback mode viewer."""
        from wavekit_mcp.viewer import Viewer

        v = Viewer(viewer_config)
        try:
            if v.mode != "fallback":
                pytest.skip("Fallback mode not used (GUI mode was available)")
            yield v
        finally:
            v.close()

    def test_fallback_mode_url(self, fallback_viewer):
        """Test that fallback mode returns file:// URL."""
        assert fallback_viewer.url.startswith("file://")

    def test_push_state_empty(self, fallback_viewer):
        """Test push_state with empty content in fallback mode."""
        # Should not raise
        fallback_viewer.push_state()
