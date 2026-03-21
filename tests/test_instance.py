"""Integration tests for Viewer (requires Surfer binary).

Run with: pytest -m integration tests/test_instance.py

These tests require the 'surfer' or 'surver' binary to be installed and available in PATH.
"""

import pytest
import shutil
import tempfile
from pathlib import Path


# Check if Surfer is available
SURFER_AVAILABLE = shutil.which("surfer") is not None
SURVER_AVAILABLE = shutil.which("surver") is not None


@pytest.fixture
def viewer_config():
    """Create test viewer configuration."""
    from wavekit_mcp.viewer import ViewerConfig

    return ViewerConfig()


@pytest.fixture
def viewer(viewer_config):
    """Create and start a Viewer for testing."""
    from wavekit_mcp.viewer import Viewer

    v = Viewer(viewer_config)
    try:
        v.start()
        yield v
    finally:
        v.close()


@pytest.mark.integration
@pytest.mark.skipif(not (SURFER_AVAILABLE or SURVER_AVAILABLE), reason="Neither surfer nor surver binary found")
class TestViewer:
    """Integration tests for Viewer."""

    def test_start_stop(self, viewer_config):
        """Test starting and stopping Surfer."""
        from wavekit_mcp.viewer import Viewer

        viewer = Viewer(viewer_config)

        url = viewer.start()
        # URL can be gui://surfer (GUI mode) or http://localhost:PORT (server mode)
        assert url.startswith("gui://") or url.startswith("http://localhost:")
        assert viewer.is_running
        assert viewer.mode in ("gui", "server")

        viewer.close()
        assert not viewer.is_running
        assert viewer.mode is None

    def test_url_property(self, viewer):
        """Test URL property."""
        url = viewer.url
        assert url.startswith("gui://") or url.startswith("http://localhost:")

    def test_mode_property(self, viewer):
        """Test mode property."""
        assert viewer.mode in ("gui", "server")


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
            v.start()
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
            v.start()
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
            gui_viewer_with_waveforms._wcp.add_variables(["top.clk", "top.data[7:0]"])
        )
        assert len(ids) == 2

        # Verify items are added
        item_list = gui_viewer_with_waveforms._loop.run_until_complete(gui_viewer_with_waveforms._wcp.get_item_list())
        assert len(item_list) == 2

    def test_pull_state(self, gui_viewer_with_waveforms, sample_vcd):
        """Test pull_state after loading and adding variables."""
        gui_viewer_with_waveforms._loop.run_until_complete(gui_viewer_with_waveforms._wcp.load(sample_vcd))
        gui_viewer_with_waveforms._loop.run_until_complete(gui_viewer_with_waveforms._wcp.add_variables(["top.clk"]))

        gui_viewer_with_waveforms.pull_state()

        # pull_state updates self.top_group in place
        assert gui_viewer_with_waveforms.top_group is not None
        # Should have at least one item
        assert gui_viewer_with_waveforms.top_group.children or len(list(gui_viewer_with_waveforms.top_group.walk())) >= 1


@pytest.mark.integration
@pytest.mark.skipif(not SURVER_AVAILABLE, reason="Surver binary not found")
class TestViewerServer:
    """Integration tests for server mode."""

    @pytest.fixture
    def server_viewer(self, viewer_config):
        """Create a server mode viewer."""
        from wavekit_mcp.viewer import Viewer

        v = Viewer(viewer_config)
        try:
            v.start()
            if v.mode != "server":
                pytest.skip("Server mode not available (GUI mode was used)")
            yield v
        finally:
            v.close()

    def test_server_mode_url(self, server_viewer):
        """Test that server mode returns HTTP URL."""
        assert server_viewer.url.startswith("http://localhost:")

    def test_push_state_empty(self, server_viewer):
        """Test push_state with empty content in server mode."""
        # Should not raise
        server_viewer.push_state()

    def test_pull_state_not_supported(self, server_viewer):
        """Test that pull_state raises error in server mode."""
        with pytest.raises(RuntimeError, match="not supported in server mode"):
            server_viewer.pull_state()

    def test_focus_not_supported(self, server_viewer):
        """Test that focus raises error in server mode."""
        from wavekit_mcp.viewer import WaveformItem

        item = WaveformItem(item_id="test", _waveform=None)
        with pytest.raises(RuntimeError, match="not supported in server mode"):
            server_viewer.focus(item)
