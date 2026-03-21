"""Integration tests for ViewerInstance (requires Surfer binary).

Run with: pytest -m integration tests/test_instance.py

These tests require the 'surfer' binary to be installed and available in PATH.
"""

import pytest
import asyncio
import shutil
import tempfile
from pathlib import Path


# Check if Surfer is available
SURFER_AVAILABLE = shutil.which("surfer") is not None


@pytest.fixture
def viewer_config():
    """Create test viewer configuration."""
    from wavekit_mcp.viewer import ViewerConfig

    return ViewerConfig(
        headless=True,
        wcp_port=0,  # Auto-assign
        http_port=0,  # Auto-assign
    )


@pytest.fixture
async def viewer_instance(viewer_config):
    """Create and start a ViewerInstance for testing."""
    from wavekit_mcp.viewer import ViewerInstance

    viewer = ViewerInstance("test_viewer", viewer_config)

    try:
        await viewer.start()
        yield viewer
    finally:
        await viewer.stop()


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found")
class TestViewerInstance:
    """Integration tests for ViewerInstance."""

    @pytest.mark.asyncio
    async def test_start_stop(self, viewer_config):
        """Test starting and stopping Surfer."""
        from wavekit_mcp.viewer import ViewerInstance

        viewer = ViewerInstance("test_start_stop", viewer_config)

        url = await viewer.start()
        assert url.startswith("http://localhost:")
        assert viewer.is_running

        await viewer.stop()
        assert not viewer.is_running

    @pytest.mark.asyncio
    async def test_url_property(self, viewer_instance):
        """Test URL property."""
        url = viewer_instance.url
        assert url.startswith("http://localhost:")

    @pytest.mark.asyncio
    async def test_get_item_list_empty(self, viewer_instance):
        """Test get_item_list on empty viewer."""
        ids = await viewer_instance._wcp.get_item_list()
        assert ids == []

    @pytest.mark.asyncio
    async def test_clear(self, viewer_instance):
        """Test clear command."""
        await viewer_instance._wcp.clear()  # Should not raise

    @pytest.mark.asyncio
    async def test_set_cursor(self, viewer_instance):
        """Test set_cursor command."""
        await viewer_instance.set_cursor(1000)  # Should not raise

    @pytest.mark.asyncio
    async def test_zoom_to_fit(self, viewer_instance):
        """Test zoom_to_fit command."""
        await viewer_instance.zoom_to_fit()  # Should not raise


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found")
class TestViewerInstanceWithWaveforms:
    """Integration tests requiring waveform data."""

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

    @pytest.mark.asyncio
    async def test_load_vcd(self, viewer_instance, sample_vcd):
        """Test loading a VCD file."""
        result = await viewer_instance._wcp.load(sample_vcd)
        # Should have some response
        assert result is not None

    @pytest.mark.asyncio
    async def test_add_variables(self, viewer_instance, sample_vcd):
        """Test adding variables after loading VCD."""
        await viewer_instance._wcp.load(sample_vcd)

        ids = await viewer_instance._wcp.add_variables(["top.clk", "top.data[7:0]"])
        assert len(ids) == 2

        # Verify items are added
        item_list = await viewer_instance._wcp.get_item_list()
        assert len(item_list) == 2

    @pytest.mark.asyncio
    async def test_pull_state(self, viewer_instance, sample_vcd):
        """Test pull_state after loading and adding variables."""
        await viewer_instance._wcp.load(sample_vcd)
        await viewer_instance._wcp.add_variables(["top.clk"])

        state = await viewer_instance.pull_state()

        assert "top_group" in state
        assert "markers" in state
        # Should have at least one item
        assert state["top_group"].children or len(list(state["top_group"].walk())) >= 1


@pytest.mark.integration
@pytest.mark.skipif(not SURFER_AVAILABLE, reason="Surfer binary not found")
class TestViewerRegistry:
    """Integration tests for ViewerRegistry with viewer workers."""

    @pytest.fixture
    def config(self):
        """Create minimal config for testing."""
        from wavekit_mcp.config import Config

        return Config()  # Use defaults

    @pytest.mark.asyncio
    async def test_create_viewer(self, config):
        """Test creating a viewer through registry."""
        from wavekit_mcp.viewer import ViewerRegistry

        registry = ViewerRegistry(config)
        viewer_id, viewer = await registry.create_viewer()

        try:
            assert viewer.is_running
            assert viewer.url.startswith("http://")
        finally:
            await registry.close_all()

    @pytest.mark.asyncio
    async def test_list_viewers(self, config):
        """Test listing viewers."""
        from wavekit_mcp.viewer import ViewerRegistry

        registry = ViewerRegistry(config)

        try:
            assert registry.list_viewers() == []

            vid1, _ = await registry.create_viewer()
            vid2, _ = await registry.create_viewer()

            viewers = registry.list_viewers()
            assert len(viewers) == 2
            assert vid1 in viewers
            assert vid2 in viewers

        finally:
            await registry.close_all()

    @pytest.mark.asyncio
    async def test_close_viewer(self, config):
        """Test closing a viewer."""
        from wavekit_mcp.viewer import ViewerRegistry

        registry = ViewerRegistry(config)
        viewer_id, viewer = await registry.create_viewer()

        assert viewer.is_running

        await registry.close_viewer(viewer_id)

        assert not viewer.is_running
        assert viewer_id not in registry.list_viewers()

    @pytest.mark.asyncio
    async def test_close_all(self, config):
        """Test closing all viewers."""
        from wavekit_mcp.viewer import ViewerRegistry

        registry = ViewerRegistry(config)

        await registry.create_viewer()
        await registry.create_viewer()

        assert len(registry.list_viewers()) == 2

        await registry.close_all()

        assert len(registry.list_viewers()) == 0
