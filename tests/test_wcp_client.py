"""Unit tests for WCP client with mock server."""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


class MockWcpServer:
    """Mock WCP server for testing."""

    def __init__(self):
        self.responses = []
        self.requests = []

    def add_response(self, response: dict):
        """Add a response to be returned."""
        self.responses.append(response)

    def clear(self):
        """Clear all recorded requests and responses."""
        self.requests.clear()
        self.responses.clear()


class TestWcpClient:
    """Tests for WcpClient class."""

    def test_init(self):
        """Initialize WcpClient."""
        from wavekit_mcp.viewer import WcpClient

        client = WcpClient("localhost", 12345)
        assert client._host == "localhost"
        assert client._port == 12345
        assert not client.connected

    @pytest.mark.asyncio
    async def test_connect_handshake(self):
        """Test connection with handshake."""
        from wavekit_mcp.viewer import WcpClient

        # Create mock reader/writer
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        # Greeting response
        greeting_response = {
            "type": "greeting",
            "version": "0",
            "commands": ["get_item_list", "add_variables"]
        }
        mock_reader.readuntil.return_value = (json.dumps(greeting_response) + "\0").encode()

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()

            assert client.connected
            # Verify greeting was sent
            assert mock_writer.write.called

    @pytest.mark.asyncio
    async def test_get_item_list(self):
        """Test get_item_list command."""
        from wavekit_mcp.viewer import WcpClient

        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        # Responses: greeting + get_item_list
        responses = [
            {"type": "greeting", "version": "0", "commands": []},
            {"type": "get_item_list", "ids": [1, 2, 3]}
        ]
        mock_reader.readuntil.side_effect = [
            (json.dumps(r) + "\0").encode() for r in responses
        ]

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()

            ids = await client.get_item_list()
            assert ids == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_add_variables(self):
        """Test add_variables command."""
        from wavekit_mcp.viewer import WcpClient

        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        responses = [
            {"type": "greeting", "version": "0", "commands": []},
            {"type": "add_variables", "ids": [10, 11]}
        ]
        mock_reader.readuntil.side_effect = [
            (json.dumps(r) + "\0").encode() for r in responses
        ]

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()

            ids = await client.add_variables(["top.clk", "top.data"])
            assert ids == [10, 11]

    @pytest.mark.asyncio
    async def test_error_response(self):
        """Test error response handling."""
        from wavekit_mcp.viewer import WcpClient, WcpError

        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        responses = [
            {"type": "greeting", "version": "0", "commands": []},
            {"type": "error", "message": "Variable not found"}
        ]
        mock_reader.readuntil.side_effect = [
            (json.dumps(r) + "\0").encode() for r in responses
        ]

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()

            with pytest.raises(WcpError, match="Variable not found"):
                await client.add_variables(["nonexistent"])

    @pytest.mark.asyncio
    async def test_set_cursor(self):
        """Test set_cursor command."""
        from wavekit_mcp.viewer import WcpClient

        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        responses = [
            {"type": "greeting", "version": "0", "commands": []},
            {"type": "set_cursor"}  # Ack
        ]
        mock_reader.readuntil.side_effect = [
            (json.dumps(r) + "\0").encode() for r in responses
        ]

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()

            await client.set_cursor(1000)  # Should not raise

    @pytest.mark.asyncio
    async def test_close(self):
        """Test close connection."""
        from wavekit_mcp.viewer import WcpClient

        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_writer.wait_closed = AsyncMock()

        responses = [
            {"type": "greeting", "version": "0", "commands": []}
        ]
        mock_reader.readuntil.side_effect = [
            (json.dumps(r) + "\0").encode() for r in responses
        ]

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()
            assert client.connected

            await client.close()
            assert not client.connected
            assert mock_writer.close.called


class TestWcpClientCommands:
    """Test all WCP commands."""

    @pytest.fixture
    async def connected_client(self):
        """Create a connected client with mock transport."""
        from wavekit_mcp.viewer import WcpClient

        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        greeting = {"type": "greeting", "version": "0", "commands": []}
        mock_reader.readuntil.return_value = (json.dumps(greeting) + "\0").encode()

        with patch('asyncio.open_connection', return_value=(mock_reader, mock_writer)):
            client = WcpClient("localhost", 12345)
            await client.connect()
            yield client, mock_reader, mock_writer

    @pytest.mark.asyncio
    async def test_get_item_info(self, connected_client):
        """Test get_item_info command."""
        client, reader, _ = connected_client

        response = {
            "type": "get_item_info",
            "results": [
                {"id": 1, "name": "top.clk", "type": "Variable"}
            ]
        }
        reader.readuntil.return_value = (json.dumps(response) + "\0").encode()

        result = await client.get_item_info([1])
        assert len(result) == 1
        assert result[0]["name"] == "top.clk"

    @pytest.mark.asyncio
    async def test_clear(self, connected_client):
        """Test clear command."""
        client, reader, _ = connected_client

        reader.readline.return_value = (json.dumps({"type": "clear"}) + "\n").encode()
        await client.clear()  # Should not raise

    @pytest.mark.asyncio
    async def test_add_markers(self, connected_client):
        """Test add_markers command."""
        client, reader, _ = connected_client

        response = {"type": "add_markers", "ids": [100, 101]}
        reader.readuntil.return_value = (json.dumps(response) + "\0").encode()

        ids = await client.add_markers([
            {"time": 100, "name": "start"},
            {"time": 500, "name": "end"}
        ])
        assert ids == [100, 101]

    @pytest.mark.asyncio
    async def test_zoom_to_fit(self, connected_client):
        """Test zoom_to_fit command."""
        client, reader, _ = connected_client

        reader.readline.return_value = (json.dumps({"type": "zoom_to_fit"}) + "\n").encode()
        await client.zoom_to_fit()  # Should not raise

    @pytest.mark.asyncio
    async def test_load(self, connected_client):
        """Test load command."""
        client, reader, _ = connected_client

        response = {"type": "load", "waveforms_loaded": True}
        reader.readuntil.return_value = (json.dumps(response) + "\0").encode()

        result = await client.load("/path/to/file.vcd")
        assert result.get("waveforms_loaded") is True
