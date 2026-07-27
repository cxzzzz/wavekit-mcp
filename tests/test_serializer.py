from __future__ import annotations

from wavekit_mcp.config import Config
from wavekit_mcp.serializer import serialize_result


def test_serializer_returns_repr_for_result():
    class Thing:
        def __repr__(self):
            return "Thing(value=1)"

    assert serialize_result(Thing(), Config()) == "Thing(value=1)"
    assert serialize_result(3, Config()) == "3"
    assert serialize_result("abc", Config()) == "'abc'"


def test_serializer_truncates_repr():
    cfg = Config()
    cfg.limits.result_str_max = 5

    assert serialize_result("abcdef", cfg) == "'abcd...[3 chars omitted]"


def test_serializer_preserves_none():
    assert serialize_result(None, Config()) is None
