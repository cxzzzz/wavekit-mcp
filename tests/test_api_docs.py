from wavekit_mcp.server import get_api_docs


def test_get_api_docs_pattern_topic_mentions_exports():
    docs = get_api_docs("Pattern")
    assert "MatchStatus" in docs
    assert "collect" in docs
    assert "MatchRecords" in docs


def test_get_api_docs_waveform_topic_does_not_crash():
    docs = get_api_docs("Waveform")
    assert "Waveform" in docs
    assert "Public methods" in docs or "pydoc" in docs
