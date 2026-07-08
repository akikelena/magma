from magma.phase1_documentation_detection.source_parser import detect_source_kind, parse_source


def test_detects_markdown():
    assert detect_source_kind("# Title\n\nbody text\n") == "markdown"


def test_detects_html():
    assert detect_source_kind("<html><body><p>hi</p></body></html>") == "html"


def test_detects_openapi():
    assert detect_source_kind('{"openapi": "3.0.0"}') == "openapi"


def test_detects_unstructured():
    assert detect_source_kind("Just a sentence with no markup.") == "unstructured"


def test_parse_markdown_tracks_heading_path():
    md = "# A\n\n## B\n\nsome text\n"
    kind, sections = parse_source(md)
    assert kind == "markdown"
    assert sections[0].heading_path == ["A", "B"]
    assert sections[0].content == "some text"


def test_parse_markdown_code_block_flagged():
    md = "# A\n\n```\ncode here\n```\n"
    _, sections = parse_source(md)
    assert any(s.is_code_block for s in sections)
