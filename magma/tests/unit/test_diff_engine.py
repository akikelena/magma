from magma.phase1_documentation_detection.diff_engine import compute_diff, is_editorial_only


def test_identical_text_has_no_diff():
    d = compute_diff("same\n", "same\n")
    assert d.is_identical


def test_changed_text_is_not_identical():
    d = compute_diff("a\n", "b\n")
    assert not d.is_identical
    assert d.lines_added == 1 and d.lines_removed == 1


def test_whitespace_only_change_is_editorial():
    d = compute_diff("a\n\n", "a\n")
    assert is_editorial_only(d)


def test_semantic_change_is_not_editorial():
    d = compute_diff("limit: int\n", "top_k: int\n")
    assert not is_editorial_only(d)
