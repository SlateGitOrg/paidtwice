import pytest

from paidtwice.distance import bounded_distance, damerau_levenshtein


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("1234", "1243", 1),
        ("1234", "2143", 2),
        ("abc", "abc", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("ca", "abc", 3),
        ("1045", "104", 1),
    ],
)
def test_damerau_levenshtein(a, b, expected):
    assert damerau_levenshtein(a, b) == expected
    assert damerau_levenshtein(b, a) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [("1234", "1243"), ("1234", "2143"), ("INV1234", "INV9999"), ("abcdef", "ab"), ("x", "x")],
)
def test_bounded_distance_matches_the_full_distance_up_to_the_cap(a, b):
    full = damerau_levenshtein(a, b)
    assert bounded_distance(a, b, 2) == min(full, 3)
