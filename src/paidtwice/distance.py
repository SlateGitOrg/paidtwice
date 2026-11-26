"""String distances used to compare invoice numbers."""

from __future__ import annotations


def damerau_levenshtein(a: str, b: str) -> int:
    """Damerau-Levenshtein distance, optimal string alignment variant.

    Insertions, deletions, substitutions and transpositions of two adjacent
    characters each cost 1, and no substring is edited more than once. That is
    the right model for keying errors: ``1234`` -> ``1243`` is one edit, while
    ``1234`` -> ``2143`` is two.
    """
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    two_back: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, two_back[j - 2] + 1)
            cur[j] = best
        two_back, prev = prev, cur
    return prev[lb]


def bounded_distance(a: str, b: str, cap: int) -> int:
    """Return ``min(damerau_levenshtein(a, b), cap + 1)``, stopping early.

    A length gap above ``cap`` rules the pair out at once. After that the scan
    stops when the minima of two consecutive rows both exceed ``cap``: every
    later cell is built from those two rows plus 1, so it cannot come back
    under the cap.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    if a == b:
        return 0
    if la == 0 or lb == 0:
        return min(max(la, lb), cap + 1)
    two_back: list[int] = []
    prev = list(range(lb + 1))
    prev_min = 0
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, two_back[j - 2] + 1)
            cur[j] = best
            if best < row_min:
                row_min = best
        if row_min > cap and prev_min > cap:
            return cap + 1
        two_back, prev, prev_min = prev, cur, row_min
    return min(prev[lb], cap + 1)
