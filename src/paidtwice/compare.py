"""Comparison vectors: one discrete level per field for every candidate pair."""

from __future__ import annotations

import numpy as np

from paidtwice.data import Payments
from paidtwice.distance import bounded_distance

FIELDS = ("invoice", "amount", "date", "vendor", "po", "method")
LEVELS: dict[str, tuple[str, ...]] = {
    "invoice": ("canonical_equal", "dl1", "dl2", "different"),
    "amount": ("exact", "within_0.5pct", "different"),
    "date": ("le10d", "le60d", "gt60d"),
    "vendor": ("same_id", "same_cluster", "different"),
    "po": ("same", "missing", "different"),
    "method": ("same", "different"),
}
# Fields whose levels are ordered from strongest to weakest evidence.
ORDINAL_FIELDS = ("invoice", "amount", "date", "vendor")


def invoice_levels(keys_a: np.ndarray, keys_b: np.ndarray) -> np.ndarray:
    """0 canonical equal, 1 Damerau-Levenshtein 1, 2 distance 2, 3 further apart."""
    out = np.full(len(keys_a), 3, dtype=np.int8)
    for i, (x, y) in enumerate(zip(keys_a, keys_b, strict=True)):
        out[i] = min(bounded_distance(x, y, 2), 3)
    return out


def comparison_vectors(p: Payments, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Level index per field for each pair ``(a[i], b[i])``; shape ``(n_pairs, 6)``."""
    n = len(a)
    out = np.zeros((n, len(FIELDS)), dtype=np.int8)
    out[:, 0] = invoice_levels(p.invoice_key[a], p.invoice_key[b])

    ca, cb = p.amount_cents[a], p.amount_cents[b]
    diff = np.abs(ca - cb)
    largest = np.maximum(np.abs(ca), np.abs(cb))
    out[:, 1] = np.where(diff == 0, 0, np.where(diff * 200 <= largest, 1, 2))

    gap = np.abs(p.day[a] - p.day[b])
    out[:, 2] = np.where(gap <= 10, 0, np.where(gap <= 60, 1, 2))

    out[:, 3] = np.where(
        p.vendor_id[a] == p.vendor_id[b], 0, np.where(p.cluster[a] == p.cluster[b], 1, 2)
    )

    po_a, po_b = p.po[a], p.po[b]
    missing = (po_a == "") | (po_b == "")
    out[:, 4] = np.where(missing, 1, np.where(po_a == po_b, 0, 2))

    out[:, 5] = np.where(p.method[a] == p.method[b], 0, 1)
    return out


def describe(vector: np.ndarray) -> str:
    """Human-readable matched levels, e.g. ``invoice=canonical_equal;amount=exact;...``."""
    return ";".join(f"{f}={LEVELS[f][int(v)]}" for f, v in zip(FIELDS, vector, strict=True))
