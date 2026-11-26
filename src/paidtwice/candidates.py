"""Blocking: the small set of payment pairs worth comparing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from paidtwice.data import CANDIDATES, Payments, read_csv

BLOCK_NAMES = ("cluster_amount", "amount_date", "invoice_number")
DATE_WINDOW_DAYS = 120


@dataclass
class CandidateSet:
    """Candidate pairs as index arrays into the payment table, ``a < b``."""

    a: np.ndarray
    b: np.ndarray
    flags: np.ndarray  # bit k set when block k produced the pair
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.a)

    def codes(self, n: int) -> np.ndarray:
        return self.a.astype(np.int64) * n + self.b


def _groups(keys: np.ndarray) -> list[np.ndarray]:
    """Index arrays of every key value that occurs at least twice."""
    codes, _ = pd.factorize(pd.Series(keys), sort=False)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    cuts = np.flatnonzero(np.diff(sorted_codes)) + 1
    return [g for g in np.split(order, cuts) if len(g) > 1]


def _all_pairs(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a_parts, b_parts = [], []
    for g in groups:
        i, j = np.triu_indices(len(g), k=1)
        a_parts.append(g[i])
        b_parts.append(g[j])
    if not a_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(a_parts), np.concatenate(b_parts)


def _ordered(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.minimum(a, b).astype(np.int64), np.maximum(a, b).astype(np.int64)
    return lo, hi


def _block_cluster_amount(p: Payments) -> tuple[np.ndarray, np.ndarray]:
    rounded = (np.abs(p.amount_cents) + 50) // 100  # absolute, so a credit memo meets its invoice
    keys = np.array([f"{c}|{r}" for c, r in zip(p.cluster, rounded, strict=True)], dtype=object)
    return _all_pairs(_groups(keys))


def _block_amount_date(p: Payments, window: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = _all_pairs(_groups(p.amount_cents))
    keep = np.abs(p.day[a] - p.day[b]) <= window
    return a[keep], b[keep]


def _block_invoice(p: Payments) -> tuple[np.ndarray, np.ndarray]:
    usable = np.flatnonzero(p.invoice_key != "")
    groups = _groups(p.invoice_key[usable])
    return _all_pairs([usable[g] for g in groups])


def build_candidates(p: Payments, window_days: int = DATE_WINDOW_DAYS) -> CandidateSet:
    """Union of three blocks, de-duplicated.

    * same vendor cluster and absolute amount rounded to $1 (so a credit memo is
      compared with the invoice it cancels);
    * same exact amount and invoice dates at most ``window_days`` apart;
    * same canonical invoice number, whatever the vendor.
    """
    n = len(p)
    blocks = [
        _block_cluster_amount(p),
        _block_amount_date(p, window_days),
        _block_invoice(p),
    ]
    frames = []
    per_block = {}
    for bit, (name, (x, y)) in enumerate(zip(BLOCK_NAMES, blocks, strict=True)):
        lo, hi = _ordered(x, y)
        codes = np.unique(lo * n + hi)
        per_block[name] = int(len(codes))
        frames.append(pd.DataFrame({"code": codes, "flag": np.uint8(1 << bit)}))
    merged = pd.concat(frames).groupby("code", sort=True)["flag"].sum()
    codes = merged.index.to_numpy(dtype=np.int64)
    flags = merged.to_numpy(dtype=np.uint8)
    result = CandidateSet(codes // n, codes % n, flags)
    exclusive = {
        name: int(np.count_nonzero(flags == (1 << bit))) for bit, name in enumerate(BLOCK_NAMES)
    }
    total = p.total_pairs
    result.report = {
        "payments": n,
        "all_pairs": total,
        "pairs_per_block": per_block,
        "pairs_only_in_block": exclusive,
        "candidate_pairs": int(len(result)),
        "share_of_all_pairs": len(result) / total,
        "reduction_ratio": 1 - len(result) / total,
    }
    return result


def write_candidates(cand: CandidateSet, p: Payments, data_dir: Path) -> None:
    names = [
        ",".join(name for bit, name in enumerate(BLOCK_NAMES) if flag >> bit & 1)
        for flag in cand.flags
    ]
    pd.DataFrame({"payment_a": p.ids[cand.a], "payment_b": p.ids[cand.b], "blocks": names}).to_csv(
        data_dir / CANDIDATES, index=False
    )


def load_candidates(p: Payments, data_dir: Path) -> CandidateSet:
    frame = read_csv(data_dir / CANDIDATES)
    a = p.index_of(frame["payment_a"])
    b = p.index_of(frame["payment_b"])
    flags = np.array(
        [sum(1 << BLOCK_NAMES.index(x) for x in cell.split(",")) for cell in frame["blocks"]],
        dtype=np.uint8,
    )
    return CandidateSet(a, b, flags)
