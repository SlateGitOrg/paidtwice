"""Suppression rules: legitimate look-alikes that must not reach the review queue.

Each rule is a pure function of the two payments (plus, for two rules, the
other payments of the same vendor) and carries a reason code. No rule looks at
the learned weights or at any label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from paidtwice.data import Payments

RECURRING_SERIES = "recurring_series"
INSTALLMENT_SUFFIX = "installment_suffix"
CREDIT_MEMO = "credit_memo"
CREDIT_MEMO_REBILL = "credit_memo_rebill"
SEQUENTIAL_INVOICE = "sequential_invoice"
RULES = (RECURRING_SERIES, INSTALLMENT_SUFFIX, CREDIT_MEMO, CREDIT_MEMO_REBILL, SEQUENTIAL_INVOICE)

SERIES_MIN_LENGTH = 4
SERIES_GAP_DAYS = (25, 35)
SERIES_SUPPRESS_MIN_DAYS = 25

_NUMBERED = re.compile(r"^(?P<pre>[A-Z]*)(?P<num>\d+)(?P<suf>[A-Z]*)$")


@dataclass
class SuppressionContext:
    """Per-payment facts the rules need, computed once from the whole ledger."""

    series_id: np.ndarray
    prefix: np.ndarray
    number: np.ndarray  # -1 when the canonical number is not prefix + digits + suffix
    suffix: np.ndarray
    memos: dict[tuple[str, int], list[int]]  # (cluster, |amount|) -> memo days


def find_series(p: Payments) -> np.ndarray:
    """Id of the recurring series each payment belongs to, -1 for none.

    A series is a chain of payments by one vendor cluster for one exact positive
    amount in which each payment follows the previous one by 25-35 days, with at
    least four links. A stray extra payment inside a month starts a chain of its
    own and so never joins the series.
    """
    lo, hi = SERIES_GAP_DAYS
    series = np.full(len(p), -1, dtype=np.int64)
    by_key: dict[tuple[str, int], list[int]] = {}
    for i in np.flatnonzero(p.amount_cents > 0):
        by_key.setdefault((p.cluster[i], int(p.amount_cents[i])), []).append(int(i))
    next_id = 0
    for members in by_key.values():
        if len(members) < SERIES_MIN_LENGTH:
            continue
        members.sort(key=lambda i: (p.day[i], i))
        chains: list[list[int]] = []
        for i in members:
            for chain in chains:
                if lo <= p.day[i] - p.day[chain[-1]] <= hi:
                    chain.append(i)
                    break
            else:
                chains.append([i])
        for chain in chains:
            if len(chain) >= SERIES_MIN_LENGTH:
                series[chain] = next_id
                next_id += 1
    return series


def build_context(p: Payments) -> SuppressionContext:
    prefix = np.empty(len(p), dtype=object)
    number = np.full(len(p), -1, dtype=np.int64)
    suffix = np.empty(len(p), dtype=object)
    for i, key in enumerate(p.invoice_key):
        match = _NUMBERED.match(key)
        prefix[i], suffix[i] = (match.group("pre"), match.group("suf")) if match else ("", "")
        if match:
            number[i] = int(match.group("num"))
    memos: dict[tuple[str, int], list[int]] = {}
    for i in np.flatnonzero(p.credit_memo & (p.amount_cents < 0)):
        memos.setdefault((p.cluster[i], -int(p.amount_cents[i])), []).append(int(p.day[i]))
    return SuppressionContext(find_series(p), prefix, number, suffix, memos)


def evaluate_rules(
    p: Payments, ctx: SuppressionContext, a: np.ndarray, b: np.ndarray
) -> dict[str, np.ndarray]:
    """Boolean array per rule: which of the pairs ``(a[i], b[i])`` the rule suppresses."""
    gap = np.abs(p.day[a] - p.day[b])
    same_cluster = p.cluster[a] == p.cluster[b]

    recurring = (
        (ctx.series_id[a] >= 0)
        & (ctx.series_id[a] == ctx.series_id[b])
        & (gap >= SERIES_SUPPRESS_MIN_DAYS)
    )
    installment = (
        same_cluster
        & (p.installment[a] > 0)
        & (p.installment[b] > 0)
        & (p.installment[a] != p.installment[b])
        & (p.invoice_key[a] == p.invoice_key[b])
    )
    memo_pair = p.credit_memo[a] | p.credit_memo[b]
    sequential = (
        (ctx.number[a] >= 0)
        & (ctx.number[b] >= 0)
        & (ctx.prefix[a] == ctx.prefix[b])
        & (ctx.suffix[a] == ctx.suffix[b])
        & (np.abs(ctx.number[a] - ctx.number[b]) == 1)
    )
    rebill = np.zeros(len(a), dtype=bool)
    same_amount = (p.amount_cents[a] == p.amount_cents[b]) & (p.amount_cents[a] > 0)
    for k in np.flatnonzero(same_cluster & same_amount & ~memo_pair):
        days = ctx.memos.get((p.cluster[a[k]], int(p.amount_cents[a[k]])))
        if days:
            low, high = sorted((int(p.day[a[k]]), int(p.day[b[k]])))
            rebill[k] = any(low <= d <= high for d in days)
    return {
        RECURRING_SERIES: recurring,
        INSTALLMENT_SUFFIX: installment,
        CREDIT_MEMO: memo_pair,
        CREDIT_MEMO_REBILL: rebill,
        SEQUENTIAL_INVOICE: sequential,
    }


def reason_codes(flags: dict[str, np.ndarray]) -> list[str]:
    """Comma-joined reason codes per pair, empty string when no rule fires."""
    n = len(next(iter(flags.values())))
    return [",".join(rule for rule in RULES if flags[rule][i]) for i in range(n)]


def any_rule(flags: dict[str, np.ndarray]) -> np.ndarray:
    return np.logical_or.reduce(list(flags.values()))
