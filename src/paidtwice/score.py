"""Ranking: comparison vectors, summed weights, suppression and the review queue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from paidtwice.candidates import CandidateSet
from paidtwice.compare import comparison_vectors, describe
from paidtwice.data import Payments
from paidtwice.em import FSModel
from paidtwice.suppress import RULES, any_rule, build_context, evaluate_rules, reason_codes

QUEUE_COLUMNS = [
    "rank",
    "payment_a",
    "payment_b",
    "vendor_a",
    "vendor_b",
    "invoice_a",
    "invoice_b",
    "date_a",
    "date_b",
    "amount_a",
    "amount_b",
    "score",
    "matched_levels",
    "reason_codes",
    "amount_at_risk",
]


@dataclass
class ScoreResult:
    queue: pd.DataFrame
    suppressed: pd.DataFrame
    summary: dict


def amount_at_risk(cents_a: np.ndarray, cents_b: np.ndarray) -> np.ndarray:
    """Dollars at stake if the pair is a duplicate: the smaller absolute amount."""
    return np.minimum(np.abs(cents_a), np.abs(cents_b)) / 100.0


def _frame(p: Payments, cand: CandidateSet, rows: np.ndarray, vectors, scores, reasons):
    a, b = cand.a[rows], cand.b[rows]
    frame = p.frame
    return pd.DataFrame(
        {
            "payment_a": p.ids[a],
            "payment_b": p.ids[b],
            "vendor_a": p.vendor_id[a],
            "vendor_b": p.vendor_id[b],
            "invoice_a": p.invoice_raw[a],
            "invoice_b": p.invoice_raw[b],
            "date_a": frame["invoice_date"].to_numpy()[a],
            "date_b": frame["invoice_date"].to_numpy()[b],
            "amount_a": frame["amount"].to_numpy()[a],
            "amount_b": frame["amount"].to_numpy()[b],
            "score": np.round(scores[rows], 3),
            "matched_levels": [describe(v) for v in vectors[rows]],
            "reason_codes": [reasons[i] for i in rows],
            "amount_at_risk": amount_at_risk(p.amount_cents[a], p.amount_cents[b]),
        }
    )


def rank_pairs(
    p: Payments,
    cand: CandidateSet,
    model: FSModel,
    *,
    budget: int = 200,
    suppress: bool = True,
) -> ScoreResult:
    """Score every candidate pair and return the top ``budget`` as the review queue.

    With ``suppress`` the pairs a rule explains are removed before ranking and
    returned separately; without it they stay in the ranking and their reason
    codes are shown for information. Ties on score are broken by amount at risk,
    then by payment ids, so the queue is deterministic.
    """
    vectors = comparison_vectors(p, cand.a, cand.b)
    scores = model.score(vectors)
    flags = evaluate_rules(p, build_context(p), cand.a, cand.b)
    reasons = reason_codes(flags)
    explained = any_rule(flags)

    kept = np.flatnonzero(~explained) if suppress else np.arange(len(cand))
    risk = amount_at_risk(p.amount_cents[cand.a], p.amount_cents[cand.b])
    order = np.lexsort((p.ids[cand.b[kept]], p.ids[cand.a[kept]], -risk[kept], -scores[kept]))
    top = kept[order][:budget]

    queue = _frame(p, cand, top, vectors, scores, reasons)
    queue.insert(0, "rank", np.arange(1, len(queue) + 1))
    queue = queue[QUEUE_COLUMNS]
    removed = np.flatnonzero(explained) if suppress else np.empty(0, dtype=np.int64)
    suppressed = _frame(p, cand, removed, vectors, scores, reasons)
    summary = {
        "candidate_pairs": int(len(cand)),
        "suppression": "on" if suppress else "off",
        "pairs_suppressed": int(len(removed)),
        "suppressed_by_rule": {r: int(flags[r].sum()) for r in RULES} if suppress else {},
        "ranked_pairs": int(len(kept)),
        "budget": budget,
        "queue_size": int(len(queue)),
        "amount_at_risk_in_queue": float(queue["amount_at_risk"].sum()),
        "score_range_in_queue": (
            [float(queue["score"].min()), float(queue["score"].max())] if len(queue) else []
        ),
    }
    return ScoreResult(queue, suppressed, summary)
