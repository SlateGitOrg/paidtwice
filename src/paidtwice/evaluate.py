"""Evaluation against the planted truth. This is the only place truth.csv is read."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from paidtwice.candidates import CandidateSet
from paidtwice.data import TRUTH, Payments, read_csv
from paidtwice.score import amount_at_risk


@dataclass
class Truth:
    """Planted duplicate pairs and look-alike pairs, as sets of ordered id pairs."""

    duplicates: dict[tuple[str, str], str]  # pair -> mechanism
    traps: dict[tuple[str, str], str]  # pair -> trap type

    @classmethod
    def load(cls, data_dir: Path) -> Truth:
        frame = read_csv(data_dir / TRUTH)
        pay = frame[frame["entity"] == "payment"]
        duplicates: dict[tuple[str, str], str] = {}
        traps: dict[tuple[str, str], str] = {}
        for _, group in pay.groupby("group_id"):
            ids = sorted(group["entity_id"])
            label = group["label"].iloc[0]
            target = duplicates if "duplicate" in set(group["role"]) else traps
            for pair in combinations(ids, 2):
                target[pair] = label
        return cls(duplicates, traps)


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _queue_pairs(queue: pd.DataFrame) -> list[tuple[str, str]]:
    return [_pair(a, b) for a, b in zip(queue["payment_a"], queue["payment_b"], strict=True)]


def _risk(p: Payments, pair: tuple[str, str]) -> float:
    i, j = p.index_of(list(pair))
    return float(amount_at_risk(p.amount_cents[i : i + 1], p.amount_cents[j : j + 1])[0])


def evaluate(
    p: Payments,
    truth: Truth,
    cand: CandidateSet,
    queue: pd.DataFrame,
    queue_no_suppress: pd.DataFrame | None = None,
    budget: int = 200,
) -> dict:
    """Metrics for the review queue at ``budget`` and the suppression ablation."""
    top = _queue_pairs(queue.head(budget))
    found = {pair for pair in top if pair in truth.duplicates}
    n_dups = len(truth.duplicates)
    dollars = {pair: _risk(p, pair) for pair in truth.duplicates}
    mechanisms = sorted(set(truth.duplicates.values()))
    recall_by_mechanism = {}
    for mech in mechanisms:
        members = [pair for pair, m in truth.duplicates.items() if m == mech]
        recall_by_mechanism[mech] = {
            "planted": len(members),
            "found": sum(pair in found for pair in members),
            "recall": sum(pair in found for pair in members) / len(members),
        }

    candidate_pairs = {_pair(p.ids[a], p.ids[b]) for a, b in zip(cand.a, cand.b, strict=True)}
    in_blocks = sum(pair in candidate_pairs for pair in truth.duplicates)

    def trap_counts(frame: pd.DataFrame | None) -> dict[str, int] | None:
        if frame is None:
            return None
        counts = {kind: 0 for kind in sorted(set(truth.traps.values()))}
        for pair in _queue_pairs(frame.head(budget)):
            if pair in truth.traps:
                counts[truth.traps[pair]] += 1
        return counts

    return {
        "budget": budget,
        "queue_size": len(top),
        "planted_duplicates": n_dups,
        "true_positives": len(found),
        "precision": len(found) / len(top) if top else 0.0,
        "recall": len(found) / n_dups,
        "dollar_recall": sum(dollars[pair] for pair in found) / sum(dollars.values()),
        "recall_by_mechanism": recall_by_mechanism,
        "blocking_recall": in_blocks / n_dups,
        "trap_false_positives_with_suppression": trap_counts(queue),
        "trap_false_positives_without_suppression": trap_counts(queue_no_suppress),
    }


def render_markdown(result: dict) -> str:
    """Markdown summary of ``evaluate`` output."""
    lines = [
        f"## Review queue at budget {result['budget']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Planted duplicate payments | {result['planted_duplicates']} |",
        f"| True duplicates in the queue | {result['true_positives']} |",
        f"| Precision at budget | {result['precision']:.3f} |",
        f"| Recall | {result['recall']:.3f} |",
        f"| Dollar recall | {result['dollar_recall']:.3f} |",
        f"| Blocking recall | {result['blocking_recall']:.3f} |",
        "",
        "## Recall by duplicate mechanism",
        "",
        "| Mechanism | Planted | Found | Recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mech, row in result["recall_by_mechanism"].items():
        lines.append(f"| {mech} | {row['planted']} | {row['found']} | {row['recall']:.3f} |")
    with_rules = result["trap_false_positives_with_suppression"]
    without = result["trap_false_positives_without_suppression"]
    lines += [
        "",
        "## Look-alikes in the queue",
        "",
        "| Look-alike type | With suppression | Without suppression |",
        "| --- | ---: | ---: |",
    ]
    for kind in with_rules:
        other = "n/a" if without is None else str(without[kind])
        lines.append(f"| {kind} | {with_rules[kind]} | {other} |")
    total_without = "n/a" if without is None else str(sum(without.values()))
    lines.append(f"| total | {sum(with_rules.values())} | {total_without} |")
    return "\n".join(lines) + "\n"
