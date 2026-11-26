import json

import numpy as np
from conftest import make_payments, pay

from paidtwice.candidates import BLOCK_NAMES, build_candidates


def _pairs(cand, p):
    return {
        (p.ids[a], p.ids[b]): int(f) for a, b, f in zip(cand.a, cand.b, cand.flags, strict=True)
    }


def test_each_block_finds_its_pairs_and_pairs_are_not_repeated():
    p = make_payments(
        [
            pay("V1", "1001", "2024-01-01", "100.40"),  # P000
            pay("V1", "2002", "2024-06-01", "100.20"),  # P001: same vendor, same dollar as P000
            pay("V2", "3003", "2024-01-20", "555.55"),  # P002
            pay(
                "V3", "4004", "2024-03-01", "555.55"
            ),  # P003: same exact amount, 41 days after P002
            pay("V4", "5005", "2024-01-01", "777.00"),  # P004
            pay("V5", "5005", "2025-06-01", "888.00"),  # P005: same canonical number, other vendor
            pay("V6", "6006", "2024-01-01", "300.00"),  # P006
            pay("V7", "6007", "2024-11-01", "300.00"),  # P007: same amount, 305 days apart
        ]
    )
    cand = build_candidates(p)
    pairs = _pairs(cand, p)
    assert pairs[("P000", "P001")] == 1  # cluster + rounded amount
    assert pairs[("P002", "P003")] == 2  # exact amount within the window
    assert pairs[("P004", "P005")] == 4  # canonical invoice across vendors
    assert ("P006", "P007") not in pairs
    assert len(pairs) == len(cand) == 3
    assert cand.report["pairs_per_block"] == dict(zip(BLOCK_NAMES, (1, 1, 1), strict=True))


def test_a_pair_found_by_two_blocks_is_kept_once_with_both_flags():
    p = make_payments(
        [pay("V1", "1001", "2024-01-01", "100.00"), pay("V1", "1001", "2024-01-05", "100.00")]
    )
    cand = build_candidates(p)
    assert len(cand) == 1
    assert int(cand.flags[0]) == 7


def test_vendor_cluster_not_vendor_id_defines_the_first_block():
    rows = [pay("V1", "1001", "2024-01-01", "250.00"), pay("V2", "9009", "2024-09-01", "250.30")]
    separate = build_candidates(make_payments(rows))
    merged = build_candidates(make_payments(rows, clusters={"V1": "CL-A", "V2": "CL-A"}))
    assert len(separate) == 0
    assert len(merged) == 1


def test_report_gives_the_reduction_ratio():
    p = make_payments(
        [pay("V1", "1", amount="10.00"), pay("V1", "1", amount="10.00"), pay("V2", "2")]
    )
    cand = build_candidates(p)
    assert cand.report["all_pairs"] == 3
    assert cand.report["candidate_pairs"] == 1
    assert np.isclose(cand.report["reduction_ratio"], 1 - 1 / 3)


def test_blocking_keeps_a_tiny_share_of_pairs_and_nearly_every_duplicate(pipeline):
    p, cand, truth = pipeline["payments"], pipeline["cand"], pipeline["truth"]
    assert len(cand) <= 0.005 * p.total_pairs
    report = json.loads((pipeline["dir"] / "blocking_report.json").read_text())
    assert report["candidate_pairs"] == len(cand)
    assert report["reduction_ratio"] > 0.999
    keys = {tuple(sorted((p.ids[a], p.ids[b]))) for a, b in zip(cand.a, cand.b, strict=True)}
    recall = sum(pair in keys for pair in truth.duplicates) / len(truth.duplicates)
    assert recall >= 0.98


def test_a_credit_memo_is_compared_with_the_invoice_it_cancels():
    p = make_payments(
        [pay("V1", "8001", "2024-02-01", "900.00"), pay("V1", "CM-8001", "2024-02-10", "-900.00")]
    )
    cand = build_candidates(p)
    assert len(cand) == 1
    assert int(cand.flags[0]) == 1
