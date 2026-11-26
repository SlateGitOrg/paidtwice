"""Acceptance checks on the seeded data: the numbers the README reports."""

import pytest
from typer.testing import CliRunner

from paidtwice import data as files
from paidtwice.cli import app
from paidtwice.evaluate import evaluate
from paidtwice.suppress import RULES

BUDGET = 200


@pytest.fixture(scope="module")
def result(pipeline):
    return evaluate(
        pipeline["payments"],
        pipeline["truth"],
        pipeline["cand"],
        pipeline["queue"],
        pipeline["queue_ns"],
        BUDGET,
    )


def test_review_budget_200_precision_recall_and_dollar_recall(result):
    assert result["queue_size"] == BUDGET
    assert result["precision"] >= 0.85
    assert result["recall"] >= 0.90
    assert result["dollar_recall"] >= 0.95
    assert result["blocking_recall"] >= 0.98


def test_every_mechanism_has_recall_of_at_least_80_percent(result):
    by_mechanism = result["recall_by_mechanism"]
    assert len(by_mechanism) == 6
    for name, row in by_mechanism.items():
        assert row["recall"] >= 0.80, name


def test_suppression_keeps_all_look_alikes_out_of_the_queue(result):
    assert result["trap_false_positives_with_suppression"] == {
        "credit_memo": 0,
        "installment": 0,
        "recurring": 0,
    }


def test_without_suppression_look_alikes_flood_the_queue(result):
    assert sum(result["trap_false_positives_without_suppression"].values()) >= 20


def test_queue_is_ranked_and_complete(pipeline):
    queue = pipeline["queue"]
    scores = queue["score"].astype(float)
    assert list(queue["rank"]) == [str(i) for i in range(1, BUDGET + 1)]
    assert scores.is_monotonic_decreasing
    assert (queue["reason_codes"] == "").all()
    assert set(queue.columns) >= {"score", "matched_levels", "reason_codes", "amount_at_risk"}
    assert (queue["amount_at_risk"].astype(float) > 0).all()


def test_no_suppress_queue_shows_the_rules_that_would_have_fired(pipeline):
    flagged = pipeline["queue_ns"]["reason_codes"]
    assert (flagged != "").sum() >= 20
    codes = {c for cell in flagged for c in cell.split(",") if c}
    assert codes <= set(RULES)


def test_suppressed_file_lists_every_removed_pair_with_a_reason(pipeline):
    suppressed = files.read_csv(pipeline["dir"] / files.SUPPRESSED)
    assert len(suppressed) > 1500
    assert (suppressed["reason_codes"] != "").all()


def test_evaluation_markdown_was_written(pipeline):
    text = (pipeline["dir"] / files.EVALUATION).read_text()
    assert "Precision at budget" in text and "Without suppression" in text


def test_same_seed_gives_an_identical_review_queue(pipeline, tmp_path):
    runner = CliRunner()
    for step in (["generate"], ["vendors"], ["candidates"], ["fit"], ["score"]):
        assert runner.invoke(app, [*step, "-d", str(tmp_path)]).exit_code == 0
    assert (tmp_path / files.QUEUE).read_bytes() == (pipeline["dir"] / files.QUEUE).read_bytes()
    assert (tmp_path / files.MODEL).read_text() == (pipeline["dir"] / files.MODEL).read_text()


def test_steps_report_a_missing_earlier_step(tmp_path):
    result = CliRunner().invoke(app, ["candidates", "-d", str(tmp_path)])
    assert result.exit_code != 0
    assert "not found" in str(result.exception)


def test_cli_prints_the_em_weight_table_and_blocking_counts(pipeline):
    assert "log2 weight" in pipeline["outputs"]["fit"]
    assert "invoice canonical_equal" in pipeline["outputs"]["fit"]
    assert "reduction ratio" in pipeline["outputs"]["candidates"]
    assert "recurring_series" in pipeline["outputs"]["score"]


def test_credit_memo_rule_fires_on_the_seeded_memo_pairs(pipeline):
    """Each of the 60 triples gives two memo pairs (invoice-memo and memo-rebill)."""
    suppressed = files.read_csv(pipeline["dir"] / files.SUPPRESSED)
    memo_rows = suppressed[suppressed["reason_codes"].str.contains("credit_memo")]
    by_rule = memo_rows["reason_codes"].str.split(",").explode().value_counts()
    assert by_rule["credit_memo"] == 120
    assert by_rule["credit_memo_rebill"] == 60
    truth = pipeline["truth"]
    memo_pairs = {(a, b) for (a, b), kind in truth.traps.items() if kind == "credit_memo"}
    keys = {
        tuple(sorted(x)) for x in zip(suppressed["payment_a"], suppressed["payment_b"], strict=True)
    }
    assert len(memo_pairs & keys) == 180
