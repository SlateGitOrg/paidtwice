import pandas as pd
import pytest

from paidtwice.distance import damerau_levenshtein
from paidtwice.generate import MECHANISMS, GeneratorConfig, generate_dataset
from paidtwice.normalize import canonical_invoice


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset()


def test_sizes_match_the_documented_data(dataset):
    assert len(dataset.vendors) == 4000
    assert len(dataset.payments) == 30000
    assert dataset.vendors["vendor_id"].is_unique and dataset.payments["payment_id"].is_unique
    amounts = dataset.payments["amount"].astype(float)
    assert amounts[amounts > 0].between(50, 250_000).all()
    assert dataset.payments["department"].nunique() == 20
    po_share = (dataset.payments["po_number"] != "").mean()
    assert 0.55 < po_share < 0.65
    assert set(dataset.payments["method"]) == {"check", "ach", "pcard"}


def test_planted_counts(dataset):
    pay = dataset.truth[dataset.truth["entity"] == "payment"]
    dups = pay[pay["role"] == "duplicate"]
    assert len(dups) == 180
    assert dups["label"].value_counts().to_dict() == MECHANISMS
    assert (pay["label"] == "recurring").sum() == 300
    assert pay[pay["label"] == "recurring"].groupby("group_id").size().eq(12).all()
    assert (pay["role"] == "installment").sum() == 160
    assert (pay["label"] == "credit_memo").sum() == 180
    vendors = dataset.truth[dataset.truth["entity"] == "vendor"]
    assert (vendors["role"] == "duplicate_record").sum() == 120
    assert vendors[vendors["role"] == "duplicate_record"]["group_id"].nunique() == 100


def _duplicate_pairs(dataset):
    truth = dataset.truth[dataset.truth["entity"] == "payment"]
    payments = dataset.payments.set_index("payment_id")
    for _, group in truth[truth["label"].isin(MECHANISMS)].groupby("group_id"):
        original = group[group["role"] == "original"]["entity_id"].iloc[0]
        duplicate = group[group["role"] == "duplicate"]["entity_id"].iloc[0]
        yield group["label"].iloc[0], payments.loc[original], payments.loc[duplicate]


def test_each_mechanism_has_its_signature(dataset):
    gaps = {m: [] for m in MECHANISMS}
    for mech, first, second in _duplicate_pairs(dataset):
        gap = abs((pd.Timestamp(second.invoice_date) - pd.Timestamp(first.invoice_date)).days)
        gaps[mech].append(gap)
        ca, cb = canonical_invoice(first.invoice_number), canonical_invoice(second.invoice_number)
        if mech in ("punctuation_variant", "ocr_confusable"):
            assert first.invoice_number != second.invoice_number and ca == cb
        elif mech == "digit_transposition":
            assert 1 <= damerau_levenshtein(ca, cb) <= 2
        elif mech == "duplicate_vendor_record":
            assert first.vendor_id != second.vendor_id and ca == cb
        elif mech == "check_and_pcard":
            assert {first.method, second.method} == {"check", "pcard"}
        elif mech == "exact_resubmission":
            assert first.invoice_number == second.invoice_number
            assert first.vendor_id == second.vendor_id and first.amount == second.amount
    assert min(gaps["exact_resubmission"]) >= 45 and max(gaps["exact_resubmission"]) <= 90
    assert max(gaps["check_and_pcard"]) <= 14


def test_recurring_series_have_equal_amounts_and_monthly_gaps(dataset):
    truth = dataset.truth[dataset.truth["label"] == "recurring"]
    pay = dataset.payments.set_index("payment_id")
    for _, group in truth.groupby("group_id"):
        rows = pay.loc[group["entity_id"]].sort_values("invoice_date")
        assert rows["amount"].nunique() == 1 and rows["vendor_id"].nunique() == 1
        days = pd.to_datetime(rows["invoice_date"]).diff().dt.days.dropna()
        assert days.between(26, 34).all()


def test_credit_memo_triples_net_to_one_payment(dataset):
    truth = dataset.truth[dataset.truth["label"] == "credit_memo"]
    pay = dataset.payments.set_index("payment_id")
    for _, group in truth.groupby("group_id"):
        rows = pay.loc[group["entity_id"]]
        amounts = sorted(rows["amount"].astype(float))
        assert amounts[0] == -amounts[1] == -amounts[2]
        assert rows["invoice_number"].str.startswith("CM").sum() == 1


def test_same_seed_gives_identical_data_and_other_seeds_differ():
    a, b = generate_dataset(GeneratorConfig(seed=3)), generate_dataset(GeneratorConfig(seed=3))
    other = generate_dataset(GeneratorConfig(seed=4))
    for name in ("vendors", "payments", "truth"):
        assert getattr(a, name).equals(getattr(b, name))
    assert not a.payments.equals(other.payments)


def test_vendor_names_are_unique_businesses_apart_from_duplicate_records(dataset):
    from paidtwice.normalize import normalize_vendor_name

    truth = dataset.truth[dataset.truth["entity"] == "vendor"].set_index("entity_id")
    masters = dataset.vendors[dataset.vendors["vendor_id"].map(truth["role"]) == "master"][
        "name"
    ].map(normalize_vendor_name)
    assert masters.is_unique
    assert len(masters) == 3880
