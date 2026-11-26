import numpy as np
from conftest import make_payments, pay

from paidtwice.suppress import (
    CREDIT_MEMO,
    CREDIT_MEMO_REBILL,
    INSTALLMENT_SUFFIX,
    RECURRING_SERIES,
    SEQUENTIAL_INVOICE,
    build_context,
    evaluate_rules,
    find_series,
    reason_codes,
)


def _fired(p, i, j):
    flags = evaluate_rules(p, build_context(p), np.array([i]), np.array([j]))
    return {rule for rule, hit in flags.items() if hit[0]}


def _rent(months=6, amount="2500.00", vendor="V1", start="2024-01-02"):
    import datetime as dt

    first = dt.date.fromisoformat(start)
    return [
        pay(vendor, f"{7000 + 100 * m}", (first + dt.timedelta(days=30 * m)).isoformat(), amount)
        for m in range(months)
    ]


def test_find_series_needs_four_payments_about_a_month_apart():
    assert (find_series(make_payments(_rent(6))) >= 0).all()
    assert (find_series(make_payments(_rent(3))) == -1).all()


def test_series_ignores_other_amounts_and_irregular_gaps():
    rows = _rent(5) + [
        pay("V1", "1", "2024-02-10", "2500.01"),
        pay("V1", "2", "2024-04-05", "99.00"),
    ]
    series = find_series(make_payments(rows))
    assert (series[:5] >= 0).all() and (series[5:] == -1).all()
    weekly = [pay("V1", str(i), f"2024-01-{1 + 7 * i:02d}", "50.00") for i in range(4)]
    assert (find_series(make_payments(weekly)) == -1).all()


def test_recurring_pairs_a_month_or_more_apart_are_suppressed():
    p = make_payments(_rent(6))
    assert RECURRING_SERIES in _fired(p, 0, 1)
    assert RECURRING_SERIES in _fired(p, 0, 5)


def test_rent_paid_twice_in_one_month_is_still_flagged():
    rows = _rent(6) + [pay("V1", "7100", "2024-02-05", "2500.00")]  # second February payment
    p = make_payments(rows)
    original_february = 1
    extra = 6
    assert RECURRING_SERIES not in _fired(p, original_february, extra)
    assert RECURRING_SERIES in _fired(p, 0, original_february)  # the real series is untouched


def test_installments_with_different_suffixes_are_suppressed():
    p = make_payments(
        [
            pay("V1", "5555-1", "2024-01-01", "400.00"),
            pay("V1", "5555-2", "2024-01-31", "400.00"),
            pay("V1", "5555-2", "2024-03-01", "400.00"),
        ]
    )
    assert INSTALLMENT_SUFFIX in _fired(p, 0, 1)
    assert INSTALLMENT_SUFFIX not in _fired(p, 1, 2)  # same suffix twice is a real duplicate


def test_credit_memo_and_rebill_net_to_one_payment():
    p = make_payments(
        [
            pay("V1", "8001", "2024-02-01", "900.00"),
            pay("V1", "CM-8001", "2024-02-10", "-900.00"),
            pay("V1", "8001", "2024-02-20", "900.00"),
            pay("V1", "8001", "2024-02-25", "900.00"),  # a further, un-netted payment
        ]
    )
    assert CREDIT_MEMO_REBILL in _fired(p, 0, 2)
    assert CREDIT_MEMO in _fired(p, 0, 1)
    assert CREDIT_MEMO_REBILL not in _fired(p, 2, 3)  # no memo between these two dates
    assert CREDIT_MEMO_REBILL in _fired(p, 0, 3)  # the memo still sits between 1st and 4th


def test_a_credit_memo_for_another_amount_does_not_explain_a_rebill():
    p = make_payments(
        [
            pay("V1", "8001", "2024-02-01", "900.00"),
            pay("V1", "CM-1", "2024-02-10", "-100.00"),
            pay("V1", "8001", "2024-02-20", "900.00"),
        ]
    )
    assert _fired(p, 0, 2) == set()


def test_sequential_invoice_numbers_are_suppressed_but_not_other_gaps():
    p = make_payments(
        [
            pay("V1", "INV-1044", amount="10.00"),
            pay("V1", "INV-1045", amount="10.00"),
            pay("V1", "INV-1047", amount="10.00"),
            pay("V1", "INV-1043A", amount="10.00"),
        ]
    )
    assert SEQUENTIAL_INVOICE in _fired(p, 0, 1)
    assert SEQUENTIAL_INVOICE not in _fired(p, 1, 2)
    assert SEQUENTIAL_INVOICE not in _fired(p, 0, 3)  # different suffix


def test_digit_transposition_is_never_sequential():
    p = make_payments([pay("V1", "1234"), pay("V1", "1243")])
    assert _fired(p, 0, 1) == set()


def test_reason_codes_join_the_rules_that_fired():
    p = make_payments([pay("V1", "5555-1"), pay("V1", "5555-2")])
    flags = evaluate_rules(p, build_context(p), np.array([0]), np.array([1]))
    assert reason_codes(flags) == [INSTALLMENT_SUFFIX]
