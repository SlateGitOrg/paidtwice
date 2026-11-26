import pytest

from paidtwice.normalize import (
    canonical_invoice,
    is_credit_memo_number,
    normalize_vendor_name,
    parse_invoice,
    split_installment,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("INV-00123", "INV123"),
        ("00123", "123"),
        ("1O45", "1045"),
        ("inv 123", "INV123"),
        ("INV.0123", "INV123"),
        ("I045", "1045"),
        ("10S5", "1055"),
        ("B123", "8123"),
        ("12Z4", "1224"),
        ("L99", "199"),
        ("0", "0"),
        ("X007", "X7"),
    ],
)
def test_canonical_invoice(raw, expected):
    assert canonical_invoice(raw) == expected


def test_variants_of_one_invoice_share_a_canonical_form():
    forms = {
        canonical_invoice(x) for x in ("INV-00123", "INV00123", "inv 123", "INV.123", "INV0123")
    }
    assert forms == {"INV123"}


def test_ocr_map_ignores_letters_that_do_not_touch_a_digit():
    assert canonical_invoice("BILL-STOCK") == "BILLSTOCK"
    assert canonical_invoice("INV-1045") == "INV1045"
    assert canonical_invoice("SOLID") == "SOLID"


def test_installment_suffix_is_split_from_the_base():
    assert split_installment("12345-2") == ("12345", 2)
    assert split_installment("12345/12") == ("12345", 12)
    assert parse_invoice("12345-2").canonical == "12345"
    assert parse_invoice("12345-2").installment == 2


@pytest.mark.parametrize("raw", ["INV-00123", "12345-13", "12345-0", "INV-12", "12345", "12345-02"])
def test_numbers_that_are_not_installments(raw):
    assert split_installment(raw)[1] is None


def test_credit_memo_detection_by_prefix_or_negative_amount():
    assert is_credit_memo_number("CM-1050")
    assert is_credit_memo_number("cm 1050")
    assert not is_credit_memo_number("CMX-1050")
    assert not is_credit_memo_number("INV-1050")
    assert parse_invoice("INV-1050", amount=-20.0).credit_memo
    assert parse_invoice("CM1050", amount=20.0).credit_memo
    assert not parse_invoice("INV-1050", amount=20.0).credit_memo


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Acme Supply, Inc.", "ACME SUPPLY"),
        ("ACME SUPPLY INCORPORATED", "ACME SUPPLY"),
        ("Smith & Sons Co.", "SMITH AND SONS"),
        ("O'Brien Labs LLC", "OBRIEN LABS"),
        ("Acme, Supply Corp", "ACME SUPPLY"),
        ("Co", "CO"),
    ],
)
def test_normalize_vendor_name(raw, expected):
    assert normalize_vendor_name(raw) == expected
