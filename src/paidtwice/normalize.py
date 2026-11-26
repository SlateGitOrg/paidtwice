"""Canonical forms for invoice numbers and vendor names."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Letters that OCR and tired typists confuse with digits. They are only
# rewritten when they touch a digit, so "INV" and "BILL" keep their letters.
OCR_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2"}

_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_DIGIT_RUN = re.compile(r"\d+")
# "-1".."-12" or "/1".."/12" at the very end, behind a base that contains a digit.
_INSTALLMENT = re.compile(r"^(?P<base>.*?)\s*[-/]\s*(?P<n>[1-9]|1[0-2])$")
_CM_PREFIX = re.compile(r"^CM[^A-Z0-9]*\d")


@dataclass(frozen=True)
class ParsedInvoice:
    """An invoice number split into its comparable parts."""

    canonical: str
    installment: int | None
    credit_memo: bool


def _strip_leading_zeros(text: str) -> str:
    return _DIGIT_RUN.sub(lambda m: m.group().lstrip("0") or "0", text)


def _ocr_to_digits(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        near_digit = (i > 0 and text[i - 1].isdigit()) or (
            i + 1 < len(text) and text[i + 1].isdigit()
        )
        out.append(OCR_TO_DIGIT[ch] if ch in OCR_TO_DIGIT and near_digit else ch)
    return "".join(out)


def split_installment(raw: str) -> tuple[str, int | None]:
    """Split ``"12345-2"`` into ``("12345", 2)``; other numbers come back whole."""
    text = raw.strip()
    match = _INSTALLMENT.match(text)
    if match and any(c.isdigit() for c in match.group("base")):
        return match.group("base"), int(match.group("n"))
    return text, None


def canonical_invoice(raw: str) -> str:
    """Canonical invoice number with any installment suffix removed.

    Steps: uppercase, drop punctuation and spaces, rewrite OCR-confusable
    letters that touch a digit, then strip leading zeros of every digit run.
    """
    base, _ = split_installment(raw)
    text = _NON_ALNUM.sub("", base.upper())
    return _strip_leading_zeros(_ocr_to_digits(text))


def is_credit_memo_number(raw: str) -> bool:
    """True for numbers that start with ``CM`` followed by a digit."""
    return bool(_CM_PREFIX.match(raw.strip().upper()))


def parse_invoice(raw: str, amount: float | None = None) -> ParsedInvoice:
    """Canonical form, installment suffix and credit-memo flag of one invoice."""
    _, installment = split_installment(raw)
    memo = is_credit_memo_number(raw) or (amount is not None and amount < 0)
    return ParsedInvoice(canonical_invoice(raw), installment, memo)


LEGAL_SUFFIXES = frozenset(
    {
        "INC",
        "INCORPORATED",
        "LLC",
        "LTD",
        "LIMITED",
        "CORP",
        "CORPORATION",
        "CO",
        "COMPANY",
        "LP",
        "LLP",
        "PLC",
        "PC",
    }
)
ARTICLES = frozenset({"THE", "A", "AN"})


def normalize_vendor_name(name: str) -> str:
    """Vendor name without case, punctuation, leading articles or legal suffixes."""
    text = name.upper().replace("&", " AND ").replace("'", "")
    tokens = re.sub(r"[^A-Z0-9]+", " ", text).split()
    while tokens and tokens[0] in ARTICLES:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)
