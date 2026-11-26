"""Seeded generator for the vendor master, the payment ledger and the planted truth.

Properties of the synthetic data (all documented in the README):

* 4,000 vendor records, 120 of them second or third records of 100 vendors.
* 30,000 payments over two fiscal years, lognormal amounts clipped to $50-$250,000,
  20 departments, a purchase order on about 60%, paid by check, ACH or p-card.
* 180 planted duplicate payments in six mechanisms.
* Legitimate look-alikes that a fuzzy matcher flags: 300 recurring monthly
  payments across 25 vendors, 80 equal installment pairs and 60 credit memo
  plus re-bill triples.

The truth file is written next to the data and read only by ``evaluate`` and
the tests.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from paidtwice.vendor_gen import VendorMaster, build_vendor_master

MECHANISMS = {
    "punctuation_variant": 50,
    "ocr_confusable": 30,
    "digit_transposition": 25,
    "duplicate_vendor_record": 35,
    "check_and_pcard": 20,
    "exact_resubmission": 20,
}
TRAP_TYPES = ("recurring", "installment", "credit_memo")

# Letters a vendor may use in an invoice-number prefix: none of the letters OCR confuses with
# digits, and no M so that a prefix can never read as the credit memo marker "CM".
PREFIX_LETTERS = "ACDEFGHJKNPQRTUVWXY"
SEPARATORS = ("-", "", "-", "")
PADDINGS = (0, 5, 6, 0)
_TO_LETTER = {"0": ("O",), "1": ("I", "L"), "2": ("Z",), "5": ("S",), "8": ("B",)}
_RECURRING_KIND = {"Properties": "rent", "Utilities": "utilities", "Software": "software"}


@dataclass(frozen=True)
class GeneratorConfig:
    """Sizes of the synthetic data set. Defaults match the documented data."""

    seed: int = 42
    n_vendor_records: int = 4000
    n_duplicate_records: int = 120
    n_duplicated_vendors: int = 100
    n_payments: int = 30000
    first_day: dt.date = dt.date(2023, 7, 1)
    n_days: int = 731
    mechanisms: dict[str, int] = field(default_factory=lambda: dict(MECHANISMS))
    n_recurring_vendors: int = 25
    recurring_length: int = 12
    n_installment_pairs: int = 80
    n_credit_memo_triples: int = 60


@dataclass
class Dataset:
    vendors: pd.DataFrame
    payments: pd.DataFrame
    truth: pd.DataFrame


@dataclass
class _Style:
    prefix: str
    sep: str
    pad: int

    def format(self, n: int) -> str:
        digits = str(n).zfill(self.pad) if self.pad else str(n)
        return f"{self.prefix}{self.sep if self.prefix else ''}{digits}"


@dataclass
class _Row:
    vendor: str
    number: int
    style: _Style
    day: int
    amount: float
    dept: str
    po: str
    method: str
    raw: str = ""
    group: str = ""
    role: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        if not self.raw:
            self.raw = self.style.format(self.number)


def _random_style(rng: np.random.Generator, numeric_only: float = 0.08) -> _Style:
    """A vendor's invoice-number format: a 3-letter code (or none), a separator, a padding."""
    prefix = ""
    if rng.random() >= numeric_only:
        prefix = "".join(PREFIX_LETTERS[int(k)] for k in rng.integers(0, len(PREFIX_LETTERS), 3))
    return _Style(
        prefix,
        SEPARATORS[int(rng.integers(0, len(SEPARATORS)))],
        PADDINGS[int(rng.integers(0, len(PADDINGS)))],
    )


# ---------------------------------------------------------------- invoice-number variants


def _punctuation_variant(row: _Row, rng: np.random.Generator) -> str:
    """Same number written differently: separator, padding or case."""
    style, digits = row.style, str(row.number)
    for _ in range(50):
        sep = rng.choice(["", "-", " ", "."]) if style.prefix else ""
        pad = int(rng.choice([0, len(digits) + 1, len(digits) + 2]))
        text = f"{style.prefix}{sep}{digits.zfill(pad) if pad else digits}"
        if style.prefix and rng.random() < 0.3:
            text = text.lower()
        if text != row.raw:
            return text
    raise AssertionError("no variant found")


def _ocr_variant(row: _Row, rng: np.random.Generator) -> str:
    """Replace one digit by the letter it is mistaken for."""
    match = list(re.finditer(r"\d+", row.raw))[-1]
    positions = [i for i in range(match.start(), match.end()) if row.raw[i] in _TO_LETTER]
    pos = positions[int(rng.integers(0, len(positions)))]
    options = _TO_LETTER[row.raw[pos]]
    return row.raw[:pos] + options[int(rng.integers(0, len(options)))] + row.raw[pos + 1 :]


def _swappable(digits: str) -> list[int]:
    return [i for i in range(len(digits) - 1) if digits[i] != digits[i + 1]]


def _transposition_variant(row: _Row, rng: np.random.Generator, swaps: int) -> str | None:
    """Swap one (``swaps=1``) or two disjoint (``swaps=2``) pairs of adjacent digits."""
    digits = list(str(row.number))
    chosen: list[int] = []
    for i in rng.permutation(_swappable("".join(digits))):
        i = int(i)
        if all(abs(i - c) > 1 for c in chosen):
            chosen.append(i)
        if len(chosen) == swaps:
            break
    if len(chosen) < swaps:
        return None
    for i in chosen:
        digits[i], digits[i + 1] = digits[i + 1], digits[i]
    if digits[0] == "0":
        return None
    number = int("".join(digits))
    return _Style(row.style.prefix, row.style.sep, row.style.pad).format(number)


# ---------------------------------------------------------------- generation


def generate_dataset(config: GeneratorConfig | None = None) -> Dataset:
    """Build the whole synthetic data set deterministically from ``config.seed``."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    master = build_vendor_master(
        rng, cfg.n_vendor_records, cfg.n_duplicate_records, cfg.n_duplicated_vendors
    )
    vendor_ids = list(master.frame["vendor_id"])
    n_dups = sum(cfg.mechanisms.values())
    n_trap_rows = (
        cfg.n_recurring_vendors * cfg.recurring_length
        + cfg.n_installment_pairs * 2
        + cfg.n_credit_memo_triples * 3
    )
    n_background = cfg.n_payments - n_dups - n_trap_rows
    rows = _background(rng, cfg, vendor_ids, master, n_background)
    groups = 0

    def new_group() -> str:
        nonlocal groups
        groups += 1
        return f"G{groups:05d}"

    used: set[int] = set()
    for mechanism, count in cfg.mechanisms.items():
        for _ in range(count):
            original = _pick_original(rng, rows, master, mechanism, cfg, used)
            gid = new_group()
            rows[original].group, rows[original].role, rows[original].label = (
                gid,
                "original",
                mechanism,
            )
            dup = _make_duplicate(rng, rows[original], master, mechanism)
            dup.group, dup.role, dup.label = gid, "duplicate", mechanism
            rows.append(dup)
    avoid = {r.vendor for r in rows if r.group}
    _add_recurring(rng, cfg, rows, master, new_group, avoid)
    _add_installments(rng, cfg, rows, vendor_ids, new_group)
    _add_credit_memo_triples(rng, cfg, rows, vendor_ids, new_group)
    assert len(rows) == cfg.n_payments, (len(rows), cfg.n_payments)
    return _assemble(rng, cfg, master, rows)


def _background(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    vendor_ids: list[str],
    master: VendorMaster,
    n_background: int,
) -> list[_Row]:
    weights = rng.lognormal(0.0, 0.9, len(vendor_ids))
    for k, vid in enumerate(vendor_ids):
        if vid in master.duplicate_records:
            weights[k] *= 0.3
    counts = rng.multinomial(n_background, weights / weights.sum())
    rows: list[_Row] = []
    for vid, count in zip(vendor_ids, counts, strict=True):
        if count == 0:
            continue
        style = _random_style(rng)
        days = np.sort(rng.integers(0, cfg.n_days, int(count)))
        number = int(rng.integers(1000, 2_000_000))
        steps = rng.choice([1, 2, 3, 5, 8, 13], size=int(count), p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        billed: set[int] = set()
        for day, step in zip(days, steps, strict=True):
            number += int(step)
            amount = _amount(rng)
            while round(amount) in billed:  # a vendor does not bill the same dollars twice
                amount = _amount(rng)
            billed.add(round(amount))
            rows.append(
                _Row(
                    vendor=vid,
                    number=number,
                    style=style,
                    day=int(day),
                    amount=amount,
                    dept=f"D{int(rng.integers(1, 21)):02d}",
                    po=_po(rng, 0.6),
                    method=str(rng.choice(["check", "ach", "pcard"], p=[0.35, 0.5, 0.15])),
                )
            )
    return rows


def _amount(rng: np.random.Generator) -> float:
    value = float(np.clip(rng.lognormal(7.3, 1.4), 50.0, 250_000.0))
    if rng.random() < 0.12:  # round-number invoices (fixed-price services)
        step = 50 if value < 1000 else 250 if value < 10_000 else 1000
        return float(max(step, round(value / step) * step))
    return round(value, 2)


def _po(rng: np.random.Generator, probability: float) -> str:
    return f"PO{int(rng.integers(100000, 999999))}" if rng.random() < probability else ""


def _has_ocr_digit(row: _Row) -> bool:
    return any(ch in _TO_LETTER for ch in str(row.number))


def _pick_original(
    rng: np.random.Generator,
    rows: list[_Row],
    master: VendorMaster,
    mechanism: str,
    cfg: GeneratorConfig,
    used: set[int],
) -> int:
    """Choose a plain background invoice to be paid a second time."""
    limit = cfg.n_days - 100
    for idx in rng.permutation(len(rows)):
        idx = int(idx)
        row = rows[idx]
        if idx in used or row.group or row.day > limit:
            continue
        if mechanism == "ocr_confusable" and not _has_ocr_digit(row):
            continue
        if mechanism == "digit_transposition" and not _swappable(str(row.number)):
            continue
        if (
            mechanism == "duplicate_vendor_record"
            and len(master.records_of(master.true_vendor[row.vendor])) < 2
        ):
            continue
        used.add(idx)
        return idx
    raise RuntimeError(f"no eligible original for {mechanism}")


def _make_duplicate(
    rng: np.random.Generator, original: _Row, master: VendorMaster, mechanism: str
) -> _Row:
    gap_ranges = {
        "punctuation_variant": (0, 26),
        "ocr_confusable": (0, 26),
        "digit_transposition": (0, 26),
        "duplicate_vendor_record": (0, 31),
        "check_and_pcard": (0, 15),
        "exact_resubmission": (45, 91),
    }
    low, high = gap_ranges[mechanism]
    dup = _Row(
        vendor=original.vendor,
        number=original.number,
        style=original.style,
        day=original.day + int(rng.integers(low, high)),
        amount=original.amount,
        dept=original.dept if rng.random() < 0.7 else f"D{int(rng.integers(1, 21)):02d}",
        po=original.po if rng.random() < 0.85 else "",
        method=original.method,
        raw=original.raw,
    )
    if mechanism == "punctuation_variant":
        dup.raw = _punctuation_variant(original, rng)
    elif mechanism == "ocr_confusable":
        dup.raw = _ocr_variant(original, rng)
    elif mechanism == "digit_transposition":
        for _ in range(100):
            swaps = 2 if rng.random() < 0.3 else 1
            text = _transposition_variant(original, rng, swaps)
            if text is not None:
                dup.raw = text
                break
        else:
            raise AssertionError("no transposition found")
    elif mechanism == "duplicate_vendor_record":
        records = [
            v
            for v in master.records_of(master.true_vendor[original.vendor])
            if v != original.vendor
        ]
        dup.vendor = records[int(rng.integers(0, len(records)))]
        if rng.random() < 0.3:
            dup.raw = _punctuation_variant(original, rng)
    elif mechanism == "check_and_pcard":
        if rng.random() < 0.6:
            original.method, dup.method = "check", "pcard"
        else:
            original.method, dup.method = "pcard", "check"
        if rng.random() < 0.25:  # the card processor's surcharge
            dup.amount = round(original.amount * 1.003, 2)
    return dup


def _add_recurring(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    rows: list[_Row],
    master: VendorMaster,
    new_group,
    avoid: set[str],
) -> None:
    """Monthly rent, utilities and subscriptions: same vendor, same amount, ~30 days apart."""
    candidates = [
        v
        for v, line in master.line_of_business.items()
        if line in _RECURRING_KIND
        and v not in avoid
        and v not in master.duplicate_records
        and len(master.records_of(master.true_vendor[v])) == 1
    ]
    chosen = rng.choice(len(candidates), size=cfg.n_recurring_vendors, replace=False)
    for k in chosen:
        vid = candidates[int(k)]
        kind = _RECURRING_KIND[master.line_of_business[vid]]
        low, high = {"rent": (4000, 60000), "utilities": (500, 8000), "software": (1000, 30000)}[
            kind
        ]
        amount = round(float(rng.uniform(low, high)), 2)
        style = _Style(_random_style(rng, 0.0).prefix, "-", 6)
        number = int(rng.integers(1000, 80000))
        blanket_po = _po(rng, 0.7)
        anchor = int(rng.integers(0, cfg.n_days - 30 * cfg.recurring_length - 5))
        gid = new_group()
        for month in range(cfg.recurring_length):
            rows.append(
                _Row(
                    vendor=vid,
                    number=number + month,
                    style=style,
                    day=anchor + 30 * month + int(rng.integers(-2, 3)) + 2,
                    amount=amount,
                    dept=f"D{int(rng.integers(1, 21)):02d}",
                    po=blanket_po,
                    method="ach",
                    group=gid,
                    role="series",
                    label="recurring",
                )
            )


def _add_installments(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    rows: list[_Row],
    vendor_ids: list[str],
    new_group,
) -> None:
    """One invoice paid in two equal halves, numbered ``-1`` and ``-2``."""
    for _ in range(cfg.n_installment_pairs):
        vid = vendor_ids[int(rng.integers(0, len(vendor_ids)))]
        style = _Style(_random_style(rng, 0.0).prefix, "-", 0)
        number = int(rng.integers(1000, 80000))
        half = round(max(_amount(rng), 100.0) / 2, 2)
        day = int(rng.integers(0, cfg.n_days - 45))
        po = _po(rng, 0.6)
        dept = f"D{int(rng.integers(1, 21)):02d}"
        method = str(rng.choice(["check", "ach"]))
        mark = "/" if rng.random() < 0.25 else "-"
        gid = new_group()
        for part, gap in ((1, 0), (2, int(rng.integers(28, 36)))):
            rows.append(
                _Row(
                    vendor=vid,
                    number=number,
                    style=style,
                    day=day + gap,
                    amount=half,
                    dept=dept,
                    po=po,
                    method=method,
                    raw=f"{style.format(number)}{mark}{part}",
                    group=gid,
                    role="installment",
                    label="installment",
                )
            )


def _add_credit_memo_triples(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    rows: list[_Row],
    vendor_ids: list[str],
    new_group,
) -> None:
    """An invoice, the credit memo that cancels it and the corrected re-bill."""
    for _ in range(cfg.n_credit_memo_triples):
        vid = vendor_ids[int(rng.integers(0, len(vendor_ids)))]
        style = _Style(_random_style(rng, 0.0).prefix, "-", 0)
        number = int(rng.integers(1000, 80000))
        amount = _amount(rng)
        day = int(rng.integers(0, cfg.n_days - 60))
        memo_day = day + int(rng.integers(3, 21))
        rebill_day = memo_day + int(rng.integers(1, 21))
        po = _po(rng, 0.6)
        dept = f"D{int(rng.integers(1, 21)):02d}"
        method = str(rng.choice(["check", "ach"]))
        rebill_raw = style.format(number) if rng.random() < 0.5 else style.format(number) + "A"
        gid = new_group()
        for role, raw, d, amt in (
            ("original", style.format(number), day, amount),
            ("credit_memo", f"CM-{number}", memo_day, -amount),
            ("rebill", rebill_raw, rebill_day, amount),
        ):
            rows.append(
                _Row(
                    vendor=vid,
                    number=number,
                    style=style,
                    day=d,
                    amount=amt,
                    dept=dept,
                    po=po,
                    method=method,
                    raw=raw,
                    group=gid,
                    role=role,
                    label="credit_memo",
                )
            )


def _assemble(
    rng: np.random.Generator, cfg: GeneratorConfig, master: VendorMaster, rows: list[_Row]
) -> Dataset:
    tiebreak = rng.random(len(rows))
    order = sorted(range(len(rows)), key=lambda i: (rows[i].day, tiebreak[i]))
    payments, truth = [], []
    for seq, i in enumerate(order, start=1):
        row = rows[i]
        pid = f"PAY-{seq:06d}"
        payments.append(
            {
                "payment_id": pid,
                "vendor_id": row.vendor,
                "invoice_number": row.raw,
                "invoice_date": (cfg.first_day + dt.timedelta(days=row.day)).isoformat(),
                "amount": f"{row.amount:.2f}",
                "department": row.dept,
                "po_number": row.po,
                "method": row.method,
            }
        )
        if row.group:
            truth.append(("payment", pid, row.group, row.role, row.label))
    for vid, true_key in master.true_vendor.items():
        role = "duplicate_record" if vid in master.duplicate_records else "master"
        truth.append(("vendor", vid, true_key, role, master.duplicate_records.get(vid, "")))
    truth_frame = pd.DataFrame(truth, columns=["entity", "entity_id", "group_id", "role", "label"])
    return Dataset(master.frame, pd.DataFrame(payments), truth_frame)


def write_dataset(dataset: Dataset, out_dir) -> None:
    """Write ``vendors.csv``, ``payments.csv`` and ``truth.csv`` into ``out_dir``."""
    from pathlib import Path

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    dataset.vendors.to_csv(path / "vendors.csv", index=False)
    dataset.payments.to_csv(path / "payments.csv", index=False)
    dataset.truth.to_csv(path / "truth.csv", index=False)
