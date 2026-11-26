"""File layout of a working directory and the prepared payment table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paidtwice.normalize import parse_invoice

VENDORS = "vendors.csv"
PAYMENTS = "payments.csv"
TRUTH = "truth.csv"
CLUSTERS = "vendor_clusters.csv"
CANDIDATES = "candidates.csv"
BLOCKING_REPORT = "blocking_report.json"
MODEL = "model.json"
QUEUE = "queue.csv"
QUEUE_NO_SUPPRESS = "queue_no_suppress.csv"
SUPPRESSED = "suppressed.csv"
EVALUATION = "evaluation.md"


@dataclass
class Payments:
    """Payments as parallel arrays, in the order of ``payments.csv``."""

    ids: np.ndarray
    vendor_id: np.ndarray
    cluster: np.ndarray
    invoice_raw: np.ndarray
    invoice_key: np.ndarray
    installment: np.ndarray  # 0 when the number has no installment suffix
    credit_memo: np.ndarray
    amount_cents: np.ndarray
    day: np.ndarray
    po: np.ndarray
    method: np.ndarray
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def total_pairs(self) -> int:
        n = len(self)
        return n * (n - 1) // 2

    def index_of(self, payment_ids) -> np.ndarray:
        return pd.Index(self.ids).get_indexer(payment_ids)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with every column as text and blanks kept as empty strings."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the earlier pipeline step first")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare_payments(payments: pd.DataFrame, clusters: pd.DataFrame) -> Payments:
    """Attach vendor clusters and parse invoice numbers and amounts."""
    cluster_of = dict(zip(clusters["vendor_id"], clusters["cluster_id"], strict=True))
    amounts = payments["amount"].astype(float).to_numpy()
    parsed = [
        parse_invoice(raw, amount)
        for raw, amount in zip(payments["invoice_number"], amounts, strict=True)
    ]
    days = (pd.to_datetime(payments["invoice_date"]) - pd.Timestamp("1970-01-01")).dt.days
    return Payments(
        ids=payments["payment_id"].to_numpy(dtype=object),
        vendor_id=payments["vendor_id"].to_numpy(dtype=object),
        cluster=np.array([cluster_of[v] for v in payments["vendor_id"]], dtype=object),
        invoice_raw=payments["invoice_number"].to_numpy(dtype=object),
        invoice_key=np.array([p.canonical for p in parsed], dtype=object),
        installment=np.array([p.installment or 0 for p in parsed], dtype=np.int16),
        credit_memo=np.array([p.credit_memo for p in parsed], dtype=bool),
        amount_cents=np.rint(amounts * 100).astype(np.int64),
        day=days.to_numpy(dtype=np.int64),
        po=payments["po_number"].to_numpy(dtype=object),
        method=payments["method"].to_numpy(dtype=object),
        frame=payments,
    )


def load_payments(data_dir: Path) -> Payments:
    """Load ``payments.csv`` together with the clusters written by ``vendors``."""
    return prepare_payments(read_csv(data_dir / PAYMENTS), read_csv(data_dir / CLUSTERS))
