from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from paidtwice import data as files
from paidtwice.candidates import load_candidates
from paidtwice.cli import app
from paidtwice.em import FSModel
from paidtwice.evaluate import Truth


def make_payments(rows: list[dict], clusters: dict[str, str] | None = None) -> files.Payments:
    """Payments from short dicts; vendor ``V1`` is its own cluster unless ``clusters`` says more."""
    defaults = {"po_number": "", "method": "ach", "department": "D01"}
    frame = pd.DataFrame(
        [
            {"payment_id": f"P{i:03d}", **defaults, **{k: str(v) for k, v in row.items()}}
            for i, row in enumerate(rows)
        ]
    )
    vendors = sorted(set(frame["vendor_id"]))
    mapping = clusters or {}
    table = pd.DataFrame(
        {"vendor_id": vendors, "cluster_id": [mapping.get(v, f"CL-{v}") for v in vendors]}
    )
    return files.prepare_payments(frame, table)


def pay(vendor="V1", number="1001", date="2024-01-10", amount="100.00", **extra) -> dict:
    return {
        "vendor_id": vendor,
        "invoice_number": number,
        "invoice_date": date,
        "amount": amount,
        **extra,
    }


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """The whole pipeline on the default seeded data, run through the CLI once."""
    data = tmp_path_factory.mktemp("paidtwice")
    runner = CliRunner()
    steps = [
        ["generate", "-d", str(data)],
        ["vendors", "-d", str(data)],
        ["candidates", "-d", str(data)],
        ["fit", "-d", str(data)],
        ["score", "-d", str(data)],
        ["score", "-d", str(data), "--no-suppress"],
        ["evaluate", "-d", str(data)],
    ]
    outputs = {}
    for step in steps:
        result = runner.invoke(app, step)
        assert result.exit_code == 0, (step, result.output)
        outputs[" ".join(step[:1] + step[3:])] = result.output
    p = files.load_payments(data)
    return {
        "dir": data,
        "outputs": outputs,
        "payments": p,
        "cand": load_candidates(p, data),
        "model": FSModel.load(data / files.MODEL),
        "queue": files.read_csv(data / files.QUEUE),
        "queue_ns": files.read_csv(data / files.QUEUE_NO_SUPPRESS),
        "truth": Truth.load(data),
        "truth_frame": files.read_csv(data / files.TRUTH),
        "vendors": files.read_csv(data / files.VENDORS),
        "clusters": files.read_csv(data / files.CLUSTERS),
    }
