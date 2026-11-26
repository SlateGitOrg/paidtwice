"""Synthetic vendor master with planted duplicate vendor records."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ADJECTIVES = (
    "Ironwood Bluepeak Summit Harbor Cedar Granite Meridian Pioneer Redwood Silverline Northgate "
    "Westbrook Eastfield Lakeshore Stonebridge Brightwater Falcon Juniper Keystone Lantern Maple "
    "Nimbus Orchard Pinnacle Quarry Riverbend Sterling Timberline Union Vanguard Willow Alder "
    "Beacon Crestview Dunmore Elmhurst Foxglove Glenview Hollis Ivory Jasper Kingsley Larkspur "
    "Mercer Norwood Oakmont Prairie Quincy Rosewood Sagebrush Tidewater Upland Valleyview "
    "Wexford Yarrow Zenith Ashford Bramble Copperfield Driftwood"
).split()
SECTORS = (
    "Industrial Medical Digital Scientific Laboratory Campus Municipal Regional Applied Advanced "
    "Dynamic Precision Integrated Coastal Central Global Allied Premier Standard Universal "
    "Clinical Civic Academic Technical Mechanical Environmental Urban Rural Metro National "
    "Interstate Pacific Atlantic Northern Southern Eastern Western Modern Classic Prime Select "
    "Apex Core Unified Superior Reliable Trusted Express Rapid"
).split()
LINES = (
    "Supply Systems Partners Logistics Services Solutions Equipment Instruments Properties "
    "Utilities Software Consulting Catering Printing Furnishings Maintenance Security Travel "
    "Staffing Telecom Electric Water Energy Analytics Networks Facilities Textiles Chemicals "
    "Pharma Optics Fabrication Transport Rentals Labs Publishing Hardware Cleaning Landscaping "
    "Elevators Controls"
).split()
LEGAL = ("Inc.", "LLC", "Corp.", "Co.", "Ltd.", "LLP", "")

# The same business as it appears in a second vendor record.
_LEGAL_FORMS = {"Inc.": "Incorporated", "Corp.": "Corporation", "Co.": "Company", "Ltd.": "Limited"}


@dataclass
class VendorMaster:
    """Vendor records plus the truth about which records are one business."""

    frame: pd.DataFrame
    true_vendor: dict[str, str]
    duplicate_records: dict[str, str] = field(default_factory=dict)  # record -> sharing mode
    line_of_business: dict[str, str] = field(default_factory=dict)
    _by_true: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for vendor, true_key in self.true_vendor.items():
            self._by_true.setdefault(true_key, []).append(vendor)

    def records_of(self, true_key: str) -> list[str]:
        return self._by_true[true_key]


def _name_variant(name: str, rng: np.random.Generator) -> str:
    """Another way a clerk could type the same business name."""
    parts = name.split()
    kind = int(rng.integers(0, 5))
    if kind == 0 and parts[-1] in _LEGAL_FORMS:
        return " ".join(parts[:-1] + [_LEGAL_FORMS[parts[-1]]])
    if kind == 1:
        return name.upper().replace(".", "")
    if kind == 2:
        return "The " + name
    if kind == 3:  # a dropped letter in the first word: normalised names no longer agree
        word = parts[0]
        cut = int(rng.integers(1, len(word) - 1))
        return " ".join([word[:cut] + word[cut + 1 :]] + parts[1:])
    return name.replace(" ", ", ", 1)


def build_vendor_master(
    rng: np.random.Generator, n_records: int, n_duplicate_records: int, n_duplicated_vendors: int
) -> VendorMaster:
    """Create ``n_records`` vendor rows, ``n_duplicate_records`` of them second records."""
    n_true = n_records - n_duplicate_records
    combos = rng.choice(len(ADJECTIVES) * len(SECTORS) * len(LINES), size=n_true, replace=False)
    zips = rng.integers(10000, 99999, size=n_true)
    tax = rng.choice(900_000_000, size=n_true + n_duplicate_records, replace=False) + 100_000_000
    bank = (
        rng.choice(9_000_000_000, size=n_true + n_duplicate_records, replace=False) + 1_000_000_000
    )

    rows: list[dict[str, str]] = []
    line_of: dict[str, str] = {}
    for k in range(n_true):
        a, rest = divmod(int(combos[k]), len(SECTORS) * len(LINES))
        s, ln = divmod(rest, len(LINES))
        legal = LEGAL[int(rng.integers(0, len(LEGAL)))]
        name = f"{ADJECTIVES[a]} {SECTORS[s]} {LINES[ln]}" + (f" {legal}" if legal else "")
        tid = str(tax[k])
        rows.append(
            {
                "_true": f"T{k + 1:05d}",
                "_mode": "",
                "name": name,
                "tax_id": f"{tid[:2]}-{tid[2:]}",
                "bank_account": str(bank[k]),
                "zip": f"{zips[k]:05d}",
            }
        )
        line_of[rows[-1]["_true"]] = LINES[ln]

    duplicated = rng.choice(n_true, size=n_duplicated_vendors, replace=False)
    extra = np.concatenate(
        [duplicated, duplicated[: n_duplicate_records - n_duplicated_vendors]]
    )  # the first few vendors get a third record
    for j, k in enumerate(extra):
        base = rows[int(k)]
        mode = ("tax", "bank", "both")[int(rng.choice(3, p=[0.45, 0.45, 0.10]))]
        fresh_tax = str(tax[n_true + j])
        rows.append(
            {
                "_true": base["_true"],
                "_mode": mode,
                "name": _name_variant(base["name"], rng),
                "tax_id": base["tax_id"]
                if mode in ("tax", "both")
                else f"{fresh_tax[:2]}-{fresh_tax[2:]}",
                "bank_account": base["bank_account"]
                if mode in ("bank", "both")
                else str(bank[n_true + j]),
                "zip": base["zip"]
                if rng.random() < 0.6
                else f"{int(rng.integers(10000, 99999)):05d}",
            }
        )

    order = rng.permutation(len(rows))
    shuffled = [rows[int(i)] for i in order]
    true_vendor: dict[str, str] = {}
    dup_records: dict[str, str] = {}
    lines: dict[str, str] = {}
    out_rows = []
    for pos, row in enumerate(shuffled):
        vid = f"V{pos + 1:05d}"
        true_vendor[vid] = row["_true"]
        if row["_mode"]:
            dup_records[vid] = row["_mode"]
        lines[vid] = line_of[row["_true"]]
        out_rows.append(
            {"vendor_id": vid, **{c: row[c] for c in ("name", "tax_id", "bank_account", "zip")}}
        )
    frame = pd.DataFrame(out_rows)
    return VendorMaster(frame, true_vendor, dup_records, lines)
