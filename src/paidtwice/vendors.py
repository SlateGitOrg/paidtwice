"""Vendor master de-duplication with union-find."""

from __future__ import annotations

import pandas as pd

from paidtwice.normalize import normalize_vendor_name


class UnionFind:
    """Disjoint sets over ``0..n-1`` with path halving and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum()).upper()


def cluster_vendors(vendors: pd.DataFrame) -> pd.DataFrame:
    """Group vendor records that are the same business.

    Two records are linked when they share a tax ID, share a bank account, or
    share a normalised name and ZIP code. Links are transitive. Blank tax IDs,
    bank accounts and ZIP codes never link anything.
    """
    ids = list(vendors["vendor_id"])
    uf = UnionFind(len(ids))
    seen: dict[tuple[str, str, str], int] = {}
    keyed = zip(
        vendors["tax_id"], vendors["bank_account"], vendors["name"], vendors["zip"], strict=True
    )
    for i, (tax, bank, name, zip_code) in enumerate(keyed):
        keys = []
        if _digits(tax):
            keys.append(("tax", _digits(tax), ""))
        if _digits(bank):
            keys.append(("bank", _digits(bank), ""))
        if zip_code.strip() and normalize_vendor_name(name):
            keys.append(("name_zip", normalize_vendor_name(name), zip_code.strip()))
        for key in keys:
            if key in seen:
                uf.union(i, seen[key])
            else:
                seen[key] = i
    roots = [uf.find(i) for i in range(len(ids))]
    first: dict[int, str] = {}
    for vendor_id, root in zip(ids, roots, strict=True):
        if root not in first or vendor_id < first[root]:
            first[root] = vendor_id
    sizes = pd.Series(roots).map(pd.Series(roots).value_counts())
    return pd.DataFrame(
        {
            "vendor_id": ids,
            "cluster_id": [f"CL-{first[r]}" for r in roots],
            "cluster_size": sizes.to_numpy(),
        }
    )
