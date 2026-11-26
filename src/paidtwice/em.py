"""Fellegi-Sunter model with probabilities learned by EM.

Every candidate pair has a comparison vector ``x`` with one discrete level per
field. The pair is a duplicate (class M) with probability ``pi`` and, given the
class, the fields are independent: ``P(x | M) = prod_k m_k[x_k]`` and
``P(x | U) = prod_k u_k[x_k]``. EM maximises the likelihood of the observed
vectors over ``pi``, ``m`` and ``u`` without any labels.

Initialisation (deterministic)
    * ``pi = 1e-3``.
    * ``m_k``: the strongest-agreement level gets 0.8; the rest share 0.2 with
      geometrically decreasing shares (1/2, 1/4, ...).
    * ``u_k``: the weighted empirical level frequencies, which is almost exactly
      the non-match distribution because duplicates are a tiny fraction of pairs.

Convergence
    Stop when no ``m`` or ``u`` entry moves by more than ``tol`` (1e-9) and
    ``pi`` moves by less than ``tol`` relative to itself, or after ``max_iter``
    iterations. A pseudo-count of 0.5 per level (``smoothing``) keeps every
    probability positive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from paidtwice.compare import FIELDS, LEVELS, ORDINAL_FIELDS

LN2 = float(np.log(2.0))


@dataclass
class FSModel:
    """Fitted match (m) and non-match (u) level probabilities."""

    pi: float
    m: dict[str, list[float]]
    u: dict[str, list[float]]
    iterations: int = 0
    converged: bool = True
    info: dict = field(default_factory=dict)

    def log2_weights(self) -> dict[str, np.ndarray]:
        return {f: np.log2(np.array(self.m[f])) - np.log2(np.array(self.u[f])) for f in FIELDS}

    def score(self, vectors: np.ndarray) -> np.ndarray:
        """Total match weight in bits: the sum of ``log2(m/u)`` over the observed levels."""
        weights = self.log2_weights()
        total = np.zeros(len(vectors))
        for k, f in enumerate(FIELDS):
            total += weights[f][vectors[:, k]]
        return total

    def to_json(self) -> dict:
        return {
            "pi": self.pi,
            "iterations": self.iterations,
            "converged": self.converged,
            "info": self.info,
            "fields": {
                f: {
                    "levels": list(LEVELS[f]),
                    "m": self.m[f],
                    "u": self.u[f],
                    "log2_weight": [float(w) for w in self.log2_weights()[f]],
                }
                for f in FIELDS
            },
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> FSModel:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `paidtwice fit` first")
        raw = json.loads(path.read_text())
        return cls(
            pi=raw["pi"],
            m={f: raw["fields"][f]["m"] for f in FIELDS},
            u={f: raw["fields"][f]["u"] for f in FIELDS},
            iterations=raw["iterations"],
            converged=raw["converged"],
            info=raw.get("info", {}),
        )


def _initial_m(n_levels: int) -> np.ndarray:
    rest = 0.5 ** np.arange(1, n_levels)
    return np.concatenate([[0.8], 0.2 * rest / rest.sum()])


def fit_em(
    patterns: np.ndarray,
    counts: np.ndarray,
    *,
    tol: float = 1e-9,
    max_iter: int = 2000,
    smoothing: float = 0.5,
    pi_init: float = 1e-3,
) -> FSModel:
    """Fit ``pi``, ``m`` and ``u`` to weighted comparison-vector patterns.

    ``patterns`` has one row per distinct vector and ``counts`` the (possibly
    fractional) number of pairs showing it.
    """
    sizes = [len(LEVELS[f]) for f in FIELDS]
    total = float(counts.sum())
    m = [_initial_m(s) for s in sizes]
    u = []
    for k, s in enumerate(sizes):
        freq = np.bincount(patterns[:, k], weights=counts, minlength=s) + smoothing
        u.append(freq / freq.sum())
    pi = pi_init
    onehot = [np.eye(s)[patterns[:, k]] for k, s in enumerate(sizes)]

    converged = False
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        log_m = sum(np.log(m[k])[patterns[:, k]] for k in range(len(sizes)))
        log_u = sum(np.log(u[k])[patterns[:, k]] for k in range(len(sizes)))
        log_ratio = np.log(pi) + log_m - np.log1p(-pi) - log_u
        resp = 1.0 / (1.0 + np.exp(-np.clip(log_ratio, -700, 700)))

        w_match = counts * resp
        w_non = counts * (1.0 - resp)
        new_pi = float(w_match.sum() / total)
        new_pi = min(max(new_pi, 1e-12), 1 - 1e-12)
        new_m, new_u = [], []
        for k in range(len(sizes)):
            sm = onehot[k].T @ w_match + smoothing
            su = onehot[k].T @ w_non + smoothing
            new_m.append(sm / sm.sum())
            new_u.append(su / su.sum())
        shift = max(
            max(float(np.abs(a - b).max()) for a, b in zip(new_m + new_u, m + u, strict=True)),
            abs(new_pi - pi) / pi,
        )
        m, u, pi = new_m, new_u, new_pi
        if shift < tol:
            converged = True
            break

    return FSModel(
        pi=pi,
        m={f: [float(x) for x in m[k]] for k, f in enumerate(FIELDS)},
        u={f: [float(x) for x in u[k]] for k, f in enumerate(FIELDS)},
        iterations=iterations,
        converged=converged,
        info={"tol": tol, "max_iter": max_iter, "smoothing": smoothing, "pi_init": pi_init},
    )


def enforce_ordinal_weights(model: FSModel) -> list[str]:
    """Pool adjacent violators on the weights of the ordered fields.

    The levels of an ordinal field run from strongest to weakest agreement, so
    their weights must not increase. When sparse data breaks that (blocking
    leaves a field almost no disagreeing pairs, so its rare levels rest on a
    handful of pairs), the violating neighbours are pooled: they get one common
    ``m/u`` ratio, the pooled ``m`` and ``u`` masses are kept, and each
    distribution stays normalised. Returns the fields that needed it.
    """
    changed = []
    for f in ORDINAL_FIELDS:
        m, u = model.m[f], model.u[f]
        blocks = [[k] for k in range(len(m))]
        pooled = True
        while pooled:
            pooled = False
            for i in range(len(blocks) - 1):
                left = sum(m[k] for k in blocks[i]) / sum(u[k] for k in blocks[i])
                right = sum(m[k] for k in blocks[i + 1]) / sum(u[k] for k in blocks[i + 1])
                if left < right:
                    blocks[i : i + 2] = [blocks[i] + blocks[i + 1]]
                    pooled = True
                    break
        if len(blocks) < len(m):
            changed.append(f)
            for block in blocks:
                ratio = sum(m[k] for k in block) / sum(u[k] for k in block)
                for k in block:
                    m[k] = u[k] * ratio
    return changed
