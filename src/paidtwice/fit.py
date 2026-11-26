"""Training data for EM: the candidate pairs that no suppression rule explains."""

from __future__ import annotations

import numpy as np

from paidtwice.candidates import CandidateSet
from paidtwice.compare import FIELDS, LEVELS, comparison_vectors
from paidtwice.data import Payments
from paidtwice.em import FSModel, enforce_ordinal_weights, fit_em
from paidtwice.suppress import any_rule, build_context, evaluate_rules


def encode(vectors: np.ndarray) -> np.ndarray:
    """One integer per comparison vector (mixed-radix over the field levels)."""
    base = np.array([len(LEVELS[f]) for f in FIELDS])
    strides = np.concatenate([[1], np.cumprod(base[:-1])])
    return (vectors.astype(np.int64) * strides).sum(axis=1)


def decode(codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode`."""
    base = [len(LEVELS[f]) for f in FIELDS]
    out = np.zeros((len(codes), len(base)), dtype=np.int8)
    rest = codes.copy()
    for k, size in enumerate(base):
        out[:, k] = rest % size
        rest //= size
    return out


def fit_model(p: Payments, cand: CandidateSet, *, use_suppression: bool = True) -> FSModel:
    """Learn match weights from the candidate pairs alone, with no labels.

    Candidates that a suppression rule explains as a legitimate pattern (rent
    series, installments, credit memo re-bills, sequential numbers) are left out
    of the training set: the rules decide what is explained, EM models the rest.
    The remaining pairs are collapsed to their distinct comparison vectors with
    counts, and EM runs on those.

    The model is fitted on blocked pairs, so fields that blocking forces to agree
    (mostly the amount) carry little weight, and the ordinal fields are pooled
    where sparse levels contradict their order.
    """
    train = np.ones(len(cand), dtype=bool)
    if use_suppression:
        train = ~any_rule(evaluate_rules(p, build_context(p), cand.a, cand.b))
    vectors = comparison_vectors(p, cand.a[train], cand.b[train])
    unique, counts = np.unique(encode(vectors), return_counts=True)
    model = fit_em(decode(unique), counts.astype(float))
    pooled = enforce_ordinal_weights(model)
    model.info.update(
        {
            "candidate_pairs": int(len(cand)),
            "training_pairs": int(train.sum()),
            "distinct_patterns": int(len(unique)),
            "suppressed_before_fit": bool(use_suppression),
            "pooled_fields": pooled,
        }
    )
    return model
