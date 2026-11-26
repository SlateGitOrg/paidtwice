import numpy as np

from paidtwice.compare import FIELDS, LEVELS
from paidtwice.em import FSModel, enforce_ordinal_weights, fit_em
from paidtwice.fit import decode, encode


def _mixture(rng, n=20000, pi=0.1):
    """Vectors drawn from a known Fellegi-Sunter model, as pattern counts."""
    m = {f: np.array([0.7] + [0.3 / (len(LEVELS[f]) - 1)] * (len(LEVELS[f]) - 1)) for f in FIELDS}
    u = {f: np.array([0.05] + [0.95 / (len(LEVELS[f]) - 1)] * (len(LEVELS[f]) - 1)) for f in FIELDS}
    is_match = rng.random(n) < pi
    vectors = np.zeros((n, len(FIELDS)), dtype=np.int8)
    for k, f in enumerate(FIELDS):
        probs = np.where(is_match[:, None], m[f], u[f])
        vectors[:, k] = (probs.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    codes, counts = np.unique(encode(vectors), return_counts=True)
    return decode(codes), counts.astype(float), m, u


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(1)
    vectors = np.column_stack([rng.integers(0, len(LEVELS[f]), 50) for f in FIELDS]).astype(np.int8)
    assert (decode(encode(vectors)) == vectors).all()


def test_em_recovers_the_parameters_of_a_known_mixture():
    patterns, counts, m_true, u_true = _mixture(np.random.default_rng(7))
    model = fit_em(patterns, counts)
    assert model.converged
    assert abs(model.pi - 0.1) < 0.02
    for f in FIELDS:
        assert abs(model.m[f][0] - m_true[f][0]) < 0.06
        assert abs(model.u[f][0] - u_true[f][0]) < 0.03


def test_probabilities_are_normalised_and_positive():
    patterns, counts, _, _ = _mixture(np.random.default_rng(3), n=5000)
    model = fit_em(patterns, counts)
    for f in FIELDS:
        assert np.isclose(sum(model.m[f]), 1.0) and np.isclose(sum(model.u[f]), 1.0)
        assert min(model.m[f]) > 0 and min(model.u[f]) > 0


def test_em_is_deterministic():
    patterns, counts, _, _ = _mixture(np.random.default_rng(5), n=5000)
    a, b = fit_em(patterns, counts), fit_em(patterns, counts)
    assert a.m == b.m and a.u == b.u and a.iterations == b.iterations


def test_score_is_the_sum_of_log2_weights():
    patterns, counts, _, _ = _mixture(np.random.default_rng(5), n=5000)
    model = fit_em(patterns, counts)
    weights = model.log2_weights()
    vector = np.array([[0, 1, 2, 0, 1, 0]], dtype=np.int8)
    expected = sum(weights[f][v] for f, v in zip(FIELDS, vector[0], strict=True))
    assert np.isclose(model.score(vector)[0], expected)


def test_model_roundtrips_through_json(tmp_path):
    patterns, counts, _, _ = _mixture(np.random.default_rng(5), n=5000)
    model = fit_em(patterns, counts)
    model.save(tmp_path / "model.json")
    loaded = FSModel.load(tmp_path / "model.json")
    assert loaded.m == model.m and loaded.u == model.u and loaded.pi == model.pi


def test_pooling_makes_ordinal_weights_monotone_and_keeps_masses():
    model = FSModel(
        pi=0.1,
        m={f: [1 / len(LEVELS[f])] * len(LEVELS[f]) for f in FIELDS},
        u={f: [1 / len(LEVELS[f])] * len(LEVELS[f]) for f in FIELDS},
    )
    model.m["amount"] = [0.2, 0.3, 0.5]  # the weakest level looks the strongest
    model.u["amount"] = [0.5, 0.3, 0.2]
    assert enforce_ordinal_weights(model) == ["amount"]
    weights = model.log2_weights()["amount"]
    assert weights[0] >= weights[1] >= weights[2]
    assert np.isclose(sum(model.m["amount"]), 1.0)
    assert np.isclose(sum(model.u["amount"]), 1.0)


def test_learned_invoice_weights_on_the_seeded_data(pipeline):
    model = pipeline["model"]
    assert model.converged
    assert model.m["invoice"][0] > 0.8
    assert model.u["invoice"][0] < 0.01
    w = model.log2_weights()["invoice"]
    assert w[0] > w[1] > w[2] > w[3]
    assert w[3] < 0 < w[2]


def test_learned_vendor_and_po_agreement_are_strong_evidence(pipeline):
    w = pipeline["model"].log2_weights()
    assert w["vendor"][0] > 10 and w["vendor"][2] < 0
    assert w["po"][0] > 10
    assert w["date"][0] > w["date"][2]
