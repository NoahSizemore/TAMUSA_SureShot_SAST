"""
Regression tests for the SureShot pipeline.

Run: pytest -q

These guard the invariants that, when broken, produce results that look fine
and are wrong. The split-leakage and calibration-independence tests are the
important ones -- both failure modes inflate metrics silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from sureshot import calibrate, data, metrics
from sureshot.features import FEATURE_NAMES, extract_features, is_probably_rust


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def test_feature_vector_has_stable_shape():
    """Matrix shape must never drift, whatever the input."""
    for src in (b"", b"fn a(){}", b"garbage ]]] {{{", b"\xff\xfe\x00bad utf8"):
        f = extract_features(src)
        assert set(f) == set(FEATURE_NAMES)
        assert len(f) == 122


def test_extract_never_raises():
    for src in (b"", b"\x00" * 100, "fn ü(){}".encode(), b"fn " * 10000):
        extract_features(src)


def test_counts_are_correct():
    src = b"""
pub unsafe fn f(p: *const u8, q: *mut u8) -> u8 {
    if p.is_null() { return 0; }
    let a = *p;
    let b = a.checked_add(1).unwrap();
    unsafe { *q = b; }
    b
}
"""
    f = extract_features(src)
    assert f["tok_raw_const_ptr"] == 1
    assert f["tok_raw_mut_ptr"] == 1
    assert f["tok_unsafe_fn"] == 1
    assert f["tok_unsafe_block"] == 1
    assert f["tok_ptr_null_check"] == 1
    assert f["tok_checked_op"] == 1
    assert f["tok_unwrap"] == 1


def test_per_line_features_are_ratios():
    f = extract_features(b"fn a() {\n    let x = 1;\n}\n")
    for k in FEATURE_NAMES:
        if k.endswith("_per_line"):
            assert 0.0 <= f[k] <= 10.0


def test_error_recovery_still_yields_features():
    """Slide 5 requires accepting code with errors. Verify we do."""
    f = extract_features(b"fn broken( { let x = ; unsafe { *p }")
    assert f["has_parse_error"] == 1.0
    assert f["tok_unsafe_block"] == 1.0


# --------------------------------------------------------------------------
# the Scenario 2 parser gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("src,expected", [
    (b"pub fn a(x: u32) -> u32 { x + 1 }", True),
    (b"fn broken( { let x = ;", True),                    # errors OK
    (b"def foo(x):\n    return x * 2\n", False),          # Python
    (b"int main(void){ return 0; }", False),              # C
    (b'{"a": 1, "b": [2,3]}', False),                     # JSON
    (b"   ", False),                                      # empty
])
def test_rust_gate(src, expected):
    accepted, _ = is_probably_rust(src)
    assert accepted is expected


# --------------------------------------------------------------------------
# splitting -- the highest-value tests in this file
# --------------------------------------------------------------------------

def _toy(n=400, n_repos=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    groups = np.array([f"repo{i % n_repos}" for i in range(n)])
    y = (X[:, 0] + rng.normal(0, 0.5, n) > 1.2).astype(np.int32)
    return data.Dataset(X, y, groups, [f"f{i}" for i in range(8)], None)


def test_split_has_no_repo_leakage():
    ds = data.split(_toy())
    tr = set(ds.groups[ds.train_idx])
    ca = set(ds.groups[ds.calib_idx])
    te = set(ds.groups[ds.test_idx])
    assert not (tr & ca) and not (tr & te) and not (ca & te)


def test_split_covers_every_row_exactly_once():
    ds = data.split(_toy())
    allidx = np.concatenate([ds.train_idx, ds.calib_idx, ds.test_idx])
    assert len(allidx) == len(ds.y)
    assert len(set(allidx)) == len(ds.y)


def test_split_is_deterministic():
    a, b = data.split(_toy(), seed=7), data.split(_toy(), seed=7)
    assert np.array_equal(a.test_idx, b.test_idx)


def test_split_detects_injected_leakage():
    """If the assertion is ever removed, this test fails loudly."""
    ds = _toy()
    ds.groups = np.array(["same"] * len(ds.y))   # one repo -> cannot split
    with pytest.raises((AssertionError, ValueError)):
        data.split(ds)


def test_scale_pos_weight_uses_train_only():
    ds = data.split(_toy())
    y = ds.y[ds.train_idx]
    assert ds.scale_pos_weight() == pytest.approx(
        (len(y) - y.sum()) / max(y.sum(), 1))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert metrics.ece(y, p, 15) < 0.02


def test_ece_large_for_confidently_wrong():
    y = np.zeros(1000, int)
    p = np.full(1000, 0.95)
    assert metrics.ece(y, p) > 0.9


def test_both_binning_strategies_run():
    rng = np.random.default_rng(1)
    p, y = rng.uniform(size=500), rng.integers(0, 2, 500)
    assert metrics.ece(y, p, strategy="quantile") >= 0
    assert metrics.ece(y, p, strategy="uniform") >= 0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    m, lo, hi = metrics.ece_bootstrap(y, p, n_boot=100)
    assert lo <= m <= hi


def test_threshold_picked_on_given_data_only():
    rng = np.random.default_rng(3)
    p = rng.uniform(size=500)
    y = (p > 0.6).astype(int)
    thr = metrics.pick_threshold(y, p)
    assert 0.4 < thr < 0.8


def test_risk_coverage_is_monotone_in_coverage():
    rng = np.random.default_rng(4)
    p, y = rng.uniform(size=500), rng.integers(0, 2, 500)
    rc = metrics.risk_coverage(y, p, 0.5)
    covs = [c for c, _, _ in rc]
    assert covs == sorted(covs, reverse=True)
    assert all(0 < c <= 1 for c in covs)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["platt", "isotonic", "temperature"])
def test_calibrators_output_valid_probabilities(name):
    rng = np.random.default_rng(5)
    s = rng.uniform(0.01, 0.99, 800)
    y = (rng.uniform(size=800) < s**2).astype(int)
    out = calibrate.CALIBRATORS[name]().fit(s, y).transform(s)
    assert out.shape == s.shape
    assert np.all((out >= 0) & (out <= 1))


def test_calibration_improves_ece_on_overconfident_scores():
    rng = np.random.default_rng(6)
    s = rng.uniform(0.01, 0.99, 3000)
    y = (rng.uniform(size=3000) < s * 0.3).astype(int)   # badly overconfident
    before = metrics.ece(y, s)
    after = metrics.ece(y, calibrate.PlattCalibrator().fit(s, y).transform(s))
    assert after < before


def test_temperature_scaling_preserves_ranking():
    """One parameter, monotone: AUROC must be unchanged."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7)
    s = rng.uniform(0.01, 0.99, 1000)
    y = (rng.uniform(size=1000) < s).astype(int)
    out = calibrate.TemperatureCalibrator().fit(s, y).transform(s)
    assert roc_auc_score(y, s) == pytest.approx(roc_auc_score(y, out), abs=1e-9)


# --------------------------------------------------------------------------
# the model contract -- run this against YOUR model
# --------------------------------------------------------------------------

class _DummyModel:
    """Minimal model satisfying the contract. Yours must do at least this."""

    def fit(self, X, y, X_val=None, y_val=None):
        self.mean_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.mean_)
        return np.column_stack([1 - p, p])


def test_dummy_model_satisfies_protocol():
    assert isinstance(_DummyModel(), calibrate.ScoreModel)


def test_pipeline_runs_end_to_end_with_any_conforming_model():
    from sureshot import pipeline
    ds = data.split(_toy(n=600, n_repos=30))
    out = pipeline.run(_DummyModel(), ds, label="dummy", verbose=False)
    assert "metrics" in out and "uncalibrated" in out["metrics"]
    assert 0 <= out["operating_point"]["f1"] <= 1


def test_positive_scores_accepts_one_or_two_columns():
    class OneCol:
        def predict_proba(self, X):
            return np.full(len(X), 0.3)
    s = calibrate.positive_scores(OneCol(), np.zeros((10, 3)))
    assert s.shape == (10,) and np.allclose(s, 0.3)
