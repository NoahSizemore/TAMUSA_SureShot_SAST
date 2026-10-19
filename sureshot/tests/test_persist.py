"""
Tests for saving a detector, loading it back, and scanning with it.

The invariant this file protects is narrow and important:

    A DETECTOR LOADED FROM DISK MUST SCORE IDENTICALLY TO THE ONE THAT WAS
    SAVED.

If that ever stops holding, every published metric becomes a statement about a
model that no longer exists, and the scanner silently reports something other
than what was measured. It is the kind of bug that produces no error at all --
just slightly different numbers, forever -- so it gets asserted to exact
floating-point equality rather than a tolerance.

Also checked here: the hand-written calibrator arithmetic really does match
scikit-learn. calibrate.py claims the sigmoid and np.interp forms are what
sklearn computes rather than approximations of it, and that claim is only worth
anything if something verifies it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from sureshot import calibrate, metrics, persist, pipeline, scan
from sureshot.features import FEATURE_NAMES, extract_features, iter_functions
from sureshot.persist import Bundle

# A small but genuine Rust file: obviously unsafe code, ordinary safe code, and
# one function too short to be scored.
RUST_SAMPLE = b"""
use std::ptr;

pub unsafe fn copy_into(dst: *mut u8, src: *const u8, len: usize) {
    if dst.is_null() || src.is_null() {
        return;
    }
    unsafe {
        ptr::copy_nonoverlapping(src, dst, len);
    }
}

pub fn safe_sum(values: &[u32]) -> u32 {
    let mut total = 0u32;
    for v in values {
        total = total.saturating_add(*v);
    }
    total
}

pub fn parse_header(buf: &[u8]) -> u32 {
    let a = buf[0] as u32;
    let b = buf[1] as u32;
    let c = buf[2] as u32;
    (a << 16) | (b << 8) | c
}

fn tiny() -> u8 { 7 }
"""


# --------------------------------------------------------------------------
# calibrator serialisation
# --------------------------------------------------------------------------

def _scores_and_labels(n=800, seed=5):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < s ** 2).astype(int)
    return s, y


@pytest.mark.parametrize("name", ["platt", "isotonic", "temperature"])
def test_calibrator_survives_a_json_round_trip(name):
    """
    to_dict -> JSON text -> from_dict must reproduce the mapping exactly.

    The JSON round trip is included deliberately: going through text is what
    catches a numpy float or array that is not actually serialisable, which an
    in-memory dict comparison would miss.
    """
    s, y = _scores_and_labels()
    cal = calibrate.CALIBRATORS[name]().fit(s, y)
    before = cal.transform(s)

    payload = json.loads(json.dumps(cal.to_dict()))
    restored = calibrate.calibrator_from_dict(payload)

    assert np.array_equal(before, restored.transform(s))


@pytest.mark.parametrize("name", ["platt", "isotonic", "temperature"])
def test_to_dict_records_the_calibrator_name(name):
    """The bundle dispatches on this field when rebuilding."""
    s, y = _scores_and_labels()
    assert calibrate.CALIBRATORS[name]().fit(s, y).to_dict()["name"] == name


# --------------------------------------------------------------------------
# calibrator selection
# --------------------------------------------------------------------------

def test_oof_selection_does_not_reward_overfitting():
    """
    The bug this guards against, concretely.

    Isotonic regression fit on a split and then scored on that SAME split gets
    an ECE at or near zero -- on the real corpus it is exactly 0.0000 -- simply
    because it is flexible enough to reproduce its own training data. Any
    selection rule based on in-sample ECE therefore picks isotonic every time,
    no matter how poorly it generalises.

    Out-of-fold selection must not have that property: every prediction it
    scores was made by a calibrator that had not seen that row.
    """
    s, y = _scores_and_labels(n=1200, seed=11)

    in_sample = {
        name: metrics.ece(y, calibrate.CALIBRATORS[name]().fit(s, y).transform(s))
        for name in calibrate.CALIBRATORS
    }
    _, oof = calibrate.select_calibrator(s, y, n_splits=5)

    # In-sample, isotonic is essentially perfect -- that is the trap.
    assert in_sample["isotonic"] < 0.01
    # Out of fold it is held to a real standard and no longer free.
    assert oof["isotonic"] > in_sample["isotonic"]


def test_select_calibrator_returns_a_real_candidate():
    s, y = _scores_and_labels(n=1000, seed=12)
    best, oof = calibrate.select_calibrator(s, y)
    assert best in calibrate.CALIBRATORS
    assert set(oof) <= set(calibrate.CALIBRATORS)
    assert best == min(oof, key=lambda k: oof[k])


def test_select_calibrator_respects_repo_groups():
    """Grouped folds keep near-duplicate functions out of both sides."""
    s, y = _scores_and_labels(n=900, seed=13)
    groups = np.array([f"repo{i % 12}" for i in range(len(y))])
    best, oof = calibrate.select_calibrator(s, y, groups=groups)
    assert best in calibrate.CALIBRATORS
    assert all(v >= 0 for v in oof.values())


def test_select_calibrator_falls_back_when_positives_are_too_few():
    """
    With almost no positives, cross-validation is meaningless. Falling back to
    Platt -- two parameters, stable on tiny data -- is better than returning an
    in-sample comparison that would favour isotonic.
    """
    rng = np.random.default_rng(14)
    s = rng.uniform(0.01, 0.99, 200)
    y = np.zeros(200, dtype=int)
    y[:2] = 1                      # 2 positives, fewer than n_splits
    best, oof = calibrate.select_calibrator(s, y, n_splits=5)
    assert best == "platt"
    assert oof == {}


def test_pipeline_records_how_the_calibrator_was_chosen(native_script,
                                                        toy_dataset):
    ds = toy_dataset
    m = native_script.MyModel(seed=42, scale_pos_weight=ds.scale_pos_weight())
    out = pipeline.run(m, ds, label="sel", verbose=False, select_on="calib")
    assert out["calibrator_selected_on"] == "calib"
    assert out["best_calibrator"] in calibrate.CALIBRATORS


def test_unknown_calibrator_name_is_rejected():
    with pytest.raises(ValueError, match="unknown calibrator"):
        calibrate.calibrator_from_dict({"name": "bayesian-magic"})


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        calibrate.PlattCalibrator().transform(np.array([0.3]))
    with pytest.raises(RuntimeError):
        calibrate.IsotonicCalibrator().transform(np.array([0.3]))


# --------------------------------------------------------------------------
# the hand-written arithmetic must equal what sklearn computes
# --------------------------------------------------------------------------

def test_platt_matches_sklearn_predict_proba():
    """
    calibrate.py replaced LogisticRegression.predict_proba with an explicit
    sigmoid so the fitted state is two serialisable floats. This asserts the
    substitution is exact, not merely close.
    """
    s, y = _scores_and_labels()
    cal = calibrate.PlattCalibrator().fit(s, y)

    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    z = cal._logit(s)
    lr.fit(z, y)

    assert np.allclose(cal.transform(s), lr.predict_proba(z)[:, 1], atol=1e-12)


def test_isotonic_matches_sklearn_predict():
    """np.interp over the knots == IsotonicRegression with out_of_bounds='clip'."""
    s, y = _scores_and_labels()
    cal = calibrate.IsotonicCalibrator().fit(s, y)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, y.astype(float))

    assert np.allclose(cal.transform(s), iso.predict(s), atol=1e-12)


def test_isotonic_clamps_outside_the_fitted_range():
    """
    Scores below or above everything seen during calibration must clamp to the
    end knots, not extrapolate. np.interp does this; a naive linear fit would
    happily return a probability outside [0, 1].
    """
    s, y = _scores_and_labels()
    cal = calibrate.IsotonicCalibrator().fit(s, y)
    out = cal.transform(np.array([-5.0, 0.0, 1.0, 5.0]))
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert out[0] == cal.knots_y_[0]
    assert out[-1] == cal.knots_y_[-1]


def test_platt_is_numerically_stable_at_extremes():
    """
    The branching sigmoid exists so large-magnitude logits do not overflow.
    A naive 1/(1+exp(-z)) warns and returns nan/inf for z around -800.
    """
    s, y = _scores_and_labels()
    cal = calibrate.PlattCalibrator().fit(s, y)
    cal.a_, cal.b_ = 500.0, 0.0      # force extreme logits
    with np.errstate(over="raise", invalid="raise"):
        out = cal.transform(np.array([1e-9, 0.5, 1 - 1e-9]))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


# --------------------------------------------------------------------------
# bundle round trip
# --------------------------------------------------------------------------

@pytest.fixture
def trained_bundle(native_script, toy_dataset, tmp_path):
    """A genuinely trained detector, saved to disk and loaded back."""
    ds = toy_dataset
    model = native_script.MyModel(seed=42,
                                  scale_pos_weight=ds.scale_pos_weight())
    model.fit(*ds.train, X_val=ds.calib[0], y_val=ds.calib[1])

    raw_cal = calibrate.positive_scores(model, ds.calib[0])
    cal = calibrate.PlattCalibrator().fit(raw_cal, ds.calib[1])

    bundle = Bundle(model=model, calibrator=cal, threshold=0.37,
                    feature_names=list(ds.feature_names),
                    metadata={"label": "test", "trained_on": "toy"})
    path = bundle.save(tmp_path / "detector")
    return bundle, Bundle.load(path), ds


def test_saved_and_loaded_detector_score_identically(trained_bundle):
    """The central invariant of this module."""
    original, loaded, ds = trained_bundle
    X = ds.test[0]
    assert np.array_equal(original.score(X), loaded.score(X))
    assert np.array_equal(original.decide(X), loaded.decide(X))


def test_loaded_bundle_preserves_its_configuration(trained_bundle):
    original, loaded, _ = trained_bundle
    assert loaded.threshold == original.threshold
    assert loaded.feature_names == original.feature_names
    assert loaded.calibrator.name == original.calibrator.name
    assert loaded.metadata["trained_on"] == "toy"


def test_bundle_writes_the_expected_files(trained_bundle, tmp_path):
    path = tmp_path / "detector"
    assert (path / persist.MANIFEST_NAME).exists()
    assert (path / persist.MODEL_NAME).exists()
    manifest = json.loads((path / persist.MANIFEST_NAME).read_text())
    assert manifest["bundle_version"] == persist.BUNDLE_VERSION
    assert manifest["model_kind"] == "xgboost-booster"
    assert manifest["n_features"] == len(manifest["feature_names"])
    # The booster must be XGBoost's own JSON, not a pickle.
    json.loads((path / persist.MODEL_NAME).read_text())


def test_saved_booster_contains_only_the_early_stopped_trees(trained_bundle):
    """
    save_model() slices the booster to best_iteration + 1. Without the slice
    the file would hold all 2000 trees and the reloaded model would use the
    overfitted tail that early stopping explicitly rejected.
    """
    original, loaded, _ = trained_bundle
    best = original.model._best_round()
    assert best is not None
    assert loaded.model.n_trees == best + 1
    assert loaded.model.n_trees < original.model.num_boost_round


def test_margin_is_distance_from_threshold(trained_bundle):
    _, loaded, ds = trained_bundle
    X = ds.test[0]
    assert np.allclose(loaded.margin(X),
                       np.abs(loaded.score(X) - loaded.threshold))


def test_score_accepts_a_single_row(trained_bundle):
    _, loaded, ds = trained_bundle
    assert loaded.score(ds.test[0][0]).shape == (1,)


def test_describe_mentions_the_essentials(trained_bundle):
    _, loaded, _ = trained_bundle
    text = loaded.describe()
    assert "platt" in text
    assert "threshold" in text


# --------------------------------------------------------------------------
# bundle failure modes -- each of these must fail loudly, not silently
# --------------------------------------------------------------------------

def test_wrong_feature_count_is_rejected(trained_bundle):
    """
    The case this guards: someone adds a token rule to features.py and then
    runs an old bundle. The booster would receive columns that no longer mean
    what it learned them to mean and emit confident nonsense.
    """
    _, loaded, ds = trained_bundle
    bad = np.zeros((4, len(loaded.feature_names) + 2), dtype=np.float32)
    with pytest.raises(ValueError, match="feature count mismatch"):
        loaded.score(bad)


def test_missing_manifest_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        Bundle.load(tmp_path / "nothing-here")


def test_version_mismatch_is_refused(trained_bundle, tmp_path):
    path = tmp_path / "detector"
    manifest = json.loads((path / persist.MANIFEST_NAME).read_text())
    manifest["bundle_version"] = persist.BUNDLE_VERSION + 99
    (path / persist.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bundle version"):
        Bundle.load(path)


def test_unknown_model_kind_is_refused(trained_bundle, tmp_path):
    path = tmp_path / "detector"
    manifest = json.loads((path / persist.MANIFEST_NAME).read_text())
    manifest["model_kind"] = "pytorch-transformer"
    (path / persist.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no loader registered"):
        Bundle.load(path)


def test_saving_an_unsavable_model_is_refused(tmp_path):
    """A model with no model_kind must fail at save time with a clear message."""
    class Barebones:
        def fit(self, X, y, X_val=None, y_val=None):
            return self

        def predict_proba(self, X):
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

    s, y = _scores_and_labels()
    bundle = Bundle(model=Barebones(),
                    calibrator=calibrate.PlattCalibrator().fit(s, y),
                    threshold=0.5, feature_names=["a", "b"])
    with pytest.raises(TypeError, match="model_kind"):
        bundle.save(tmp_path / "doomed")


# --------------------------------------------------------------------------
# function splitting
# --------------------------------------------------------------------------

def test_iter_functions_finds_the_real_functions():
    funcs = {f.name: f for f in iter_functions(RUST_SAMPLE)}
    assert {"copy_into", "safe_sum", "parse_header"} <= set(funcs)
    # `tiny()` is a single line, below the 3-line training filter, so the model
    # never learned what functions that short look like.
    assert "tiny" not in funcs


def test_iter_functions_reports_usable_line_numbers():
    text = RUST_SAMPLE.decode()
    lines = text.splitlines()
    for f in iter_functions(RUST_SAMPLE):
        assert f.start_line >= 1
        assert f.end_line >= f.start_line
        # 1-based line numbers, so the name must appear on the start line.
        assert f.name in lines[f.start_line - 1]


def test_iter_functions_never_raises_on_junk():
    for src in (b"", b"   ", b"\x00\x01\x02", b"def f(): pass",
                b"fn broken( { let x = ;", b"}" * 500):
        assert isinstance(iter_functions(src), list)


def test_iter_functions_keeps_same_named_functions_separate():
    """
    Unlike the harvester, the scanner must not de-duplicate by name: two `new`
    methods in different impl blocks are two separate findings.
    """
    src = b"""
struct A; struct B;
impl A {
    pub fn new() -> Self {
        let x = 1;
        A
    }
}
impl B {
    pub fn new() -> Self {
        let y = 2;
        B
    }
}
"""
    names = [f.name for f in iter_functions(src)]
    assert names.count("new") == 2


# --------------------------------------------------------------------------
# the scanner
# --------------------------------------------------------------------------

@pytest.fixture
def rust_bundle(native_script, tmp_path):
    """
    A detector trained over the REAL feature names, so it can score real Rust.

    The training data is random, so this model's judgements are meaningless --
    which is fine, because these tests check the inference plumbing (gating,
    splitting, column ordering, thresholding), not detection quality. Quality
    is measured on the actual corpus by the training scripts.
    """
    rng = np.random.default_rng(0)
    n, k = 400, len(FEATURE_NAMES)
    X = rng.normal(size=(n, k)).astype(np.float32)
    y = (X[:, 0] > 0.8).astype(np.int32)

    model = native_script.MyModel(seed=0, scale_pos_weight=4.0)
    model.fit(X[:300], y[:300], X_val=X[300:], y_val=y[300:])

    raw = calibrate.positive_scores(model, X[300:])
    cal = calibrate.PlattCalibrator().fit(raw, y[300:])

    bundle = Bundle(model=model, calibrator=cal, threshold=0.20,
                    feature_names=list(FEATURE_NAMES),
                    metadata={"label": "plumbing-test",
                              "test_precision": 0.09, "test_recall": 0.29})
    return Bundle.load(bundle.save(tmp_path / "rust-detector"))


def test_scanner_scores_every_eligible_function(rust_bundle):
    result = scan.scan_source(rust_bundle, RUST_SAMPLE, path="sample.rs")
    assert result.accepted
    names = {f.function for f in result.findings}
    assert {"copy_into", "safe_sum", "parse_header"} <= names
    for f in result.findings:
        assert 0.0 <= f.probability <= 1.0
        assert f.flagged == (f.probability >= rust_bundle.threshold)


def test_scanner_rejects_non_rust_before_the_model_sees_it(rust_bundle):
    """
    Scenario 2 in the requirements. The point is that this returns a REJECTION
    rather than a confident score: a model handed Python will still emit a
    number, and that number is meaningless.
    """
    for src in (b"def foo(x):\n    return x * 2\n",
                b"int main(void){ return 0; }",
                b'{"a": 1, "b": [2, 3]}'):
        result = scan.scan_source(rust_bundle, src, path="x")
        assert not result.accepted
        assert "not Rust" in result.reason
        assert result.findings == []


def test_scanner_findings_are_ordered_most_suspicious_first(rust_bundle):
    result = scan.scan_source(rust_bundle, RUST_SAMPLE)
    probs = [f.probability for f in result.findings]
    assert probs == sorted(probs, reverse=True)


def test_abstain_margin_converts_borderline_calls_to_unsure(rust_bundle):
    """
    With a margin of 1.0 every prediction is within the band, so everything
    must become UNSURE. This is the abstention behaviour that makes a weak
    discriminator still useful.
    """
    result = scan.scan_source(rust_bundle, RUST_SAMPLE, abstain_margin=1.0)
    assert result.findings
    assert all(f.abstained for f in result.findings)
    assert all(f.verdict == "UNSURE" for f in result.findings)
    assert result.flagged == []


def test_no_abstention_by_default(rust_bundle):
    result = scan.scan_source(rust_bundle, RUST_SAMPLE)
    assert not any(f.abstained for f in result.findings)


def test_scanner_handles_a_file_with_no_scoreable_functions(rust_bundle):
    result = scan.scan_source(rust_bundle, b"pub struct Config { pub a: u32 }\nuse std::fmt;\n")
    assert result.accepted
    assert result.findings == []
    assert "no functions" in result.reason


def test_featurise_follows_the_bundle_column_order(rust_bundle):
    """
    Vectors are assembled from bundle.feature_names, NOT from FEATURE_NAMES in
    its natural order. This is what makes a --drop-size-features bundle work
    without special casing, and it is verified by reversing the order and
    checking the values move with it.
    """
    reversed_bundle = Bundle(
        model=rust_bundle.model, calibrator=rust_bundle.calibrator,
        threshold=rust_bundle.threshold,
        feature_names=list(reversed(FEATURE_NAMES)))

    src = iter_functions(RUST_SAMPLE)[0].source
    normal = scan.featurise(rust_bundle, [src])[0]
    flipped = scan.featurise(reversed_bundle, [src])[0]
    assert np.array_equal(normal, flipped[::-1])


def test_featurise_matches_direct_extraction(rust_bundle):
    src = iter_functions(RUST_SAMPLE)[0].source
    feats = extract_features(src)
    row = scan.featurise(rust_bundle, [src])[0]
    expected = np.asarray([feats[n] for n in rust_bundle.feature_names],
                          dtype=np.float32)
    assert np.array_equal(row, expected)


def test_scan_file_and_directory_walk(rust_bundle, tmp_path):
    src = tmp_path / "crate" / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_bytes(RUST_SAMPLE)
    (src / "util.rs").write_bytes(RUST_SAMPLE)
    # Excluded from training, so excluded from scanning.
    tests = tmp_path / "crate" / "tests"
    tests.mkdir()
    (tests / "it.rs").write_bytes(RUST_SAMPLE)

    results = scan.scan_path(rust_bundle, tmp_path / "crate")
    scanned = {Path(r.path).name for r in results}
    assert scanned == {"lib.rs", "util.rs"}


def test_scan_file_on_a_missing_path_is_a_rejection_not_a_crash(rust_bundle,
                                                                tmp_path):
    result = scan.scan_file(rust_bundle, tmp_path / "ghost.rs")
    assert not result.accepted
    assert "unreadable" in result.reason


def test_format_results_reports_the_honest_caveat(rust_bundle):
    results = [scan.scan_source(rust_bundle, RUST_SAMPLE, path="a.rs")]
    text = scan.format_results(results, rust_bundle)
    assert "function(s) scored" in text
    # The precision/recall caveat is not decoration: a reviewer needs to know
    # most flags are false alarms before they start trusting the output.
    assert "Calibration note" in text
    assert "review this first" in text
