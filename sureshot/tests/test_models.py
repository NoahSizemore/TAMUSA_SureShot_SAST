"""
Tests for the actual model classes in scripts/.

test_pipeline.py checks the harness using a _DummyModel. That proves the
harness works; it proves nothing about the models that ship. STRUCTURE.md says
to run the contract test against the model, and these are that test.

The failure this file is written to catch: a model that looks fine on reading
but cannot complete a single run. That is not hypothetical here -- the
`num_boost_round = 2000,` trailing comma turned an int into a one-element tuple
and made scripts/train_XGBoost.py unrunnable, and nothing in the suite noticed,
because nothing in the suite ever instantiated the class.
"""

from __future__ import annotations

import numpy as np
import pytest

from sureshot import calibrate, persist, pipeline


# --------------------------------------------------------------------------
# hyperparameter sanity -- cheap, and catches the typo class of bug
# --------------------------------------------------------------------------

def test_native_hyperparameters_have_correct_types(native_script):
    """
    A trailing comma silently produces a tuple. These asserts cost nothing and
    turn a confusing deep-library failure into an obvious one.
    """
    m = native_script.MyModel(seed=42, scale_pos_weight=16.0)
    assert isinstance(m.num_boost_round, int), \
        "num_boost_round must be an int -- a trailing comma makes it a tuple"
    assert isinstance(m.early_stopping_rounds, int)
    for key in ("max_depth", "min_child_weight"):
        assert isinstance(m.params[key], int), f"{key} must be an int"
    for key in ("eta", "subsample", "colsample_bytree", "lambda",
                "scale_pos_weight"):
        assert isinstance(m.params[key], (int, float)), f"{key} must be numeric"
    assert m.params["objective"] == "binary:logistic"
    # AUPRC, not AUROC: at 1:16 imbalance ROC flatters the model because the
    # large negative class keeps the false-positive rate small regardless.
    assert m.params["eval_metric"] == "aucpr"


def test_scale_pos_weight_reaches_the_booster(native_script):
    m = native_script.MyModel(seed=1, scale_pos_weight=16.5)
    assert m.params["scale_pos_weight"] == 16.5


# --------------------------------------------------------------------------
# the two contracts
# --------------------------------------------------------------------------

@pytest.fixture(params=["native", "reference"])
def model_class(request, native_script, reference_script):
    """Run every contract test against both shipped models."""
    if request.param == "native":
        return native_script.MyModel
    return reference_script.ReferenceModel


def test_model_satisfies_score_model_protocol(model_class):
    assert isinstance(model_class(), calibrate.ScoreModel)


def test_model_satisfies_savable_model_protocol(model_class):
    """Needed by persist.Bundle.save(): a model_kind tag and save_model()."""
    m = model_class()
    assert isinstance(m, persist.SavableModel)
    assert isinstance(m.model_kind, str) and m.model_kind
    assert m.model_kind in persist.MODEL_LOADERS, \
        f"no loader registered for {m.model_kind!r}, so it could never be read back"


def test_predict_proba_shape_and_range(model_class, toy_dataset):
    ds = toy_dataset
    X_tr, y_tr = ds.train
    X_te, _ = ds.test
    m = model_class(seed=42, scale_pos_weight=ds.scale_pos_weight()).fit(
        X_tr, y_tr, X_val=ds.calib[0], y_val=ds.calib[1])

    p = m.predict_proba(X_te)
    assert p.shape == (len(X_te), 2)
    assert np.all((p >= 0) & (p <= 1))
    # Column 1 is P(vulnerable) and the rows must be a distribution.
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)


def test_fit_returns_self_for_chaining(model_class, toy_dataset):
    ds = toy_dataset
    m = model_class()
    assert m.fit(*ds.train) is m


def test_predict_before_fit_raises(native_script):
    """A clear error beats a confusing one from inside XGBoost."""
    with pytest.raises(RuntimeError):
        native_script.MyModel().predict_proba(np.zeros((3, 8), dtype=np.float32))


def test_fit_works_without_a_validation_set(model_class, toy_dataset):
    """
    early_stopping_rounds must be suppressed when no eval set is given.
    XGBoost raises if asked to early-stop with nothing to watch.
    """
    ds = toy_dataset
    m = model_class(seed=0, scale_pos_weight=ds.scale_pos_weight())
    m.fit(*ds.train)
    p = m.predict_proba(ds.test[0])
    assert p.shape == (len(ds.test[0]), 2)


# --------------------------------------------------------------------------
# end to end through the real harness
# --------------------------------------------------------------------------

def test_model_runs_end_to_end_through_pipeline(model_class, toy_dataset):
    ds = toy_dataset
    m = model_class(seed=42, scale_pos_weight=ds.scale_pos_weight())
    out = pipeline.run(m, ds, label="contract-test", verbose=False)

    assert out["best_calibrator"] in calibrate.CALIBRATORS
    assert 0 <= out["operating_point"]["f1"] <= 1
    for block in out["metrics"].values():
        assert 0 <= block["auroc"] <= 1
        assert 0 <= block["auprc"] <= 1

    # The toy label depends on feature 0, so a working model must beat chance.
    # This is the assertion that would catch a model wired up backwards.
    assert out["metrics"]["uncalibrated"]["auroc"] > 0.6


def test_feature_importances_align_with_feature_names(model_class, toy_dataset):
    """
    pipeline.run only prints the importance table and runs the size-confound
    check when len(importances) == len(feature_names). A length mismatch makes
    that analysis silently vanish rather than fail, so it is asserted here.
    """
    ds = toy_dataset
    m = model_class(seed=42, scale_pos_weight=ds.scale_pos_weight())
    m.fit(*ds.train, X_val=ds.calib[0], y_val=ds.calib[1])

    imp = m.feature_importances_
    assert imp is not None
    assert len(imp) == len(ds.feature_names)
    assert np.all(np.asarray(imp) >= 0)


def test_native_importances_are_normalised(native_script, toy_dataset):
    ds = toy_dataset
    m = native_script.MyModel(seed=42, scale_pos_weight=ds.scale_pos_weight())
    m.fit(*ds.train, X_val=ds.calib[0], y_val=ds.calib[1])
    assert np.isclose(np.sum(m.feature_importances_), 1.0, atol=1e-6)


def test_unfitted_importances_are_none(native_script):
    assert native_script.MyModel().feature_importances_ is None


# --------------------------------------------------------------------------
# early stopping
# --------------------------------------------------------------------------

def test_early_stopping_stops_well_short_of_the_ceiling(native_script,
                                                        toy_dataset):
    """
    num_boost_round is a ceiling, not a target. If early stopping were not
    working the booster would contain all 2000 trees, which on this toy data
    would be heavy overfitting -- and, more importantly, would mean the saved
    model and the evaluated model disagree about how many trees to use.
    """
    ds = toy_dataset
    m = native_script.MyModel(seed=42, scale_pos_weight=ds.scale_pos_weight())
    m.fit(*ds.train, X_val=ds.calib[0], y_val=ds.calib[1])

    best = m._best_round()
    assert best is not None, "early stopping did not record a best iteration"
    assert best < m.num_boost_round - 1
    assert "calib" in m.evals_result, \
        "the validation curve should be recorded for inspection"
