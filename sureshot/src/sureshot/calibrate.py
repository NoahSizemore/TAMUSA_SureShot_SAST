"""
The calibration layer, and the interface the model must satisfy.

This is the part of SureShot that is the actual contribution: it maps a raw
classifier score onto a probability you can trust, and reports how much that
mapping improved things.

DESIGN NOTE, worth reading before you write the model.

Calibrators here fit on SCORES, not on estimators. The obvious alternative --
sklearn's CalibratedClassifierCV wrapping the model -- forces the model to
implement the full sklearn estimator protocol (predict, classes_,
__sklearn_is_fitted__, get_params) or it fails deep inside cross_val_predict
with an unhelpful error. None of that boilerplate has anything to do with
modelling.

Fitting on scores is mathematically identical: Platt scaling is a logistic
regression on one feature (the score), and isotonic regression is a monotonic
step function over it. Neither needs to know what produced the score.

The result is that the model needs exactly two methods:

    fit(X, y, X_val=None, y_val=None) -> self
    predict_proba(X) -> ndarray (n, 2), column 1 = P(vulnerable)

A raw xgboost Booster, a hand-rolled gradient booster, or a neural net all plug
in the same way.

SECOND DESIGN NOTE: fit with a library, predict with explicit arithmetic.

Every calibrator below is fitted using scikit-learn, but none of them *predicts*
through scikit-learn. Immediately after fitting, each one copies the handful of
numbers that define the learned mapping into plain attributes (`a_`/`b_` for
Platt, the knot arrays for isotonic, `T` for temperature) and `transform()` is
implemented directly on top of those numbers.

Two things fall out of that, and both matter:

1. SERIALISATION BECOMES TRIVIAL AND SAFE. A fitted calibrator is now just a
   few floats, so `to_dict()` / `from_dict()` write and read JSON. The
   alternative -- pickling the sklearn estimator -- produces a file that is
   tied to the exact scikit-learn version that wrote it, silently breaks on
   upgrade, and executes arbitrary code on load. A SAST tool that ships a
   pickled model is a SAST tool with a deserialisation vulnerability in it.

2. THERE IS ONLY ONE PREDICTION CODE PATH. A calibrator that was just fitted
   and a calibrator that was loaded from disk run the exact same lines of
   arithmetic, so a saved model cannot quietly disagree with the model whose
   metrics you published. If prediction went through sklearn when fresh and
   through restored attributes when loaded, that gap would be untested and
   would eventually drift.

The arithmetic is not an approximation of what sklearn does, it is what sklearn
does: binary `LogisticRegression.predict_proba` is `sigmoid(X @ coef_ +
intercept_)`, and `IsotonicRegression.predict` with `out_of_bounds="clip"` is
linear interpolation between the fitted knots, clamped at the ends -- which is
exactly `np.interp`. `tests/test_persist.py` asserts this equivalence against
sklearn directly, so the claim is checked rather than merely believed.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@runtime_checkable
class ScoreModel(Protocol):
    """The contract the model must satisfy. Two methods, that is all."""

    def fit(self, X, y, X_val=None, y_val=None): ...
    def predict_proba(self, X) -> np.ndarray: ...


def positive_scores(model, X) -> np.ndarray:
    """
    P(vulnerable) as a 1-D array.

    Accepts either an (n, 2) probability matrix or a plain (n,) score vector,
    so a model that returns a single column still works.
    """
    p = np.asarray(model.predict_proba(X), dtype=float)
    return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel()


class PlattCalibrator:
    """
    Platt scaling: a logistic regression fit on the raw score.

        p_calibrated = sigmoid(a * logit(score) + b)

    Two parameters, so it is stable on small calibration sets. This is the
    safer default below roughly 1000 calibration samples, which is where this
    project sits.

    It fits on the logit of the score rather than the score itself: the model
    output is already squashed into [0, 1], and re-squashing a squashed value
    compresses the tails, which is exactly where the interesting predictions are.
    """

    name = "platt"

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        # The entire fitted state, once fit() has run: a slope and an intercept
        # on the logit of the score. These two floats are what gets serialised.
        self.a_: float | None = None
        self.b_: float | None = None

    def _logit(self, p):
        p = np.clip(np.asarray(p, float).ravel(), self.eps, 1 - self.eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, scores, y):
        self.lr.fit(self._logit(scores), np.asarray(y).ravel())
        # Pull the two learned numbers out of the estimator. From here on the
        # estimator object is never consulted again -- see the module docstring.
        self.a_ = float(self.lr.coef_.ravel()[0])
        self.b_ = float(self.lr.intercept_.ravel()[0])
        return self

    def transform(self, scores) -> np.ndarray:
        if self.a_ is None:
            raise RuntimeError("PlattCalibrator.fit() has not been called")
        # sigmoid(a * logit(score) + b), computed in a numerically stable way.
        # np.exp(-z) overflows for large negative z, so we branch on the sign:
        # for z >= 0 use 1/(1+e^-z); for z < 0 use e^z/(1+e^z). Both are the
        # same function, but each form only ever exponentiates a negative
        # number, which cannot overflow.
        z = self.a_ * self._logit(scores).ravel() + self.b_
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def to_dict(self) -> dict:
        """Serialise to plain JSON-safe values."""
        return {"name": self.name, "eps": self.eps, "a": self.a_, "b": self.b_}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        obj = cls(eps=float(d.get("eps", 1e-6)))
        obj.a_, obj.b_ = float(d["a"]), float(d["b"])
        return obj


class IsotonicCalibrator:
    """
    Isotonic regression: a non-decreasing step function fit on the score.

    Non-parametric and more flexible than Platt, so it wins on larger
    calibration sets. It overfits badly on small ones -- watch for an MCE far
    worse than Platt's while ECE looks comparable. That pattern means a few
    bins have collapsed onto extreme values.
    """

    name = "isotonic"

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # The fitted state: the (x, y) corners of the step/ramp function.
        # sklearn calls these the thresholds; everything between two knots is a
        # straight line, and everything outside the range is clamped.
        self.knots_x_: np.ndarray | None = None
        self.knots_y_: np.ndarray | None = None

    def fit(self, scores, y):
        self.iso.fit(np.asarray(scores, float).ravel(),
                     np.asarray(y, float).ravel())
        self.knots_x_ = np.asarray(self.iso.X_thresholds_, dtype=float)
        self.knots_y_ = np.asarray(self.iso.y_thresholds_, dtype=float)
        return self

    def transform(self, scores) -> np.ndarray:
        if self.knots_x_ is None:
            raise RuntimeError("IsotonicCalibrator.fit() has not been called")
        # np.interp is linear interpolation between the knots, and for inputs
        # below knots_x_[0] or above knots_x_[-1] it returns the first/last y
        # value. That clamping behaviour is precisely out_of_bounds="clip", so
        # this reproduces IsotonicRegression.predict() exactly.
        return np.interp(np.asarray(scores, float).ravel(),
                         self.knots_x_, self.knots_y_)

    def to_dict(self) -> dict:
        return {"name": self.name,
                "knots_x": self.knots_x_.tolist(),
                "knots_y": self.knots_y_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        obj = cls()
        obj.knots_x_ = np.asarray(d["knots_x"], dtype=float)
        obj.knots_y_ = np.asarray(d["knots_y"], dtype=float)
        return obj


class TemperatureCalibrator:
    """
    Temperature scaling: divide the logit by a single learned scalar T.

        p_calibrated = sigmoid(logit(score) / T)

    One parameter, so it cannot change the ranking at all -- AUROC and AUPRC
    are identical before and after. It is the weakest of the three for tree
    ensembles, and it is here precisely because it is weak: if temperature
    scaling matches Platt on the data, the miscalibration was pure
    over-confidence rather than a shape problem. That is a useful thing to be
    able to say in a writeup.
    """

    name = "temperature"

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.T = 1.0

    def _logit(self, p):
        p = np.clip(np.asarray(p, float).ravel(), self.eps, 1 - self.eps)
        return np.log(p / (1 - p))

    def fit(self, scores, y):
        z = self._logit(scores)
        y = np.asarray(y, float).ravel()
        best, best_nll = 1.0, np.inf
        for T in np.linspace(0.05, 10.0, 400):
            p = np.clip(1.0 / (1.0 + np.exp(-z / T)), self.eps, 1 - self.eps)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best, best_nll = T, nll
        self.T = float(best)
        return self

    def transform(self, scores) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self._logit(scores) / self.T))

    def to_dict(self) -> dict:
        return {"name": self.name, "eps": self.eps, "T": float(self.T)}

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureCalibrator":
        obj = cls(eps=float(d.get("eps", 1e-6)))
        obj.T = float(d["T"])
        return obj


CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
}


def calibrator_from_dict(d: dict):
    """
    Rebuild a fitted calibrator from the dict its own to_dict() produced.

    This is the inverse half of the serialisation contract and the reason the
    `name` field is written into every payload: the saved bundle records WHICH
    calibrator won on the calibration split, and the scanner has to reconstruct
    that specific one. Dispatching on a stored string keeps the decision in the
    data rather than hard-coding "always Platt" into the inference path.
    """
    name = d["name"]
    if name not in CALIBRATORS:
        raise ValueError(
            f"unknown calibrator {name!r}; expected one of {sorted(CALIBRATORS)}")
    return CALIBRATORS[name].from_dict(d)


def select_calibrator(scores_cal, y_cal, groups=None, n_splits: int = 5,
                      seed: int = 0) -> tuple[str, dict[str, float]]:
    """
    Choose which calibrator to ship, using no test data at all.

    Returns (winning_name, {name: out-of-fold ECE}).

    WHY NOT JUST COMPARE ECE ON THE CALIBRATION SPLIT

    Because the calibrators were FIT on that split, so the comparison is
    in-sample and the most flexible candidate wins by construction. Measured
    that way on the real corpus, isotonic regression scores an ECE of exactly
    0.0000 -- it is a step function with enough knots to reproduce its own
    training data perfectly. That is not evidence that it calibrates well; it
    is the definition of overfitting, and it would pick isotonic every single
    time regardless of how badly it generalises.

    So this does K-fold cross-validation INSIDE the calibration split: fit each
    calibrator on K-1 folds, predict the held-out fold, and accumulate
    out-of-fold predictions until every calibration row has a prediction made
    by a calibrator that never saw it. The ECE of those pooled out-of-fold
    predictions is an honest estimate of how each candidate generalises, and it
    is computed without touching the test split.

    The winner is then refitted on the whole calibration split by the caller.
    Selecting by cross-validation and refitting on all available data is the
    standard pattern -- the CV estimate chooses the model, the full fit
    maximises the data the chosen model learns from.

    GROUPING

    When `groups` (repository labels) is supplied, the folds are grouped by
    repository, for the same reason the main train/calib/test split is: Rust
    codebases are full of near-duplicate helper functions, and a fold boundary
    that cuts through one repository puts copies of the same function on both
    sides. That would flatter the flexible calibrator all over again, just at a
    smaller scale.
    """
    scores_cal = np.asarray(scores_cal, float).ravel()
    y_cal = np.asarray(y_cal).ravel()

    from .metrics import ece  # local import: metrics imports nothing from here

    # Enough of each class to make stratified folds meaningful?
    n_pos = int(y_cal.sum())
    if n_pos < n_splits or (len(y_cal) - n_pos) < n_splits:
        # Too few positives to cross-validate. Fall back to Platt, which has
        # two parameters and is the safe choice on small data -- rather than
        # silently returning an in-sample comparison that favours isotonic.
        return "platt", {}

    X = scores_cal.reshape(-1, 1)
    if groups is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        groups = np.asarray(groups).ravel()
        # StratifiedGroupKFold needs at least n_splits distinct groups.
        if len(set(groups)) >= n_splits:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=seed)
            folds = list(splitter.split(X, y_cal, groups))
        else:
            groups = None
    if groups is None:
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        folds = list(splitter.split(X, y_cal))

    oof = {name: np.full(len(y_cal), np.nan) for name in CALIBRATORS}
    for tr, te in folds:
        # A fold with only one class cannot fit a logistic regression.
        if len(set(y_cal[tr])) < 2:
            continue
        for name, cls in CALIBRATORS.items():
            try:
                c = cls().fit(scores_cal[tr], y_cal[tr])
                oof[name][te] = c.transform(scores_cal[te])
            except Exception:
                # A candidate that cannot be fitted is simply not a candidate.
                oof[name][te] = np.nan

    scored: dict[str, float] = {}
    for name, p in oof.items():
        ok = ~np.isnan(p)
        # Require most rows to have an out-of-fold prediction before trusting
        # the estimate for this candidate.
        if ok.sum() >= 0.5 * len(y_cal):
            scored[name] = ece(y_cal[ok], p[ok])

    if not scored:
        return "platt", {}
    return min(scored, key=lambda k: scored[k]), scored


def fit_all_calibrators(scores_cal, y_cal) -> dict:
    """
    Fit every calibrator on the held-out calibration split.

    These scores must come from the CALIBRATION split, and the ECE you report
    must be measured on the TEST split. Fitting a calibrator and then measuring
    its ECE on the same data gives an optimistically biased number -- it is the
    calibration-layer equivalent of testing on the training set.
    """
    return {name: cls().fit(scores_cal, y_cal) for name, cls in CALIBRATORS.items()}
