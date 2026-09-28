"""
The calibration layer, and the interface your model must satisfy.

This is the part of SureShot that is the actual contribution: it maps a raw
classifier score onto a probability you can trust, and reports how much that
mapping improved things.

DESIGN NOTE, worth reading before you write your model.

Calibrators here fit on SCORES, not on estimators. The obvious alternative --
sklearn's CalibratedClassifierCV wrapping your model -- forces your model to
implement the full sklearn estimator protocol (predict, classes_,
__sklearn_is_fitted__, get_params) or it fails deep inside cross_val_predict
with an unhelpful error. None of that boilerplate has anything to do with
modelling.

Fitting on scores is mathematically identical: Platt scaling is a logistic
regression on one feature (the score), and isotonic regression is a monotonic
step function over it. Neither needs to know what produced the score.

The result is that your model needs exactly two methods:

    fit(X, y, X_val=None, y_val=None) -> self
    predict_proba(X) -> ndarray (n, 2), column 1 = P(vulnerable)

A raw xgboost Booster, a hand-rolled gradient booster, or a neural net all plug
in the same way.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@runtime_checkable
class ScoreModel(Protocol):
    """The contract your model must satisfy. Two methods, that is all."""

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

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)

    def _logit(self, p):
        p = np.clip(np.asarray(p, float).ravel(), self.eps, 1 - self.eps)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, scores, y):
        self.lr.fit(self._logit(scores), np.asarray(y).ravel())
        return self

    def transform(self, scores) -> np.ndarray:
        return self.lr.predict_proba(self._logit(scores))[:, 1]


class IsotonicCalibrator:
    """
    Isotonic regression: a non-decreasing step function fit on the score.

    Non-parametric and more flexible than Platt, so it wins on larger
    calibration sets. It overfits badly on small ones -- watch for an MCE far
    worse than Platt's while ECE looks comparable. That pattern means a few
    bins have collapsed onto extreme values.
    """

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores, y):
        self.iso.fit(np.asarray(scores, float).ravel(),
                     np.asarray(y, float).ravel())
        return self

    def transform(self, scores) -> np.ndarray:
        return self.iso.predict(np.asarray(scores, float).ravel())


class TemperatureCalibrator:
    """
    Temperature scaling: divide the logit by a single learned scalar T.

        p_calibrated = sigmoid(logit(score) / T)

    One parameter, so it cannot change the ranking at all -- AUROC and AUPRC
    are identical before and after. It is the weakest of the three for tree
    ensembles, and it is here precisely because it is weak: if temperature
    scaling matches Platt on your data, the miscalibration was pure
    over-confidence rather than a shape problem. That is a useful thing to be
    able to say in a writeup.
    """

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


CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "temperature": TemperatureCalibrator,
}


def fit_all_calibrators(scores_cal, y_cal) -> dict:
    """
    Fit every calibrator on the held-out calibration split.

    These scores must come from the CALIBRATION split, and the ECE you report
    must be measured on the TEST split. Fitting a calibrator and then measuring
    its ECE on the same data gives an optimistically biased number -- it is the
    calibration-layer equivalent of testing on your training set.
    """
    return {name: cls().fit(scores_cal, y_cal) for name, cls in CALIBRATORS.items()}
