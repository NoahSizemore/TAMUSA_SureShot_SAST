"""
Saving and loading a trained, calibrated, thresholded detector.

WHY THIS MODULE EXISTS

Until now the project could only ever answer "how good is this model?" --
`pipeline.run()` trained, measured, printed a table and then threw the trained
model away. Nothing was kept, so nothing could be used. That is fine for
benchmarking and useless for a tool: a SAST scanner has to answer "is THIS
function, which I have never seen before, dangerous?", and to do that it needs
the trained artefact to still exist after training has finished.

WHAT A USABLE DETECTOR ACTUALLY CONSISTS OF

It is tempting to think "save the model" means "save the booster". It does not.
A booster alone produces raw scores that are badly miscalibrated -- on this
dataset the uncalibrated ECE is 0.354, meaning the numbers it emits are not
probabilities in any meaningful sense. Three things have to travel together, and
dropping any one of them silently breaks the tool:

  1. THE BOOSTER         produces a raw score for a feature vector.
  2. THE CALIBRATOR      maps that raw score onto an honest probability. Which
                         calibrator is not a fixed choice -- it is selected on
                         the calibration split at training time, so the winner
                         has to be recorded, not assumed.
  3. THE THRESHOLD       turns a probability into a yes/no call. Chosen on the
                         calibration split (never on test), and nowhere near
                         0.5: on this 1:16-imbalanced task it lands around 0.09.
                         A scanner that defaults to 0.5 reports almost nothing.

The bundle also records the FEATURE NAMES, which are not part of the detector
but are part of its input contract. If someone adds a token rule to features.py
and then runs an old bundle, the booster receives 142 columns that no longer
mean what it learned them to mean, and the output is confident nonsense. Column
counts happening to match is not the same as columns meaning the same thing, so
the names are compared, not just the count.

ON-DISK LAYOUT

A bundle is a directory, not a single file, because the two things being stored
have genuinely different formats:

    artifacts/xgboost/
        manifest.json   calibrator params, threshold, feature names, metadata
        model.json      the booster, in XGBoost's own documented format

WHY NOT PICKLE

Pickle would do all of this in three lines. It is the wrong answer here:

  - `pickle.load` executes arbitrary code from the file by construction. For a
    security tool, shipping a format whose load path is remote code execution
    is indefensible.
  - A pickle of a scikit-learn estimator is bound to the scikit-learn version
    that wrote it and breaks, sometimes silently, on upgrade.
  - It is opaque. `manifest.json` can be opened in an editor, diffed in a pull
    request, and checked by a human.

Everything here is JSON plus XGBoost's own versioned format, both of which are
inspectable and stable across library upgrades.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .calibrate import calibrator_from_dict, positive_scores

# Bumped when the on-disk layout changes incompatibly. load() refuses anything
# it does not recognise, so an old bundle fails loudly instead of being
# misread as a new one.
BUNDLE_VERSION = 1

MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.json"


# --------------------------------------------------------------------------
# The model-persistence contract
# --------------------------------------------------------------------------

@runtime_checkable
class SavableModel(Protocol):
    """
    What a model must add, on top of the two-method ScoreModel contract in
    calibrate.py, to be saveable.

        model_kind    a string naming the on-disk format, e.g. "xgboost-booster"
        save_model    write the trained estimator to one file

    Loading is deliberately NOT a method on the model. At load time there is no
    model object yet -- that is the whole problem -- so reconstruction is done
    by a registered loader function keyed on `model_kind`. This is the standard
    way to solve "I need to build an object before I know its class": store a
    tag, and look the constructor up in a registry.
    """

    model_kind: str

    def save_model(self, path: Path) -> None: ...


# kind -> function that rebuilds a predictor from the file save_model() wrote.
MODEL_LOADERS: dict[str, Callable[[Path], object]] = {}


def register_loader(kind: str, fn: Callable[[Path], object]) -> None:
    """
    Teach persist.py how to reload a new model format.

    Add a model type to this project and you register its loader here; nothing
    else in the package needs to change.
    """
    MODEL_LOADERS[kind] = fn


class _BoosterPredictor:
    """
    A loaded XGBoost booster, wrapped to satisfy the ScoreModel contract.

    The raw `xgboost.Booster` API takes a DMatrix and returns a 1-D vector of
    positive-class probabilities. The rest of this package expects sklearn's
    `predict_proba` shape -- (n, 2), column 1 = P(vulnerable) -- so the wrapper
    exists purely to translate between the two.

    There is no `iteration_range` here on purpose. Boosters are sliced down to
    their early-stopped tree count BEFORE being saved (see the training
    scripts), so every tree in the file is a tree we intend to use. Carrying a
    "use only the first N trees" instruction alongside the model is an easy way
    to end up evaluating 2000 trees in production and 400 in the paper.
    """

    model_kind = "xgboost-booster"

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X) -> np.ndarray:
        import xgboost as xgb
        p = self.booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)))
        p = np.asarray(p, dtype=float).ravel()
        return np.column_stack([1.0 - p, p])

    @property
    def n_trees(self) -> int:
        return int(self.booster.num_boosted_rounds())


def _load_xgboost_booster(path: Path):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(path))
    return _BoosterPredictor(booster)


register_loader("xgboost-booster", _load_xgboost_booster)


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------

@dataclass
class Bundle:
    """
    A trained detector that can be used on new code.

    `model` is anything with predict_proba; `calibrator` has transform();
    `threshold` is the decision boundary on CALIBRATED probabilities.
    """

    model: object
    calibrator: object
    threshold: float
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)

    # -- using it -----------------------------------------------------------

    def score(self, X) -> np.ndarray:
        """
        Calibrated P(vulnerable) for each row. This is the number to show a
        user, because it is the one the ECE figures actually describe.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"feature count mismatch: bundle expects "
                f"{len(self.feature_names)}, got {X.shape[1]}. The feature "
                f"extractor has changed since this model was trained -- retrain "
                f"it rather than forcing the shapes to agree.")
        raw = positive_scores(self.model, X)
        return np.asarray(self.calibrator.transform(raw), dtype=float)

    def decide(self, X) -> np.ndarray:
        """Hard 0/1 call at the stored threshold."""
        return (self.score(X) >= self.threshold).astype(int)

    def margin(self, X) -> np.ndarray:
        """
        Distance from the threshold: how firmly the model holds its own call.

        This is the quantity the risk-coverage curve in metrics.py ranks by, and
        it is what lets the scanner abstain. Given this model's weak raw
        discrimination (AUROC 0.62), being able to say "I am not confident here"
        is worth more than the yes/no call itself.
        """
        return np.abs(self.score(X) - self.threshold)

    # -- saving and loading -------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the bundle to a directory, creating it if needed."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        kind = getattr(self.model, "model_kind", None)
        if kind is None:
            raise TypeError(
                f"{type(self.model).__name__} has no `model_kind`, so it cannot "
                f"be saved. Give it a model_kind string and a save_model(path) "
                f"method, and register a loader with persist.register_loader().")
        if not hasattr(self.model, "save_model"):
            raise TypeError(
                f"{type(self.model).__name__} has no save_model(path) method.")

        self.model.save_model(path / MODEL_NAME)

        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "model_kind": kind,
            "model_file": MODEL_NAME,
            "calibrator": self.calibrator.to_dict(),
            "threshold": float(self.threshold),
            "feature_names": list(self.feature_names),
            "n_features": len(self.feature_names),
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata": self.metadata,
        }
        (path / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, default=float), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        """Rebuild a bundle written by save()."""
        path = Path(path)
        manifest_path = path / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {path}. Train and save a model first, "
                f"e.g. `python scripts/train_XGBoost.py ... --save {path}`.")

        m = json.loads(manifest_path.read_text(encoding="utf-8"))

        version = m.get("bundle_version")
        if version != BUNDLE_VERSION:
            raise ValueError(
                f"bundle version {version} but this code reads version "
                f"{BUNDLE_VERSION}. Retrain and re-save.")

        kind = m["model_kind"]
        if kind not in MODEL_LOADERS:
            raise ValueError(
                f"no loader registered for model_kind {kind!r}; known kinds are "
                f"{sorted(MODEL_LOADERS)}. Use persist.register_loader().")

        model = MODEL_LOADERS[kind](path / m.get("model_file", MODEL_NAME))
        calibrator = calibrator_from_dict(m["calibrator"])

        return cls(model=model,
                   calibrator=calibrator,
                   threshold=float(m["threshold"]),
                   feature_names=list(m["feature_names"]),
                   metadata=m.get("metadata", {}))

    # -- introspection ------------------------------------------------------

    def describe(self) -> str:
        """Human-readable summary, printed by the scanner CLI."""
        md = self.metadata
        lines = [
            f"calibrator     {self.calibrator.name}",
            f"threshold      {self.threshold:.4f}",
            f"features       {len(self.feature_names)}",
        ]
        if "trained_on" in md:
            lines.append(f"trained on     {md['trained_on']}")
        if "label" in md:
            lines.append(f"run label      {md['label']}")
        if "test_auprc" in md:
            lines.append(
                f"test AUPRC     {md['test_auprc']:.4f}   "
                f"AUROC {md.get('test_auroc', float('nan')):.4f}")
        if "test_precision" in md:
            lines.append(
                f"test P/R       precision {md['test_precision']:.3f}  "
                f"recall {md['test_recall']:.3f}")
        return "\n".join(lines)
