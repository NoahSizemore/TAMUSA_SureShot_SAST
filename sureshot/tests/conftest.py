"""
Shared pytest fixtures.

Two things live here because both test_models.py and test_persist.py need them:
a toy dataset small enough to train on in under a second, and a way to import
the training scripts.

WHY IMPORTING THE SCRIPTS NEEDS A HELPER

`scripts/` is deliberately not a package -- the scripts are entry points, not
library code. On top of that, `train_scikit-learn.py` contains a hyphen, which
is not a legal character in a Python identifier, so `import train_scikit-learn`
is a syntax error no matter what the path looks like.

`importlib.util.spec_from_file_location` sidesteps both problems: it loads a
module from an explicit file path under a name we choose. This is the standard
way to test a script that was never meant to be imported, and it means the
model classes get tested as they actually ship rather than being copied into
the test file where they could drift out of sync.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sureshot import data

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_script(filename: str, module_name: str):
    """Import a module from an explicit path, bypassing normal import rules."""
    path = SCRIPTS / filename
    if not path.exists():
        pytest.skip(f"{path} not found")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Registering in sys.modules before exec lets the module use dataclasses,
    # typing.get_type_hints and anything else that looks itself up by name.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def native_script():
    """scripts/train_XGBoost.py, imported as a module."""
    return _load_script("train_XGBoost.py", "sureshot_train_xgboost")


@pytest.fixture(scope="session")
def reference_script():
    """scripts/train_scikit-learn.py, imported as a module."""
    return _load_script("train_scikit-learn.py", "sureshot_train_reference")


def make_toy_dataset(n: int = 600, n_repos: int = 30, n_features: int = 8,
                     seed: int = 0, positive_rate: float = 0.25):
    """
    A split, repo-grouped Dataset that trains fast.

    Deliberately learnable: the label depends on feature 0, so a working model
    scores clearly better than chance and a broken one is obvious. The real
    corpus is for measuring quality; this is for checking the plumbing.

    `positive_rate` is kept well above the real 0.058 base rate on purpose --
    at a realistic imbalance a 600-row toy would land only a handful of
    positives in the calibration split, and the calibrators would be fitting
    noise rather than exercising their code paths.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    groups = np.array([f"repo{i % n_repos}" for i in range(n)])
    cut = np.quantile(X[:, 0] + rng.normal(0, 0.5, n), 1 - positive_rate)
    y = (X[:, 0] + rng.normal(0, 0.5, n) > cut).astype(np.int32)
    ds = data.Dataset(X, y, groups, [f"f{i}" for i in range(n_features)], None)
    return data.split(ds, seed=42)


@pytest.fixture
def toy_dataset():
    return make_toy_dataset()
