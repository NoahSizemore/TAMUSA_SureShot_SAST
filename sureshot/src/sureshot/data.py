"""
Data loading, feature matrix construction, and splitting.

Both the reference pipeline and your own must import from here, so that any
comparison between them differs ONLY in the model. If you copy this logic into
your own script and it drifts, you are no longer comparing models -- you are
comparing two different pipelines, and the comparison is meaningless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .features import FEATURE_NAMES, extract_features

# Raw size features. Positives in commit-mined data skew long (~2.4x), so these
# are the ones to ablate when checking whether the model learned "big function"
# rather than "vulnerable".
SIZE_FEATURES = {
    "loc", "loc_nonblank", "chars", "avg_line_len", "max_line_len", "ident_count",
}


class Dataset:
    """A loaded, featurised, split dataset. Immutable once constructed."""

    def __init__(self, X, y, groups, names, df):
        self.X, self.y, self.groups = X, y, groups
        self.feature_names = names
        self.df = df
        self.train_idx: np.ndarray | None = None
        self.calib_idx: np.ndarray | None = None
        self.test_idx: np.ndarray | None = None

    # -- convenience accessors so callers never index by hand ---------------
    @property
    def train(self):
        return self.X[self.train_idx], self.y[self.train_idx]

    @property
    def calib(self):
        return self.X[self.calib_idx], self.y[self.calib_idx]

    @property
    def test(self):
        return self.X[self.test_idx], self.y[self.test_idx]

    def summary(self) -> str:
        lines = [
            f"rows={len(self.y)}  pos={int(self.y.sum())}  "
            f"repos={len(set(self.groups))}  features={len(self.feature_names)}"
        ]
        for nm, ix in (("train", self.train_idx), ("calib", self.calib_idx),
                       ("test", self.test_idx)):
            if ix is None:
                continue
            lines.append(
                f"  {nm:6s} n={len(ix):5d}  pos={int(self.y[ix].sum()):4d}  "
                f"repos={len(set(self.groups[ix]))}  "
                f"prevalence={self.y[ix].mean():.4f}"
            )
        return "\n".join(lines)

    def scale_pos_weight(self) -> float:
        """The imbalance correction XGBoost needs. Computed on TRAIN only."""
        y = self.y[self.train_idx]
        pos = int(y.sum())
        return float((len(y) - pos) / max(pos, 1))


def load(path: str | Path, source_type: str | None = None,
         drop_size_features: bool = False) -> Dataset:
    """
    Load a parquet corpus and build the feature matrix.

    source_type: pass "rustsec" or "synthetic" to filter. Metrics from the two
    must never be pooled, so filtering here is safer than filtering downstream.
    """
    df = pd.read_parquet(path)
    if source_type is not None:
        df = df[df.source_type == source_type].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"no rows with source_type={source_type!r}")

    rows = [extract_features(s) for s in df["source"]]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0.0).to_numpy(np.float32)
    names = list(FEATURE_NAMES)

    if drop_size_features:
        keep = [i for i, n in enumerate(names) if n not in SIZE_FEATURES]
        X, names = X[:, keep], [names[i] for i in keep]

    return Dataset(X, df["label"].to_numpy(np.int32),
                   df["repo"].to_numpy(), names, df)


def split(ds: Dataset, seed: int = 42, test_frac: float = 0.20,
          calib_frac: float = 0.20) -> Dataset:
    """
    Three-way split GROUPED BY REPOSITORY, mutating ds in place.

    Splitting by function instead of repository is the single most common way
    to fake good results on this task: Rust codebases are full of near-duplicate
    helpers, and a random split puts copies of the same function on both sides
    of the boundary. The model memorises and the F1 looks excellent.

    The assertion at the end is not optional. Leave it in.
    """
    idx = np.arange(len(ds.y))
    rest_frac = test_frac + calib_frac

    gss = GroupShuffleSplit(n_splits=1, test_size=rest_frac, random_state=seed)
    tr, rest = next(gss.split(idx, ds.y, ds.groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac / rest_frac,
                             random_state=seed)
    c_rel, t_rel = next(gss2.split(rest, ds.y[rest], ds.groups[rest]))

    ds.train_idx, ds.calib_idx, ds.test_idx = tr, rest[c_rel], rest[t_rel]

    train_repos = set(ds.groups[ds.train_idx])
    for nm, ix in (("calib", ds.calib_idx), ("test", ds.test_idx)):
        overlap = train_repos & set(ds.groups[ix])
        assert not overlap, f"repo leakage train/{nm}: {sorted(overlap)[:5]}"
    overlap = set(ds.groups[ds.calib_idx]) & set(ds.groups[ds.test_idx])
    assert not overlap, f"repo leakage calib/test: {sorted(overlap)[:5]}"

    return ds
