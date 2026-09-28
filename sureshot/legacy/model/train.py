"""
SureShot SAST - training, calibration and evaluation.

Pipeline:
  parquet -> feature matrix -> repo-grouped 3-way split
          -> XGBoost -> calibration (Platt / isotonic) -> metrics

Two rules this script enforces, because violating either produces numbers that
look excellent and mean nothing:

  1. SPLIT BY REPOSITORY, never by function. Rust codebases are full of
     near-duplicate helpers; a random function split leaks them across the
     boundary and the model memorises rather than generalises.

  2. The CALIBRATION split is used only to fit the calibrator and pick the
     threshold. Fitting a calibrator and then measuring ECE on the same data
     gives an optimistically biased estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extract"))
from features import FEATURE_NAMES, extract_features  # noqa: E402


# --------------------------------------------------------------------------
# calibration metrics
# --------------------------------------------------------------------------

def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15,
        strategy: str = "quantile") -> float:
    """
    Expected Calibration Error.

    strategy="quantile" (equal-mass bins) is the more robust estimator and the
    one to lead with. "uniform" (equal-width) is included because most papers
    report it. ECE is biased downward on small test sets and moves with bin
    count, so bootstrap a CI before quoting a headline number.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            return float(abs(p.mean() - y.mean()))
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > edges[0] else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        total += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(total)


def mce(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error (equal-width bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            worst = max(worst, abs(y[m].mean() - p[m].mean()))
    return float(worst)


def ece_bootstrap(y, p, n_boot=500, n_bins=15, seed=0):
    """Bootstrap CI for ECE - required when the test set is small."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [ece(y[i], p[i], n_bins) for i in
            (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [extract_features(s) for s in df["source"]]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0.0).to_numpy(np.float32)
    return X, df["label"].to_numpy(np.int32), df["repo"].to_numpy()


def grouped_split(groups, y, seed=42):
    """60/20/20 train/calibration/test, split on repository."""
    idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    tr, rest = next(gss.split(idx, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    c_rel, t_rel = next(gss2.split(rest, y[rest], groups[rest]))
    return tr, rest[c_rel], rest[t_rel]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-size-features", action="store_true",
                    help="ablation: remove raw size features to test the "
                         "length confound in commit-mined data")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print(f"loaded {len(df)} rows | pos={int(df.label.sum())} | repos={df.repo.nunique()}")

    X, y, groups = build_matrix(df)
    names = list(FEATURE_NAMES)
    if args.drop_size_features:
        drop = {"loc", "loc_nonblank", "chars", "avg_line_len",
                "max_line_len", "ident_count"}
        keep = [i for i, n in enumerate(names) if n not in drop]
        X, names = X[:, keep], [names[i] for i in keep]
        print(f"ablation: dropped size features, {len(names)} remain")

    tr, ca, te = grouped_split(groups, y, args.seed)
    for nm, ix in (("train", tr), ("calib", ca), ("test", te)):
        print(f"  {nm:6s} n={len(ix):5d}  pos={int(y[ix].sum()):4d}  "
              f"repos={len(set(groups[ix]))}")

    overlap = set(groups[tr]) & set(groups[te])
    assert not overlap, f"repo leakage across splits: {overlap}"
    print("  leakage check: no repo appears in more than one split\n")

    pos, neg = int(y[tr].sum()), int((y[tr] == 0).sum())
    clf = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr",
        n_estimators=2000, early_stopping_rounds=50,
        max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=1.0, scale_pos_weight=neg / max(pos, 1),
        tree_method="hist", random_state=args.seed, n_jobs=4,
    )
    clf.fit(X[tr], y[tr], eval_set=[(X[ca], y[ca])], verbose=False)
    print(f"trained: {clf.best_iteration + 1} rounds\n")

    p_raw = clf.predict_proba(X[te])[:, 1]

    # sklearn >=1.6 replaced cv="prefit" with FrozenEstimator
    frozen = FrozenEstimator(clf)
    platt = CalibratedClassifierCV(frozen, method="sigmoid").fit(X[ca], y[ca])
    iso = CalibratedClassifierCV(frozen, method="isotonic").fit(X[ca], y[ca])
    p_platt = platt.predict_proba(X[te])[:, 1]
    p_iso = iso.predict_proba(X[te])[:, 1]

    print(f"{'model':12s} {'AUPRC':>7s} {'AUROC':>7s} {'Brier':>7s} "
          f"{'ECE-q':>7s} {'ECE-u':>7s} {'MCE':>7s}")
    results = {}
    for nm, p in (("uncalibrated", p_raw), ("platt", p_platt), ("isotonic", p_iso)):
        r = {
            "auprc": average_precision_score(y[te], p),
            "auroc": roc_auc_score(y[te], p),
            "brier": brier_score_loss(y[te], p),
            "ece_quantile": ece(y[te], p, 15, "quantile"),
            "ece_uniform": ece(y[te], p, 15, "uniform"),
            "mce": mce(y[te], p),
        }
        results[nm] = r
        print(f"{nm:12s} {r['auprc']:7.4f} {r['auroc']:7.4f} {r['brier']:7.4f} "
              f"{r['ece_quantile']:7.4f} {r['ece_uniform']:7.4f} {r['mce']:7.4f}")

    best = min(("platt", "isotonic"), key=lambda k: results[k]["ece_quantile"])
    p_best = {"platt": p_platt, "isotonic": p_iso}[best]
    m, lo, hi = ece_bootstrap(y[te], p_best)
    print(f"\nbest calibrator: {best}   ECE {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"ECE reduction vs uncalibrated: "
          f"{results['uncalibrated']['ece_quantile'] - results[best]['ece_quantile']:+.4f}")

    # threshold chosen on the CALIBRATION split, never on test
    p_cal = {"platt": platt, "isotonic": iso}[best].predict_proba(X[ca])[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y[ca], (p_cal >= t).astype(int), zero_division=0) for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    pr, rc, f1, _ = precision_recall_fscore_support(
        y[te], (p_best >= thr).astype(int), average="binary", zero_division=0)
    print(f"\nthreshold {thr:.2f} (picked on calibration split)")
    print(f"test precision={pr:.3f}  recall={rc:.3f}  F1={f1:.3f}")

    # selective prediction: abstain in the uncertain band
    lo_b, hi_b = thr * 0.5, min(thr * 1.5, 0.95)
    conf = (p_best < lo_b) | (p_best > hi_b)
    if conf.sum():
        acc = ((p_best[conf] >= thr).astype(int) == y[te][conf]).mean()
        print(f"abstention band [{lo_b:.2f},{hi_b:.2f}]: "
              f"coverage={conf.mean():.1%}  accuracy on covered={acc:.3f}")

    print("\n=== top 15 features by gain ===")
    imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])[:15]
    for n, v in imp:
        print(f"  {v:.4f}  {n}")

    size_feats = {"loc", "loc_nonblank", "chars", "avg_line_len",
                  "max_line_len", "ident_count"}
    size_share = sum(v for n, v in zip(names, clf.feature_importances_)
                     if n in size_feats)
    print(f"\nsize-feature share of total importance: {size_share:.1%}")
    if size_share > 0.25:
        print("  WARNING: size features dominate. In commit-mined data the "
              "positive class skews long, so the model may be learning "
              "'big function' rather than 'vulnerable'. Re-run with "
              "--drop-size-features to check.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "data": str(args.data), "n": len(df),
            "splits": {"train": len(tr), "calib": len(ca), "test": len(te)},
            "metrics": results, "best_calibrator": best,
            "ece_ci": [m, lo, hi], "threshold": thr,
            "test_precision": pr, "test_recall": rc, "test_f1": f1,
            "size_feature_share": size_share,
            "top_features": imp,
        }, indent=2, default=float))
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
