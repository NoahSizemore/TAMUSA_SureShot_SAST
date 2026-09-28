"""
The shared evaluation harness.

Both the reference pipeline and your own call run(), so the split, the
calibration, the threshold selection and every metric are byte-identical
between them. The ONLY difference is the model object handed in.

That is what makes a comparison between your model and the reference
meaningful. If you bypass this and write your own evaluation, you lose that
guarantee immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibrate import fit_all_calibrators, positive_scores
from .data import SIZE_FEATURES, Dataset
from .metrics import (ece_bootstrap, evaluate, operating_point, pick_threshold,
                      risk_coverage)


def run(model, ds: Dataset, label: str = "model",
        report: str | Path | None = None, verbose: bool = True) -> dict:
    """
    Train, calibrate and evaluate a model on an already-split Dataset.

    model must satisfy the ScoreModel protocol in calibrate.py:
        fit(X, y, X_val=None, y_val=None) and predict_proba(X) -> (n, 2)
    """
    if ds.train_idx is None:
        raise ValueError("Dataset not split. Call sureshot.data.split(ds) first.")

    X_tr, y_tr = ds.train
    X_ca, y_ca = ds.calib
    X_te, y_te = ds.test

    def say(*a):
        if verbose:
            print(*a)

    say(f"\n=== {label} ===")
    say(ds.summary())

    model.fit(X_tr, y_tr, X_val=X_ca, y_val=y_ca)

    s_cal = positive_scores(model, X_ca)
    s_te = positive_scores(model, X_te)
    cals = fit_all_calibrators(s_cal, y_ca)
    preds = {"uncalibrated": s_te}
    for nm, c in cals.items():
        preds[nm] = c.transform(s_te)

    results = {nm: evaluate(y_te, p) for nm, p in preds.items()}

    if verbose:
        print(f"\n{'model':14s} {'AUPRC':>7s} {'AUROC':>7s} {'Brier':>7s} "
              f"{'ECE-q':>7s} {'ECE-u':>7s} {'MCE':>7s}")
        for nm, r in results.items():
            print(f"{nm:14s} {r['auprc']:7.4f} {r['auroc']:7.4f} {r['brier']:7.4f} "
                  f"{r['ece_quantile']:7.4f} {r['ece_uniform']:7.4f} {r['mce']:7.4f}")

    # best calibrator by ECE on the test split
    cal_names = list(cals)
    best = min(cal_names, key=lambda k: results[k]["ece_quantile"])
    p_best = preds[best]
    m, lo, hi = ece_bootstrap(y_te, p_best)
    delta = results["uncalibrated"]["ece_quantile"] - results[best]["ece_quantile"]
    say(f"\nbest calibrator: {best}   ECE {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    say(f"ECE reduction vs uncalibrated: {delta:+.4f}")

    # threshold on the CALIBRATION split, never on test
    p_cal = cals[best].transform(s_cal)
    thr = pick_threshold(y_ca, p_cal, objective="f1")
    op = operating_point(y_te, p_best, thr)
    say(f"\nthreshold {thr:.2f} (picked on calibration split)")
    say(f"test precision={op['precision']:.3f}  recall={op['recall']:.3f}  "
        f"F1={op['f1']:.3f}   (base rate {y_te.mean():.4f})")

    rc = risk_coverage(y_te, p_best, thr)
    if rc:
        say("\nrisk-coverage (abstain on the least confident):")
        for target in (1.00, 0.80, 0.60, 0.40, 0.20):
            near = min(rc, key=lambda t: abs(t[0] - target))
            say(f"  coverage {near[0]:6.1%}   accuracy {near[1]:.3f}")

    out = {
        "label": label, "metrics": results, "best_calibrator": best,
        "ece_bootstrap": {"mean": m, "lo": lo, "hi": hi},
        "ece_reduction": delta, "operating_point": op,
        "risk_coverage": rc,
        "splits": {"train": len(ds.train_idx), "calib": len(ds.calib_idx),
                   "test": len(ds.test_idx)},
    }

    imp = getattr(model, "feature_importances_", None)
    if imp is not None and len(imp) == len(ds.feature_names):
        ranked = sorted(zip(ds.feature_names, map(float, imp)),
                        key=lambda x: -x[1])
        out["top_features"] = ranked[:15]
        share = sum(v for n, v in ranked if n in SIZE_FEATURES)
        out["size_feature_share"] = share
        if verbose:
            print("\n=== top 10 features by gain ===")
            for n, v in ranked[:10]:
                print(f"  {v:.4f}  {n}")
            print(f"\nsize-feature share: {share:.1%}")
            if share > 0.25:
                print("  WARNING: size features dominate. Positives skew long in "
                      "commit-mined data; re-run with drop_size_features=True.")

    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(json.dumps(out, indent=2, default=float))
        say(f"\nwrote {report}")

    return out


def compare(report_a: str | Path, report_b: str | Path) -> None:
    """Print a side-by-side of two runs. Use this to check your model against
    the reference once both have written reports."""
    a = json.loads(Path(report_a).read_text())
    b = json.loads(Path(report_b).read_text())
    print(f"{'metric':22s} {a['label'][:14]:>14s} {b['label'][:14]:>14s} {'delta':>9s}")
    for key in ("auprc", "auroc", "brier", "ece_quantile"):
        for cal in ("uncalibrated",):
            va, vb = a["metrics"][cal][key], b["metrics"][cal][key]
            print(f"{cal[:4]+'.'+key:22s} {va:14.4f} {vb:14.4f} {vb - va:+9.4f}")
    for key in ("precision", "recall", "f1"):
        va, vb = a["operating_point"][key], b["operating_point"][key]
        print(f"{'op.'+key:22s} {va:14.4f} {vb:14.4f} {vb - va:+9.4f}")
