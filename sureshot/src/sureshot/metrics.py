"""
Calibration and discrimination metrics.

sklearn ships Brier score but not ECE, so the calibration metrics here are
hand-written. This module is pure functions over (y_true, p_pred) -- it knows
nothing about models, so your pipeline and the reference pipeline are scored by
identical code.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)


def ece(y, p, n_bins: int = 15, strategy: str = "quantile") -> float:
    """
    Expected Calibration Error: mean gap between predicted confidence and
    observed frequency, weighted by bin population.

    strategy="quantile" uses equal-MASS bins (each holds the same number of
    samples). This is the more robust estimator and the one to lead with, because
    predictions cluster hard near zero on an imbalanced task and equal-width bins
    leave most bins nearly empty.

    strategy="uniform" uses equal-WIDTH bins. Report it too, since most papers do.

    Caveat to carry into any writeup: ECE is biased downward on small test sets
    and moves with bin count. Quote the bootstrap CI, not the point estimate.
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
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if not m.any():
            continue
        total += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(total)


def mce(y, p, n_bins: int = 15) -> float:
    """Maximum Calibration Error: the worst single bin, equal-width."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            worst = max(worst, abs(y[m].mean() - p[m].mean()))
    return float(worst)


def ece_bootstrap(y, p, n_boot: int = 500, n_bins: int = 15,
                  seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean and 95% CI for ECE. Required when the test set is small."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(ece(y[i], p[i], n_bins))
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def reliability_curve(y, p, n_bins: int = 15):
    """
    Per-bin (mean_predicted, observed_frequency, count) for a reliability
    diagram. Perfect calibration is the identity line.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if m.any():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def evaluate(y, p) -> dict[str, float]:
    """The standard metric block. Use this everywhere so runs are comparable."""
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_quantile": ece(y, p, 15, "quantile"),
        "ece_uniform": ece(y, p, 15, "uniform"),
        "mce": mce(y, p),
        "prevalence": float(np.mean(y)),
    }


def pick_threshold(y_cal, p_cal, objective: str = "f1",
                   target_precision: float = 0.80) -> float:
    """
    Choose the decision threshold on the CALIBRATION split. Never on test --
    picking a threshold on test and then reporting test F1 is leakage.

    objective="f1"        maximise F1
    objective="precision" cheapest threshold hitting target_precision
    """
    grid = np.linspace(0.02, 0.98, 97)
    if objective == "f1":
        scores = [f1_score(y_cal, (p_cal >= t).astype(int), zero_division=0)
                  for t in grid]
        return float(grid[int(np.argmax(scores))])

    best = None
    for t in grid:
        pr, rc, _, _ = precision_recall_fscore_support(
            y_cal, (p_cal >= t).astype(int), average="binary", zero_division=0)
        if pr >= target_precision:
            if best is None or rc > best[1]:
                best = (float(t), rc)
    return best[0] if best else float(grid[-1])


def operating_point(y, p, thr: float) -> dict[str, float]:
    """Precision / recall / F1 at a fixed threshold."""
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (p >= thr).astype(int), average="binary", zero_division=0)
    return {"threshold": float(thr), "precision": float(pr),
            "recall": float(rc), "f1": float(f1)}


def risk_coverage(y, p, thr: float, n_points: int = 21):
    """
    Selective prediction curve: accuracy as a function of coverage.

    Rank predictions by how far they sit from the decision threshold -- that
    distance is the model's confidence in its own call -- then answer only the
    most confident fraction and abstain on the rest.

    Sweeping a symmetric band around the threshold does NOT work here: the
    threshold sits near 0.09 on this imbalanced task, so the lower edge runs off
    the end of [0, 1] immediately and coverage barely moves. Ranking by distance
    avoids that entirely and gives an evenly spaced curve.

    Returns a list of (coverage, accuracy, min_confidence_margin).
    This is the strongest single figure this project can produce: a model with
    weak raw accuracy is still useful if it knows when to abstain.
    """
    y, p = np.asarray(y), np.asarray(p, float)
    margin = np.abs(p - thr)
    order = np.argsort(-margin)          # most confident first
    correct = ((p >= thr).astype(int) == y).astype(float)[order]

    out = []
    for frac in np.linspace(1.0, 0.05, n_points):
        k = max(int(round(frac * len(y))), 1)
        out.append((float(k / len(y)),
                    float(correct[:k].mean()),
                    float(margin[order][k - 1])))
    return out
