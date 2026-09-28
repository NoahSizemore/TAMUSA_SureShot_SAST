#!/usr/bin/env python3
"""
YOUR MODEL GOES HERE.

Everything except the model is already done and imported from the sureshot
package: loading, featurisation, repo-grouped splitting, calibration, metrics,
threshold selection, risk-coverage. You implement one class.

The contract is two methods:

    fit(X, y, X_val=None, y_val=None) -> self
    predict_proba(X) -> ndarray (n, 2), column 1 = P(vulnerable)

Optionally expose `feature_importances_` (length == len(ds.feature_names)) and
the pipeline will print the importance table and run the size-confound check.

Run it:
    python scripts/train_mine.py --data data/processed/rustsec_v0.1.parquet \
        --source-type rustsec --report reports/mine.json

Compare against the reference:
    python -c "from sureshot.pipeline import compare; \
        compare('reports/reference.json', 'reports/mine.json')"

Because both scripts call sureshot.pipeline.run() with the same seed, the
splits and metrics are identical. Any difference in the numbers is caused by
your model and nothing else. That is the whole point of this layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb

from sureshot import data, pipeline


class MyModel:
    """
    TIER 2 (recommended): the native XGBoost API.

    The sklearn wrapper hides the booster. Using DMatrix + xgb.train() gives you
    the real interface: custom objectives, custom eval functions, the full
    evals_result history you can plot, and direct access to the Booster.

    Things to work out for yourself as you fill this in:

      - Why eval_metric "aucpr" and not "auc"? (Hint: at 1:16 imbalance the
        false-positive rate denominator is huge, so ROC looks good for free.)
      - What does scale_pos_weight actually do to the gradient? Derive it.
      - Why max_depth 5 and not 10, on ~290 training positives?
      - early_stopping watches the calibration split here. What does that cost
        you in terms of split independence, and is it acceptable?
      - What does `base_score` default to, and does it matter when the positive
        class is 6% of the data?

    Delete my scaffolding freely. Only fit/predict_proba must survive.
    """

    def __init__(self, seed: int = 42, scale_pos_weight: float = 1.0):
        self.seed = seed
        self.booster: xgb.Booster | None = None
        self.evals_result: dict = {}
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            # TODO: tune these yourself and justify each one.
            "max_depth": 5,
            "eta": 0.05,                       # native API name for learning_rate
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "lambda": 1.0,                     # native API name for reg_lambda
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "seed": seed,
            "nthread": 4,
        }
        self.num_boost_round = 2000
        self.early_stopping_rounds = 50

    def fit(self, X, y, X_val=None, y_val=None):
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]
        if X_val is not None:
            evals.append((xgb.DMatrix(X_val, label=y_val), "calib"))

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            evals_result=self.evals_result,
            verbose_eval=False,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("call fit() first")
        # iteration_range stops at the best round found by early stopping;
        # without it you silently predict with the overfitted final trees.
        best = getattr(self.booster, "best_iteration", None)
        rng = (0, best + 1) if best is not None else None
        p = self.booster.predict(xgb.DMatrix(X), iteration_range=rng)
        return np.column_stack([1.0 - p, p])

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """
        The native API keys importance by feature name ('f0', 'f1', ...), not
        by position, and OMITS features that were never split on. Rebuild a
        dense vector or the pipeline's importance table will misalign.
        """
        if self.booster is None:
            return None
        gain = self.booster.get_score(importance_type="gain")
        n = self.booster.num_features()
        out = np.zeros(n, dtype=float)
        for k, v in gain.items():
            out[int(k[1:])] = v
        total = out.sum()
        return out / total if total > 0 else out


# ---------------------------------------------------------------------------
# TIER 3, if you want to go further: replace MyModel entirely with your own
# gradient booster. The skeleton below is the shape of it. Everything
# downstream keeps working, because the contract is still fit/predict_proba.
#
# class MyGradientBooster:
#     """Gradient boosting on regression trees, logistic loss.
#
#     For log-loss with raw score F:
#         p        = sigmoid(F)
#         gradient = p - y
#         hessian  = p * (1 - p)
#     Each tree fits the Newton step -g/h, leaf value = -sum(g)/(sum(h)+lambda).
#     Weight the positive class by multiplying its gradient and hessian by
#     scale_pos_weight -- work out why that is the right place to apply it.
#     """
#     def fit(self, X, y, X_val=None, y_val=None): ...
#     def predict_proba(self, X): ...
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--source-type", default=None,
                    choices=[None, "rustsec", "synthetic"])
    ap.add_argument("--seed", type=int, default=42,
                    help="keep this equal to the reference run or the "
                         "comparison is meaningless")
    ap.add_argument("--drop-size-features", action="store_true")
    ap.add_argument("--report", type=Path, default=Path("reports/mine.json"))
    args = ap.parse_args()

    ds = data.load(args.data, args.source_type, args.drop_size_features)
    data.split(ds, seed=args.seed)

    model = MyModel(seed=args.seed, scale_pos_weight=ds.scale_pos_weight())
    pipeline.run(model, ds, label="mine", report=args.report)


if __name__ == "__main__":
    main()
