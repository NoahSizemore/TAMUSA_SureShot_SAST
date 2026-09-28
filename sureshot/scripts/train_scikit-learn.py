#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import xgboost as xgb

from sureshot import data, pipeline


class ReferenceModel:
    """
    XGBoost via the scikit-learn wrapper. Satisfies the ScoreModel protocol.

    Hyperparameter reasoning, since you will be justifying your own:

    eval_metric="aucpr"     Area under precision-recall, not ROC. At 1:16
                            imbalance ROC is misleadingly flattering, because
                            the huge negative class makes the false-positive
                            rate look small no matter what.
    scale_pos_weight        neg/pos on the TRAINING split. Without it the model
                            predicts "safe" everywhere and scores 94% accuracy.
    max_depth=5             Shallow. With ~290 training positives, depth 8+
                            memorises individual functions.
    min_child_weight=5      Refuses splits backed by too few samples. Second
                            line of defence against memorising rare positives.
    learning_rate=0.05      Slow, paired with many rounds and early stopping.
    subsample /             Row and column sampling per tree. Decorrelates the
    colsample_bytree=0.8    ensemble on a small, noisy dataset.
    early_stopping_rounds   Watches the CALIBRATION split. Note this means the
                            calibration split is lightly used for model
                            selection as well as calibration -- acceptable, but
                            worth stating in a writeup.
    """

    def __init__(self, seed: int = 42, scale_pos_weight: float = 1.0):
        self.clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=2000,
            early_stopping_rounds=50,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        evals = [(X_val, y_val)] if X_val is not None else None
        self.clf.fit(X, y, eval_set=evals, verbose=False)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--source-type", default=None,
                    choices=[None, "rustsec", "synthetic"],
                    help="filter the corpus; never pool the two")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-size-features", action="store_true")
    ap.add_argument("--report", type=Path, default=Path("reports/reference.json"))
    args = ap.parse_args()

    ds = data.load(args.data, args.source_type, args.drop_size_features)
    data.split(ds, seed=args.seed)

    model = ReferenceModel(seed=args.seed,
                           scale_pos_weight=ds.scale_pos_weight())
    pipeline.run(model, ds, label="reference-xgb", report=args.report)


if __name__ == "__main__":
    main()
