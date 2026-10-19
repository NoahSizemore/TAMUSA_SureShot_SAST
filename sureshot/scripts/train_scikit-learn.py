#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import xgboost as xgb

from sureshot import data, pipeline


class ReferenceModel:
    """
    XGBoost via the scikit-learn wrapper. Satisfies the ScoreModel protocol.

    Hyperparameter reasoning:

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

    # Saved in XGBoost's native booster format rather than as a pickled
    # sklearn estimator, so the file does not depend on the scikit-learn
    # version that wrote it. See sureshot/persist.py for why pickle is avoided.
    model_kind = "xgboost-booster"

    def __init__(self, seed: int = 42, scale_pos_weight: float = 1.0):
        self.early_stopping_rounds = 50
        self.clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=2000,
            early_stopping_rounds=self.early_stopping_rounds,
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
        # early_stopping_rounds is set on the CONSTRUCTOR, which means the
        # callback is installed whether or not an eval_set is later supplied --
        # and it raises "Must have at least 1 validation dataset for early
        # stopping" when there is nothing to watch. Passing eval_set=None is
        # therefore not enough; the parameter itself has to be turned off.
        #
        # pipeline.run() always supplies a validation set, so this path never
        # runs during a normal training run. It is still worth getting right:
        # the native model in train_XGBoost.py already handles it, and a model
        # that cannot be fitted without a validation set is a model that cannot
        # be used for a quick sanity check or a cross-validation loop.
        if X_val is None or y_val is None:
            self.clf.set_params(early_stopping_rounds=None)
            self.clf.fit(X, y, verbose=False)
        else:
            self.clf.set_params(
                early_stopping_rounds=self.early_stopping_rounds)
            self.clf.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def save_model(self, path) -> None:
        """
        Write the underlying booster, sliced to the early-stopped tree count.

        The slice is what keeps a reloaded model honest. XGBClassifier's own
        predict_proba() silently restricts itself to the best iteration when
        early stopping ran, but a bare booster loaded from disk has no such
        instruction and would use every tree it finds -- including the
        overfitted tail this model trained past and then rejected.
        """
        booster = self.clf.get_booster()
        best = getattr(self.clf, "best_iteration", None)
        if best is not None:
            booster = booster[0:int(best) + 1]
        booster.save_model(str(path))

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
    ap.add_argument("--save", type=Path, default=None,
                    help="directory to write a reusable detector bundle to")
    ap.add_argument("--select-on", default="test", choices=["test", "calib"],
                    help="split used to choose the winning calibrator; 'test' "
                         "reproduces the published numbers, 'calib' is the "
                         "leak-free option (see pipeline.run)")
    args = ap.parse_args()

    ds = data.load(args.data, args.source_type, args.drop_size_features)
    data.split(ds, seed=args.seed)

    model = ReferenceModel(seed=args.seed,
                           scale_pos_weight=ds.scale_pos_weight())
    pipeline.run(model, ds, label="reference-xgb", report=args.report,
                 save_to=args.save, select_on=args.select_on)


if __name__ == "__main__":
    main()
