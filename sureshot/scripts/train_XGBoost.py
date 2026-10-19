#!/usr/bin/env python3
"""
The native-API XGBoost detector.

This is the counterpart to scripts/train_scikit-learn.py. Both train XGBoost on
identical data through the identical harness; the difference is that this one
drives the library through its own `xgb.train` / `Booster` API instead of the
scikit-learn compatibility wrapper.

Using the native API is not cosmetic. It gives direct access to the Booster,
which is what makes the early-stopped tree count visible and sliceable at save
time (see save_model below), and it keeps the evaluation history in
`self.evals_result` so the learning curve can be inspected afterwards.

With the hyperparameters currently set here matching the reference, this script
is the equivalence check STRUCTURE.md asks for: identical settings driven
through a different API should reproduce the reference metrics almost exactly,
which is what proves the harness itself is sound before anything is tuned.
"""

# imports
import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb

from sureshot import data, pipeline


# building XGBoost model
class MyModel:
    # Tells sureshot.persist which on-disk format this model saves itself in,
    # and therefore which registered loader can rebuild it later. persist.py
    # ships a loader for "xgboost-booster".
    model_kind = "xgboost-booster"

    def __init__(self, seed: int = 42, scale_pos_weight: float = 1.0):
        self.seed = seed
        self.booster: xgb.Booster | None = None
        self.evals_result: dict = {}
        self.params = {
            "objective": "binary:logistic", # using logistic regression objective for binary classification
            "eval_metric": "aucpr", # using area under the precision-recall curve as evaluation metric
            "max_depth": 5, # shallow: with ~290 training positives, depth 8+ memorises individual functions
            "eta": 0.05, # learning rate for the model
            "subsample": 0.8, # fraction of samples to use for each tree
            "colsample_bytree": 0.8, # fraction of columns to use for each tree
            "min_child_weight": 5, # sum of Hessians
            "lambda": 1.0, # using 1.0 for regularization
            "scale_pos_weight": scale_pos_weight, # adjusting for class imbalance
            "tree_method": "hist", # using histogram-based tree method for efficiency
            "seed": seed, # seed 42 for reproducibility
            "nthread": 4,
        }
        # Keep these as plain ints. A trailing comma here makes the value a
        # one-element TUPLE rather than a number, and xgb.train then fails deep
        # inside the library with an error that does not mention this line.
        self.num_boost_round = 2000      # ceiling only; early stopping finds the real count
        self.early_stopping_rounds = 50

    def fit(self, X, y, X_val=None, y_val=None):
        # create DMatrix objects for training and validation data
        dtrain = xgb.DMatrix(X, label=y)
        # create evaluation list for training and validation data
        evals = [(dtrain, "train")]
        # add validation data to the evaluation list if provided
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "calib"))
        # train the XGBoost model with the specified parameters and data
        self.booster = xgb.train(
            self.params, # parameters for the XGBoost model
            dtrain, # training data in DMatrix format
            num_boost_round=self.num_boost_round, # number of boosting rounds (trees) to train
            evals=evals, # list of evaluation datasets for monitoring training progress
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None, # must be none when no validation set is present
            evals_result=self.evals_result, # mutated in place and stores previous rounds (think learning curve)
            verbose_eval=False, # set to True if you want to see training progress
        )
        return self # exists for method chaining

    # -- the early-stopped tree count ---------------------------------------
    #
    # Early stopping does NOT delete the trees built after the best round; it
    # only records which round was best. The booster still physically contains
    # every tree it built. Two places therefore have to agree on how many of
    # them to actually use:
    #
    #   predict_proba()  passes iteration_range, limiting the trees used for
    #                    this one prediction.
    #   save_model()     slices the booster, physically dropping the surplus
    #                    trees from the saved file.
    #
    # They are equivalent, and tests/test_persist.py asserts that a saved and
    # reloaded model scores identically to the in-memory one. The slicing
    # matters because a saved file carries no `best_iteration` instruction to
    # the prediction call -- anything loading it would use every tree,
    # including the overfitted tail, and quietly score differently from the
    # model whose metrics were published.

    def _best_round(self) -> int | None:
        """The round early stopping settled on, or None if it never ran."""
        # getattr returns the default when best_iteration is undefined, which
        # is the case whenever fit() was called without a validation set.
        best = getattr(self.booster, "best_iteration", None)
        return None if best is None else int(best)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # ensure the model has been fitted before making predictions
        if self.booster is None:
            raise RuntimeError("call fit() first")
        # determine the best iteration for prediction, if available
        best = self._best_round()
        # (0, 0) is XGBoost's "use every tree"; (0, best+1) keeps the first
        # best+1 trees, since best_iteration is a 0-based index.
        rng = (0, best + 1) if best is not None else (0, 0)
        # make predictions using the best iteration range, if available
        p = self.booster.predict(xgb.DMatrix(X), iteration_range=rng)
        # return the predicted probabilities for both classes in 1-D array
        return np.column_stack([1.0 - p, p])

    def save_model(self, path) -> None:
        """
        Write the trained booster to `path` in XGBoost's native JSON format.

        Called by sureshot.persist.Bundle.save(). The booster is sliced to the
        early-stopped tree count first, so the file holds exactly the trees
        that produced the published metrics -- see the comment block above.
        """
        if self.booster is None:
            raise RuntimeError("call fit() first")
        best = self._best_round()
        booster = self.booster if best is None else self.booster[0:best + 1]
        booster.save_model(str(path))

    @property
    def feature_importances_(self) -> np.ndarray | None:
            # return the feature importances as a normalized array, or None if the model is not fitted
            if self.booster is None:
                return None
            # get the feature importances from the booster based on gain
            gain = self.booster.get_score(importance_type="gain")
            # initialize an output array with zeros for all features
            n = self.booster.num_features()
            out = np.zeros(n, dtype=float)
            # get_score only reports features the model actually split on, keyed
            # by DMatrix's auto-generated names "f0", "f1", ... -- so k[1:] is
            # the column index. Features never split on keep their zero.
            for k, v in gain.items():
                out[int(k[1:])] = v
            total = out.sum()
            # normalize the feature importances by the total gain
            return out / total if total > 0 else out

def main():
    # create argument parser for training the model
    #
    # The flag SPELLINGS deliberately match scripts/train_scikit-learn.py
    # (hyphens, not underscores). The two scripts are meant to be run
    # back-to-back with the same arguments and then compared, so a CLI that
    # differs between them makes that comparison needlessly error-prone.
    parser = argparse.ArgumentParser(description="=== Train XGBoost model (native API) ===")
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset. Required.")
    parser.add_argument("--source-type", default=None, choices=["rustsec", "synthetic"],
                        help="filter the corpus; never pool the two")
    parser.add_argument("--drop-size-features", action="store_true",
                        help="ablation: drop raw size features to test the length confound")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (prefer 42).")
    parser.add_argument("--report", type=Path, default=Path("reports/mine.json"),
                        help="where to write the metrics JSON")
    parser.add_argument("--save", type=Path, default=None,
                        help="directory to write a reusable detector bundle to, "
                             "e.g. artifacts/xgboost (omit to train without saving)")
    parser.add_argument("--select-on", default="test", choices=["test", "calib"],
                        help="split used to choose the winning calibrator; "
                             "'test' reproduces the published numbers, 'calib' "
                             "is the leak-free option (see pipeline.run)")
    args = parser.parse_args()

    # load the dataset and split it into training and testing sets
    ds = data.load(args.data, args.source_type, args.drop_size_features)
    data.split(ds, seed=args.seed)

    # initialize the model with the specified seed and scale_pos_weight
    model = MyModel(seed=args.seed, scale_pos_weight=ds.scale_pos_weight())
    pipeline.run(model, ds, label="xgboost-native", report=args.report,
                 save_to=args.save, select_on=args.select_on)

# run main
if __name__ == "__main__":
    main()
