#!/usr/bin/env python3

# imports
import numpy as np
import xgboost as xgb
import argparse
from sureshot import data, pipeline


# building XGBoost model 
class MyModel:
    def __init__(self, seed: int = 42, scale_pos_weight: float = 1.0):
        self.seed = seed
        self.booster: xgb.Booster | None = None
        self.evals_result: dict = {}
        self.params = {
            "objective": "binary:logistic", # using logistic regression objective for binary classification
            "eval_metric": "aucpr", # using area under the precision-recall curve as evaluation metric
            "max_depth": 3, # starting with max depth three for more conservative model
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
        self.num_boost_round = 2000,      # ceiling only; early stopping finds the real count
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
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None, # must be none when no vslaidation set is present
            evals_result=self.evals_result, # mutated in place and stores previous rounds (think learning curve)
            verbose_eval=False, # set to True if you want to see training progress
        )
        return self # exsists for method chaining

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # ensure the model has been fitted before making predictions
        if self.booster is None:
            raise RuntimeError("call fit() first")
        # convert the input data to a DMatrix object for prediction
        best = getattr(self.booster, "best_iteration", None)
        # determine the best iteration for prediction, if available
        rng = (0, best + 1) if best is not None else (0, 0)
        # make predictions using the best iteration range, if available
        p = self.booster.predict(xgb.DMatrix(X), iteration_range=rng)
        # return the predicted probabilities for both classes in 1-D array
        return np.column_stack([1.0 - p, p])

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
            for k, v in gain.items():
                out[int(k[1:])] = v
            total = out.sum()
            # normalize the feature importances by the total gain
            return out / total if total > 0 else out

def main():
    # create argument parser for training the model
    parser = argparse.ArgumentParser(description="=== Train XGBoost model ===")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset. Required.")
    parser.add_argument("--source_type", type=str, required=True, choices=["rustsec", "synthetic"], help="Type of the data source (synthetic or not). Required.")
    parser.add_argument("--drop_size_features", action="store_true", help="ablation: drop raw size features to test the length confound")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (prefer 42).")
    args = parser.parse_args()

    # load the dataset and split it into training and testing sets
    ds = data.load(args.data, args.source_type, args.drop_size_features)
    data.split(ds, seed=args.seed)

    # initialize the model with the specified seed and scale_pos_weight
    model = MyModel(seed=args.seed, scale_pos_weight=ds.scale_pos_weight())
    pipeline.run(model, ds, label="mine", report="reports/mine.json")

# run main
if __name__ == "__main__":
    main()
    