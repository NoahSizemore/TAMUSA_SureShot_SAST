# Repository structure

The layout below exists for one reason: so that the model and the reference
model are evaluated by **identical** code, and any difference in the numbers is
caused by the model and nothing else.

```
sureshot/
├── pyproject.toml              installable package definition
├── run_model.sh                >>> START HERE <<< runs everything, logs to .out
│
├── src/sureshot/               THE PACKAGE — shared infrastructure
│   ├── __init__.py
│   ├── features.py             Rust source -> 142 numbers, function splitter,
│   │                           + non-Rust gate
│   ├── data.py                 load, featurise, repo-grouped split
│   ├── metrics.py              ECE, MCE, bootstrap, threshold, risk-coverage
│   ├── calibrate.py            Platt / isotonic / temperature, ScoreModel
│   │                           contract, out-of-fold calibrator selection
│   ├── persist.py              save/load a trained detector (Bundle)
│   ├── scan.py                 inference: score unseen Rust
│   └── pipeline.py             run() — the shared evaluation harness
│
├── scripts/                    ENTRY POINTS — model definitions only
│   ├── train_scikit-learn.py   baseline (sklearn XGBClassifier)
│   ├── train_XGBoost.py        (native xgb.train API)
│   └── scan.py                 the user-facing scanner CLI
│
├── harvest/                    dataset construction (run once, then leave alone)
│   ├── analyze_advisories.py   measure advisory-db yield
│   ├── rustsec.py              commit-mine RustSec into labelled functions
│   └── synth.py                synthetic fixture generator
│
├── tests/                      95 tests; run before every commit
│   ├── conftest.py             toy dataset + script-importing helpers
│   ├── test_pipeline.py        harness invariants (splits, metrics, calibration)
│   ├── test_models.py          the shipped model classes, both of them
│   └── test_persist.py         save/load identity, selection, the scanner
│
├── data/
│   ├── raw/advisory-db/        cloned RustSec database (gitignored)
│   ├── interim/                fix_commits.json worklist
│   └── processed/              *.parquet corpora
│
├── demo/                       a Rust file for run_model.sh to scan; NOT corpus
│   └── vulnerable_buffer.rs
│
├── reports/                    metrics JSON, one per run
│
├── artifacts/                  saved detector bundles + captured stdout
│                               (committed: ~45 KB each, so the scanner runs
│                               without retraining)
│
└── legacy/                     ORIGINAL monolithic version, kept for reference
    ├── extract/features.py
    └── model/train.py
```

## Why this shape

**`src/` layout, not a flat package.** With a flat layout, `import sureshot`
resolves to whatever directory you happen to be standing in, so a script run
from `scripts/` and the same script run from the repo root can import different
code. Putting the package under `src/` and installing it makes the import path
unambiguous from any working directory.

**Entry points contain only the model.** `scripts/train_scikit-learn.py` is
about 100 lines and roughly 40 of them are a class. Everything else is
imported. The file should look the same. If you find yourself copying a split
function or an ECE implementation into the script, stop, that is exactly the
drift this layout prevents.

**Training measures; `scan.py` runs.** `pipeline.run()` answers "how good is
this detector?" and `scripts/scan.py` answers "what is wrong with this file?".
They share the feature extractor and the saved bundle, so the thing being
scanned is scored by the same code that was benchmarked.

**`legacy/` is kept, not deleted.** The original `model/train.py` is the version
that produced the numbers in the README. It still runs standalone. Keep it for
comparison, as you asked.

## Setup

```bash
cd sureshot
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,plot]"
pytest -q                                              # expect 95 passed
```

The `-e` matters. Editable install means edits to `src/sureshot/` take effect
immediately with no reinstall.

## Running everything: `./run_model.sh`

Once the venv exists, this is the only command most people need:

```bash
./run_model.sh            # from anywhere; the script locates itself
./run_model.sh --help     # options
```

It runs six steps and saves each step's stdout to a `.out` file **next to the
artifact that step produced**, so the log that made a model always sits beside
that model:

| Step | Produces |
|---|---|
| 1. test suite | `artifacts/tests.out` |
| 2. baseline, sklearn API | `artifacts/reference/` + `train.out` |
| 3. native xgb API | `artifacts/xgboost/` + `train.out` |
| 4. ablation, size features dropped | `artifacts/xgboost-nosize/` + `train.out` |
| 5. side-by-side comparison | `artifacts/compare.out` |
| 6. demo scan on `demo/vulnerable_buffer.rs` | `artifacts/scan.out` |

Steps 2–4 also write metrics JSON to `reports/`.

Useful flags: `--seed N`, `--data PATH`, `--source-type rustsec|synthetic`,
`--select-on test|calib`, `--scan PATH`, `--skip-tests`.

Three details in that script are deliberate and worth knowing before editing it:

- **The tests run first and abort the run on failure.** They take ~3 seconds and
  they check the invariants that otherwise produce results which look fine and
  are wrong — repo leakage across splits, a calibrator measured on the data it
  was fitted on. Training on a broken harness costs a minute and yields numbers
  you cannot trust.
- **It calls `.venv/Scripts/python.exe` directly instead of `activate`.**
  Activation is shell- and platform-specific and sourcing it from a
  non-interactive script is a classic "works in my terminal, fails in the
  runner". The script probes for the Windows path, then the POSIX one, then
  falls back to `python`.
- **`scan.py` exiting 1 is not a failure.** Exit 1 means "something was
  flagged", which is a normal scan outcome, so only exit ≥ 2 is treated as an
  error. This is why the step helper recovers the real status from
  `PIPESTATUS[0]` rather than letting `pipefail` decide.

`demo/vulnerable_buffer.rs` exists only to give step 6 something to scan. It is
not part of the corpus and is not used in training. The directory is called
`demo/` rather than `examples/` on purpose: `scan.iter_rust_files()` skips
directories named `examples`, `tests`, `benches` and `fuzz`, because the model
was trained only on production code.

## The contract the model must satisfy

Two methods. That is the entire interface.

```python
class MyModel:
    def fit(self, X, y, X_val=None, y_val=None) -> "MyModel": ...
    def predict_proba(self, X) -> np.ndarray:  # shape (n, 2), col 1 = P(vulnerable)
        ...
```

Optional: a `feature_importances_` property of length `len(ds.feature_names)`.
Supply it and the pipeline prints the importance table and runs the
size-confound check automatically.

To be SAVEABLE, a model adds two more things:

```python
class MyModel:
    model_kind = "xgboost-booster"        # names the on-disk format

    def save_model(self, path) -> None:   # write the estimator to one file
        ...
```

Loading is deliberately not a method, because at load time there is no model
object yet. `persist.py` keeps a registry mapping `model_kind` to a loader
function, and ships one for `"xgboost-booster"`. A new model format calls
`persist.register_loader()` and nothing else in the package changes.

Calibrators fit on **scores**, not on the estimator, so you do not need
`predict`, `classes_`, `get_params`, or any other sklearn boilerplate. This was
a deliberate change: the sklearn-native approach (`CalibratedClassifierCV` +
`FrozenEstimator`) demands the full estimator protocol and fails deep inside
`cross_val_predict` with an unhelpful error when it is not satisfied.

## Writing the model

```python
from sureshot import data, pipeline

ds = data.load("data/processed/rustsec_v0.1.parquet", source_type="rustsec")
data.split(ds, seed=42)

model = MyModel(scale_pos_weight=ds.scale_pos_weight())
pipeline.run(model, ds, label="mine", report="reports/mine.json",
             save_to="artifacts/xgboost")
```

Then compare:

```bash
python scripts/train_scikit-learn.py --data data/processed/rustsec_v0.1.parquet \
    --source-type rustsec --report reports/reference.json \
    --save artifacts/reference
python scripts/train_XGBoost.py --data data/processed/rustsec_v0.1.parquet \
    --source-type rustsec --report reports/mine.json \
    --save artifacts/xgboost

python -c "from sureshot.pipeline import compare; \
    compare('reports/reference.json','reports/mine.json')"
```

**Use the same `--seed` for both.** Different seeds mean different repo splits,
and the comparison becomes noise.

Both scripts take the same flags, deliberately, so they can be run back to back
without re-reading the CLI.

## Using the detector

`--save` writes a bundle: the booster, the calibrator that won, the threshold,
and the feature names it was trained on. All four are needed, and the reason is
worth internalising — a bare booster emits raw scores whose ECE is 0.40, which
are not probabilities in any useful sense, and a scanner defaulting to a 0.5
threshold on a task whose threshold is 0.10 reports almost nothing.

```bash
python scripts/scan.py --model artifacts/xgboost src/lib.rs
python scripts/scan.py --model artifacts/xgboost some/crate/src      # walks it
cat lib.rs | python scripts/scan.py --model artifacts/xgboost -      # stdin
python scripts/scan.py --model artifacts/xgboost --json src/         # for CI
```

Exit codes are CI-shaped: `0` nothing flagged, `1` something flagged, `2` usage
or load error. `--abstain-margin 0.05` makes the scanner answer UNSURE near the
threshold instead of guessing, which is the risk-coverage result from
`metrics.py` turned into a feature.

Bundles are versioned and validated on load. A bundle whose feature names no
longer match `features.py` is refused rather than silently fed mismatched
columns — add a token rule and you must retrain, because the old booster's
columns no longer mean what it learned them to mean.

## Rules that are not style preferences

1. **Split by repository, never by function.** `data.split()` asserts this.
   Leave the assertion in. Random function-level splits put near-duplicate
   helpers on both sides of the boundary and inflate F1 dramatically.
2. **The calibration split fits the calibrator and picks the threshold. The
   test split measures.** Never merge them.
3. **Never pool `rustsec` and `synthetic` metrics.** Pass `--source-type`.
   Synthetic scores 1.000 on everything and will destroy any pooled number.
4. **Report precision and recall separately, not only F1.** For a SAST tool a
   false alarm and a missed vulnerability have very different costs.
5. **Quote the ECE bootstrap CI, not the point estimate.** ECE is biased
   downward on small test sets and moves with bin count.
6. **Never pick a calibrator by its ECE on the split it was fitted on.** That
   comparison is in-sample, and the most flexible candidate wins by
   construction: measured that way isotonic regression scores an ECE of exactly
   0.0000 here. `calibrate.select_calibrator()` cross-validates inside the
   calibration split instead.
7. **Changing `features.py` invalidates every saved bundle.** The feature count
   is pinned in `tests/test_pipeline.py` so the change has to be deliberate.
   Retrain and re-save; do not just bump the constant.

## Verified baseline

`rustsec_v0.1`, seed 42, repo-grouped 4223/2047/1892 split, **142 features**,
xgboost 3.4.1 / scikit-learn 1.9.1:

| | AUPRC | AUROC | Brier | ECE-q | MCE |
|---|---|---|---|---|---|
| Uncalibrated | 0.0705 | 0.6338 | 0.2066 | 0.3955 | 0.6392 |
| Platt | 0.0705 | 0.6338 | 0.0441 | 0.0226 | 0.0923 |
| Isotonic | 0.0632 | 0.6129 | 0.0471 | 0.0211 | 0.4524 |
| Temperature | 0.0705 | 0.6338 | 0.1925 | 0.3220 | 0.8429 |

Best ECE 0.0242, 95% CI [0.0177, 0.0317]. Reduction vs uncalibrated: 0.3744.
Threshold 0.10 → precision 0.078, recall 0.230, F1 0.116 (base rate 0.046).

Early stopping settles at **14 trees**, far short of the 2000 ceiling. The
validation AUPRC stops improving almost immediately, which is the clearest
single indicator of how weak the signal in this corpus is.

`scripts/train_XGBoost.py` reproduces the reference to within 0.0000 on every
metric, which confirms the native-API model is equivalent to the sklearn
reference before anything is tuned. Verify that yourself first — it is the
proof that the harness is sound — then start modifying.

### Why these differ from the figures in README.md

README.md quotes AUPRC 0.0733 / AUROC 0.6206, measured when the extractor
emitted **122** features. Commit `4a1fee0` added 10 token rules, taking it to
142, and nothing was re-run. The table above is the current state.

The uncalibrated AUROC moved (0.6206 → 0.6338), and no change to a calibrator
can affect an uncalibrated metric — which is how you can tell the difference
comes from the feature set and library versions, not from the calibration
refactor. The repo-grouped split is byte-identical across both runs.
