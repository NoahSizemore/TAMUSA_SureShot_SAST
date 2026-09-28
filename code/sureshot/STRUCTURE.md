# Repository structure

The layout below exists for one reason: so that your model and the reference
model are evaluated by **identical** code, and any difference in the numbers is
caused by the model and nothing else.

```
sureshot/
├── pyproject.toml              installable package definition
│
├── src/sureshot/               THE PACKAGE — shared infrastructure
│   ├── __init__.py
│   ├── features.py             Rust source -> 122 numbers, + non-Rust gate
│   ├── data.py                 load, featurise, repo-grouped split
│   ├── metrics.py              ECE, MCE, bootstrap, threshold, risk-coverage
│   ├── calibrate.py            Platt / isotonic / temperature + ScoreModel contract
│   └── pipeline.py             run() — the shared evaluation harness
│
├── scripts/                    ENTRY POINTS — model definitions only
│   ├── train_reference.py      baseline (sklearn XGBClassifier)
│   └── train_mine.py           >>> YOUR FILE <<<
│
├── harvest/                    dataset construction (run once, then leave alone)
│   ├── analyze_advisories.py   measure advisory-db yield
│   ├── rustsec.py              commit-mine RustSec into labelled functions
│   └── synth.py                synthetic fixture generator
│
├── tests/
│   └── test_pipeline.py        30 tests; run before every commit
│
├── data/
│   ├── raw/advisory-db/        cloned RustSec database (gitignored)
│   ├── interim/                fix_commits.json worklist
│   └── processed/              *.parquet corpora
│
├── reports/                    metrics JSON, one per run
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

**Entry points contain only the model.** `scripts/train_reference.py` is about
100 lines and roughly 40 of them are a class. Everything else is imported. Your
file should look the same. If you find yourself copying a split function or an
ECE implementation into your script, stop — that is exactly the drift this
layout prevents.

**`legacy/` is kept, not deleted.** The original `model/train.py` is the version
that produced the numbers in the README. It still runs standalone. Keep it for
comparison, as you asked.

## Setup

```bash
cd sureshot
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev,plot]"
pytest -q                                              # expect 30 passed
```

The `-e` matters. Editable install means edits to `src/sureshot/` take effect
immediately with no reinstall.

## The contract your model must satisfy

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

Calibrators fit on **scores**, not on your estimator, so you do not need
`predict`, `classes_`, `get_params`, or any other sklearn boilerplate. This was
a deliberate change: the sklearn-native approach (`CalibratedClassifierCV` +
`FrozenEstimator`) demands the full estimator protocol and fails deep inside
`cross_val_predict` with an unhelpful error when it is not satisfied.

## Writing your model

```python
from sureshot import data, pipeline

ds = data.load("data/processed/rustsec_v0.1.parquet", source_type="rustsec")
data.split(ds, seed=42)

model = MyModel(scale_pos_weight=ds.scale_pos_weight())
pipeline.run(model, ds, label="mine", report="reports/mine.json")
```

Then compare:

```bash
python scripts/train_reference.py --data data/processed/rustsec_v0.1.parquet \
    --source-type rustsec --report reports/reference.json
python scripts/train_mine.py --data data/processed/rustsec_v0.1.parquet \
    --source-type rustsec --report reports/mine.json

python -c "from sureshot.pipeline import compare; \
    compare('reports/reference.json','reports/mine.json')"
```

**Use the same `--seed` for both.** Different seeds mean different repo splits,
and the comparison becomes noise.

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

## Verified baseline

`rustsec_v0.1`, seed 42, repo-grouped 4223/2047/1892 split:

| | AUPRC | AUROC | Brier | ECE-q | MCE |
|---|---|---|---|---|---|
| Uncalibrated | 0.0733 | 0.6206 | 0.1889 | 0.3539 | 0.7155 |
| Platt | 0.0733 | 0.6206 | 0.0441 | 0.0200 | 0.0929 |
| Isotonic | 0.0644 | 0.6045 | 0.0458 | 0.0185 | 0.6667 |
| Temperature | 0.0733 | 0.6206 | 0.1851 | 0.3034 | 0.8004 |

Best ECE 0.0215, 95% CI [0.0149, 0.0290]. Reduction vs uncalibrated: 0.3354.
Threshold 0.09 → precision 0.088, recall 0.287, F1 0.135 (base rate 0.046).

`scripts/train_mine.py` as shipped reproduces these to within 0.0000 on every
metric, which confirms the native-API scaffold is equivalent to the sklearn
reference before you change anything. Verify that yourself first — it is your
proof that the harness is sound — then start modifying.
