# What changed, and why

A walkthrough of the step from "we can benchmark a model" to "we can ship a
scanner". Written to be read top to bottom once, then used as a reference.

Every number quoted here was produced by running the code in this repo on
`data/processed/rustsec_v0.1.parquet` with `--seed 42`.

---

## 1. The short version

| | Before | After |
|---|---|---|
| `scripts/train_XGBoost.py` | crashed on every run | runs, and matches the reference exactly |
| Trained model after training | discarded | saved to `artifacts/` and reusable |
| Scanning unseen Rust | not possible | `scripts/scan.py` |
| Tests | 30, none touching the shipped models | 95, both models covered |
| Feature count | code said 142, tests and docs said 122 | 142 everywhere |

Four new files, three bugs fixed, two bugs found that were never reported
before.

---

## 2. Bugs that were already there

### 2.1 The one-character bug that made the script unrunnable

```python
self.num_boost_round = 2000,      # <- trailing comma
```

That comma makes the value the tuple `(2000,)`, not the integer `2000`.
`xgb.train` then fails somewhere deep inside XGBoost with a message that never
mentions this line. The script could not complete a single run.

You fixed this yourself while this work was in progress. It is documented here
because the interesting part is not the comma, it is **why nothing caught it**:
the test suite had 30 tests and not one of them ever constructed `MyModel`. All
of them exercised a `_DummyModel` defined inside the test file. A test suite
that only tests a stand-in proves the harness works and says nothing about the
code that ships.

`tests/test_models.py` now asserts the types of the hyperparameters directly:

```python
assert isinstance(m.num_boost_round, int), \
    "num_boost_round must be an int -- a trailing comma makes it a tuple"
```

That assert looks almost too obvious to write. It is the cheapest possible
guard against an entire family of typo, and it turns a confusing library-
internal failure into a one-line explanation.

### 2.2 The reference model could not be fitted without a validation set

Found by the new tests, not by reading the code.

```python
# train_scikit-learn.py, before
self.clf = xgb.XGBClassifier(..., early_stopping_rounds=50)

def fit(self, X, y, X_val=None, y_val=None):
    evals = [(X_val, y_val)] if X_val is not None else None
    self.clf.fit(X, y, eval_set=evals, verbose=False)
```

`early_stopping_rounds` is set on the **constructor**, so the early-stopping
callback is installed whether or not an eval set is later supplied. Passing
`eval_set=None` is not enough to switch it off, and XGBoost raises:

```
ValueError: Must have at least 1 validation dataset for early stopping.
```

`pipeline.run()` always passes a validation set, so this never fired during
normal training — it was latent. It still matters: the native model in
`train_XGBoost.py` already handled this case correctly, so the two models
behaved differently, and a model that cannot be fitted without a validation set
cannot be used in a cross-validation loop or a quick sanity check.

The fix turns the parameter off for that call rather than just omitting the
data:

```python
if X_val is None or y_val is None:
    self.clf.set_params(early_stopping_rounds=None)
    self.clf.fit(X, y, verbose=False)
```

### 2.3 The feature count had silently changed

`tests/test_pipeline.py` asserted `len(f) == 122`. The extractor emits 142.

Commit `4a1fee0` ("Update features.py") added 10 token rules. Each rule
contributes two features — a raw count and a per-line ratio — so the count went
from 122 to 142, and the test, `STRUCTURE.md` and `README.md` were all left
behind.

This was not a cosmetic slip. It means **every metric in `README.md` describes a
different model than the one in the repo** (see §7).

The fix splits the assertion in two, because the two things being checked are
different:

```python
def test_feature_vector_has_stable_shape():
    # the invariant: shape must not vary with input
    assert len(f) == len(FEATURE_NAMES)

def test_feature_count_matches_its_parts():
    # the pinned value, with the arithmetic that justifies it
    assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT
    assert len(FEATURE_NAMES) == 2 * len(TOKEN_RULES) + len(AST_NODE_TYPES) + 16
```

Pinning the number is now load-bearing rather than pedantic: saved model
bundles record the feature names they were trained on, so changing the feature
set invalidates every bundle produced before the change.

---

## 3. New: saving a trained detector (`src/sureshot/persist.py`)

### What was missing

`pipeline.run()` trained a model, measured it, printed a table, and returned a
dict of metrics. The trained model itself went out of scope and was garbage
collected. The project could say how good a detector was and could never run
one.

### The key idea: a detector is three things, not one

The instinct is "save the model" = "save the booster". That is wrong, and each
missing piece breaks the tool in a different way:

1. **The booster** turns a feature vector into a raw score. On its own those
   scores have an ECE of **0.3955** — they are not probabilities in any useful
   sense, they are just ordered numbers.
2. **The calibrator** maps raw score to honest probability, dropping ECE to
   **0.0211**. Which calibrator wins is *decided at training time*, so it has to
   be recorded rather than assumed.
3. **The threshold** turns a probability into a decision. It is **0.10** here,
   not 0.5, because the base rate is 0.046. A scanner that defaults to 0.5
   reports essentially nothing and looks like it is working.

A bundle also stores the **feature names**, which are not part of the detector
but are part of its input contract. Matching column *counts* is not the same as
matching column *meanings*, so names are compared, not just lengths.

### On-disk layout

```
artifacts/xgboost/
    manifest.json    calibrator params, threshold, feature names, metadata
    model.json       the booster, in XGBoost's own format
```

A directory rather than one file, because the two payloads have genuinely
different natural formats. Total size: about 45 KB, small enough to commit, so
the scanner runs without anyone retraining.

### Why not pickle

`pickle` would do all of this in three lines. It is the wrong tool here:

- **`pickle.load` executes arbitrary code from the file.** For a security tool,
  shipping a model format whose load path is remote code execution is
  indefensible.
- **It binds the file to the library version that wrote it.** A pickled
  scikit-learn estimator breaks, sometimes silently, on upgrade.
- **It is opaque.** `manifest.json` can be opened, diffed in a pull request,
  and read by a human.

### The trick that makes JSON possible: fit with a library, predict with arithmetic

This is the change inside `calibrate.py`, and it is the one most worth
understanding.

Each calibrator still *fits* with scikit-learn. But immediately after fitting it
copies the handful of numbers that define the learned mapping into plain
attributes, and `transform()` is implemented on those numbers directly:

| Calibrator | Fitted state | `transform()` is |
|---|---|---|
| Platt | `a_`, `b_` (two floats) | `sigmoid(a * logit(s) + b)` |
| Isotonic | knot arrays | `np.interp(s, knots_x, knots_y)` |
| Temperature | `T` (one float) | `sigmoid(logit(s) / T)` |

This is **not an approximation** of what scikit-learn does, it is what
scikit-learn does. Binary `LogisticRegression.predict_proba` *is*
`sigmoid(X @ coef_ + intercept_)`. `IsotonicRegression.predict` with
`out_of_bounds="clip"` *is* linear interpolation between fitted knots, clamped
at the ends, which is exactly `np.interp`. `tests/test_persist.py` asserts both
equivalences against scikit-learn to `1e-12`, so the claim is checked rather
than believed.

Two consequences, and the second is the important one:

1. A fitted calibrator is a few floats, so `to_dict()` / `from_dict()` is JSON.
2. **There is only one prediction code path.** A calibrator that was just
   fitted and one loaded from disk run the same lines of arithmetic. If
   prediction went through sklearn when fresh and through restored attributes
   when loaded, the gap between them would be untested and would eventually
   drift — and a saved model quietly disagreeing with its own published metrics
   is a bug that produces no error message, just wrong numbers forever.

### The tree-slicing detail

Early stopping does **not** delete the trees built after the best round. It
records which round was best; the booster still physically holds all of them.
Here, early stopping settles at **14 trees** out of a 2000 ceiling.

Two places have to agree on how many trees to use:

- `predict_proba()` passes `iteration_range=(0, best+1)`.
- `save_model()` **slices** the booster: `self.booster[0:best + 1]`.

The slice is what keeps a reloaded model honest. A saved file carries no
`best_iteration` instruction to whoever loads it, so anything reading it would
use all 2000 trees — including the overfitted tail that early stopping
explicitly rejected — and score differently from the published model.

Verified on the real corpus: a saved-and-reloaded detector scores
**bit-identically** (`max_abs_diff = 0.000e+00`) to the in-memory one, for all
three calibrators, with 14 trees loaded.

### Extending it

`model_kind` is a string naming the on-disk format; `persist.MODEL_LOADERS`
maps it to a function that rebuilds a predictor. Loading is deliberately *not*
a method on the model, because at load time there is no model object yet — that
is the whole problem. Storing a tag and looking the constructor up in a
registry is the standard solution to "I need to build an object before I know
its class".

---

## 4. New: scanning unseen code (`src/sureshot/scan.py`, `scripts/scan.py`)

The pipeline, in order:

```
source bytes
  -> is_probably_rust()      reject non-Rust BEFORE the model
  -> iter_functions()        cut the file into functions
  -> feature vector          in the order the BUNDLE declares
  -> booster                 raw score
  -> calibrator              honest probability
  -> threshold               flag / do not flag
  -> margin                  optionally abstain
```

The order matters at both ends.

**The gate is first** because a model handed Python will happily emit a
confident number about it. Garbage in, confident garbage out. This is Scenario 2
in the requirements, and the behaviour is to return a *rejection*, not a score:

```
-- notrust.rs
   SKIPPED: not Rust: no Rust keywords found
```

**Abstention is last** because it is where this detector is genuinely strong.
Raw discrimination is weak (AUROC 0.63), but the calibrated probabilities are
trustworthy (ECE 0.02), so "I do not know" is a conclusion this model can
actually support. `--abstain-margin 0.05` turns the risk-coverage analysis in
`metrics.py` into a feature.

### Function splitting

Training data arrived pre-cut by the harvester. At scan time the user hands over
a file, so `features.iter_functions()` does the same cutting with a tree-sitter
query. A regex cannot do this: finding the end of a function means balancing
braces, and braces appear inside strings, chars and comments.

It differs from the harvester's version in one deliberate way — it does **not**
de-duplicate by name. Two methods called `new` in different `impl` blocks are
two separate findings for a scanner, where the harvester only wanted one row
per name.

`min_lines=3` mirrors the training filter. Functions shorter than that were
dropped from the corpus, so the model never learned what they look like and
scoring them would be extrapolation.

### Features are built from the bundle's list, not the extractor's

```python
rows.append([feats[n] for n in bundle.feature_names])
```

Not `for n in FEATURE_NAMES`. The two differ whenever a model was trained with
`--drop-size-features`, which yields 136 columns instead of 142. Reading the
order from the bundle means ablation models load and run with no special
casing, and a genuine mismatch is caught by name rather than slipping through
because the counts happened to agree.

### The honesty footer

Every report ends with:

```
Calibration note: at this threshold the detector measured precision 0.078 and
recall 0.230 on held-out repositories.
Expect roughly 8 in 100 flags to be real, and expect it to miss about 77% of
real issues.
Treat flags as 'review this first', not as proof of a bug.
```

This is not self-deprecation, it is the product. A tool that reports 8%
precision *and says so* is usable for triage. The same tool presenting flags as
findings would be worse than nothing, because a reviewer would calibrate their
trust wrongly and stop reading.

Exit codes are CI-shaped: `0` clean, `1` something flagged, `2` usage error.

---

## 5. The subtle one: how the calibrator gets chosen

Worth reading even if you skip everything else, because the first fix was wrong.

`pipeline.run()` picked the winning calibrator by its ECE **on the test split**:

```python
best = min(cal_names, key=lambda k: results[k]["ece_quantile"])
```

That is a model-selection decision made on test data. Rule 2 in `STRUCTURE.md`
says the calibration split fits and the test split measures. All three
calibrators are still *fit* on the calibration split, so this is a mild leak
rather than a broken experiment — but the reported ECE for the winner is
optimistic by roughly the spread between candidates, because we got to look at
the answer before picking.

**The obvious fix does not work.** Comparing the three by their ECE on the
calibration split gives:

```
platt=0.0208, isotonic=0.0000, temperature=0.3072
```

Isotonic scores **exactly zero**. Not because it is well calibrated — because it
was *fitted* on that split and is flexible enough to reproduce its own training
data perfectly. Any in-sample comparison picks isotonic every time, regardless
of how it generalises. This is overfitting in its purest observable form, and
it is a genuinely easy trap to fall into: the code looks right, the rule
"select on calibration, measure on test" sounds like it is being followed, and
the number it produces is beautiful and meaningless.

`calibrate.select_calibrator()` does it properly: K-fold cross-validation
**inside** the calibration split, so every scored prediction was made by a
calibrator that had not seen that row. Folds are grouped by repository, for the
same reason the main split is — Rust codebases are full of near-duplicate
helpers, and a fold boundary cutting through a repository re-creates the same
flattery at smaller scale. The winner is then refitted on the whole calibration
split: CV chooses the model, the full fit maximises the data it learns from.

Honest numbers:

```
calibrator chosen by out-of-fold ECE within the calibration split:
platt=0.0234, isotonic=0.0212, temperature=0.3062
```

Isotonic still wins, now legitimately, by a real margin of 0.0022.

Available as `--select-on calib`. **The default is still `test`**, so every
number already committed stays reproducible. Changing the default changes every
published figure in the repo, which is a decision for the team, not something a
library function should do quietly.

---

## 6. Tests: 30 to 95

| File | Covers |
|---|---|
| `test_pipeline.py` | harness invariants — split leakage, metrics, calibration (existing, plus the feature-count fix) |
| `test_models.py` | the two shipped model classes, parameterised so every contract test runs against both |
| `test_persist.py` | save/load identity, calibrator serialisation, selection, function splitting, the scanner |
| `conftest.py` | toy dataset and the script-import helper |

Two things are worth calling out.

**Importing the training scripts needs a helper.** `scripts/` is deliberately
not a package, and `train_scikit-learn.py` contains a hyphen, which is not a
legal Python identifier — `import train_scikit-learn` is a syntax error no
matter what the path looks like. `conftest.py` uses
`importlib.util.spec_from_file_location` to load a module from an explicit path
under a chosen name. This matters because it means the model classes are tested
**as they ship**, rather than being copied into the test file where they would
drift.

**The central assertion is exact equality, not a tolerance:**

```python
def test_saved_and_loaded_detector_score_identically(trained_bundle):
    assert np.array_equal(original.score(X), loaded.score(X))
```

If a saved detector scores even slightly differently from the one that was
measured, every published metric becomes a statement about a model that no
longer exists. There is no tolerance at which that is acceptable, so none is
allowed.

---

## 7. Results

### The two models are equivalent

```
metric                  reference-xgb xgboost-native     delta
unca.auprc                     0.0705         0.0705   +0.0000
unca.auroc                     0.6338         0.6338   +0.0000
unca.brier                     0.2066         0.2066   +0.0000
unca.ece_quantile              0.3955         0.3955   +0.0000
op.precision                   0.0778         0.0778   +0.0000
op.recall                      0.2299         0.2299   +0.0000
op.f1                          0.1163         0.1163   +0.0000
```

With matched hyperparameters, the native `xgb.train` API and the scikit-learn
wrapper agree to four decimal places on every metric. This is the equivalence
check `STRUCTURE.md` asks for, and it is the proof that the harness is sound
before anything gets tuned.

### The numbers moved, and not because of this work

| | README (122 features) | now (142 features) | delta |
|---|---|---|---|
| uncal. AUPRC | 0.0733 | 0.0705 | −0.0028 |
| uncal. AUROC | 0.6206 | 0.6338 | +0.0132 |
| uncal. ECE-q | 0.3539 | 0.3955 | +0.0415 |
| precision | 0.0880 | 0.0778 | −0.0102 |
| recall | 0.2874 | 0.2299 | −0.0575 |
| F1 | 0.1348 | 0.1163 | −0.0185 |

The cause is the 10 token rules added in commit `4a1fee0`, plus library
versions (xgboost 3.4.1, scikit-learn 1.9.1).

How you can tell it is not the calibration refactor: **the uncalibrated AUROC
moved**, and no change to a calibrator can affect an uncalibrated metric. The
repo-grouped split is byte-identical across both runs (4223/2047/1892), so it is
not the split either.

`STRUCTURE.md` has been updated. `README.md` was left alone as you asked, and
its metrics table and its "122-feature extractor" line are both now stale.

### The size-confound ablation still holds

| | AUPRC | AUROC | best ECE | precision | recall | F1 |
|---|---|---|---|---|---|---|
| full, 142 features | 0.0705 | 0.6338 | 0.0211 | 0.078 | 0.230 | 0.116 |
| size features dropped, 136 | 0.0684 | 0.6115 | 0.0163 | 0.085 | 0.310 | 0.133 |

Dropping every raw size feature costs only **0.022 AUROC**, and size features
account for **15.9%** of total gain. The README's conclusion — the model is not
simply learning "long function = vulnerable" — survives the move to 142
features. (It previously measured 0.621 → 0.600 and 13.8%.)

Note that the ablation scores *better* at the operating point: F1 0.133 vs
0.116, and recall 0.310 vs 0.230. On a test set with 87 positives that is well
inside noise, but it is not nothing, and it is worth re-checking across seeds.

---

## 8. What I would look at next

1. **Early stopping settles at 14 trees.** Out of a 2000 ceiling. The
   validation AUPRC stops improving almost immediately, which says the signal
   is weak — not that the ceiling is wrong. No amount of hyperparameter tuning
   fixes a 14-tree plateau; the features or the labels are the constraint.
   This is the single most informative number in this document.

2. **Isotonic wins on ECE but damages ranking.** It is selected (0.0211 vs
   Platt's 0.0226) and it costs AUPRC (0.0632 vs 0.0705) and MCE (0.4524 vs
   0.0923). The practical effect is visible in a real scan — six of seven
   functions came back at *exactly* `p=0.044`, because a step function with few
   knots has few distinct outputs. For triage, ranking is the product, so
   **Platt is arguably the better choice for the shipped bundle despite the
   slightly worse ECE.** `calibrate.py` already warns about exactly this
   pattern ("an MCE far worse than Platt's while ECE looks comparable"); the
   corpus now shows it.

3. **Hand-audit ~50 positives.** `README.md` lists this as a known limitation
   and it is now the binding one. A function is labelled vulnerable because it
   was touched by a security fix, not because the flaw was verified in it. At
   AUROC 0.63 it is impossible to tell how much of the ceiling is label noise.

4. **Repeat across seeds.** Every number here is one repo-grouped split with 87
   test positives. The ablation apparently *improving* F1 is the kind of result
   that usually turns out to be noise, and `data.split()` takes a `seed`
   already.

5. **Decide on `--select-on`.** `calib` is the defensible choice for anything
   written up. It needs a deliberate re-run of every reported figure.
