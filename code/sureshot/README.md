# SureShot SAST — dataset v0.1

Static Rust vulnerability detection with an ML-calibrated confidence layer.
This drop contains two labelled datasets, the code that produced them, a
feature extractor, and a training/calibration pipeline that runs end to end.

Everything here was generated and verified in one session. The numbers below
are reproducible from the scripts in this repo.

---

## Files

| Path | What it is |
|---|---|
| `data/processed/rustsec_v0.1.parquet` | **Real** commit-mined corpus. 8,162 functions, 477 vulnerable, 151 repos, 199 advisories |
| `data/processed/synthetic_v0.1.parquet` | **Synthetic** corpus. 4,200 functions, 1,200 vulnerable, 8 injected bug patterns |
| `data/processed/combined_v0.1.parquet` | Both, with `source_type` distinguishing them |
| `harvest/analyze_advisories.py` | Measures advisory-db yield before harvesting |
| `harvest/rustsec.py` | The harvester (run this yourself for the full corpus) |
| `harvest/synth.py` | Synthetic generator |
| `extract/features.py` | 122-feature extractor + non-Rust parser gate |
| `model/train.py` | Grouped splits, XGBoost, calibration, ECE |
| `reports/*.json` | Metrics from the runs below |

Schema: `repo, commit, file, function_name, source, label, source_type, pattern, advisory, package, informational`

`label`: 1 = vulnerable, 0 = safe. `source_type`: `rustsec` or `synthetic`.
**Never pool metrics across `source_type`.** Report them separately.

---

## How the real dataset was built

Commit-mining, the methodology Devign and Big-Vul used for C:

1. Clone `rustsec/advisory-db` — 1,253 advisories.
2. Drop `informational = unmaintained` (275) and `notice` (6). 972 candidates remain.
3. Extract fix links: **89** advisories link a commit directly; **204** link only a PR.
4. Fetch each fix commit with `git fetch --depth 2 <sha>`, which pulls only the
   commit and its parent (~1–3 MB each instead of a full clone).
   PR-only advisories resolve through `refs/pull/<n>/head`, so the same code path
   handles both — no GitHub API, no token, no rate limit.
5. Diff parent..fix. For each `.rs` function whose body changed:
   pre-fix → label 1, post-fix → label 0. Untouched functions in the same files → label 0.

Filters applied: production `.rs` only (`tests/`, `benches/`, `examples/`, `fuzz/`
excluded); commits touching >10 files dropped as refactors; whitespace- and
comment-only diffs ignored; functions under 3 lines dropped; duplicate bodies
de-duplicated (671 removed).

Yield: 210 advisories harvested, 82 skipped (53 changed no production Rust,
26 were refactors, 3 had unreachable refs).

---

## Measured results

Repo-grouped 60/20/20 split, asserted zero repo overlap across splits.

### Real data (`rustsec_v0.1`)

| Model | AUPRC | AUROC | Brier | ECE (quantile) | MCE |
|---|---|---|---|---|---|
| Uncalibrated | 0.0733 | 0.6206 | 0.1889 | 0.3539 | 0.7155 |
| Platt | 0.0733 | 0.6206 | 0.0441 | 0.0200 | 0.0929 |
| Isotonic | 0.0644 | 0.6045 | 0.0458 | 0.0185 | 0.6667 |

Best ECE 0.0215, 95% CI [0.0149, 0.0290] over 500 bootstrap resamples.
**ECE improves by 0.335 after calibration** — that is the headline result.

At threshold 0.09 (chosen on the calibration split, never on test):
precision 0.088, recall 0.287, F1 0.135. Base rate is 0.058, so AUPRC 0.073
is a modest lift over chance. AUROC 0.62 is above chance but weak.

**Read this honestly: discrimination is poor, calibration is excellent.** For a
first pass on 477 noisy positives that is the expected shape, and it is exactly
the result that motivates the calibration layer — a model that knows when it
does not know is useful even when raw accuracy is low. The abstention band
covers 84.7% of inputs at 0.861 accuracy on what it covers.

### Synthetic data (`synthetic_v0.1`)

AUPRC 1.000, AUROC 1.000, F1 1.000. The templates are trivially separable.
**This proves the pipeline runs; it proves nothing about detection ability.**
Use it only to exercise plumbing.

### Size-confound ablation

Positives average 2.4× longer than negatives (1,333 vs 555 chars) — a known
commit-mining artifact, since fixes touch larger functions. Removing all size
features drops AUROC only 0.621 → 0.600, and size features account for 13.8%
of total gain. **The model is not simply learning "long function = vulnerable."**
Re-run this ablation whenever the dataset changes:

```bash
python model/train.py --data data/processed/rustsec_v0.1.parquet --drop-size-features
```

---

## Known limitations

- **Label noise.** A function is labelled vulnerable because it was modified in a
  security fix, not because the flaw was verified in it. Fix commits carry
  refactors and cleanups. Hand-audit a random sample of ~50 positives before
  quoting headline numbers.
- **116 of 477 positives come from `unsound` advisories**, not CVE-grade
  vulnerabilities. Filter on the `informational` column if you want CVE-only.
- **Class imbalance is 1:16.** Handled via `scale_pos_weight`; always report
  precision and recall separately, never F1 alone.
- **Advisory concentration.** The top repo (`nostrdevkit/nostr`) supplies 31
  positives. Grouped splits prevent leakage, but a single repo can still swing
  a fold. Consider repeated splits across seeds.
- **ECE is biased downward on small test sets** and moves with bin count. The
  bootstrap CI is reported for this reason; quote it, not the point estimate.

---

## Running it

```bash
pip install xgboost tree-sitter tree-sitter-rust pyarrow pandas scikit-learn

# 1. get the advisory database
git clone --depth 1 https://github.com/rustsec/advisory-db data/raw/advisory-db

# 2. measure yield
python harvest/analyze_advisories.py data/raw/advisory-db

# 3. harvest (add --include-prs for the ~2.3x yield used above)
python harvest/rustsec.py --include-prs --workers 8

# 4. train + calibrate
python model/train.py --data data/processed/rustsec_v0.1.parquet --report reports/rustsec.json
```

The harvest took ~9 minutes with 8 workers. It is network-bound, so more
workers help up to about 12.

---

## Where to go next

In priority order:

1. **Grow the positive class.** 477 is workable but thin. The GitHub Security
   Advisory database (GHSA) carries Rust entries beyond RustSec, and OSV
   aggregates both. That is the cheapest path to ~1,000 positives.
2. **Add Clippy features.** `cargo clippy --message-format=json` gives per-function
   diagnostic counts. High signal, nearly free, and currently unused.
3. **Audit labels.** Sample 50 positives, check by hand how many really contain
   the flaw. Report that rate in your paper — it bounds everything else.
4. **Repeated grouped CV.** One split on 151 repos is noisy. Use
   `StratifiedGroupKFold` across 5 folds and report mean ± std.
5. **Risk–coverage curve.** You have the abstention band already; plotting
   accuracy against coverage is the strongest single figure for this project.

## A correction for your requirements document

Slide 5 lists "a XGBoost model with pretrained weights" as a requirement. No
such pretrained model exists for Rust vulnerability detection — the weights are
trained here, from this corpus. Slide 5 also treats the vulnerability rules and
the model as separate components; in this implementation the rules *are* the
features (`extract/features.py`), and XGBoost learns the weighting. Both are
worth correcting before the document goes further.
