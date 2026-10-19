"""
Inference: run a saved detector over Rust source the model has never seen.

This is the module that turns the project from an experiment into a tool.
Training answers "how good is the detector?"; this answers "what is wrong with
this file?", which is the question the requirements document actually asks.

THE PIPELINE, IN ORDER

    source bytes
      -> is_probably_rust()        reject non-Rust input before the model
      -> iter_functions()          cut the file into individual functions
      -> feature vector per fn     in the order the BUNDLE declares
      -> booster                   raw score
      -> calibrator                honest probability
      -> threshold                 flag / do not flag
      -> margin                    optionally abstain when not confident

The order matters at both ends. The parser gate is first because a model fed
Python will happily emit a confident number about it -- garbage in, confident
garbage out -- and that is Scenario 2 in the requirements. The abstention check
is last because it is the one place this detector is genuinely strong: with
AUROC 0.62 the ranking is weak, but the calibrated probabilities are trustworthy
(ECE 0.02), so "I do not know" is a conclusion this model can actually support.

A NOTE ON FEATURE ORDER

Features are assembled by looking up `bundle.feature_names` one at a time,
NOT by taking `FEATURE_NAMES` in its natural order. The two differ whenever a
model was trained with --drop-size-features, which produces a bundle with 136
columns instead of 142. Building the vector from the bundle's own list means
the ablation models load and run correctly with no special casing, and a
mismatch between extractor and bundle is caught by name rather than slipping
through because the counts happened to agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import extract_features, is_probably_rust, iter_functions
from .persist import Bundle


@dataclass
class Finding:
    """One scored function."""

    function: str
    start_line: int
    end_line: int
    probability: float      # calibrated P(vulnerable)
    flagged: bool           # probability >= threshold
    margin: float           # |probability - threshold|
    abstained: bool = False  # margin too small to stand behind the call

    @property
    def verdict(self) -> str:
        if self.abstained:
            return "UNSURE"
        return "VULNERABLE" if self.flagged else "ok"


@dataclass
class ScanResult:
    """The outcome for one file (or one in-memory snippet)."""

    path: str
    accepted: bool                 # did it pass the Rust parser gate?
    reason: str                    # why accepted or rejected
    findings: list[Finding] = field(default_factory=list)

    @property
    def flagged(self) -> list[Finding]:
        return [f for f in self.findings if f.flagged and not f.abstained]

    @property
    def abstained(self) -> list[Finding]:
        return [f for f in self.findings if f.abstained]


def featurise(bundle: Bundle, sources) -> np.ndarray:
    """
    Build the feature matrix for a list of function sources, in the column
    order this bundle was trained with.

    Returns shape (len(sources), len(bundle.feature_names)).
    """
    rows = []
    for src in sources:
        feats = extract_features(src)
        missing = [n for n in bundle.feature_names if n not in feats]
        if missing:
            raise ValueError(
                f"the feature extractor no longer produces "
                f"{len(missing)} feature(s) this model was trained on, e.g. "
                f"{missing[:3]}. features.py has changed incompatibly since "
                f"training -- retrain and re-save the bundle.")
        rows.append([feats[n] for n in bundle.feature_names])
    return np.asarray(rows, dtype=np.float32)


def scan_source(bundle: Bundle, source: bytes | str, path: str = "<memory>",
                abstain_margin: float = 0.0,
                gate: bool = True) -> ScanResult:
    """
    Score every function in one blob of Rust source.

    abstain_margin
        Report UNSURE instead of a yes/no call when the calibrated probability
        sits within this distance of the threshold. 0.0 disables abstention and
        every function gets a hard call.

    gate
        Run the non-Rust parser check first. Leave this on unless you are
        deliberately scanning a pre-validated snippet.
    """
    if gate:
        accepted, reason = is_probably_rust(source)
        if not accepted:
            # Deliberately return rather than raise: scanning a directory of
            # mixed content should skip the .rs file that is actually a
            # generated blob, not abort the whole run.
            return ScanResult(path=path, accepted=False,
                              reason=f"not Rust: {reason}")
    else:
        reason = "gate skipped"

    functions = iter_functions(source)
    if not functions:
        return ScanResult(path=path, accepted=True,
                          reason="no functions of scoreable length found")

    X = featurise(bundle, [f.source for f in functions])
    probs = bundle.score(X)
    margins = np.abs(probs - bundle.threshold)

    findings = []
    for fn, p, mg in zip(functions, probs, margins):
        findings.append(Finding(
            function=fn.name,
            start_line=fn.start_line,
            end_line=fn.end_line,
            probability=float(p),
            flagged=bool(p >= bundle.threshold),
            margin=float(mg),
            abstained=bool(mg < abstain_margin),
        ))

    # Most suspicious first: that is the order a reviewer wants to read in.
    findings.sort(key=lambda f: -f.probability)
    return ScanResult(path=path, accepted=True, reason=reason, findings=findings)


def scan_file(bundle: Bundle, path: str | Path, **kw) -> ScanResult:
    """Scan one file from disk. Unreadable files become rejected results."""
    path = Path(path)
    try:
        source = path.read_bytes()
    except OSError as e:
        return ScanResult(path=str(path), accepted=False,
                          reason=f"unreadable: {e}")
    return scan_source(bundle, source, path=str(path), **kw)


def iter_rust_files(root: str | Path) -> list[Path]:
    """
    Every .rs file under `root`, or just `root` itself if it is a file.

    Skips the directories the training corpus excluded -- tests, benches,
    examples, fuzz -- plus build output. This is not tidiness: the model was
    trained only on production code, so test fixtures full of deliberate
    `unwrap()` calls are exactly the inputs it will be most wrong about.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    skip = {"tests", "benches", "examples", "fuzz", "target", ".git"}
    return sorted(p for p in root.rglob("*.rs")
                  if not skip & set(p.parts))


def scan_path(bundle: Bundle, root: str | Path, **kw) -> list[ScanResult]:
    """Scan a file or walk a directory tree."""
    return [scan_file(bundle, p, **kw) for p in iter_rust_files(root)]


def format_results(results: list[ScanResult], bundle: Bundle,
                   show_all: bool = False) -> str:
    """
    Render scan results as a terminal report.

    By default only flagged and abstained functions are listed. A SAST tool
    that prints every function it looked at buries its own output -- at a base
    rate near 0.05 the clean functions are the overwhelming majority.
    """
    lines: list[str] = []
    n_files = n_funcs = n_flagged = n_unsure = n_rejected = 0

    for r in results:
        if not r.accepted:
            n_rejected += 1
            lines.append(f"\n-- {r.path}\n   SKIPPED: {r.reason}")
            continue

        n_files += 1
        n_funcs += len(r.findings)
        n_flagged += len(r.flagged)
        n_unsure += len(r.abstained)

        shown = r.findings if show_all else (r.flagged + r.abstained)
        if not shown:
            continue

        lines.append(f"\n-- {r.path}")
        for f in sorted(shown, key=lambda x: -x.probability):
            lines.append(
                f"   {f.verdict:<11s} p={f.probability:.3f}  "
                f"margin={f.margin:.3f}  "
                f"lines {f.start_line}-{f.end_line}  {f.function}()")

    lines.append(
        f"\n{'=' * 66}\n"
        f"{n_files} file(s) scanned, {n_funcs} function(s) scored, "
        f"{n_rejected} skipped\n"
        f"{n_flagged} flagged at threshold {bundle.threshold:.3f}"
        + (f", {n_unsure} unsure" if n_unsure else ""))

    # The honesty paragraph. These numbers are from the model's own evaluation
    # and they are the difference between a tool a reviewer can calibrate their
    # trust against and one that just emits confident noise.
    md = bundle.metadata
    if "test_precision" in md:
        lines.append(
            f"\nCalibration note: at this threshold the detector measured "
            f"precision {md['test_precision']:.3f} and recall "
            f"{md['test_recall']:.3f} on held-out repositories.\n"
            f"Expect roughly {md['test_precision'] * 100:.0f} in 100 flags to "
            f"be real, and expect it to miss about "
            f"{(1 - md['test_recall']) * 100:.0f}% of real issues.\n"
            f"Treat flags as 'review this first', not as proof of a bug.")

    return "\n".join(lines)
