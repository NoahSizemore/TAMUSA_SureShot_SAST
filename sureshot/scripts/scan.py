#!/usr/bin/env python3
"""
Scan Rust source with a saved SureShot detector.

This is the tool the training scripts exist to produce. Everything else in the
repo measures a model; this runs one.

    # train once, saving the detector
    python scripts/train_XGBoost.py \
        --data data/processed/rustsec_v0.1.parquet \
        --source-type rustsec --save artifacts/xgboost

    # then scan anything
    python scripts/scan.py --model artifacts/xgboost path/to/file.rs
    python scripts/scan.py --model artifacts/xgboost some/crate/src
    cat lib.rs | python scripts/scan.py --model artifacts/xgboost -

Exit codes, chosen so this can be dropped into CI:
    0   scanned cleanly, nothing flagged
    1   at least one function flagged
    2   usage or load error (bad path, missing bundle, no scoreable input)

Note that exit 1 means "a human should look", not "a vulnerability exists". At
this detector's measured precision, most flags are false alarms; its value is
in ranking which functions to read first, and in being honest about how
confident it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sureshot.persist import Bundle
from sureshot.scan import format_results, scan_path, scan_source


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scan Rust source for likely vulnerabilities.")
    ap.add_argument("paths", nargs="+", type=str,
                    help="files or directories to scan; '-' reads stdin")
    ap.add_argument("--model", type=Path, required=True,
                    help="detector bundle directory, e.g. artifacts/xgboost")
    ap.add_argument("--abstain-margin", type=float, default=0.0,
                    help="report UNSURE when the calibrated probability sits "
                         "within this distance of the threshold (0 = always "
                         "give a hard yes/no)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold stored in the bundle; raising "
                         "it trades recall for precision")
    ap.add_argument("--all", action="store_true",
                    help="list every function scored, not just the flagged ones")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a report")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the model summary header")
    args = ap.parse_args()

    # --- load the detector ---------------------------------------------------
    try:
        bundle = Bundle.load(args.model)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: could not load detector from {args.model}\n  {e}",
              file=sys.stderr)
        return 2

    if args.threshold is not None:
        # Overriding is legitimate -- the stored threshold maximises F1, which
        # is rarely what a specific team wants -- but the stored
        # precision/recall figures no longer describe the new operating point,
        # so say so rather than letting the footer quietly mislead.
        print(f"note: overriding stored threshold {bundle.threshold:.3f} with "
              f"{args.threshold:.3f}; the measured precision/recall in this "
              f"bundle no longer apply.", file=sys.stderr)
        bundle.threshold = float(args.threshold)
        bundle.metadata.pop("test_precision", None)
        bundle.metadata.pop("test_recall", None)

    if not args.quiet and not args.json:
        print(f"=== SureShot detector: {args.model} ===")
        print(bundle.describe())

    # --- scan ----------------------------------------------------------------
    results = []
    for p in args.paths:
        if p == "-":
            source = sys.stdin.buffer.read()
            results.append(scan_source(bundle, source, path="<stdin>",
                                       abstain_margin=args.abstain_margin))
        else:
            path = Path(p)
            if not path.exists():
                print(f"error: no such path: {p}", file=sys.stderr)
                return 2
            results.extend(scan_path(bundle, path,
                                     abstain_margin=args.abstain_margin))

    if not results:
        print("error: no .rs files found to scan", file=sys.stderr)
        return 2

    # --- report --------------------------------------------------------------
    if args.json:
        payload = {
            "model": str(args.model),
            "threshold": bundle.threshold,
            "calibrator": bundle.calibrator.name,
            "files": [
                {
                    "path": r.path,
                    "accepted": r.accepted,
                    "reason": r.reason,
                    "findings": [
                        {
                            "function": f.function,
                            "start_line": f.start_line,
                            "end_line": f.end_line,
                            "probability": round(f.probability, 6),
                            "margin": round(f.margin, 6),
                            "flagged": f.flagged,
                            "abstained": f.abstained,
                            "verdict": f.verdict,
                        }
                        # --all controls verbosity in JSON too, so a CI job can
                        # keep the full scored list without parsing the report.
                        for f in (r.findings if args.all
                                  else r.flagged + r.abstained)
                    ],
                }
                for r in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_results(results, bundle, show_all=args.all))

    any_flagged = any(r.flagged for r in results)
    return 1 if any_flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
