#!/usr/bin/env bash
#
# run_model.sh -- one command that trains, evaluates and exercises everything.
#
#   ./run_model.sh
#
# Run it from anywhere; it locates itself. Every step's stdout is saved to a
# .out file next to the artifact it produced, so the log that made a model
# always sits beside that model.
#
# WHAT IT RUNS, in order:
#
#   1. the test suite                 -> artifacts/tests.out
#   2. baseline model (sklearn API)   -> artifacts/reference/      + train.out
#   3. the model (native xgb API)    -> artifacts/xgboost/        + train.out
#   4. ablation, size features gone   -> artifacts/xgboost-nosize/ + train.out
#   5. side-by-side comparison        -> artifacts/compare.out
#   6. a demo scan with the new model -> artifacts/scan.out
#
# Steps 2-4 also write metrics JSON into reports/.
#
# Why the test suite runs FIRST: it takes about three seconds and it checks the
# invariants that produce results which look fine and are wrong -- repo leakage
# between splits, calibrators fitted and measured on the same data. Training on
# a broken harness wastes a minute and produces numbers you cannot trust, so
# the script stops if the tests fail.

set -euo pipefail

# Work from the directory this script lives in, so the relative paths below
# mean the same thing no matter where it was invoked from. Without this,
# `bash sureshot/run_model.sh` and `./run_model.sh` would disagree.
cd "$(dirname "${BASH_SOURCE[0]}")"

# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------

DATA="data/processed/rustsec_v0.1.parquet"
SOURCE_TYPE="rustsec"
SEED=42
SCAN_TARGET="demo/vulnerable_buffer.rs"
SELECT_ON="test"
SKIP_TESTS=0

usage() {
    sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --data PATH          corpus to train on (default: the rustsec parquet)
  --source-type NAME   rustsec | synthetic  (never pooled)
  --seed N             repo-split seed; must match across models to compare
  --scan PATH          file or directory for the demo scan
  --select-on WHICH    test | calib  -- how the winning calibrator is chosen.
                       'test' reproduces the committed numbers; 'calib' is the
                       leak-free option (grouped out-of-fold CV).
  --skip-tests         do not run pytest first (not recommended)
  -h, --help           this text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)        DATA="$2"; shift 2 ;;
        --source-type) SOURCE_TYPE="$2"; shift 2 ;;
        --seed)        SEED="$2"; shift 2 ;;
        --scan)        SCAN_TARGET="$2"; shift 2 ;;
        --select-on)   SELECT_ON="$2"; shift 2 ;;
        --skip-tests)  SKIP_TESTS=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------
# find the interpreter
# --------------------------------------------------------------------------
#
# The venv's python is called DIRECTLY rather than going through
# `activate`. Activation is shell-specific -- the path differs between Windows
# and POSIX, and sourcing it from a non-interactive script is a common source
# of "works in my terminal, fails in the runner". Calling the interpreter by
# path has none of those problems and needs no shell state.

if   [[ -x ".venv/Scripts/python.exe" ]]; then PY=".venv/Scripts/python.exe"   # Windows
elif [[ -x ".venv/bin/python" ]];       then PY=".venv/bin/python"             # macOS / Linux
elif command -v python >/dev/null 2>&1; then PY="python"
else
    echo "error: no Python found. Create the venv first:" >&2
    echo "  python -m venv .venv && .venv/Scripts/activate   # Windows" >&2
    echo "  python -m venv .venv && source .venv/bin/activate # POSIX" >&2
    echo "  pip install -e \".[dev,plot]\"" >&2
    exit 2
fi

# Fail early with a clear message rather than deep inside an import.
if ! "$PY" -c "import sureshot, xgboost, sklearn, tree_sitter_rust" 2>/dev/null; then
    echo "error: dependencies missing from $PY" >&2
    echo "  $PY -m pip install -e \".[dev,plot]\"" >&2
    exit 2
fi

if [[ ! -f "$DATA" ]]; then
    echo "error: corpus not found: $DATA" >&2
    exit 2
fi

mkdir -p artifacts reports

# --------------------------------------------------------------------------
# step helper
# --------------------------------------------------------------------------
#
# Writes a provenance header, then streams the command's output to BOTH the
# terminal and the .out file via `tee`, so a run is watchable and recorded at
# the same time.
#
# The pipefail dance matters. With `set -o pipefail` a pipeline reports the
# first failure, which would normally be what we want -- but here the exit code
# we need is the COMMAND's, not tee's, and `scan.py` deliberately exits 1 to
# mean "something was flagged", which is a successful scan rather than an
# error. So pipefail is lifted for the pipeline and the real status recovered
# from PIPESTATUS[0].

STEP=0
run_logged() {
    local out="$1"; shift
    mkdir -p "$(dirname "$out")"
    STEP=$((STEP + 1))

    echo
    echo "=============================================================="
    echo " step $STEP: $*"
    echo " log:    $out"
    echo "=============================================================="

    {
        echo "# command : $*"
        echo "# utc     : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "# python  : $("$PY" --version 2>&1)"
        echo
    } > "$out"

    set +o pipefail
    "$@" 2>&1 | tee -a "$out"
    local rc=${PIPESTATUS[0]}
    set -o pipefail
    return "$rc"
}

train() {
    # train <bundle-dir> <report.json> <script> [extra flags...]
    local bundle="$1" report="$2" script="$3"; shift 3
    run_logged "$bundle/train.out" \
        "$PY" "$script" \
            --data "$DATA" \
            --source-type "$SOURCE_TYPE" \
            --seed "$SEED" \
            --select-on "$SELECT_ON" \
            --report "$report" \
            --save "$bundle" \
            "$@"
}

# --------------------------------------------------------------------------
# 1. tests
# --------------------------------------------------------------------------

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    run_logged "artifacts/tests.out" "$PY" -m pytest -q
else
    echo "skipping tests (--skip-tests)"
fi

# --------------------------------------------------------------------------
# 2-4. train, evaluate and save each model
# --------------------------------------------------------------------------
#
# Each of these is a single run that trains AND evaluates -- there is no
# separate eval step, because evaluation needs the model in memory. The booster
# fits on the train split, the calibrators and threshold are fitted on the
# calibration split, and every reported number comes from the test split.

train "artifacts/reference"      "reports/reference.json"  "scripts/train_scikit-learn.py"
train "artifacts/xgboost"        "reports/mine.json"       "scripts/train_XGBoost.py"

# The ablation re-trains with every raw size feature removed. Positives in
# commit-mined data are ~2.4x longer than negatives, so this is the check that
# the model learned something other than "long function = suspicious". Its
# bundle records 136 feature names instead of 142, which is also a live
# demonstration that a bundle carries its own input contract.
train "artifacts/xgboost-nosize" "reports/mine_nosize.json" "scripts/train_XGBoost.py" \
    --drop-size-features

# --------------------------------------------------------------------------
# 5. compare the two models
# --------------------------------------------------------------------------
#
# Both reports came from the same split, the same seed and the same evaluation
# code, so any delta here is caused by the model and nothing else. With matched
# hyperparameters the deltas should all be +0.0000, which is the proof that the
# harness itself is sound before anything gets tuned.

run_logged "artifacts/compare.out" "$PY" -c \
    "from sureshot.pipeline import compare; compare('reports/reference.json', 'reports/mine.json')"

# --------------------------------------------------------------------------
# 6. demo scan
# --------------------------------------------------------------------------
#
# Exit code 1 from scan.py means "something was flagged", which is a normal
# outcome and not a failure, so only 2-or-higher is treated as an error.

if [[ -e "$SCAN_TARGET" ]]; then
    rc=0
    run_logged "artifacts/scan.out" \
        "$PY" scripts/scan.py --model artifacts/xgboost --all "$SCAN_TARGET" || rc=$?
    if [[ "$rc" -gt 1 ]]; then
        echo "error: the scan failed (exit $rc)" >&2
        exit "$rc"
    fi
else
    echo
    echo "skipping demo scan: '$SCAN_TARGET' not found"
fi

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

echo
echo "=============================================================="
echo " done -- $STEP steps, no failures"
echo "=============================================================="
echo
echo " detector bundles (model.json + manifest.json + train.out)"
echo "   artifacts/reference/          baseline, sklearn XGBClassifier"
echo "   artifacts/xgboost/            native xgb.train API"
echo "   artifacts/xgboost-nosize/     ablation, size features dropped"
echo
echo " metrics JSON"
echo "   reports/reference.json  reports/mine.json  reports/mine_nosize.json"
echo
echo " captured stdout"
# Listed by what actually exists, so the summary never names a file a skipped
# step did not write.
find artifacts -name '*.out' | sort | sed 's/^/   /'
echo
echo " scan any Rust with the model just trained:"
echo "   $PY scripts/scan.py --model artifacts/xgboost <file-or-dir>"
echo
echo " how to read the numbers: ../documentation/CHANGES.md section 7,"
echo "                          STRUCTURE.md \"Verified baseline\""
