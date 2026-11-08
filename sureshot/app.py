"""
SureShot GUI: paste or upload Rust code and see what the saved detector thinks.

    pip install -r requirements-gui.txt
    streamlit run app.py

This app only EVALUATES. It loads a detector bundle that was already trained
and saved (artifacts/xgboost by default) and scores the code you give it. It
never trains, never writes to artifacts/ or reports/, and never runs the code
it is given -- input is only parsed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sureshot.persist import Bundle
from sureshot.scan import ScanResult, scan_source

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEMO = ROOT / "demo" / "vulnerable_buffer.rs"
MAX_BYTES = 1_000_000          # per input; larger files are refused, not truncated

st.set_page_config(page_title="SureShot Rust Scanner", page_icon="🦀", layout="wide")


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------

def available_models() -> list[str]:
    """Every directory under artifacts/ that holds a saved detector."""
    if not ARTIFACTS.is_dir():
        return []
    return sorted(p.name for p in ARTIFACTS.iterdir()
                  if (p / "manifest.json").is_file())


@st.cache_resource(show_spinner="Loading detector...")
def load_bundle(name: str) -> Bundle:
    return Bundle.load(ARTIFACTS / name)


# --------------------------------------------------------------------------
# sidebar: model and options
# --------------------------------------------------------------------------

st.sidebar.header("Detector")
models = available_models()
if not models:
    st.error(f"No saved detector found in {ARTIFACTS}. "
             "Train one first with run_model.sh, then reload this page.")
    st.stop()

default = models.index("xgboost") if "xgboost" in models else 0
model_name = st.sidebar.selectbox("Model", models, index=default,
                                  help="Saved detector bundles in artifacts/")
try:
    bundle = load_bundle(model_name)
except Exception as e:  # a broken bundle should not take the page down
    st.error(f"Could not load artifacts/{model_name}: {e}")
    st.stop()

md = bundle.metadata
st.sidebar.markdown(
    f"- Calibrator: **{bundle.calibrator.name}**\n"
    f"- Threshold: **{bundle.threshold:.3f}**\n"
    f"- Features: **{len(bundle.feature_names)}**")
if "test_precision" in md:
    st.sidebar.markdown(
        f"- Held-out precision: **{md['test_precision']:.3f}**\n"
        f"- Held-out recall: **{md['test_recall']:.3f}**")

st.sidebar.header("Options")
use_rules = st.sidebar.toggle(
    "Pattern rules", value=True,
    help="Also check for specific risky constructs (transmute, from_raw_parts, "
         "raw pointer copies...). Off = the model's calls only.")
abstain = st.sidebar.slider(
    "Abstain margin", 0.0, 0.10, 0.0, 0.01,
    help="Report UNSURE when the model's probability is this close to the "
         "threshold. 0 = always give a yes/no.")
show_all = st.sidebar.toggle("Show every function", value=False,
                             help="Off = only flagged, unsure and review.")


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------

st.title("SureShot Rust Scanner")
st.caption("Static vulnerability triage for Rust. Paste code or upload .rs "
           "files; each function is scored by the ML model and checked "
           "against pattern rules. Nothing you enter is executed.")

tab_paste, tab_upload = st.tabs(["Paste code", "Upload files"])

with tab_paste:
    if st.button("Load demo file"):
        st.session_state["code"] = DEMO.read_text(encoding="utf-8")
    pasted = st.text_area("Rust source", key="code", height=320,
                          placeholder="pub fn example(buf: &[u8]) -> u8 {\n"
                                      "    buf[0]\n}")

with tab_upload:
    uploads = st.file_uploader("Rust files", type=["rs"],
                               accept_multiple_files=True)

inputs: list[tuple[str, bytes]] = []
if pasted and pasted.strip():
    inputs.append(("pasted code", pasted.encode("utf-8")))
for f in uploads or []:
    inputs.append((f.name, f.getvalue()))

run = st.button("Scan", type="primary", disabled=not inputs)
if not inputs:
    st.info("Paste Rust code or upload a .rs file, then press Scan.")


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

VERDICT_ICON = {"VULNERABLE": "🔴", "UNSURE": "🟠", "review": "🟡", "ok": "🟢"}


def function_source(source: bytes, start: int, end: int) -> str:
    lines = source.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[start - 1:end])


def show_result(r: ScanResult, source: bytes) -> None:
    if not r.accepted:
        st.warning(f"Skipped: {r.reason}")
        return
    if not r.findings:
        st.info(f"Nothing to score: {r.reason}. Functions shorter than 3 "
                "lines are skipped, matching the training data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Functions scored", len(r.findings))
    c2.metric("Flagged", len(r.flagged))
    c3.metric("To review", len(r.review))
    c4.metric("Unsure", len(r.abstained))

    shown = r.findings if show_all else (r.flagged + r.abstained + r.review)
    if not shown:
        st.success("Nothing flagged in this input.")
        return

    table = pd.DataFrame([{
        "": VERDICT_ICON.get(f.verdict, ""),
        "verdict": f.verdict,
        "function": f"{f.function}()",
        "lines": f"{f.start_line}-{f.end_line}",
        "probability": round(f.probability, 3),
        "flagged by": f.source_of_flag or "-",
        "rule hits": len(f.rule_hits),
    } for f in shown])
    st.dataframe(table, hide_index=True, width="stretch")

    for f in shown:
        label = (f"{VERDICT_ICON.get(f.verdict, '')} {f.verdict}  "
                 f"{f.function}()  lines {f.start_line}-{f.end_line}  "
                 f"p={f.probability:.3f}")
        with st.expander(label, expanded=f.verdict == "VULNERABLE"):
            st.markdown(
                f"Model probability **{f.probability:.3f}** vs threshold "
                f"**{bundle.threshold:.3f}** (margin {f.margin:.3f})."
                + (" The model abstained." if f.abstained else ""))
            if f.rule_hits:
                st.dataframe(pd.DataFrame([{
                    "rule": h.rule_id, "severity": h.severity,
                    "line": h.line, "risk": h.message, "match": h.snippet,
                } for h in f.rule_hits]), hide_index=True,
                    width="stretch")
            st.code(function_source(source, f.start_line, f.end_line),
                    language="rust", line_numbers=True)


if run and inputs:
    all_results = []
    for name, source in inputs:
        st.subheader(name)
        if len(source) > MAX_BYTES:
            st.warning(f"Skipped: larger than {MAX_BYTES // 1000} KB.")
            continue
        with st.spinner(f"Scanning {name}..."):
            r = scan_source(bundle, source, path=name,
                            abstain_margin=abstain, rules=use_rules)
        all_results.append(r)
        show_result(r, source)

    # The honesty note: what a flag is worth, measured on held-out repos.
    st.divider()
    if "test_precision" in md:
        st.caption(
            f"At this threshold the model measured precision "
            f"{md['test_precision']:.3f} and recall {md['test_recall']:.3f} on "
            f"held-out repositories: roughly {md['test_precision'] * 100:.0f} "
            f"in 100 model flags are real, and it misses about "
            f"{(1 - md['test_recall']) * 100:.0f}% of real issues. Rule hits "
            f"mean a risky construct is present, not that it is misused. "
            f"Treat flags as 'review this first', not as proof of a bug.")

    payload = {
        "model": model_name,
        "threshold": bundle.threshold,
        "rules": use_rules,
        "files": [{
            "path": r.path, "accepted": r.accepted, "reason": r.reason,
            "findings": [{
                "function": f.function, "start_line": f.start_line,
                "end_line": f.end_line,
                "probability": round(f.probability, 6),
                "verdict": f.verdict, "flagged_by": f.source_of_flag,
                "rule_hits": [h.to_dict() for h in f.rule_hits],
            } for f in r.findings],
        } for r in all_results],
    }
    st.download_button("Download results (JSON)",
                       json.dumps(payload, indent=2),
                       file_name="sureshot_results.json",
                       mime="application/json")
