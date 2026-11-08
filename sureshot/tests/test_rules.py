"""
Tests for the pattern rules in sureshot/rules.py.

These need no trained model, so they are fast and pin down exactly which
constructs each rule does and does not match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sureshot import rules
from sureshot.features import iter_functions

DEMO = Path(__file__).resolve().parents[1] / "demo" / "vulnerable_buffer.rs"


def ids(source: str) -> set[str]:
    return {h.rule_id for h in rules.check(source)}


# --------------------------------------------------------------------------
# each high-severity rule fires on its construct
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id, body", [
    ("SS001", "unsafe { std::mem::transmute::<u64, f64>(v) }"),
    ("SS001", "unsafe { transmute(v) }"),
    ("SS002", "unsafe { std::slice::from_raw_parts(p, n) }"),
    ("SS002", "unsafe { slice::from_raw_parts_mut(p, n) }"),
    ("SS003", "unsafe { v.set_len(n) }"),
    ("SS004", "unsafe { std::mem::zeroed() }"),
    ("SS004", "unsafe { x.assume_init() }"),
    ("SS005", "unsafe { *v.get_unchecked(i) }"),
    ("SS005", "unsafe { std::str::from_utf8_unchecked(b) }"),
    ("SS006", "unsafe { ptr::copy_nonoverlapping(s, d, n) }"),
    ("SS006", "unsafe { ptr::write(d, v) }"),
    ("SS007", "unsafe { Box::from_raw(p) }"),
    ("SS007", "unsafe { CStr::from_ptr(p) }"),
])
def test_high_rule_fires(rule_id, body):
    src = f"fn f() {{\n    {body};\n}}\n"
    hits = [h for h in rules.check(src) if h.rule_id == rule_id]
    assert hits, f"{rule_id} did not fire on: {body}"
    assert hits[0].severity == "high"


# --------------------------------------------------------------------------
# comments and strings never fire rules
# --------------------------------------------------------------------------

def test_comments_and_strings_are_ignored():
    src = (
        "fn f() {\n"
        "    // never call transmute( or from_raw_parts( here\n"
        "    /* v.set_len(n) */\n"
        "    let s = \"x.unwrap() and ptr::write(\";\n"
        "    let r = r#\"Box::from_raw(p)\"#;\n"
        "}\n"
    )
    assert rules.check(src) == []


def test_code_only_keeps_line_numbers():
    src = "fn f() {\n  // c\n  let s = \"ab\";\n}\n"
    out = rules.code_only(src)
    assert len(out) == len(src)
    assert out.count("\n") == src.count("\n")
    assert "//" not in out and "ab" not in out


# --------------------------------------------------------------------------
# indexing rules use the syntax tree, so slice TYPES do not match
# --------------------------------------------------------------------------

def test_slice_types_are_not_indexing():
    src = ("fn f<'a>(a: &'a [u8], b: &mut [u8], c: Vec<[u8; 4]>) -> usize {\n"
           "    a.len() + b.len()\n}\n")
    assert not ids(src) & {"SS102", "SS201"}


def test_arithmetic_index_is_medium():
    src = "fn f(b: &[u8], o: usize) -> u8 {\n    b[o + 1]\n}\n"
    hits = [h for h in rules.check(src) if h.rule_id == "SS102"]
    assert hits and hits[0].severity == "medium"
    assert hits[0].line == 2


def test_plain_index_only_without_a_length_check():
    unguarded = "fn f(b: &[u8], i: usize) -> u8 {\n    b[i]\n}\n"
    guarded = ("fn f(b: &[u8], i: usize) -> u8 {\n"
               "    if i < b.len() { b[i] } else { 0 }\n}\n")
    assert "SS201" in ids(unguarded)
    assert "SS201" not in ids(guarded)


def test_constant_index_is_not_flagged():
    assert not ids("fn f(b: &[u8]) -> u8 {\n    b[0]\n}\n") & {"SS102", "SS201"}


# --------------------------------------------------------------------------
# reporting details
# --------------------------------------------------------------------------

def test_hits_are_reported_against_the_file_line():
    src = "fn f() {\n    let x = 1;\n    unsafe { transmute(x) };\n}\n"
    hit = next(h for h in rules.check(src, first_line=40) if h.rule_id == "SS001")
    assert hit.line == 42


def test_hits_are_ordered_most_severe_first():
    src = "fn f(v: Vec<u8>) {\n    v.get(0).unwrap();\n    unsafe { transmute(v) };\n}\n"
    sev = [h.severity for h in rules.check(src)]
    order = {s: i for i, s in enumerate(rules.SEVERITIES)}
    assert sev == sorted(sev, key=order.__getitem__)
    assert rules.worst(rules.check(src)) == "high"


def test_rule_ids_are_unique_and_severities_valid():
    all_ids = [r.id for r in rules.RULES]
    assert len(all_ids) == len(set(all_ids))
    assert all(r.severity in rules.SEVERITIES for r in rules.RULES)
    assert all((r.pattern is None) != (r.ast is None) for r in rules.RULES)


def test_never_raises_on_garbage():
    assert isinstance(rules.check("\x00\xff not rust {{{{ ]["), list)
    assert rules.check("") == []


# --------------------------------------------------------------------------
# the demo file: every deliberately dangerous function is caught
# --------------------------------------------------------------------------

def test_demo_file_expectations():
    src = DEMO.read_text()
    worst = {f.name: rules.worst(rules.check(f.source, f.start_line))
             for f in iter_functions(src)}
    assert worst["copy_region"] == "high"
    assert worst["as_slice"] == "high"
    assert worst["reinterpret"] == "high"
    assert worst["parse_length"] == "medium"
    assert worst["decode_name"] == "low"
    # the careful functions stay clean
    assert worst["checked_total"] is None
    assert worst["byte_at"] is None
