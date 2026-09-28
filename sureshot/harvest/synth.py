"""
SureShot SAST - synthetic dataset generator (v0.1)

Generates paired Rust functions: a SAFE variant and a MUTATED (vulnerable)
variant that differ by one injected bug pattern. Used ONLY to exercise the
pipeline (parser -> features -> XGBoost -> calibration). Metrics computed on
this data are NOT evidence of real vulnerability detection ability and must
never be pooled with metrics from the RustSec-derived corpus.

Bug patterns injected:
  BOUNDS      removed bounds check before slice index
  RAWPARTS    unvalidated length passed to slice::from_raw_parts
  INTOVF      checked_add/checked_mul replaced with raw + / *
  NULLDEREF   removed null check before raw pointer deref
  TRANSMUTE   safe cast replaced with mem::transmute
  UNINIT      MaybeUninit read before initialisation
  OFFSET      unchecked pointer .add()/.offset() arithmetic
  LOCKORDER   lock released before use of guarded data (UAF-style)

Output schema (matches the RustSec harvester):
  repo, commit, file, function_name, source, label, source_type, pattern
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Naming pools - vary identifiers so the model cannot key on literal names
# --------------------------------------------------------------------------

VERBS = ["read", "copy", "parse", "decode", "write", "fetch", "scan", "merge",
         "load", "encode", "fill", "drain", "split", "pack", "unpack", "shift"]
NOUNS = ["buffer", "frame", "chunk", "record", "packet", "header", "payload",
         "entry", "block", "slot", "segment", "token", "field", "region"]
INT_TYPES = ["u8", "u16", "u32", "u64", "usize"]

# Fake repo names so GroupShuffleSplit has groups to work with.
SYNTH_REPOS = [f"synthcrate-{i:02d}" for i in range(12)]


@dataclass
class Variant:
    """One generated function: source text plus its label metadata."""
    name: str
    source: str
    label: int          # 1 = vulnerable, 0 = safe
    pattern: str


# --------------------------------------------------------------------------
# Template helpers
# --------------------------------------------------------------------------

def _ident(rng: random.Random) -> str:
    return f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}"


def _noise_lines(rng: random.Random, n: int) -> str:
    """Filler statements so safe/vuln pairs are not trivially length-separable."""
    pool = [
        "    let _span = tracing::trace_span!(\"op\");",
        "    debug_assert!(!data.is_empty());",
        "    let started = std::time::Instant::now();",
        "    let mut retries = 0usize;",
        "    if cfg!(debug_assertions) { retries += 1; }",
        "    let _ = &retries;",
        "    let tag = 0x5Au8;",
        "    let _ = tag;",
    ]
    return "\n".join(rng.sample(pool, min(n, len(pool))))


# --------------------------------------------------------------------------
# Pattern generators. Each returns (safe_variant, vuln_variant).
# --------------------------------------------------------------------------

def gen_bounds(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    ty = rng.choice(INT_TYPES)
    noise = _noise_lines(rng, rng.randint(1, 3))
    safe = f"""pub fn {fn}(data: &[{ty}], idx: usize) -> Option<{ty}> {{
{noise}
    if idx >= data.len() {{
        return None;
    }}
    let value = data[idx];
    Some(value)
}}"""
    vuln = f"""pub fn {fn}(data: &[{ty}], idx: usize) -> Option<{ty}> {{
{noise}
    let value = data[idx];
    Some(value)
}}"""
    return (Variant(fn, safe, 0, "BOUNDS"), Variant(fn, vuln, 1, "BOUNDS"))


def gen_rawparts(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    noise = _noise_lines(rng, rng.randint(1, 2))
    safe = f"""pub unsafe fn {fn}(ptr: *const u8, len: usize, cap: usize) -> &'static [u8] {{
{noise}
    assert!(!ptr.is_null());
    let len = core::cmp::min(len, cap);
    core::slice::from_raw_parts(ptr, len)
}}"""
    vuln = f"""pub unsafe fn {fn}(ptr: *const u8, len: usize, cap: usize) -> &'static [u8] {{
{noise}
    let _ = cap;
    core::slice::from_raw_parts(ptr, len)
}}"""
    return (Variant(fn, safe, 0, "RAWPARTS"), Variant(fn, vuln, 1, "RAWPARTS"))


def gen_intovf(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    ty = rng.choice(["u32", "u64", "usize"])
    op = rng.choice([("checked_add", "+"), ("checked_mul", "*")])
    noise = _noise_lines(rng, rng.randint(1, 3))
    safe = f"""pub fn {fn}(base: {ty}, count: {ty}) -> Option<{ty}> {{
{noise}
    let total = base.{op[0]}(count)?;
    Some(total)
}}"""
    vuln = f"""pub fn {fn}(base: {ty}, count: {ty}) -> Option<{ty}> {{
{noise}
    let total = base {op[1]} count;
    Some(total)
}}"""
    return (Variant(fn, safe, 0, "INTOVF"), Variant(fn, vuln, 1, "INTOVF"))


def gen_nullderef(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    noise = _noise_lines(rng, rng.randint(1, 2))
    safe = f"""pub unsafe fn {fn}(handle: *mut u32) -> u32 {{
{noise}
    if handle.is_null() {{
        return 0;
    }}
    *handle
}}"""
    vuln = f"""pub unsafe fn {fn}(handle: *mut u32) -> u32 {{
{noise}
    *handle
}}"""
    return (Variant(fn, safe, 0, "NULLDEREF"), Variant(fn, vuln, 1, "NULLDEREF"))


def gen_transmute(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    noise = _noise_lines(rng, rng.randint(1, 2))
    safe = f"""pub fn {fn}(raw: u32) -> f32 {{
{noise}
    f32::from_bits(raw)
}}"""
    vuln = f"""pub fn {fn}(raw: u32) -> f32 {{
{noise}
    unsafe {{ core::mem::transmute::<u32, f32>(raw) }}
}}"""
    return (Variant(fn, safe, 0, "TRANSMUTE"), Variant(fn, vuln, 1, "TRANSMUTE"))


def gen_uninit(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    n = rng.choice([16, 32, 64, 128])
    noise = _noise_lines(rng, rng.randint(1, 2))
    safe = f"""pub fn {fn}(src: &[u8]) -> [u8; {n}] {{
{noise}
    let mut out = [0u8; {n}];
    let take = core::cmp::min(src.len(), {n});
    out[..take].copy_from_slice(&src[..take]);
    out
}}"""
    vuln = f"""pub fn {fn}(src: &[u8]) -> [u8; {n}] {{
{noise}
    let mut out: core::mem::MaybeUninit<[u8; {n}]> = core::mem::MaybeUninit::uninit();
    let take = core::cmp::min(src.len(), {n});
    unsafe {{
        core::ptr::copy_nonoverlapping(src.as_ptr(), out.as_mut_ptr() as *mut u8, take);
        out.assume_init()
    }}
}}"""
    return (Variant(fn, safe, 0, "UNINIT"), Variant(fn, vuln, 1, "UNINIT"))


def gen_offset(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    noise = _noise_lines(rng, rng.randint(1, 2))
    safe = f"""pub unsafe fn {fn}(base: *const u8, len: usize, off: usize) -> Option<u8> {{
{noise}
    if off >= len {{
        return None;
    }}
    Some(*base.add(off))
}}"""
    vuln = f"""pub unsafe fn {fn}(base: *const u8, len: usize, off: usize) -> Option<u8> {{
{noise}
    let _ = len;
    Some(*base.add(off))
}}"""
    return (Variant(fn, safe, 0, "OFFSET"), Variant(fn, vuln, 1, "OFFSET"))


def gen_lockorder(rng: random.Random) -> tuple[Variant, Variant]:
    fn = _ident(rng)
    noise = _noise_lines(rng, rng.randint(1, 2))
    safe = f"""pub fn {fn}(shared: &std::sync::Mutex<Vec<u8>>) -> usize {{
{noise}
    let guard = shared.lock().unwrap();
    let total = guard.len();
    drop(guard);
    total
}}"""
    vuln = f"""pub fn {fn}(shared: &std::sync::Mutex<Vec<u8>>) -> usize {{
{noise}
    let ptr = {{
        let guard = shared.lock().unwrap();
        guard.as_ptr()
    }};
    unsafe {{ *ptr as usize }}
}}"""
    return (Variant(fn, safe, 0, "LOCKORDER"), Variant(fn, vuln, 1, "LOCKORDER"))


GENERATORS = [
    gen_bounds, gen_rawparts, gen_intovf, gen_nullderef,
    gen_transmute, gen_uninit, gen_offset, gen_lockorder,
]


# --------------------------------------------------------------------------
# Distractor functions - safe code with no paired mutant, so the negative
# class is not simply "the other half of every pair".
# --------------------------------------------------------------------------

def gen_distractor(rng: random.Random) -> Variant:
    fn = _ident(rng)
    kind = rng.randint(0, 3)
    noise = _noise_lines(rng, rng.randint(1, 3))
    if kind == 0:
        src = f"""pub fn {fn}(items: &[u32]) -> u64 {{
{noise}
    items.iter().map(|v| *v as u64).sum()
}}"""
    elif kind == 1:
        src = f"""pub fn {fn}(text: &str) -> Vec<String> {{
{noise}
    text.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}}"""
    elif kind == 2:
        src = f"""pub fn {fn}(map: &std::collections::HashMap<String, u32>, key: &str) -> u32 {{
{noise}
    map.get(key).copied().unwrap_or_default()
}}"""
    else:
        src = f"""pub fn {fn}(input: &[u8]) -> Result<u32, std::io::Error> {{
{noise}
    let mut acc: u32 = 0;
    for b in input.iter().take(4) {{
        acc = acc.rotate_left(8) ^ (*b as u32);
    }}
    Ok(acc)
}}"""
    return Variant(fn, src, 0, "DISTRACTOR")


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def build(n_pairs: int, n_distractors: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    def add(v: Variant, repo: str, idx: int) -> None:
        digest = hashlib.sha1(v.source.encode()).hexdigest()[:12]
        rows.append({
            "repo": repo,
            "commit": digest,
            "file": f"src/{v.pattern.lower()}/mod_{idx:04d}.rs",
            "function_name": v.name,
            "source": v.source,
            "label": v.label,
            "source_type": "synthetic",
            "pattern": v.pattern,
        })

    for i in range(n_pairs):
        gen = GENERATORS[i % len(GENERATORS)]
        safe_v, vuln_v = gen(rng)
        # Both halves of a pair share a repo so a grouped split cannot put a
        # function and its own near-duplicate mutant on opposite sides.
        repo = rng.choice(SYNTH_REPOS)
        add(safe_v, repo, i)
        add(vuln_v, repo, i)

    for i in range(n_distractors):
        add(gen_distractor(rng), rng.choice(SYNTH_REPOS), 10_000 + i)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic Rust vuln dataset")
    ap.add_argument("--pairs", type=int, default=1200,
                    help="safe/vulnerable pairs to generate (default 1200)")
    ap.add_argument("--distractors", type=int, default=1800,
                    help="unpaired safe functions (default 1800)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/synthetic_v0.1.parquet"))
    args = ap.parse_args()

    df = build(args.pairs, args.distractors, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    pos = int(df.label.sum())
    neg = len(df) - pos
    print(f"wrote {args.out}  rows={len(df)}  pos={pos}  neg={neg}  "
          f"ratio=1:{neg / max(pos, 1):.2f}")
    print(f"repos={df.repo.nunique()}  patterns={df.pattern.nunique()}")
    print(df.pattern.value_counts().to_string())


if __name__ == "__main__":
    main()
