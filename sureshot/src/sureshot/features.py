"""
SureShot SAST - feature extraction.

Turns a Rust function into a fixed-length numeric vector. These are the
"vulnerability rules" from the requirements document, reframed as features:
each rule emits a count or ratio and XGBoost learns how to weight and combine
them, rather than a human hand-authoring the decision logic.

Contract:
  extract_features(source: bytes) -> dict[str, float]
  - deterministic
  - never raises (malformed input yields sentinel values)
  - stable key set (FEATURE_NAMES), so the matrix never changes shape
"""

from __future__ import annotations

import re
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_rust as ts_rust

RUST = Language(ts_rust.language())

# --------------------------------------------------------------------------
# Token-level rules. Regex over source text: cheap, and robust on the
# malformed input the requirements say we must accept.
# --------------------------------------------------------------------------

TOKEN_RULES: dict[str, str] = {
    # --- unsafe surface ---
    "unsafe_block":      r"\bunsafe\s*\{",
    "unsafe_fn":         r"\bunsafe\s+(?:extern\s+\"[^\"]*\"\s+)?fn\b",
    "unsafe_impl":       r"\bunsafe\s+impl\b",
    # --- additional unsafe operations ---
    "get_unchecked":       r"\.\s*get_unchecked(?:_mut)?\s*\(",
    "unwrap_unchecked":    r"\.\s*unwrap_unchecked\s*\(",
    "from_utf8_unchecked": r"\b(?:str::)?from_utf8_unchecked\s*\(",
    "unchecked_arith":     r"\.\s*unchecked_(?:add|sub|mul|div|shl|shr)\s*\(",
    # --- raw pointers ---
    "raw_const_ptr":     r"\*const\b",
    "raw_mut_ptr":       r"\*mut\b",
    "ptr_add":           r"\.\s*add\s*\(",
    "ptr_offset":        r"\.\s*offset\s*\(",
    "ptr_wrapping":      r"\.\s*wrapping_(?:add|sub|offset|mul)\s*\(",
    "ptr_read_write":    r"\bptr::(?:read|write|copy|copy_nonoverlapping)\b",
    "ptr_null_check":    r"\.\s*is_null\s*\(\s*\)",
    "ptr_read_unaligned":  r"\bptr::read_unaligned\b",
    "ptr_write_unaligned": r"\bptr::write_unaligned\b",
    "ptr_as_ref":          r"\.\s*as_ref\s*\(",
    "ptr_as_mut":          r"\.\s*as_mut\s*\(",
    # --- FFI ---
    "extern_c":          r"\bextern\s+\"C\"",
    "no_mangle":         r"#\[\s*no_mangle\s*\]",
    "libc_call":         r"\blibc::",
    "c_string":          r"\bC(?:String|Str)\b",
    "from_raw_parts":    r"\bfrom_raw_parts(?:_mut)?\s*\(",
    "cstring_from_vec":  r"\bCString::from_vec_unchecked\b",
    "from_ptr":          r"\b(?:CStr|CString)::from_ptr\b",
    # --- memory management ---
    "transmute":         r"\bmem::transmute\b|\btransmute\s*(?:::<[^>]*>)?\s*\(",
    "mem_forget":        r"\bmem::forget\b",
    "maybe_uninit":      r"\bMaybeUninit\b",
    "assume_init":       r"\bassume_init\b",
    "manually_drop":     r"\bManuallyDrop\b",
    "box_raw":           r"\bBox::(?:from_raw|into_raw|leak)\b",
    "alloc_call":        r"\balloc::(?:alloc|dealloc|realloc)\b",
    "set_len":           r"\.\s*set_len\s*\(",
    # --- panic surface ---
    "unwrap":            r"\.\s*unwrap\s*\(\s*\)",
    "expect":            r"\.\s*expect\s*\(",
    "panic_macro":       r"\bpanic!\s*\(",
    "unreachable":       r"\bunreachable!\s*\(",
    "todo_unimpl":       r"\b(?:todo|unimplemented)!\s*\(",
    "array_index":       r"\w\s*\[\s*[a-z_]\w*\s*\]",
    # --- integer safety ---
    "as_cast":           r"\bas\s+(?:u8|u16|u32|u64|usize|i8|i16|i32|i64|isize)\b",
    "checked_op":        r"\.\s*checked_(?:add|sub|mul|div|shl|shr)\s*\(",
    "saturating_op":     r"\.\s*saturating_(?:add|sub|mul)\s*\(",
    "wrapping_arith":    r"\.\s*wrapping_(?:add|sub|mul)\s*\(",
    "overflowing_op":    r"\.\s*overflowing_(?:add|sub|mul)\s*\(",
    # --- concurrency ---
    "unsafe_send_sync":  r"\bunsafe\s+impl\s+(?:\w+\s+)?(?:Send|Sync)\b",
    "static_mut":        r"\bstatic\s+mut\b",
    "unsafe_cell":       r"\bUnsafeCell\b",
    "lock_acquire":      r"\.\s*(?:lock|read|write|borrow_mut)\s*\(\s*\)",
    "atomic_op":         r"\bAtomic(?:Usize|U8|U16|U32|U64|Bool|Ptr)\b",
    "ordering_relaxed":  r"\bOrdering::Relaxed\b",
    # --- error handling ---
    "question_mark":     r"\?\s*;",
    "result_type":       r"\bResult\s*<",
    "option_type":       r"\bOption\s*<",
    # --- lifetimes / borrows ---
    "explicit_lifetime": r"&\s*'[a-z_]\w*",
    "static_lifetime":   r"'static\b",
    "mut_ref_param":     r"&\s*mut\s+",
    "clone_call":        r"\.\s*clone\s*\(\s*\)",
}

_COMPILED = {k: re.compile(v) for k, v in TOKEN_RULES.items()}

# --------------------------------------------------------------------------
# AST-level rules via tree-sitter.
# --------------------------------------------------------------------------

AST_NODE_TYPES = [
    "if_expression", "match_expression", "for_expression",
    "while_expression", "loop_expression", "call_expression",
    "unary_expression", "closure_expression", "macro_invocation",
    "unsafe_block", "reference_expression", "index_expression",
]

FEATURE_NAMES: list[str] = (
    [f"tok_{k}" for k in TOKEN_RULES]
    + [f"tok_{k}_per_line" for k in TOKEN_RULES]
    + [f"ast_{t}" for t in AST_NODE_TYPES]
    + [
        "loc", "loc_nonblank", "chars", "avg_line_len", "max_line_len",
        "max_nest_depth", "cyclomatic", "param_count", "generic_param_count",
        "has_parse_error", "error_node_count", "ident_count",
        "unsafe_line_ratio", "comment_line_ratio",
        "checked_to_unchecked_ratio", "deref_density",
    ]
)


def _walk(node, depth=0):
    """Iterative DFS yielding (node, depth); avoids recursion limits."""
    stack = [(node, depth)]
    while stack:
        n, d = stack.pop()
        yield n, d
        for child in reversed(n.children):
            stack.append((child, d + 1))


def extract_features(source: bytes | str) -> dict[str, float]:
    """Extract the full feature vector. Never raises."""
    if isinstance(source, str):
        source = source.encode("utf-8", errors="replace")

    feats: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    try:
        text = source.decode("utf-8", errors="replace")
    except Exception:
        return feats

    lines = text.splitlines()
    n_lines = max(len(lines), 1)
    nonblank = [ln for ln in lines if ln.strip()]

    # --- token rules ---
    for key, rx in _COMPILED.items():
        try:
            c = float(len(rx.findall(text)))
        except Exception:
            c = 0.0
        feats[f"tok_{key}"] = c
        feats[f"tok_{key}_per_line"] = c / n_lines

    # --- size / shape ---
    feats["loc"] = float(len(lines))
    feats["loc_nonblank"] = float(len(nonblank))
    feats["chars"] = float(len(text))
    feats["avg_line_len"] = float(sum(len(l) for l in lines)) / n_lines
    feats["max_line_len"] = float(max((len(l) for l in lines), default=0))
    feats["comment_line_ratio"] = (
        sum(1 for l in nonblank if l.strip().startswith("//")) / max(len(nonblank), 1)
    )

    # --- derived ratios ---
    checked = feats["tok_checked_op"] + feats["tok_saturating_op"]
    unchecked = feats["tok_wrapping_arith"] + feats["tok_as_cast"]
    feats["checked_to_unchecked_ratio"] = checked / (unchecked + 1.0)
    feats["unsafe_line_ratio"] = feats["tok_unsafe_block"] / n_lines
    feats["deref_density"] = (
        feats["tok_raw_const_ptr"] + feats["tok_raw_mut_ptr"]
    ) / n_lines

    # --- AST rules ---
    try:
        tree = Parser(RUST).parse(source)
        root = tree.root_node
        feats["has_parse_error"] = 1.0 if root.has_error else 0.0

        counts: dict[str, int] = {}
        max_depth = 0
        errors = 0
        idents = 0
        params = 0
        generics = 0

        for node, depth in _walk(root):
            t = node.type
            counts[t] = counts.get(t, 0) + 1
            if depth > max_depth:
                max_depth = depth
            if t == "ERROR":
                errors += 1
            elif t == "identifier":
                idents += 1
            elif t == "parameter":
                params += 1
            elif t == "type_parameters":
                generics += len(node.children)

        for t in AST_NODE_TYPES:
            feats[f"ast_{t}"] = float(counts.get(t, 0))

        feats["max_nest_depth"] = float(max_depth)
        feats["error_node_count"] = float(errors)
        feats["ident_count"] = float(idents)
        feats["param_count"] = float(params)
        feats["generic_param_count"] = float(generics)

        # cyclomatic complexity: 1 + number of branch points
        feats["cyclomatic"] = 1.0 + sum(
            counts.get(t, 0) for t in
            ("if_expression", "match_arm", "for_expression",
             "while_expression", "loop_expression", "binary_expression")
        )
    except Exception:
        feats["has_parse_error"] = 1.0

    return feats


def is_probably_rust(source: bytes | str, min_ratio: float = 0.35) -> tuple[bool, str]:
    """
    Parser gate for Scenario 2 (non-Rust input). This runs BEFORE the model:
    non-Rust code is rejected here, it never reaches inference.

    Returns (accepted, reason).
    """
    if isinstance(source, str):
        source = source.encode("utf-8", errors="replace")
    if not source.strip():
        return False, "empty input"

    try:
        tree = Parser(RUST).parse(source)
    except Exception:
        return False, "parser failure"

    total = err = 0
    for node, _ in _walk(tree.root_node):
        total += 1
        if node.type in ("ERROR", "MISSING"):
            err += 1
    if total == 0:
        return False, "no parseable nodes"

    good = 1.0 - (err / total)
    text = source.decode("utf-8", errors="replace")
    has_rust_kw = bool(re.search(r"\b(fn|let|impl|struct|enum|trait|use|mod|pub)\b", text))

    if good < min_ratio:
        return False, f"parse quality {good:.2f} below {min_ratio}"
    if not has_rust_kw:
        return False, "no Rust keywords found"
    return True, f"parse quality {good:.2f}"


if __name__ == "__main__":
    demo = b"""
pub fn read_at(buf: &[u8], i: usize) -> u8 {
    buf[i]
}
"""
    f = extract_features(demo)
    print(f"{len(f)} features")
    for k, v in sorted(f.items()):
        if v:
            print(f"  {k:32s} {v:.3f}")
