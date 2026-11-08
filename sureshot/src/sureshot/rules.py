"""
Pattern rules: exact, explainable checks that run alongside the ML model.

WHY THIS EXISTS

The ML detector ranks functions by how much they resemble code that was later
patched in a security fix. That is useful for triage, but its recall is low
(about 0.23 at the operating threshold) and it cannot say WHY a function looks
risky. A reviewer handed `p=0.116` has nothing to look at.

Rules are the opposite trade-off. Each one names a specific construct that is
a well-known source of memory-safety or denial-of-service bugs in Rust, points
at the line, and explains the risk. They do not learn and they do not
generalise, but they never miss the pattern they describe.

The scanner reports both. A function is reported when EITHER the model flags it
OR a high-severity rule fires. Medium and low hits are shown as notes for a
reviewer, without changing the verdict.

HOW MATCHING WORKS

Rules are regular expressions, but they run on a copy of the function where
every comment and string literal has been blanked out with spaces (found with
the same tree-sitter parser the feature extractor uses). That keeps
`// never call transmute here` or `"unwrap()"` from firing a rule, while byte
offsets, and therefore line numbers, stay exactly the same.

Rules are deliberately NOT model features. Adding them as features would
retrain the model and change every published number; keeping them separate
means the measured precision/recall of the model still describes the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tree_sitter import Parser

from .features import RUST

SEVERITIES = ("high", "medium", "low")

# Node types whose text cannot contain code.
_NON_CODE_NODES = {
    "line_comment", "block_comment", "doc_comment",
    "string_literal", "raw_string_literal", "char_literal",
    "string_content",
}


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str           # "high" | "medium" | "low"
    pattern: str | None     # regex, matched against comment/string-free code
    message: str            # what the risk is, in one line
    # Rule only fires when NONE of these regexes appear anywhere in the
    # function. Used for "indexing with no bounds check in sight".
    unless_any: tuple[str, ...] = ()
    # Syntax-tree check used instead of `pattern` when a regex cannot tell
    # the construct apart from look-alikes (indexing vs. slice types).
    ast: str | None = None


RULES: tuple[Rule, ...] = (
    # ---- high: constructs whose misuse is undefined behaviour ---------------
    Rule("SS001", "high",
         r"\bmem::transmute\b|\btransmute\s*(?:::<[^>]*>)?\s*\(",
         "transmute reinterprets bits with no type or validity check"),
    Rule("SS002", "high",
         r"\bfrom_raw_parts(?:_mut)?\s*\(",
         "slice built from a raw pointer and length the compiler cannot check"),
    Rule("SS003", "high",
         r"\.\s*set_len\s*\(",
         "Vec::set_len can expose uninitialised memory or skip drops"),
    Rule("SS004", "high",
         r"\bmem::(?:zeroed|uninitialized)\b|\bassume_init(?:_ref|_mut|_read)?\b",
         "uninitialised memory treated as a valid value"),
    Rule("SS005", "high",
         r"\.\s*get_unchecked(?:_mut)?\s*\(|\.\s*unwrap_unchecked\s*\(|"
         r"\bfrom_utf8_unchecked(?:_mut)?\s*\(|\.\s*unchecked_(?:add|sub|mul|div|shl|shr)\s*\(",
         "unchecked operation: an out-of-range value is undefined behaviour"),
    Rule("SS006", "high",
         r"\bptr::(?:copy|copy_nonoverlapping|read|write|read_unaligned|write_unaligned|swap)\s*\(",
         "raw pointer copy/read/write with no bounds check"),
    Rule("SS007", "high",
         r"\b(?:Box|Rc|Arc|CString|Vec|String)::from_raw(?:_parts)?\s*\(|\bCStr::from_ptr\s*\(",
         "ownership taken from a raw pointer: double free or use-after-free if wrong"),

    # ---- medium: worth a reviewer's eyes ------------------------------------
    Rule("SS101", "medium",
         r"\bunsafe\s*\{",
         "unsafe block: the compiler's memory-safety checks are off inside it"),
    Rule("SS102", "medium", None,
         "index computed with arithmetic: panics (or wraps) on short input",
         ast="index_arith"),
    # Low, not medium: on the training split it fires on 721 functions with
    # LOWER precision than the base rate. It is shown for context only.
    Rule("SS103", "low",
         r"\.\s*unwrap\s*\(\s*\)|\.\s*expect\s*\(",
         "unwrap/expect panics on bad input instead of returning an error"),
    Rule("SS104", "medium",
         r"\bstatic\s+mut\b",
         "mutable static: data race unless every access is synchronised"),

    # ---- low: context for the reviewer --------------------------------------
    Rule("SS201", "low", None,
         "indexing by a variable with no length check anywhere in the function",
         unless_any=(r"\.\s*len\s*\(\s*\)", r"\.\s*get(?:_mut)?\s*\("),
         ast="index_plain"),
)

_COMPILED = [(r, re.compile(r.pattern) if r.pattern else None,
              [re.compile(u) for u in r.unless_any])
             for r in RULES]


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    severity: str
    line: int               # 1-based line in the scanned file
    message: str
    snippet: str            # the matched text, trimmed

    def to_dict(self) -> dict:
        return {"rule": self.rule_id, "severity": self.severity,
                "line": self.line, "message": self.message,
                "snippet": self.snippet}


def _parse(source: str):
    raw = source.encode("utf-8", errors="replace")
    try:
        return raw, Parser(RUST).parse(raw)
    except Exception:
        return raw, None


def _blank_non_code(raw: bytes, tree) -> bytes:
    buf = bytearray(raw)
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in _NON_CODE_NODES:
            for i in range(node.start_byte, node.end_byte):
                if buf[i] not in (0x0A, 0x0D):
                    buf[i] = 0x20
            continue
        stack.extend(node.children)
    return bytes(buf)


def code_only(source: str) -> str:
    """
    Return `source` with every comment and string literal replaced by spaces.

    Newlines are kept, so line numbers and byte offsets are unchanged. If the
    parser fails for any reason, the original text is returned: a rule firing
    inside a comment is a smaller problem than a scanner that crashes.
    """
    raw, tree = _parse(source)
    if tree is None:
        return source
    return _blank_non_code(raw, tree).decode("utf-8", errors="replace")


def _contains(node, types: set[str]) -> bool:
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in types:
            return True
        stack.extend(n.children)
    return False


def _index_nodes(tree, kind: str):
    """
    Yield index_expression nodes matching `kind`:
      index_arith   the index (or a range bound) is computed: buf[i + 1]
      index_plain   the index is a bare variable or field:   buf[i], buf[s.n]
    Constant indexes such as buf[0] are left alone.
    """
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        stack.extend(n.children)
        if n.type != "index_expression" or len(n.named_children) < 2:
            continue
        idx = n.named_children[1]
        if kind == "index_arith":
            if idx.type == "binary_expression" or (
                    idx.type == "range_expression"
                    and _contains(idx, {"binary_expression"})):
                yield n
        elif kind == "index_plain":
            if idx.type in ("identifier", "field_expression"):
                yield n


def check(source: str, first_line: int = 1) -> list[RuleHit]:
    """
    Run every rule over one function's source.

    first_line is the 1-based line where `source` starts in its file, so hits
    can be reported against the file rather than the function. Hits are
    returned most severe first, then in line order. Never raises.
    """
    try:
        raw, tree = _parse(source)
        code = _blank_non_code(raw, tree) if tree is not None else raw
        text = code.decode("utf-8", errors="replace")
    except Exception:
        return []

    hits: list[RuleHit] = []
    for rule, rx, unless in _COMPILED:
        if any(u.search(text) for u in unless):
            continue
        if rule.ast is not None:
            if tree is None:
                continue
            for n in _index_nodes(tree, rule.ast):
                snippet = code[n.start_byte:n.end_byte].decode(errors="replace")
                hits.append(RuleHit(rule.id, rule.severity,
                                    first_line + n.start_point[0],
                                    rule.message, " ".join(snippet.split())[:60]))
            continue
        for m in rx.finditer(text):
            line = first_line + text.count("\n", 0, m.start())
            hits.append(RuleHit(rule.id, rule.severity, line, rule.message,
                                " ".join(m.group(0).split())[:60]))

    rank = {s: i for i, s in enumerate(SEVERITIES)}
    hits.sort(key=lambda h: (rank[h.severity], h.line, h.rule_id))
    return hits


def worst(hits: list[RuleHit]) -> str | None:
    """The most severe level among `hits`, or None."""
    for s in SEVERITIES:
        if any(h.severity == s for h in hits):
            return s
    return None
