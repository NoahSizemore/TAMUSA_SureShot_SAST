"""
SureShot SAST - RustSec commit-mining harvester.

Builds a function-level labelled corpus from the RustSec advisory database
using the commit-mining methodology (cf. Devign / Big-Vul for C):

  for each advisory with a linked fix commit F:
      diff F^ .. F
      for each .rs function whose body changed:
          pre-fix  version -> label 1 (vulnerable)
          post-fix version -> label 0 (safe)
      untouched functions in the same files -> label 0 (negatives)

Fetching uses `git fetch --depth 2 <sha>`, which pulls only the fix commit and
its parent. This costs ~1-3 MB per advisory instead of a full clone, so the
whole database is harvestable on a laptop.

LABEL NOISE WARNING: a function is labelled vulnerable because it was modified
in a security fix, not because it was verified to contain the flaw. Fix commits
routinely carry refactors, test updates and unrelated cleanups. The filters
below remove the worst of it, but expect a meaningful false-positive rate in
the positive class. Hand-audit a random sample before you report headline
numbers.

Usage:
    python harvest/analyze_advisories.py data/raw/advisory-db
    python harvest/rustsec.py --limit 40 --workers 4
    python harvest/rustsec.py --include-prs          # full run, ~3x yield
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_rust as ts_rust

RUST = Language(ts_rust.language())
FUNC_QUERY = Query(RUST, "(function_item name: (identifier) @name) @func")

# --- filters -------------------------------------------------------------
MAX_FILES_IN_COMMIT = 10      # above this it is a refactor, not a fix
MIN_FUNC_LINES = 3            # drop trivial getters / one-liners
MAX_FUNC_CHARS = 20_000       # drop pathological generated code
EXCLUDE_DIR_RE = re.compile(r"(^|/)(tests?|benches?|examples?|fuzz)/")

PR_SHA_RE = re.compile(r"^From ([0-9a-f]{40}) ", re.MULTILINE)


# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> str:
    """Run a command, returning stdout. Raises on non-zero exit."""
    res = subprocess.run(
        cmd, cwd=cwd, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd[:4])} failed rc={res.returncode}: "
            f"{res.stderr.decode(errors='replace')[:300]}"
        )
    return res.stdout.decode(errors="replace")


def fetch_commit(owner: str, repo: str, sha: str, workdir: Path) -> Path:
    """Shallow-fetch a single commit and its parent into a scratch repo."""
    workdir.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "-q"], cwd=workdir)
    run(["git", "remote", "add", "origin",
         f"https://github.com/{owner}/{repo}"], cwd=workdir)
    run(["git", "fetch", "-q", "--depth", "2", "origin", sha],
        cwd=workdir, timeout=300)
    return workdir


def file_at(workdir: Path, rev: str, path: str) -> bytes | None:
    """Read a file's bytes at a given revision, or None if absent."""
    try:
        res = subprocess.run(
            ["git", "show", f"{rev}:{path}"], cwd=workdir,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60,
        )
        return res.stdout if res.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# function extraction
# --------------------------------------------------------------------------

def extract_functions(source: bytes) -> dict[str, str]:
    """
    Map function name -> source text. Tree-sitter recovers from syntax errors,
    so malformed files still yield their parseable functions.
    """
    if not source:
        return {}
    try:
        tree = Parser(RUST).parse(source)
    except Exception:
        return {}

    caps = QueryCursor(FUNC_QUERY).captures(tree.root_node)
    funcs = caps.get("func", [])
    names = caps.get("name", [])

    out: dict[str, str] = {}
    for fn_node in funcs:
        # the name node is the first name capture inside this function's span
        name = None
        for nm in names:
            if fn_node.start_byte <= nm.start_byte < fn_node.end_byte:
                name = source[nm.start_byte:nm.end_byte].decode(errors="replace")
                break
        if name is None:
            continue
        text = source[fn_node.start_byte:fn_node.end_byte].decode(errors="replace")
        if len(text) > MAX_FUNC_CHARS:
            continue
        if text.count("\n") + 1 < MIN_FUNC_LINES:
            continue
        # a name can repeat across impl blocks; keep the longest body
        if name not in out or len(text) > len(out[name]):
            out[name] = text
    return out


def normalise(text: str) -> str:
    """Strip whitespace and line comments so cosmetic edits do not count."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"//.*$", "", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# per-advisory harvest
# --------------------------------------------------------------------------

def harvest_one(entry: dict, scratch: Path) -> tuple[list[dict], str]:
    owner, repo, sha = entry["owner"], entry["repo"], entry["sha"]
    tag = f"{owner}/{repo}@{sha[:8]}"
    # `sha` may be a ref containing slashes (refs/pull/123/head), which would
    # create nested directories and collide across entries. Hash it instead.
    slug = hashlib.sha1(f"{owner}/{repo}/{sha}".encode()).hexdigest()[:16]
    workdir = scratch / slug

    try:
        fetch_commit(owner, repo, sha, workdir)
        # `sha` may be a literal SHA or a ref such as refs/pull/123/head;
        # FETCH_HEAD resolves both to the concrete commit just fetched.
        sha = run(["git", "rev-parse", "FETCH_HEAD"], cwd=workdir).strip()
        tag = f"{owner}/{repo}@{sha[:8]}"
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], f"SKIP {tag}: fetch failed ({str(exc)[:80]})"

    try:
        parent = run(["git", "rev-parse", f"{sha}^"], cwd=workdir).strip()
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], f"SKIP {tag}: no parent (root commit or shallow boundary)"

    try:
        all_changed = run(
            ["git", "diff", "--name-only", parent, sha], cwd=workdir
        ).split()
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], f"SKIP {tag}: diff failed ({str(exc)[:60]})"

    if len(all_changed) > MAX_FILES_IN_COMMIT:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], f"SKIP {tag}: {len(all_changed)} files changed (refactor)"

    rs_files = [
        f for f in all_changed
        if f.endswith(".rs") and not EXCLUDE_DIR_RE.search(f)
    ]
    if not rs_files:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], f"SKIP {tag}: no production .rs files changed"

    rows: list[dict] = []
    n_pos = 0
    for path in rs_files:
        pre_src = file_at(workdir, parent, path)
        post_src = file_at(workdir, sha, path)
        if pre_src is None or post_src is None:
            continue  # file added or deleted wholesale

        pre_funcs = extract_functions(pre_src)
        post_funcs = extract_functions(post_src)

        for name, pre_text in pre_funcs.items():
            post_text = post_funcs.get(name)
            if post_text is None:
                continue  # function removed by the fix; ambiguous, skip
            changed = normalise(pre_text) != normalise(post_text)

            if changed:
                n_pos += 1
                rows.append(_row(entry, parent, path, name, pre_text, 1, "prefix"))
                rows.append(_row(entry, sha, path, name, post_text, 0, "postfix"))
            else:
                # untouched function in a security-relevant file: a negative
                rows.append(_row(entry, sha, path, name, post_text, 0, "untouched"))

    shutil.rmtree(workdir, ignore_errors=True)
    if not rows:
        return [], f"SKIP {tag}: no matched function pairs"
    return rows, f"OK   {tag}: {n_pos} pos / {len(rows) - 2 * n_pos} neg"


def _row(entry, commit, path, name, text, label, role) -> dict:
    return {
        "repo": f"{entry['owner']}/{entry['repo']}",
        "commit": commit,
        "file": path,
        "function_name": name,
        "source": text,
        "label": label,
        "source_type": "rustsec",
        "pattern": role,
        "advisory": entry.get("advisory"),
        "package": entry.get("package"),
        "informational": entry.get("informational"),
    }


# --------------------------------------------------------------------------
# PR resolution (no GitHub API - uses the public .patch endpoint)
# --------------------------------------------------------------------------

def resolve_prs(db: Path, limit: int | None = None) -> list[dict]:
    """
    Advisories that link a PR rather than a commit roughly triple the usable
    advisory count (204 PR-only vs 89 direct-commit at time of writing).

    Rather than scraping the .patch endpoint (which redirects to
    patch-diff.githubusercontent.com and needs an extra host allowed), we emit
    a git ref. GitHub exposes every PR head at refs/pull/<n>/head, so the same
    `git fetch --depth 2` path used for direct commits resolves PRs too, with
    no API token, no rate limit and no additional hosts.

    This function does no network I/O; it only reads the advisory files.
    """
    import tomllib

    pr_re = re.compile(r"https?://github\.com/([\w.\-]+)/([\w.\-]+)/pull/(\d+)")
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for p in sorted(db.glob("*/*/RUSTSEC-*.md")):
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.startswith("```toml"):
            continue
        end = text.find("```", 7)
        try:
            meta = tomllib.loads(text[7:end])
        except Exception:
            continue
        adv = meta.get("advisory", {})
        if adv.get("informational") in ("unmaintained", "notice"):
            continue
        if re.search(r"/commit/[0-9a-f]{7,40}", text):
            continue  # already covered by the direct-commit path

        for owner, repo, num in sorted(set(pr_re.findall(text))):
            repo = repo.removesuffix(".git")
            ref = f"refs/pull/{num}/head"
            key = (owner, repo, ref)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "advisory": adv.get("id"), "package": adv.get("package"),
                "owner": owner, "repo": repo, "sha": ref, "is_pr": True,
                "informational": adv.get("informational"),
            })
            if limit and len(out) >= limit:
                return out
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest RustSec fix commits")
    ap.add_argument("--commits", type=Path,
                    default=Path("data/interim/fix_commits.json"))
    ap.add_argument("--db", type=Path, default=Path("data/raw/advisory-db"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/processed/rustsec_v0.1.parquet"))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap advisories processed (for quick runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-prs", action="store_true",
                    help="also resolve PR-linked advisories (~3x yield, slower)")
    ap.add_argument("--pr-limit", type=int, default=None)
    args = ap.parse_args()

    entries = json.loads(args.commits.read_text())
    print(f"direct-commit advisories: {len(entries)}")

    if args.include_prs:
        print("resolving PR-linked advisories via .patch endpoint ...")
        pr_entries = resolve_prs(args.db, args.pr_limit)
        print(f"  resolved {len(pr_entries)} additional fix commits from PRs")
        entries += pr_entries

    # de-duplicate on (owner, repo, sha)
    uniq, seen = [], set()
    for e in entries:
        k = (e["owner"], e["repo"], e["sha"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    entries = uniq[: args.limit] if args.limit else uniq
    print(f"processing {len(entries)} unique fix commits with {args.workers} workers\n")

    scratch = Path(tempfile.mkdtemp(prefix="sureshot_"))
    all_rows: list[dict] = []
    ok = skipped = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(harvest_one, e, scratch): e for e in entries}
            for i, fut in enumerate(cf.as_completed(futures), 1):
                try:
                    rows, msg = fut.result()
                except Exception as exc:
                    rows, msg = [], f"SKIP: worker crashed ({str(exc)[:70]})"
                all_rows.extend(rows)
                ok, skipped = (ok + 1, skipped) if rows else (ok, skipped + 1)
                print(f"[{i:3d}/{len(entries)}] {msg}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not all_rows:
        print("\nNo rows harvested.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    # drop exact duplicate function bodies (vendored/copy-pasted code)
    before = len(df)
    df = df.drop_duplicates(subset=["source", "label"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    pos = int(df.label.sum())
    neg = len(df) - pos
    print(f"\n{'=' * 60}")
    print(f"advisories: {ok} harvested, {skipped} skipped")
    print(f"rows: {len(df)} (dropped {before - len(df)} duplicate bodies)")
    print(f"positives: {pos}   negatives: {neg}   ratio 1:{neg / max(pos, 1):.1f}")
    print(f"repos: {df.repo.nunique()}")
    print(f"wrote {args.out}")
    if pos < 300:
        print(f"\nGATE: {pos} positives is below the 300 threshold. "
              f"Run with --include-prs, or take the fallback plan.")


if __name__ == "__main__":
    main()