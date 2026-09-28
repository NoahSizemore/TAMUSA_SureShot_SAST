"""Measure the realistic yield of the RustSec advisory-db before harvesting."""
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

DB = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/advisory-db")

COMMIT_RE = re.compile(r"https?://github\.com/([\w.\-]+)/([\w.\-]+)/commit/([0-9a-f]{7,40})")
PR_RE = re.compile(r"https?://github\.com/([\w.\-]+)/([\w.\-]+)/pull/(\d+)")


def parse(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("```toml"):
        return None, text
    end = text.find("```", 7)
    if end == -1:
        return None, text
    try:
        meta = tomllib.loads(text[7:end])
    except Exception:
        return None, text
    return meta, text[end + 3:]


stats = Counter()
repos_with_commits = Counter()
commit_rows = []

for p in sorted(DB.glob("*/*/RUSTSEC-*.md")):
    meta, body = parse(p)
    stats["total"] += 1
    if meta is None:
        stats["unparseable"] += 1
        continue
    adv = meta.get("advisory", {})
    info = adv.get("informational")
    if info in ("unmaintained", "notice"):
        stats[f"skip_{info}"] += 1
        continue
    if info == "unsound":
        stats["unsound_kept"] += 1
    stats["candidate"] += 1

    full = str(meta) + body
    commits = set(COMMIT_RE.findall(full))
    prs = set(PR_RE.findall(full))

    if commits:
        stats["has_commit"] += 1
        for owner, repo, sha in commits:
            repos_with_commits[f"{owner}/{repo}"] += 1
            commit_rows.append({
                "advisory": adv.get("id"),
                "package": adv.get("package"),
                "owner": owner,
                "repo": repo.removesuffix(".git"),
                "sha": sha,
                "informational": info,
                "categories": adv.get("categories", []),
            })
    elif prs:
        stats["has_pr_only"] += 1
    else:
        stats["no_fix_link"] += 1

print("=== advisory stats ===")
for k, v in stats.most_common():
    print(f"{k:22s} {v}")
print(f"\nunique (advisory,commit) pairs: {len(commit_rows)}")
print(f"unique repos with commits:     {len(repos_with_commits)}")
print("\n=== top repos by commit count ===")
for repo, n in repos_with_commits.most_common(15):
    print(f"  {n:3d}  {repo}")

import json
Path("data/interim").mkdir(parents=True, exist_ok=True)
Path("data/interim/fix_commits.json").write_text(json.dumps(commit_rows, indent=1))
print("\nwrote data/interim/fix_commits.json")
