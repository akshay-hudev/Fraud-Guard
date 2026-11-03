"""Map research_sources_sha256 values in run records to commits whose research/*.py tree has that hash."""
import hashlib, json, subprocess
from pathlib import Path, PurePosixPath
def git(*a): return subprocess.check_output(["git", *a], text=False)
commits = git("rev-list", "--all").decode().split()
by_hash = {}
for c in commits:
    names = [n for n in git("ls-tree", "-r", "--name-only", c, "research").decode().split("\n") if n.endswith(".py")]
    h = hashlib.sha256()
    for n in sorted(names, key=PurePosixPath):
        h.update(n.encode() + git("cat-file", "-p", f"{c}:{n}"))
    by_hash.setdefault(h.hexdigest()[:16], []).append(c[:7])
want = {}
for f in Path("results").rglob("*.json"):
    try: p = json.load(open(f)).get("provenance", {})
    except Exception: continue
    g = p.get("git", {}) if isinstance(p, dict) else {}
    s = g.get("research_sources_sha256")
    if s: want.setdefault(s, set()).add((f.parts[1], g.get("commit", "")[:7], g.get("dirty")))
for s, where in sorted(want.items()):
    match = by_hash.get(s)
    print(s, "->", ("matches commits " + ",".join(match[:4])) if match else "NO COMMITTED TREE MATCHES", "|", sorted(where)[:4])
