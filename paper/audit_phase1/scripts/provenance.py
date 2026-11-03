import json, sys, hashlib
from pathlib import Path
from collections import Counter, defaultdict
R = Path("results")
# current sources hash (same recipe as ledger._read_git_state)
h = hashlib.sha256()
for f in sorted(Path("research").rglob("*.py")):
    h.update(f.as_posix().encode() + f.read_bytes())
print("HEAD research_sources_sha256:", h.hexdigest()[:16])
for d in sorted(p for p in R.iterdir() if (p / "runs").exists()):
    stats = Counter(); commits = Counter(); src = Counter(); times = []
    for f in sorted((d / "runs").glob("*.json")):
        p = json.load(open(f))["provenance"]; g = p.get("git", {})
        commits[g.get("commit", "?")[:7]] += 1
        stats["dirty" if g.get("dirty") else "clean"] += 1
        src[g.get("research_sources_sha256", "none")] += 1
        times.append(p.get("created_utc"))
    print(f"\n{d.name}: n={sum(commits.values())} {dict(stats)} first={min(times)} last={max(times)}")
    print("   commits:", dict(commits.most_common(6)))
    print("   sources:", dict(src.most_common(6)))
for extra in ["cascade/runs", "latency/latency.json", "legacy/legacy_reanalysis.json", "tuning/best_hyperparameters.json",
              "kaggle/tuning/best_hyperparameters.json", "tuning_lowvolume/best_hyperparameters.json"]:
    p = R / extra
    if p.is_file():
        g = json.load(open(p))["provenance"]["git"]; print(extra, g, json.load(open(p))["provenance"]["created_utc"])
