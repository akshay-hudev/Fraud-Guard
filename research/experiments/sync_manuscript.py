"""Paste the generated tables (results/tables/*.tex) into the manuscript and its Online Resource, matched by label.

A table block in the manuscript is replaced when it carries the same \\label as a generated
table; a line ``%%TABLE:<label>`` marks where a new table goes. Prints which tables changed.

    python -m research.experiments.sync_manuscript [--check]
"""

from __future__ import annotations

import argparse
import re
import sys

from ..ledger import REPO_ROOT, RESULTS_DIR

MANUSCRIPT = REPO_ROOT / "fraud_detection_paper.tex"
SUPPLEMENT = REPO_ROOT / "supplementary_material.tex"
TABLES = {"tab:regime_ringap": "synthetic_ring_ap.tex", "tab:pooled": "synthetic_pooled.tex",
          "tab:regime_isoap": "synthetic_isolated_ap.tex", "tab:ablation": "ablation.tex", "tab:hgt_arch": "hgt_arch.tex",
          "tab:protocol": "protocol.tex", "tab:kaggle": "kaggle.tex", "tab:cascade": "cascade.tex",
          "tab:legacy": "legacy.tex", "tab:hyperparams": "hyperparameters.tex", "tab:lowvolume": "lowvolume.tex",
          "tab:dose": "dose_response.tex", "tab:kaggle_repeated": "kaggle_repeated.tex",
          "tab:kaggle_nested": "kaggle_nested.tex", "tab:lowvolume_shift": "lowvolume_shift.tex",
          "tab:modelseeds": "modelseeds.tex", "tab:regimetuned": "regimetuned.tex",
          "tab:shapes": "shapes.tex"}
BLOCK = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)


def sync(text: str) -> tuple[str, list[str]]:
    changed = []
    for label, name in TABLES.items():
        path = RESULTS_DIR / "tables" / name
        if not path.exists():
            continue
        new = path.read_text().strip()
        # A marker is a whole line, so that %%TABLE:tab:kaggle does not match %%TABLE:tab:kaggle_repeated.
        marker = re.compile(rf"^%%TABLE:{re.escape(label)}$", re.M)
        blocks = [m for m in BLOCK.finditer(text) if f"\\label{{{label}}}" in m.group(0)]
        if blocks:
            m = blocks[0]
            if m.group(0) != new:
                text = text[:m.start()] + new + text[m.end():]
                changed.append(label)
        elif marker.search(text):
            text = marker.sub(lambda _: new, text, count=1)
            changed.append(label)
    return text, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report tables that differ; exit 1 if any")
    args = ap.parse_args()
    differing = []
    for doc in [MANUSCRIPT, SUPPLEMENT]:
        if not doc.exists():
            continue
        text, changed = sync(doc.read_text())
        differing += [f"{doc.name}:{label}" for label in changed]
        if not args.check:
            doc.write_text(text)
    print("tables differing from results/tables:", differing or "none")
    if args.check:
        sys.exit(1 if differing else 0)


if __name__ == "__main__":
    main()
