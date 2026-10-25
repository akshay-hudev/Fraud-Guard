"""Paste the generated tables (results/tables/*.tex) into the manuscript, matched by label.

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
TABLES = {"tab:regime_ringap": "synthetic_ring_ap.tex", "tab:pooled": "synthetic_pooled.tex",
          "tab:regime_isoap": "synthetic_isolated_ap.tex", "tab:ablation": "ablation.tex", "tab:hgt_arch": "hgt_arch.tex",
          "tab:protocol": "protocol.tex", "tab:kaggle": "kaggle.tex", "tab:cascade": "cascade.tex",
          "tab:legacy": "legacy.tex", "tab:hyperparams": "hyperparameters.tex", "tab:lowvolume": "lowvolume.tex"}
BLOCK = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)


def sync(text: str) -> tuple[str, list[str]]:
    changed = []
    for label, name in TABLES.items():
        path = RESULTS_DIR / "tables" / name
        if not path.exists():
            continue
        new = path.read_text().strip()
        marker = f"%%TABLE:{label}"
        blocks = [m for m in BLOCK.finditer(text) if f"\\label{{{label}}}" in m.group(0)]
        if blocks:
            m = blocks[0]
            if m.group(0) != new:
                text = text[:m.start()] + new + text[m.end():]
                changed.append(label)
        elif marker in text:
            text = text.replace(marker, new)
            changed.append(label)
    return text, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report tables that differ; exit 1 if any")
    args = ap.parse_args()
    text, changed = sync(MANUSCRIPT.read_text())
    print("tables differing from results/tables:", changed or "none")
    if args.check:
        sys.exit(1 if changed else 0)
    MANUSCRIPT.write_text(text)


if __name__ == "__main__":
    main()
