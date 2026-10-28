"""Order the manuscript's pasted reference list by first citation, as sn-vancouver-num numbering requires.

    python paper/order_references.py [--check]

Fails if a cited key has no \\bibitem; reports (and, without --check, removes) entries that are not cited,
because HISS lists only works cited in the text. Exit status 1 with --check if the order or the set differs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MANUSCRIPT = Path(__file__).resolve().parents[1] / "fraud_detection_paper.tex"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = MANUSCRIPT.read_text()
    head, rest = text.split(r"\bibcommenthead", 1)
    body, tail = rest.split(r"\end{thebibliography}", 1)
    entries = {}
    for block in re.split(r"\n(?=\\bibitem)", body.strip()):
        entries[re.search(r"\\bibitem\[.*?\]\{([^}]+)\}", block, re.S).group(1)] = block.strip()
    cited = []
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", head):
        for key in (k.strip() for k in group.split(",")):
            if key not in cited:
                cited.append(key)
    missing = [k for k in cited if k not in entries]
    if missing:
        sys.exit(f"cited without a \\bibitem: {missing}")
    unused = [k for k in entries if k not in cited]
    in_order = list(entries) == cited
    print(f"{len(cited)} cited; in order of first citation: {in_order}; not cited: {unused or 'none'}")
    if args.check:
        sys.exit(0 if in_order and not unused else 1)
    ordered = "\n\n".join(entries[k] for k in cited)
    MANUSCRIPT.write_text(head + "\\bibcommenthead\n\n" + ordered + "\n\n\\end{thebibliography}" + tail)


if __name__ == "__main__":
    main()
