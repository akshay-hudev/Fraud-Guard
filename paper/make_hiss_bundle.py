"""Build the HISS upload bundle: manuscript source, class file, figures named Fig<N>, compiled PDF.

    python paper/make_hiss_bundle.py --cls <path to sn-jnl.cls> [--pdf <compiled manuscript PDF>]

HISS asks for figure files named "Fig" plus the figure number (e.g. Fig1.eps). The repository keeps descriptive
names under results/figures/; this script copies each included figure as Fig<N>.pdf, where N is the figure's
number in the manuscript (TikZ figures drawn inside the .tex are counted but need no file), and rewrites the
\\includegraphics commands to match. The Springer Nature class file is not tracked in the repository (it is
distributed with the journal's LaTeX template); pass the copy from the template download.

Output: paper/hiss_submission/ and paper/hiss_submission.zip (both git-ignored). The Online Resource
(supplementary_material.tex) is uploaded as a PDF, ESM_1.pdf.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "fraud_detection_paper.tex"
FIGURES = ROOT / "results" / "figures"
OUT = ROOT / "paper" / "hiss_submission"
INCLUDE = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"


def figure_files(tex: str) -> dict[str, str]:
    """Map each included figure file to Fig<N>.pdf, N = position of its figure environment in the text."""
    mapping = {}
    for n, block in enumerate(re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", tex, re.S), start=1):
        for name in re.findall(INCLUDE, block):
            mapping[name] = f"Fig{n}.pdf"
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", type=Path, required=True, help="sn-jnl.cls from the Springer Nature LaTeX template")
    ap.add_argument("--pdf", type=Path, help="compiled manuscript PDF to include")
    ap.add_argument("--supplement-pdf", type=Path,
                    help="compiled supplementary_material.tex, added as ESM_1.pdf (Online Resource 1; HISS wants PDF)")
    ap.add_argument("--iicol", action="store_true",
                    help="two-column layout recommended by HISS: [iicol] class option, every float spans both columns")
    args = ap.parse_args()

    tex = MANUSCRIPT.read_text()
    if args.iicol:
        tex = tex.replace(r"\documentclass[pdflatex,sn-vancouver-num]{sn-jnl}",
                          r"\documentclass[pdflatex,sn-vancouver-num,iicol]{sn-jnl}", 1)
        for env in ("figure", "table"):
            tex = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?", r"\\begin{" + env + r"*}[t]", tex)
            tex = tex.replace(r"\end{" + env + "}", r"\end{" + env + "*}")
        # Long URLs in the narrow reference column: break them at hyphens too, and allow looser lines.
        tex = tex.replace(r"\begin{thebibliography}", "\\sloppy\\def\\UrlBreaks{\\do\\/\\do\\-\\do\\.}\n\\begin{thebibliography}", 1)
    mapping = figure_files(tex)
    for old, new in mapping.items():
        tex = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)" + re.escape(old) + r"\}", lambda m: m.group(1) + new + "}", tex)
    tex = tex.replace(r"\graphicspath{{./}{results/figures/}}", r"\graphicspath{{./}}")

    staging = OUT.with_name(OUT.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / MANUSCRIPT.name).write_text(tex)
    shutil.copy(args.cls, staging / "sn-jnl.cls")     # copied before OUT is replaced, so --cls may point into it
    for old, new in mapping.items():
        shutil.copy(FIGURES / old, staging / new)
    if args.pdf:
        shutil.copy(args.pdf, staging / "fraud_detection_paper_compiled.pdf")
    if args.supplement_pdf:
        shutil.copy(args.supplement_pdf, staging / "ESM_1.pdf")
    if OUT.exists():
        shutil.rmtree(OUT)
    staging.rename(OUT)
    with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            z.write(f, f.name)
    for old, new in mapping.items():
        print(f"{new:9s} <- results/figures/{old}")
    print(f"bundle: {OUT.relative_to(ROOT)}/ and {OUT.with_suffix('.zip').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
