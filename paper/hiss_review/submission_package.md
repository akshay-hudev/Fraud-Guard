# Recommended HISS submission package

## Files to upload (Editorial Manager)

Build the bundle with
`python paper/make_hiss_bundle.py --cls <sn-jnl.cls> --iicol --pdf <compiled PDF>`.
It writes `paper/hiss_submission/` and `paper/hiss_submission.zip`, both git-ignored.

* `--iicol` produces the two-column layout HISS recommends (30 pages; compiled and checked).
* Without `--iicol` the bundle is single-column (37 pages), which is also acceptable.
* Compile the bundle once and pass that PDF with `--pdf`, so the upload's compiled PDF matches its source.

| File | Item type in Editorial Manager | Notes |
|---|---|---|
| `fraud_detection_paper.tex` | Manuscript (LaTeX source) | Single file; references pasted in (no BibTeX step); figure names rewritten to `Fig<N>.pdf` |
| `sn-jnl.cls` | LaTeX supplementary/style file | Springer Nature template class v12, from the journal template download; not tracked in git |
| `Fig3.pdf` … `Fig9.pdf` | Figure | Vector PDFs with embedded TrueType fonts. Figs. 1–2 are TikZ drawings inside the `.tex` |
| `fraud_detection_paper_compiled.pdf` | Compiled manuscript PDF | Required alongside LaTeX source |
| Cover letter | Cover letter | **To be written by the authors.** Suggested content: (1) scope fit (health information systems: payer claims data, audit decision support); (2) relation to Yoo et al. 2023 and the replication aspect; (3) disclosure of the earlier manuscript version if applicable (U-7); (4) code and data availability with the tagged repository version (U-1); (5) no competing interests; (6) not under consideration elsewhere |

## Not uploaded, but referenced and available

| Material | Location |
|---|---|
| Code, tests, configurations, run records | Public GitHub repository, after U-1 (push and tag) |
| Per-world results, tuning logs, test predictions | `results/` (predictions are git-ignored for size; regenerate with the commands in `research/README.md` or attach as an archive) |
| REFORMS checklist | `paper/reforms_checklist.md` |
| Generator datasheet | `research/DATASHEET.md` |
| Public data | Kaggle (CC0), checksums in `research/README.md` |

## Pre-upload checklist

1. U-1 to U-4 in `unresolved_issues.md` are resolved; all authors have approved.
2. `PY=.venv-research/bin/python; $PY -m pytest research/tests -q && $PY verify.py && $PY -m research.experiments.sync_manuscript --check`
   all pass.
3. The manuscript compiles without errors, undefined references, undefined citations or overfull boxes:
   `tectonic -X compile fraud_detection_paper.tex` in a directory holding the `.tex`, `sn-jnl.cls` and
   `results/figures/`.
4. The abstract has 150–250 words with the four HISS headings, and there are 4–6 keywords.
5. The bundle compiles on its own: unzip it to an empty directory and compile.
