# Recommended HISS submission package

## Files to upload (Editorial Manager)

Build the bundle with
`python paper/make_hiss_bundle.py --cls <sn-jnl.cls> --iicol --pdf <compiled PDF> --supplement-pdf <compiled supplementary_material.pdf>`.
It writes `paper/hiss_submission/` and `paper/hiss_submission.zip`, both git-ignored.

* `--iicol` produces the two-column layout HISS recommends (29 pages; compiled and checked on 2026-10-28).
* Without `--iicol` the bundle is single-column (37 pages), which is also acceptable.
* `--supplement-pdf` adds Online Resource 1 as `ESM_1.pdf` (16 pages). Compile `supplementary_material.tex`
  next to `sn-jnl.cls` and `results/figures/`, after the article, so that `paper/check_cross_references.py` can compare
  the two `.aux` files.
* Compile the bundle once and pass that PDF with `--pdf`, so the upload's compiled PDF matches its source.

| File | Item type in Editorial Manager | Notes |
|---|---|---|
| `fraud_detection_paper.tex` | Manuscript (LaTeX source) | Single file; references pasted in (no BibTeX step); figure names rewritten to `Fig<N>.pdf` |
| `sn-jnl.cls` | LaTeX supplementary/style file | Springer Nature template class v12, from the journal template download; not tracked in git |
| `Fig3.pdf` … `Fig7.pdf` | Figure | Vector PDFs with embedded TrueType fonts (Arial). Figs. 1–2 are TikZ drawings inside the `.tex` |
| `fraud_detection_paper_compiled.pdf` | Compiled manuscript PDF | Required alongside LaTeX source |
| `ESM_1.pdf` | Supplementary material (Online Resource 1) | Compiled from `supplementary_material.tex`; cited in the article as "Online Resource 1"; title page carries the article title, journal and authors |
| Cover letter | Cover letter | **Not sent:** the authors decided on 2026-10-27 to submit without one. The corresponding author has confirmed that the manuscript is not under consideration elsewhere and that the earlier version was not submitted anywhere (U-7); give the same answers if the submission form asks |

## Not uploaded, but referenced and available

| Material | Location |
|---|---|
| Code, tests, configurations, run records | Public GitHub repository, tag `hiss-submission-v1` (U-1) |
| Per-world results, tuning logs, test predictions | `results/` (predictions are git-ignored for size; regenerate with the commands in `research/README.md` or attach as an archive) |
| REFORMS checklist | `paper/reforms_checklist.md` |
| Generator datasheet | `research/DATASHEET.md` |
| Public data | Kaggle (CC0), checksums in `research/README.md` |

## Pre-upload checklist

1. U-3 and U-4 in `unresolved_issues.md` are confirmed (U-1, U-2 and U-7 are resolved), and every author has seen the final PDF.
2. `PY=.venv-research/bin/python; $PY -m pytest research/tests -q && $PY verify.py && $PY -m research.experiments.sync_manuscript --check && $PY paper/check_manuscript_numbers.py && $PY paper/order_references.py --check`
   all pass, and `$PY paper/check_cross_references.py <build dir>` passes after compiling both documents.
3. The manuscript compiles without errors, undefined references, undefined citations or overfull boxes:
   `tectonic -X compile fraud_detection_paper.tex` in a directory holding the `.tex`, `sn-jnl.cls` and
   `results/figures/`.
4. The abstract has 150–250 words with the four HISS headings, and there are 4–6 keywords.
5. The bundle compiles on its own: unzip it to an empty directory and compile.
