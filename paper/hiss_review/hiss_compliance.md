# HISS journal-compliance checklist (verified 2026-10-25)

**Journal:** *Health Information Science and Systems* (HISS), Springer Nature, hybrid publication model. This is the
journal; it is distinct from the HIS conference series and its proceedings.

**Sources read on 2026-10-25:**

* Submission guidelines: <https://link.springer.com/journal/13755/submission-guidelines>
* Aims and scope: <https://link.springer.com/journal/13755/aims-and-scope>

Both pages were read in full. The "editorial policies" URL returned 404, and the journal's peer-review model
(single- or double-anonymous) is not stated on either page. Quoted requirements are verbatim from the guidelines.

**Labels:**

* **M**: mandatory ("must", "required", "will be returned as incomplete").
* **R**: recommended ("we recommend", "should", "encouraged").

## 1. Scope

| Item | Verified text | Assessment |
|---|---|---|
| Aims and scope | Multidisciplinary journal integrating "artificial intelligence/computer science/information technology" with "health science and services". Scope point ii: "medical big data, medical/health/biomedicine information resources such as patient medical records, devices and equipments…"; iii: "data management, data mining, and knowledge discovery, all of which play a key role in decision making…"; iv: "development of new architectures and applications for health information systems." | **Plausible but not core fit.** Claims fraud detection on payer data is data mining of a health information resource for decision support. Fraud and insurance are not named, and no HISS article on claims fraud was found in a search on 2026-10-25. The revised introduction and discussion now state the health-information-systems relevance explicitly: claims data, audit workflows, investigation budgets and the cost of graph pipelines. A pre-submission enquiry to the editor is advisable (author decision). |

## 2. Manuscript format

| Requirement | Type | Verified text | Status |
|---|---|---|---|
| LaTeX with the Springer Nature template | R | "Manuscripts should be submitted in LaTeX." "We recommend using Springer Nature's LaTeX template and choosing the formatting option "[iicol]"." | Template used (`sn-jnl.cls` v12, `sn-vancouver-num`). The source stays single-column; `paper/make_hiss_bundle.py --iicol` builds the recommended two-column upload (27 pages after the second round, compiles cleanly) |
| Editable source files at every submission | M | "Failing to submit a complete set of editable source files will result in your article not being considered for review." | Bundle built by `paper/make_hiss_bundle.py`: `.tex`, `sn-jnl.cls`, figure files, compiled PDF |
| Title page: title, names, affiliations, corresponding e-mail | M | — | Present |
| ORCID iDs | R | — | **Missing**; authors to add |
| Structured abstract, 150–250 words: Purpose, Methods, Results, Conclusion | M | "Please provide a structured abstract of 150 to 250 words which should be divided into the following sections" | Present: 242 words (248 if each numeric range is counted as two words), with the four headings |
| 4–6 keywords | M | "Please provide 4 to 6 keywords which can be used for indexing purposes." | 6 keywords |
| Decimal headings, at most three levels | M | — | Two numbered levels; unnumbered run-in `\paragraph` headings only |
| Abbreviations defined at first mention | M | — | Checked; ROC-AUC and CI now defined at first use |
| Footnotes, not endnotes | M | — | None used |
| Supplementary information: cited in the text as "Online Resource" and numbered; each file includes the article title, journal name, author names and the corresponding author's affiliation and e-mail; each item has a concise caption; PDF for text | M (as read on 2026-10-25; the page redirected to a login on 2026-10-27, so it was not re-read) | — | `supplementary_material.tex` → `ESM_1.pdf` (10 pages); cited in the text as "Online Resource 1"; its title page names the article, the journal, and the authors with affiliation and e-mail addresses; the article's "Supplementary information" paragraph is its caption. Its cross-references to the article are checked by `paper/check_cross_references.py` |

## 3. References

| Requirement | Type | Verified text | Status |
|---|---|---|---|
| Numbered citations in square brackets, consecutive numbering | M | "Reference citations in the text should be identified by numbers in square brackets." "The entries in the list should be numbered consecutively." | `sn-vancouver-num`; the list is ordered by first citation (script check) |
| Only published or accepted works | M | — | arXiv preprints cited only where no archival version exists: Weber 2019, Shchur 2018, Fey 2019, Rossi 2020. GraphLand is cited as the NeurIPS 2025 paper with its arXiv identifier |
| DOIs as full links where available | R ("please always include") | "If available, please always include DOIs as full DOI links in your reference list" | **Added** for every reference with a Crossref DOI; each was verified against Crossref (`paper/hiss_review/reference_verification.md`) |
| Journal abbreviations (ISSN LTWA), or full title if unsure | M | "Always use the standard abbreviation of a journal's name according to the ISSN List of Title Word Abbreviations" | Full titles used, which the guidelines permit when unsure. Authors may abbreviate |

## 4. Tables and figures

| Requirement | Type | Status |
|---|---|---|
| Tables numbered and cited consecutively, with captions | M | Checked after revision |
| Figures cited in order, captions in the text file, Arabic numbering | M | Checked after revision |
| Figure file names "Fig" + number | M | "Name your figure files with "Fig" and the figure number, e.g., Fig1.eps." The repository keeps descriptive names; the submission bundle renames them (`Fig3.pdf` … `Fig7.pdf`). Figs. 1–2 are TikZ drawings inside the `.tex`. The Online Resource's three figures are inside `ESM_1.pdf` |
| Vector graphics with embedded fonts | M | Matplotlib PDFs with TrueType (Type 42) fonts embedded (`pdf.fonttype = 42`) |
| Lettering Helvetica/Arial, about 8–12 pt | R | Figures use DejaVu Sans at 6.5–8.5 pt. **Not changed**: a cosmetic change, best done once the layout ([iicol] or not) is decided |
| Colour that survives greyscale; no colour words in captions if printed in B&W | R | Every series has its own marker or line style. One caption (real-data figure) names colours; online colour is free of charge |

## 5. Statements and declarations

| Requirement | Type | Status |
|---|---|---|
| "Statements and Declarations" section before the references | M | Present |
| Funding | M | "did not receive support from any organization". **Authors confirm** |
| Competing interests (financial and non-financial, previous three years) | M | "no competing interests". **Authors confirm** |
| Ethics approval | M (human research) | Not applicable: synthetic data plus a public CC0 dataset with pseudonymous identifiers only. Wording made precise ("de-identified" replaced by what was verified) |
| Consent to participate / for publication | M where applicable | Not applicable; stated |
| Data availability | M (declarations) | Present; checksums in `research/README.md` |
| Code availability | M (support claims) | **Blocker:** the public repository (`origin/main` at `76ee3e9`) lacks 12 local commits, including the routing, HGT-ablation, re-analysis, nested-tuning and cost code and their result records. The authors must push and tag the submitted version |
| Author contributions (CRediT recommended) | R | Present, provisional. **All authors must confirm** |
| Use of LLMs documented in Methods | M | Section "Use of AI tools". **Authors confirm wording**, including this revision |
| Authorship criteria (ICMJE); no author changes after acceptance | M | Author list unchanged; **all authors must approve** |
| Single-journal submission; no salami slicing; raw data on request | M | **Confirmed 2026-10-27** by the corresponding author: not under consideration elsewhere; the earlier version was not submitted anywhere (U-7) |

## 6. Research data and reproducibility

| Item | Type | Status |
|---|---|---|
| Type 1 research-data policy (deposit encouraged) | R | Synthetic data regenerated from seeds; real data linked (CC0) with checksums |
| Cite datasets with DataCite fields | R | Kaggle dataset cited with creator, title, publisher, URL, licence |
| Data/code must support published claims | M | All tables regenerated from run records (`analyze.py`, `sync_manuscript --check`); see the reproducibility report |
| Reporting guideline | R (EQUATOR, as relevant) | No EQUATOR guideline fits a methods benchmark. REFORMS checklist (ML-based science) provided in `paper/reforms_checklist.md` |
