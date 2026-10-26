# Reference verification (2026-10-25)

Every entry in the manuscript's bibliography (52 entries, numbered by first citation as `sn-vancouver-num` requires)
was checked against a registry record. The check covered authors, title, venue, volume, issue, pages or article
number, and year. Where the manuscript attributes a specific claim to a work, that claim was also checked against
the work itself.

**Methods.**

* **Crossref.** `https://api.crossref.org/works/<doi>` was queried for every DOI (script in the session scratchpad,
  output reproduced below in summary).
* **DataCite.** Used for arXiv DOIs (`10.48550/arXiv.*`).
* **arXiv and conference pages.** Used for NeurIPS and ICLR papers without DOIs.
* **Claim checks.** Article pages and abstracts were read via the publisher site, OpenAlex or Semantic Scholar.

**Notation.** "Claim checked" means the sentence in the manuscript was compared with the source. "Metadata only"
means the reference is cited for a method or dataset rather than a specific result.

## New references added in this revision

| Key | Registry record | Claim in manuscript | Claim checked against |
|---|---|---|---|
| yoo2023 | Crossref 10.1109/ACCESS.2023.3305962: Yoo Y, Shin J, Kyeong S. *IEEE Access* 11:88278–88294 (2023) | ML with graph-centrality features outperformed GNNs in precision, recall and F1, training 250–300× faster | Abstract (Semantic Scholar, OpenAlex). Full text not retrievable (IEEE blocks automated download). The claim that it uses "public Medicare claims with provider fraud labels" follows the abstract's description |
| mardani2024 | Crossref 10.1109/ACCESS.2024.3425892: Mardani S, Moradi H. *IEEE Access* 12:132786–132800 (2024) | Graph transformer network F1 0.53 vs random forest 0.52 | Stated via the review (curtis2025), which the sentence also cites; primary text not read |
| mardani2026 | Crossref 10.1007/s40747-026-02336-x: Mardani S, Moradi H, Asadpour M. *Complex Intell Syst* 12(9):220 (2026) | 10% claim sample; F1 0.56; ROC-AUC similar to tree ensembles (0.87–0.88) | Article full text (Springer, open access) |
| lu2023 | Crossref 10.1186/s12911-023-02152-0: Lu J, Lin K, Chen R, Lin M, Chen X, Lu P. *BMC Med Inform Decis Mak* 23:62 (2023) | Meta-path attention over patients, doctors, departments, medicines; classifies patients; graph-only baselines | Article full text |
| hong2024 | Crossref 10.1016/j.heliyon.2024.e30045: Hong B, Lu P, Xu H, Lu J, Lin K, Yang F. *Heliyon* 10(9):e30045 (2024) | Graph structure learning for health-insurance fraud | Abstract (Europe PMC) |
| muhammad2025 | Crossref 10.1038/s41598-025-22910-6: Muhammad R, Tbaishat D, Nazir A, Yacoub S, AbdulRazek M, El-Enen MAA, Sahlol AT. *Sci Rep* 15:41734 (2025) | Heterogeneous GNNs including a graph transformer classify billed activities; GNNExplainer/PGExplainer; tabular baselines without relational features | Article full text (nature.com) |
| curtis2025 | Crossref 10.1186/s40537-025-01295-3: Curtis ED, Billion-Polak P, Khoshgoftaar TM, Furht B. *J Big Data* 12:238 (2025) | F1 0.49–0.90 across studies of the dataset; significance tests and code rare | Article full text |
| vandervorst2026 | Crossref 10.1007/s10618-026-01249-0: Vandervorst F, Deprez B, Verbeke W, Verdonck T. *Data Min Knowl Disc* 40(5):78 (2026) | Graph GBM on par with or better than the state of the art for insurance fraud | arXiv abstract (2510.05676 v2), which links the published DOI |
| bazhenov2025graphland | DataCite 10.48550/arXiv.2409.14500; arXiv comment "Accepted at NeurIPS 2025 (Datasets & Benchmarks Track)" | GBDTs with graph-based features are very strong, though not universally winning, baselines | arXiv abstract |
| platonov2023 | ICLR 2023 (iclr.cc virtual poster 11466; arXiv:2302.11640) | Duplicate nodes leaking between train and test strongly affected GNN performance on two benchmarks | arXiv abstract |
| dalpozzolo2018 | Crossref 10.1109/TNNLS.2017.2736643: *IEEE TNNLS* 29(8):3784–3797 (2018) | Investigators examine only a limited number of cases | Abstract (OpenAlex): "only a small set of transactions are timely checked by investigators" |
| nadeau2003 | Crossref 10.1023/A:1024068626366: *Machine Learning* 52(3):239–281 (2003) | Corrected resampled t-test, variance factor 1/J + n₂/n₁ | Not readable as primary text in this session. Formula confirmed by two independent secondary sources: the CRAN `correctR` vignette and the `MachineShop` documentation |

## Existing references: DOIs added

DOIs were added to 26 existing entries. For each, the Crossref (or DataCite) record matched the manuscript's
authors, title, venue and pages:

schlichtkrull2018, wang2019han, hu2020hgt, lv2021, kaufman2012leakage, kapoor2023leakage, bauder2017,
herland2018, johnson2019, thornton2013, akoglu2015, akoglu2010oddball, beutel2013copycatch, hooi2016fraudar,
wang2019fdgars, zhang2019player2vec, dou2020care, liu2021pcgnn, saito2015pr, walonoski2018synthea,
kapoor2024reforms, gebru2021datasheets, weber2019elliptic (arXiv), shchur2018pitfalls (arXiv), fey2019 (arXiv),
rossi2020tgn (arXiv). pourhabibi2020 already had its DOI.

## Existing references without a DOI

These are cited for methods or general findings and match their proceedings records:

* NeurIPS: grinsztajn2022, tang2023gadbench, altman2023aml, jesus2022baf, ke2017lightgbm, hamilton2017sage
* ICLR: xu2019gin
* PMLR: tang2022bwgnn
* JMLR: pedregosa2011, demsar2006
* *Scand J Stat* (no DOI): holm1979
* Dataset and web pages: kaggle2019, nhcaa

GADBench's claim, "tree ensembles with simple neighborhood aggregation can outperform the latest GNNs", was
re-checked against its arXiv abstract (29 models, 10 datasets).

## Corrections made

* None of the existing entries' metadata was wrong.
* One citation-adjacent number in the text was wrong and has been corrected: the Holm-adjusted p for SAGE+AG
  at δ = 0, s = 0.1 is 0.095, not 0.61 (0.61 is rounded from HGT+AG's 0.645).
