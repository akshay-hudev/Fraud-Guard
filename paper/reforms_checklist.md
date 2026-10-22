# REFORMS checklist (Kapoor et al., Science Advances 2024) — mapping to this study

Section numbers refer to `fraud_detection_paper.tex`; code paths to the repository.

| Module | Item (abridged) | Where addressed |
|---|---|---|
| 1. Study design | Population / scientific question stated | §1 (question: when does relational structure help claims-fraud detection), §3 (synthetic population), §5 (real population: 5,410 providers) |
| | Motivation for ML and for each model family | §1, §2, §4.3 |
| 2. Computational reproducibility | Code, data access, environment, seeds | §10; `research/README.md` (pinned `requirements.txt` / `requirements-lock.txt`, data checksums, commands); seeds fixed (generator 1–6, development 901/902, model seed 0, folds seed 0, tuning split seed 1000) |
| | Run-level provenance | every `results/**/runs/*.json` records git commit, config hash, data hashes, library versions |
| 3. Data quality | Source, collection, labels, missingness | §3.1 + `research/DATASHEET.md` (synthetic); §5.1 (Kaggle CC0, label provenance undocumented — limitation §9.5) |
| | Sample sizes / class balance | §6 (test periods: 4,924 claims, 384 ring, 133 isolated); §5 (506/5,410 fraudulent providers) |
| 4. Preprocessing | Exclusions, transformations, feature construction | §4.1, §4.2, §5.2; `research/synth/features.py`, `research/realdata/kaggle.py` |
| 5. Modelling | All models, tuning procedure, budgets, selection criterion | §4.3, Appendix (Table of selected hyperparameters); equal budget, validation AP, development worlds never evaluated |
| | Baselines and controls | tabular with/without graph features, MLP control (no message passing), unsupervised overlap score, hybrids |
| 6. Data leakage | Train/test separation; no test information in training | audit periods analysed separately (§3.2); emergent rings entity-disjoint; provider-disjoint folds (§5.3); training-only standardisation and label features; tests in `research/tests/test_validity.py` (§10) |
| | Leakage demonstrations | §7.2 (identifier leakage, one-directional graph), §7.3 (entity memorisation) |
| 7. Metrics and uncertainty | Metric choice justified for imbalance | §4.4 (average precision primary; F1 at validation threshold; ROC-AUC; budgeted recall in supplementary) |
| | Uncertainty and tests | 95% t-intervals over worlds/folds; paired t-tests, Holm within pre-specified families; Wilcoxon robustness (supplementary CSVs) |
| 8. Generalisability and limitations | External validity, threats | §8 (real data), §9.3, §9.5 (generator simplifications, prevalence, single tuning regime, label provenance, tuning-split overlap) |
