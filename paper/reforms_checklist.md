# REFORMS checklist (Kapoor et al., Science Advances 2024) — mapping to this study

Section numbers refer to `fraud_detection_paper.tex` (revision of 2026-10-25); code paths to the repository.

| Module | Item (abridged) | Where addressed |
|---|---|---|
| 1. Study design | Population / scientific question stated | §1 (question: when does relational structure help claims-fraud detection), §3 (synthetic population), §5 (public provider-labelled data: 5,410 providers) |
| | Motivation for ML and for each model family | §1, §2, §4.3; routing as a deployment design §4.4 |
| 2. Computational reproducibility | Code, data access, environment, seeds | §10; `research/README.md` (pinned `requirements.txt` / `requirements-lock.txt`, data checksums, commands); seeds fixed (generator 1–6, development 901/902, model seed 0, folds seed 0, tuning split seed 1000) |
| | Run-level provenance | every `results/**/runs/*.json` records git commit, config hash, data hashes, library versions; new records also a hash of the research sources at process start. The original runs were made from an uncommitted tree; retraining GB-CAG, SAGE and HGT with the committed code reproduced their stored predictions on all 54 worlds (`results/cascade/runs/*.json`, `reproduction`; checked by `verify.py`) |
| 3. Data quality | Source, collection, labels, missingness | §3.1 + `research/DATASHEET.md` (synthetic); §5.1 (Kaggle CC0, label provenance undocumented — limitation §9.6) |
| | Sample sizes / class balance | §6 (test periods: 4,924 claims, 384 ring, 133 isolated); §5 (506/5,410 fraudulent providers) |
| 4. Preprocessing | Exclusions, transformations, feature construction | §4.1, §4.2, §5.2; `research/synth/features.py`, `research/realdata/kaggle.py` |
| 5. Modelling | All models, tuning procedure, budgets, selection criterion | §4.3, Appendix (Table A2); equal budget, validation AP, development worlds never evaluated; routing escalation fraction chosen on validation AP (§4.4) |
| | Baselines and controls | tabular with/without graph features, MLP control (no message passing), unsupervised overlap score, feature-level hybrids, two-stage routing (§6.7); low-volume stress test with frozen and re-tuned hyperparameters (§6.6) |
| 6. Data leakage | Train/test separation; no test information in training | audit periods analysed separately (§3.2); emergent rings entity-disjoint; provider-disjoint folds (§5.3); training-only standardisation and label features; routing uses validation statistics only; tests in `research/tests/` (§10) |
| | Leakage demonstrations | §7.5 (identifier leakage, one-directional graph), §7.6 (entity memorisation), §9.3 (re-analysis of the preliminary fixed dataset) |
| | Tuning leakage | real-data tuning split overlaps test folds; nested per-rotation selection for tabular and MLP models changes AP by ≤ 0.010; GNNs keep the shared selection, which is expected to favour them (§8) |
| 7. Metrics and uncertainty | Metric choice justified for imbalance | §4.5 (average precision primary; F1 at validation threshold; ROC-AUC; budgeted recall and precision; ring-level detection) |
| | Uncertainty and tests | 95% t-intervals; paired t-tests, Holm within pre-specified families (all ablations form one family); Wilcoxon robustness (supplementary CSVs). Statistical unit: the six generator seeds within a regime; pooled over regimes, seed means (a seed's nine regimes share legitimate claims, isolated fraud and ring members; §3.1, §4.5). Real data: corrected resampled t-test over the overlapping rotations (Nadeau & Bengio 2003) and a provider-level bootstrap with trained models fixed (§5.3, §8) |
| 8. Generalisability and limitations | External validity, threats | §6.6 (low ring volume), §8 (public data; provider labels do not validate ring detection), §9.2 (engineered features designed with knowledge of the typology), §9.3, §9.6 (generator simplifications, prevalence, volume signal, single tuning regime, seed-level power, one training seed per world, single-regime ablations, routing scope, cost-measurement scope, undocumented data and label provenance, tuning-split overlap for the GNNs) |
