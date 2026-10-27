# Novelty audit and literature comparison (2026-10-25)

Every work below was checked against its registered metadata (Crossref; arXiv pages for preprints and
NeurIPS/ICLR papers) and against the article itself where it could be read. Claims attributed to a work are
limited to what was verified; "not verified" marks what could not be read. Search date: 2026-10-25.

## 1. Closest works

| Work (verified record) | Research question | Data and labels | Split / evaluation protocol | Methods compared | Main finding | Limitations relevant to us | How our study differs | Genuine contribution or implementation difference? |
|---|---|---|---|---|---|---|---|---|
| **Yoo, Shin, Kyeong (2023)**, *IEEE Access* 11:88278–88294, doi:10.1109/ACCESS.2023.3305962 | Do GNNs beat ML models given graph-centrality features for Medicare provider fraud? | Public Medicare inpatient/outpatient claims and beneficiary data with provider fraud flags, combined into one graph. By its description this is the Kaggle dataset we use; the authors' earlier conference paper used that dataset (Curtis et al. 2025) | Not verified (full text could not be retrieved; IEEE blocks automated access) | GNNs vs. ML models with graph-centrality features | ML with centrality features **outperformed the best GNN** in precision, recall and F1; GNN training took 250–300× longer (abstract) | One dataset; provider-level labels of undocumented origin; cannot vary or identify collusive typologies or test unseen rings | Controlled benchmark varying feature-level and structural ring signals independently; unseen-ring protocol; five heterogeneous GNNs plus hybrids under an equal tuning budget; mechanism and memorisation analyses; real-data part with provider-disjoint rotations and label-neighbourhood features | **Our real-data finding replicates theirs; it is not new.** The manuscript did not cite this paper. The controlled benchmark and mechanism analyses are what is new |
| **Mardani, Moradi (2024)**, *IEEE Access* 12:132786–132800, doi:10.1109/ACCESS.2024.3425892 | Graph attention network for provider fraud | Kaggle provider-fraud dataset (Curtis et al. 2025) | Not verified | GAT, graph transformer network (GTN), RF, XGBoost, LR | Per Curtis et al. (2025): GTN F1 0.53 vs. RF 0.52; graph and tree models performed similarly | Primary text not read; numbers taken from the review | We do not propose an architecture; we ask when structure helps | Different question; consistent evidence that graph models do not clearly beat tree ensembles on this dataset |
| **Mardani, Moradi, Asadpour (2026)**, *Complex Intell Syst* 12(9):220, doi:10.1007/s40747-026-02336-x | Does modelling interdependencies through graph embeddings improve fraud detection? | **10% uniform claim sample** of the Kaggle dataset (full data needed >100 GB memory); provider labels; also a financial payments dataset | 5-fold cross-validation; fold construction, provider disjointness and separation of tuning from testing not described; states "inductive learning" | GAF (edge-weighted GAT on meta-path graphs, hyperedge aggregation, autoencoder) vs. HAN, DOMINANT, CARE-GNN, RF, XGBoost | GAF F1 0.56, AUC 0.88; RF and XGBoost AUC 0.87; GAF highest F1 and recall | Sampling claims changes provider aggregates and topology; no tabular baseline with engineered relational features reported; code "not applicable"; no controlled typologies | Full dataset; provider-disjoint rotations; equal tuning budgets; graph-augmented tabular baselines; controlled regimes; unseen rings; released code and tests | Genuinely different question (when, not whether, structure helps). Numbers are **not comparable** (10% claim sample, unspecified folds) |
| **Lu et al. (2023)**, *BMC Med Inform Decis Mak* 23:62, doi:10.1186/s12911-023-02152-0 | Attributed heterogeneous network with hierarchical attention for health-insurance fraud | Two datasets from a Chinese municipal health-insurance bureau (2018); **patient-level** labels (e.g. repeated prescriptions, abnormal kidney-disease treatment) | Train/validation/test; ratios and entity disjointness not described | MHAMFD vs. metapath2vec, GCN, GAT, HAN | MHAMFD best (F1 about 6 points above other models on Medical-1) | **Graph-only baselines**: no tabular or tree models | We compare GNNs with tabular models with and without engineered graph features | Different evaluation question; supports our argument that GNNs are rarely compared with graph-augmented tabular baselines |
| **Hong et al. (2024)**, *Heliyon* 10(9):e30045, doi:10.1016/j.heliyon.2024.e30045 | Multi-channel heterogeneous graph structure learning (MHGSL) | Real health-insurance data (insurer not named in the abstract); patient-level | Not verified (full text blocked) | MHGSL vs. "existing methods" | Outperforms existing methods (abstract; no numbers) | — | As above | As above |
| **Muhammad et al. (2025)**, *Sci Rep* 15:41734, doi:10.1038/s41598-025-22910-6 | GNNs and explainers for activity-level claims | Private MENA payer data (0.5–6 M activities, balanced); labels are insurer **accept/reject** decisions | 80/20 split with validation hold-out; "no overlap"; entity disjointness not stated | HINormer, HybridGNN, RE-GraphSAGE vs. RF, SVM (small dataset only) | GNNs matched or surpassed RF/SVM (macro-F1 0.79–0.84) | Baselines without graph features; rejection ≠ confirmed fraud; data not public | Graph-augmented tabular baselines; entity-disjoint evaluation; public code and generator | Different setting; strengthens the case for graph-augmented baselines |
| **Vandervorst et al. (2026)**, *Data Min Knowl Disc* 40(5):78, doi:10.1007/s10618-026-01249-0 | Inductive gradient boosting on heterogeneous, dynamic graphs for insurance fraud | One open-source and one proprietary insurance-fraud dataset | Not verified | G-GBM (path-level feature concatenation into GBDT) vs. state of the art | On par with or better than the state of the art | Not healthcare-specific (per abstract) | We do not propose a model; we test when a simple graph-augmented GBDT suffices on controlled healthcare typologies | Independent support for graph-augmented tabular models |
| **Curtis et al. (2025)**, *J Big Data* 12:238, doi:10.1186/s40537-025-01295-3 | Review of ML classifiers for healthcare fraud | — | — | — | On the Kaggle dataset, reported F1 ranges from 0.49 to 0.90 across studies under differing conditions; significance testing rare; source code rarely shared | — | Our study reports uncertainty, multiplicity-corrected tests, and releases code, tests and run records | Motivates the evaluation design; not a competing contribution |

## 2. Graph versus tabular evidence outside healthcare

| Work | Finding (verified) | Relation to our study |
|---|---|---|
| Tang et al. (2023), GADBench, NeurIPS Datasets & Benchmarks | 29 models, 10 datasets: "tree ensembles with simple neighborhood aggregation can outperform the latest GNNs tailored for the GAD task" | Same pattern; no controlled typologies, not healthcare |
| Bazhenov, Platonov, Prokhorenkova (2025), GraphLand, NeurIPS 2025 Datasets & Benchmarks (arXiv:2409.14500) | GBDTs with graph-based input features can be "very strong baselines", "sometimes" | Same pattern on industrial graphs |
| Weber et al. (2019), Elliptic | Random forest on node features outperformed GCNs under a temporal split | Same pattern in AML |
| Altman et al. (2023), NeurIPS Datasets & Benchmarks | Synthetic AML transactions with known laundering typologies | Our generator is the analogous tool for healthcare provider rings, with independent feature and structure knobs |
| Platonov et al. (2023), ICLR | Duplicate nodes in Squirrel/Chameleon cause train–test leakage and change GNN rankings | Analogous to our recurring-ring-member leakage, which we quantify |
| Dal Pozzolo et al. (2018), *IEEE TNNLS* 29(8):3784–3797, doi:10.1109/TNNLS.2017.2736643 | Fraud detection evaluated under the operating conditions of an investigation team that checks a limited number of alerts | Justifies our precision and recall at a fixed investigation budget |

## 3. What is and is not novel

**Not novel; the manuscript must not claim it as new:**

* "GNNs do not beat tree ensembles given engineered neighbourhood features." This holds outside healthcare (GADBench,
  GraphLand, Elliptic) and **on the same public Medicare provider dataset** (Yoo et al. 2023; also Mardani & Moradi 2024
  as summarised by Curtis et al. 2025).
* Heterogeneous GNNs for health-insurance fraud (Lu et al. 2023; Hong et al. 2024; Muhammad et al. 2025).

**Defensibly new (no verified prior work does this):**

1. A **controlled** healthcare-claims benchmark in which the feature-level signal (δ) and the structural density (s)
   of collusive rings vary independently, with entity-disjoint ("emergent") test rings and automated leakage tests.
   We found no study that varies the two signal sources independently for patient–provider claim graphs.
2. **A map of conditions** under which relational structure adds value beyond per-entity aggregates (dense blocks,
   claims not anomalous on their own, ring members not already conspicuous by volume), traced over four ring volumes
   (16, 8, 4, 2 claims per month; Section 6.6 and Table 8 of the article). The value of structure peaks at
   intermediate volume and vanishes when rings file about one claim per member.
3. **Mechanism**: ring detection by GNNs is structural (counting); attention-based HGT needs degree encodings;
   message passing dilutes claim-level anomalies (isolated fraud).
4. **Quantified evaluation pitfalls**: recurring ring members inflate ring AP by up to 0.60 for an attribute-only
   model, by amounts that differ between methods; and generated worlds that share their background data make pooled
   tests anti-conservative unless the generator seed is the unit (Platonov et al. 2023 is the closest analogue for
   the first).
5. An evaluation of **two-stage routing** (GB → GNN) under validation-only selection.
6. A real-data check under a stricter protocol than prior work on the dataset (provider-disjoint rotations, equal tuning
   budgets, corrected tests, graph-augmented tabular and hybrid baselines), with a descriptive explanation of the null
   result (no label assortativity through shared beneficiaries). This is a **replication and extension** of
   Yoo et al. (2023), not an independent discovery.

**Changes made to the manuscript:** Yoo et al. (2023), Mardani & Moradi (2024), Mardani et al. (2026), Lu et al. (2023),
Muhammad et al. (2025), Curtis et al. (2025), Vandervorst et al. (2026), GraphLand (2025), Platonov et al. (2023),
Dal Pozzolo et al. (2018) and Nadeau & Bengio (2003) were added. The related work now has a paragraph on graph
learning for health-insurance fraud. The contributions list states the real-data result as consistent with prior work
on the same dataset. The research gap is narrowed to controlled conditions, mechanism and memorisation.
