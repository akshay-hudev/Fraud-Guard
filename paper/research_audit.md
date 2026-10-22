# Research Audit: "When Do Graph Neural Networks Outperform Tabular Models in Healthcare Fraud Detection? A Comparative Study with Heterogeneous Graph Transformers"

**Phase 1: audit only. No remediation has been performed.**

| Item | Value |
|---|---|
| Audit date | 2026-10-19 |
| Repository | `akshay-hudev/Fraud-Guard`, branch `streamlit-deployment`, HEAD `f16ccc7` |
| Manuscript audited | Working-tree (uncommitted) `fraud_detection_paper.tex`, 1,110 lines (diff vs HEAD: +1,050 / −331) |
| Scope | Source code, data, configs, tests, CI, logs, model artifacts, figures, references, manuscript, git history |
| Changes to repository | None. This file is the only addition. |
| Independent checks | Executed read-only against committed artifacts from a scratch directory outside the repo (Section 5; code in Appendix A) |

---

## How to read this document

**Severity**
- **Critical**: blocks submission. Either research integrity is at stake, or the validity of a headline claim is destroyed.
- **High**: invalidates or seriously undermines a main claim or method.
- **Medium**: weakens rigor or reproducibility; reviewers will very likely object.
- **Low**: presentation, wording, minor correctness.

**Confidence**
- **High**: directly verified by inspection and/or re-execution.
- **Medium**: strong inference from evidence.
- **Low**: plausible; needs confirmation.

**Evidence tags**
- **[VERIFIED]**: fact established in this audit.
- **[CONTRADICTED]**: manuscript claim contradicted by repository evidence.
- **[UNSUPPORTED]**: manuscript claim with no supporting evidence anywhere in the repository or its history.
- **[RISK]**: plausible threat to validity, not yet demonstrated.
- **[UNKNOWN]**: cannot be determined from the repository.
- **[verify]**: bibliographic or literature statement made from auditor knowledge. It must be checked against the publisher record before acting on it.

**References.** Paths are repo-relative. `tex:L123` means line 123 of the working-tree `fraud_detection_paper.tex`. `git:<sha>:<path>` means a file as of that commit.

---

## Table of contents

1. Executive summary
2. Scope, method, and limits of this audit
3. Critical blockers (B-01 to B-13)
4. Detailed audit by dimension
   - 4.1 Research contribution and novelty
   - 4.2 Data validity
   - 4.3 Leakage and validity
   - 4.4 Methodology
   - 4.5 Evaluation
   - 4.6 Reproducibility
   - 4.7 Manuscript quality
   - 4.8 Deployment claims (Section 5 and Appendix A)
5. Independent verification results
6. Arithmetic consistency of the manuscript's own tables
7. Evidence-to-claim map
8. Alternative explanations for apparent performance
9. Research-gap analysis and competing research
10. Proposed experiments, ranked
11. Reproducibility requirements
12. Publication strategy
13. Staged publication-readiness roadmap
14. Strongest defensible research direction
15. Complete issue register: mandatory fixes vs optional improvements
- Appendix A: Reproduction code for the audit checks
- Appendix B: Provenance timeline of the manuscript's numbers

---

## 1. Executive summary

**Verdict: the manuscript is not submittable, and it cannot be repaired by editing.** Its quantitative content has to be regenerated from new, corrected experiments. The research question is good and publishable if answered rigorously. The current evidence does not answer it.

### Principal findings

1. **The reported results have no generating evidence.** [VERIFIED] Every number in Tables 2–9, the hybrid-deployment claims, and the SHAP percentages lacks any code or output in the repository or its history that produced it.
   - Table 2's values appear verbatim as hard-coded literals in a since-deleted figure script (`git:26425e5:generate_paper_figures.py`). One example is `hgt_scores = [95.35, 90.03, 0.980]`.
   - RGCN, HAN, and SimpleHGN are not implemented anywhere.
   - The repository's own logs state that HGT was never run (`logs/statistical_tests.json`: "HGT not run").
   - The repository's actual logged results contradict the paper. Logged RF/GB F1 is 1.000; the paper reports 98.50/98.83.
2. **The manuscript's numbers are arithmetically impossible, regardless of the code.** [VERIFIED] GB's mixed-test recall (99.45%) exceeds both its isolated recall (99.30%) and its ring recall (72.40%). Mixed recall is a weighted average of the two subset recalls, so this cannot happen. In addition, 20 of the 21 subset recalls in Table 3 cannot occur as 5-seed means on the repository's test partition (Section 6).
3. **There is direct label leakage.** [VERIFIED] `ring_id` is non-zero only for ring-fraud claims. It is feature #4 of 27 in every tabular model, is the largest coefficient in the committed Logistic Regression, and is in the SHAP top-5.
4. **The dataset cannot test the hypothesis.** [VERIFIED]
   - Ring-fraud claims are drawn from the same inflated distribution as isolated fraud. They are not "individually borderline" as tex:L258-266 states.
   - A rule using three thresholds taken from the generator's code reaches F1 = 0.988 with zero false positives and 95.5% ring recall.
   - Gradient Boosting with only 3 raw claim features reaches F1 = 1.000 and ring recall = 1.000.
   - Tabular models are at ceiling, so no graph model can show an advantage.
5. **The implemented HGT cannot use graph structure.** [VERIFIED]
   - All edges are one-directional. Claim nodes receive messages only from patients, and patient nodes receive nothing.
   - In layer 2 this raises a `KeyError` that is silently swallowed, so the "2-layer" model is effectively one layer.
   - Randomizing doctor features, or deleting `treated_by` edges, changes claim logits by exactly **0.0**.
   - Every mechanistic claim (the `treated_by` ablation at −8.4 F1, the depth ablation, "coordination signal propagation") is impossible with this code.
6. **When the HGT is actually trained, it is equivalent to an MLP and is worse than GB.** [VERIFIED] This required bypassing an environment defect. HGT: F1 0.933 ± 0.021 vs GB 1.000; ring recall 0.981 ± 0.043 vs 1.000. Zeroing the HGT's only neighbor input does not change its performance.
7. **Evaluation and reproducibility are defective.** [VERIFIED]
   - The ring-recall evaluation code misaligns 144 of 420 test positions.
   - The committed environment cannot execute the GNN path (torch 2.2.2 vs NumPy 2.3.5 ABI clash; `NeighborLoader` dependencies missing).
   - Paper and code differ on features (27 vs 36), edge types (4 vs 3), stratification, hyperparameters, seeds, software versions, and deployment configuration.
8. **The bibliography has integrity problems.** At least one reference appears not to exist in the stated form ("Tiong et al. 2021"). Several have wrong author lists, or are cited for claims they do not make [verify].

### What is genuinely salvageable

- The research question: regime-dependence of relational models for provider-ring fraud versus isolated fraud.
- The train-only aggregation skeleton in `training/src/data/preprocessor.py`.
- Dataset SHA-256 logging. The logged hash matches the committed data [VERIFIED].
- The deployment system, as engineering. It is not a research contribution of this paper.

### Issue counts (Section 15)

| | Critical | High | Medium | Low | Total |
|---|---|---|---|---|---|
| Count | 12 | 21 | 19 | 9 | 61 |

There are 61 distinct issues: 48 mandatory and 13 optional. The register lists 63 IDs because two mandatory items (M-18, M-21) are merged into other blockers and counted only once.

---

## 2. Scope, method, and limits of this audit

### Inspected

- **Manuscript and notes**: the working-tree manuscript; `paper/temporal_gnn_additions.txt`.
- **Pipeline code**: everything under `training/`. This includes:
  - the generator (`training/generate_dataset.py`) and the older generator (`training/data/generate_synthetic.py`);
  - `src/data/preprocessor.py` and `src/data/graph_builder.py`;
  - `src/models/baseline.py` and `src/models/gnn.py`;
  - `src/training/train_gnn.py`;
  - `src/utils/metrics.py` and `src/analysis/shap_comparison.py`;
  - `scripts/run_pipeline.py` and `scripts/train_gnn.py`;
  - `threshold_analysis.py` and `train_real_world.py`;
  - tests.
- **Root wrappers**: the root-level wrapper scripts and `verify.py`.
- **Artifacts and logs**: all of `logs/`, `models/`, `data/raw/`, `data/processed/` (including `hetero_graph.pt`), `figures/`, and `training/models/`.
- **Deployment**: `backend/` (main API, predictors, drift, security, config), `frontend/app.py` (page structure and hard-coded metrics), `docker/`, `Dockerfile`.
- **Infrastructure**: `.github/workflows/*`, `requirements*.txt`, `.venv` package versions.
- **History**: git history, used to trace the origin of every headline number (Appendix B).

### Executed (read-only, outside the repository)

- Dataset hash verification; distributional and separability analysis; split composition; entity overlap.
- Inference with the committed models on the committed `X_test.npy`.
- Retraining of LR/RF/GB with the paper's stated hyperparameters: with and without `ring_id`, and with 3 raw features only.
- Temporal vs random split; "leaky" vs train-only aggregates.
- HGT information-flow tests using the repository's own `build_gnn_model`.
- HGT training with the repository's `GNNTrainer` (5 seeds). The only patch replaced `tensor.numpy()` calls, which crash in the committed environment.
- An alignment test of the repository's ring-recall evaluation code.
- A GRIM-style arithmetic feasibility analysis of the manuscript's tables.

### Not done, and why

- **The repository's pipeline was not run.** It overwrites `logs/`, `models/`, and `data/`, which would modify project state.
- **The manuscript was not compiled.** Its figure files do not exist in the repository.
- **No external bibliographic databases were queried in this session.** Statements about references are marked [verify] where they rest on auditor knowledge.
- **The authors were not interviewed.** Where the origin of a number is unknown, it is reported as [UNSUPPORTED], not as intentional.

---

## 3. Critical blockers

Each blocker lists severity, confidence, evidence, scientific impact, and corrective action.

### B-01: Reported results have no generating evidence (research integrity)

**Severity:** Critical · **Confidence:** High · [VERIFIED] / [UNSUPPORTED]

**Evidence: hard-coded literals in a deleted figure script**
- `git:26425e5:generate_paper_figures.py` (added 2026-04-20; deleted the same day in `432518b`, "Fixed the paper"):
  - L20–21: `before = [99.9, 99.9, 99.9, 99.9]` and `after = [94.55, 98.95, 99.05, 95.35]`.
  - L61–64: `lr_scores = [94.55, 90.73, 0.965]`, `rf_scores = [98.95, 98.50, 0.998]`, `gb_scores = [99.05, 98.83, 0.997]`, `hgt_scores = [95.35, 90.03, 0.980]`. These are exactly the Acc/F1/AUC values in Table 2 (tex:L476-490).
  - L104: `cm = np.array([[340, 10], [12, 58]])`, commented "estimated from 99.05% accuracy on 420 test samples". That matrix gives (340+58)/420 = **94.8%** accuracy, not 99.05%.

**Evidence: claimed experiments that were never implemented or run**
- **GNN baselines are not implemented.** A repository-wide search finds RGCN/HAN/SimpleHGN only as names in a "skipped" stub (`training/scripts/run_pipeline.py:254-266`). There is no `RGCNConv`, `HANConv`, or SimpleHGN code.
- **HGT never produced results.**
  - `logs/statistical_tests.json`: `"hgt_ring_recall_per_seed": []`, `"note": "HGT not run - install torch-sparse and re-run"`.
  - `models/comparison.json`: `"gnn_hgt": {}`.
  - `logs/model_results_table.csv`: empty `gnn_hgt` row.
  - `logs/ring_fraud_evaluation.json`: HGT = 0.0. This value comes from the fallback at `training/src/models/baseline.py:253-255`, not from a model.

**Evidence: the repository's logged results contradict the paper**

| Model | Repository logs | Paper Table 2 (F1) |
|---|---|---|
| LR | F1 0.9474, identical across all 5 seeds (std 0) | 90.73 ± 0.81 |
| RF | F1 1.0000 in every seed | 98.50 ± 0.29 |
| GB | F1 1.0000 in 4 seeds, 0.9922 in 1 | 98.83 ± 0.31 |

Sources: `models/baseline/*_metrics.json`, `logs/tabular_5seed_results.json`.

**Evidence: ablation and mechanism numbers written as string literals**
- `training/src/analysis/shap_comparison.py:84-99` writes `"delta_f1": -8.4 / -4.7 / -1.2` into `logs/shap_comparison.json`.
- `:105-112` writes the narrative "This explains HGT's +14.2pp recall advantage".
- `:156-157` draws hard-coded bars `[8.4, 4.7, 1.2, 10.6]` into `figures/fig_shap_comparison.pdf`.
- No ablation was ever executed.

**Evidence: latency and hybrid results**
- No benchmarking script exists.
- `training/threshold_analysis.py:28-29` hard-codes `GB_P50_MS = 30.0` and `HGT_P50_MS = 50.0`.
- `:77-79` substitutes GB scores for HGT scores when HGT predictions are missing. They are always missing.

**Evidence: HGT-like numbers exist only as mocks and hand-entered values**
- `training/src/utils/metrics.py:459-472`, `MOCK_RESULTS["gnn_hgt"]`: accuracy .9571, recall .8843, F1 .8971, AUC .9801. These are close to the paper's HGT row (95.35 / 88.45 / 90.03 / 0.980).
- `backend/main.py:449-455` hard-codes a mock HGT row.
- `frontend/app.py:199-203` and `:459-461` hard-code HGT metrics and `"best_model": "gnn_hgt"`.
- `training/models/comparison.json` contains `"precision": 0.9910`. Python's `json` module would emit `0.991`. The trailing zero suggests manual entry, and no generating script was found.

**Impact**
- Every result table (2–9), every result figure (Figs. 5–10), and the abstract, contributions, discussion, and conclusion lack an evidentiary basis.
- Submitting them would meet the definition of data fabrication in COPE guidance and Springer Nature's research-integrity policy, however the values originated (placeholders, drafting aids, earlier versions).

**Corrective action**
1. Remove every number, table, and figure not produced by a logged run.
2. Build an experiment ledger. Every table cell must be generated by a script from saved per-seed predictions, stamped with git SHA, dataset hash, and config hash.
3. Delete `MOCK_RESULTS` and every hard-coded metric from the code paths that feed figures, logs, or the UI.
4. All co-authors, including the supervising author, should review this finding before any further submission activity.

---

### B-02: Label leakage through `ring_id`

**Severity:** Critical · **Confidence:** High · [VERIFIED]

**Evidence**
- `training/generate_dataset.py:141` writes `ring_id` into every claim. Legitimate and isolated-fraud claims always get 0 (`:154`, `:163`); ring claims get 1–5 (`:173`).
- The preprocessor drop list (`training/src/data/preprocessor.py:212-216`) does not include `ring_id`.
- `data/processed/feature_names.json` lists `ring_id` at index 3. The same list is used by the deployed models (`training/models/baseline/feature_names.json`).
- In committed `X_test.npy`, every row whose scaled `ring_id` exceeds the minimum is fraud (21/21).
- The committed LR's largest |coefficient| is `ring_id` (5.55).
- `logs/shap_comparison.json` ranks `ring_id` 5th (RF) and 4th (GB).

**Impact**
- Every tabular ring-recall figure is invalid, as is the SHAP analysis.
- At serving time `ring_id` is filled with 0 for every input (`backend/production_predictor.py` aligns to `feature_names` and fills missing columns with 0.0). The served model therefore behaves differently from the evaluated one.

**Corrective action**
1. Exclude `ring_id`, `fraud_prone`, and every generator ground-truth column from features, using an explicit feature allowlist.
2. Store ground truth in a separate metadata table keyed by `claim_id`.
3. Add a unit test that fails if any blocklisted or label-derived column reaches `X`.

---

### B-03: The synthetic data contradicts the described design and cannot test the hypothesis

**Severity:** Critical · **Confidence:** High · [CONTRADICTED]

**Evidence: described vs actual generation**

| | Manuscript (tex:L249-266) | Code (`training/generate_dataset.py:118-127`) |
|---|---|---|
| Isolated fraud amount | U($15k–$50k) | U($4,500–$22,000) |
| Isolated fraud procedures | U(12–20) | 4–11 |
| Isolated fraud stay | U(10–25) days | 1–15 days |
| Ring fraud | U($8k–$20k), "individually borderline … insufficient to trigger tabular anomaly detectors" | Same distribution as isolated fraud |
| Legitimate claims | (not stated) | U($120–$8,500), procedures 1–5, stay 0–5 |

**Evidence: observed data** (`data/raw/claims.csv`)
- Median amount: ring $12,634, isolated $14,190, legitimate $4,187.
- Single-feature AUC, ring vs legitimate: amount 0.937, procedures 0.941.
- A rule taken from the generator's bounds, *amount > 8,500 ∨ procedures > 5 ∨ days > 5*, gives TP 369, FP 0, FN 9. That is precision 1.000, recall 0.976, F1 0.988, and ring recall 0.955.
- Retrained with the paper's hyperparameters on train only, GB reaches F1 = 1.000 and ring recall = 1.000 in each of these settings:
  - all 27 features;
  - 26 features (`ring_id` removed);
  - only `claim_amount`, `num_procedures`, `days_in_hospital`.

**Impact**
- The central claim is contradicted: tabular models fail on ring fraud and HGT gains +14.2 pp recall.
- With tabular models at ceiling, no regime boundary can exist in this dataset.

**Corrective action**
1. Redesign the generator so that ring-fraud claims are marginally indistinguishable from legitimate claims, or differ by a tunable effect size δ.
2. Put the detectable signal in structure: dense doctor–patient blocks, shared-patient overlap, lockstep timing, referral cycles.
3. Sweep the regime parameters (E3, E7) and document the generator in a datasheet.

---

### B-04: The HGT cannot access relational information, and its depth silently collapses

**Severity:** Critical · **Confidence:** High · [VERIFIED]

**Evidence: graph construction**
- `training/src/data/graph_builder.py:115-154` creates only `patient→claim` (filed_claim), `claim→doctor` (treated_by), `doctor→hospital` (works_at), and `patient→doctor` (visited). There are no reverse relations.
- Consequences:
  - The only relation *into* claim nodes is `filed_claim` from patients.
  - Patient nodes have no incoming relation.
  - Doctor and hospital information can never reach a claim node.

**Evidence: layer collapse**
- Under PyG 2.8.0 (the repository's venv), `HGTConv` returns outputs only for destination types: `{doctor, hospital, claim}`.
- Layer 2 therefore receives no `patient` entry and raises `KeyError: 'patient'`.
- That error is caught at `training/src/models/gnn.py:93-97`. The log message says "using initial projections", but the code actually keeps layer-1's output.
- Net effect: 2 layers ≡ 1 layer, and any depth ≥ 1 is identical.
- Under older PyG versions the symptom may differ (a crash, or `None` outputs). The information-flow limitation is version-independent.

**Evidence: information-flow tests** (repository model, eval mode)

| Intervention | max \|Δ claim logit\| |
|---|---|
| Randomize doctor + hospital features (×100 scale) | **0.0** |
| Delete `treated_by` edges | **0.0** |
| Delete `works_at` edges | **0.0** |
| Delete `visited` edges | **0.0** |
| Randomize patient features | 0.78 |

**Evidence: what the model reduces to**
- Each claim has exactly one patient, so the attention softmax over a single neighbor is 1.
- The HGT therefore reduces to an MLP over 9 claim features plus a linear transform of 4 patient-demographic features.
- Trained for 5 seeds: F1 0.933 ± 0.021. With patient features zeroed: F1 0.931 ± 0.016.

**Impact**
- Tables 5 and 6, the ablation heatmap, and the mechanism narrative (tex:L614-641, L822-836) describe effects that this code cannot produce.
- Any "HGT vs tabular" comparison is really an MLP with 13 inputs against trees with 27 inputs, including the leaked label.

**Corrective action**
1. Add reverse relations (`torch_geometric.transforms.ToUndirected()` or explicit `rev_*` edge types).
2. Delete the `try/except`.
3. Assert that every node type has an incoming relation.
4. Add a unit test: perturbing doctor features must change claim logits.
5. Consider doctor–doctor shared-patient edges or claim–claim meta-path edges as the ring channel.
6. Make graph construction time-respecting (see B-10).

---

### B-05: Claimed baselines and experiments do not exist

**Severity:** Critical · **Confidence:** High · [UNSUPPORTED]

None of the following has code or output anywhere in the repository or its history:

- RGCN, HAN, SimpleHGN (Tables 2, 3, 8; App. B).
- Depth, heads, edge-type, and feature-group ablations (Tables 4–7; Fig. 8).
- Temporal vs random split (Table 9; Fig. 7).
- The leaky-protocol run (Table 2, last row).
- Latency profiling over 1,000 requests (Table 8; Fig. 10; tex:L761-764).
- The hybrid pipeline (tex:L796-803).
- The ring-subgraph visualization with attention (Fig. 5).
- The isolated-vs-ring comparison (Fig. 4).

**Corrective action:** implement each experiment, or remove every claim tied to it (E5, E9, E15).

---

### B-06: Statistical claims were not performed, and the design is underpowered

**Severity:** Critical · **Confidence:** High · [CONTRADICTED] / [VERIFIED]

**Evidence**
- tex:L290-292 claims paired t-tests at α = 0.05. None exist: `logs/statistical_tests.json` reports `"t_statistic": "N/A"`, and the paper reports no p-values.
- Seeds are inconsistent: the paper states {42, 123, 456, 789, 1337}; the code uses `[0, 1, 2, 3, 4]` (`training/src/models/baseline.py:329`).
- The test set is small:
  - 420 claims with 64 positives: 43 isolated and 21 ring.
  - 3–6 ring claims per ring.
  - One ring claim equals **4.76 pp** of ring recall. The headline +14.2 pp is **three claims**.
- Seeds vary only model initialization. Data and split are fixed, so seed standard deviation understates true uncertainty.
- The reported confidence intervals use `1.96·sd/√5` (`training/scripts/run_pipeline.py:278-281`; `training/src/utils/metrics.py:104-114`). With n = 5 the correct critical value is t₀.₉₇₅,₄ = 2.776, so these intervals are too narrow by about 30%.

**Corrective action** (E6)
- Use ≥ 10 independently generated datasets × ≥ 5 model seeds.
- Report 95% CIs (t-based or bootstrap).
- Run paired tests across dataset replicates (Wilcoxon signed-rank or paired t) with Holm correction across comparisons.
- Report effect sizes and per-subset counts.

---

### B-07: The ring-recall evaluation code misaligns labels with predictions

**Severity:** Critical · **Confidence:** High · [VERIFIED]

**Evidence**
- These functions re-sort `features_raw.csv` by `claim_date` using pandas' default unstable sort, then align the result *positionally* with `X_test` / `y_test`:
  - `training/src/models/baseline.py:372-390` (`_load_test_claims`), used by `evaluate_five_seeds` and `evaluate_test`;
  - `training/scripts/run_pipeline.py:148-165`;
  - `training/threshold_analysis.py:48-63`.
- Verified result:
  - 144 of 420 test positions refer to a different claim after the re-sort.
  - The resulting "ring" mask selects 15 positions, which cover only 14 of the 21 true ring-fraud claims.
- Every logged ring metric is therefore computed on the wrong claims: `logs/ring_fraud_evaluation.json`, `logs/per_ring_recall.json`, `ring_recall` in `logs/tabular_5seed_results.json`, and `logs/threshold_sweep.json`.
- The graph builder's re-sort (`graph_builder.py:84-86`) likewise changes node order relative to the CSV. Test membership happens to be identical.

**Corrective action**
1. Carry `claim_id` through every artifact and align by ID, never by position.
2. Use `sort_values(["claim_date", "claim_id"], kind="stable")`.
3. Add a test that asserts ID alignment.

---

### B-08: The committed environment cannot run the GNN pipeline

**Severity:** Critical · **Confidence:** High · [VERIFIED]

**Evidence**
- `.venv` contains Python 3.12.13 and torch 2.2.2, which was built against NumPy 1.x, alongside NumPy 2.3.5.
  - Result: `tensor.numpy()` raises `RuntimeError: Numpy is not available`.
  - `train_gnn.py:41-43` calls `.numpy()`, so GNN training cannot complete in this environment.
- `NeighborLoader` is used by default (`USE_INDUCTIVE_MODE = True`, `gnn.py:19`; `train_gnn.py:113-130`). It raises "`NeighborSampler` requires either `pyg-lib` or `torch-sparse`".
- Declared versions disagree everywhere:

| Source | Python | PyTorch | PyG | scikit-learn |
|---|---|---|---|---|
| Paper (tex:L452-457) | 3.11 | 2.2.0 | 2.5.0 | 1.4 |
| `requirements_frozen.txt` | — | 2.10.0 | 2.7.0 | 1.8.0 |
| `.venv` | 3.12.13 | 2.2.2 | 2.8.0.post1 | 1.9.1 |
| `requirements.txt` | unpinned (`>=`) | | | |

**Corrective action**
1. Produce one lock file and one container (pinned image digest), and use it for every reported run.
2. Write library versions into every log.
3. Install `pyg-lib`, or drop `NeighborLoader` for this graph size.

---

### B-09: Paper and code disagree on core methodology

**Severity:** Critical · **Confidence:** High · [CONTRADICTED]

This blocker covers mismatches beyond B-02 and B-04.

| Aspect | Manuscript | Code / artifacts |
|---|---|---|
| Feature count | 36 in groups of 7/15/14 (tex:L296-318) | **27** (`data/processed/feature_names.json`), including `ring_id` |
| Procedure/diagnosis codes | Encoded features (tex:L300-301) | Dropped (`preprocessor.py:212-215`) |
| Doctor/hospital fraud rate | Train-computed features (tex:L305-307, L152-159) | Explicitly excluded (`preprocessor.py:126`, "NO fraud_rate") |
| Patient distinct doctors/hospitals; tenure; chronic flag | Features | Absent from the tabular set (tenure appears only as a graph node feature) |
| Group 3 ratios | 14 | 3 ratios (plus 3 calendar features) |
| Split | "stratified by fraud label within each partition" (tex:L277-279) | Purely positional (`preprocessor.py:68-91`). Fraud rate: train 19.0%, val 17.9%, test 15.2% |
| LR | L2, C = 1, max_iter 1000 | Committed: saga, **L1**, C = 1, max_iter 3000 |
| RF | 200 trees, depth 12 | Committed: **100 trees, depth 8** |
| GB | 150 / 0.1 / depth 5 / subsample 0.8 | Committed: **100 / 0.05 / depth 3 / subsample 0.9** |
| Tuning | Not mentioned | Grid search (`baseline.py:100-162`). Comment says "cv=5" but uses one `PredefinedSplit` fold. Validation F1 is 1.0 for RF and GB, so the search is degenerate (all ties) |
| Edge types | 3 relations, 6,300 edges (tex:L242-243, L418-425) | **4** relations, **6,200** edges |
| GNN mode | "transductive" (tex:L875-877) | Default inductive `NeighborLoader`, over a graph that still contains all nodes |
| Seeds | {42, 123, 456, 789, 1337} | `[0…4]` (tabular); no torch seeding anywhere |
| Hardware | i7-12700 (tex:L452); "4-core CPU" (Table 8 caption) | Unknown |

**Corrective action:** regenerate the Methods section from one versioned experiment configuration, cite that configuration in the paper, and have CI check that paper constants match the config.

---

### B-10: Entity leakage and transductive overlap; "newly formed rings" were never tested

**Severity:** Critical · **Confidence:** High · [VERIFIED]

**Evidence: test entities are already known from training**
- 100% of test claims' doctors and hospitals appear in training, as do 91.4% of patients.
- 100% of test ring claims are filed by doctors who already had ring-fraud claims in training. For patients the figure is 81%.
- Rings are static over the whole 24-month span.

**Evidence: the graph contains future information**
- The graph contains all validation and test claims.
- `visited` patient→doctor edges are built from all claims, including future ones (`graph_builder.py:146`).
- Claim features are standardized using statistics from all claims (`graph_builder.py:57-61`). The train-subset column means deviate by up to 0.19 SD.

**Impact**
- In this setup, "ring detection" means recognizing entities that are already labelled.
- The claim that HGT detects "ring members whose fraud_rate is not yet elevated (newly formed rings)" (`logs/shap_comparison.json`, `key_finding`) has never been tested.
- Once reverse edges are added, the future `visited` edges and future claim nodes become a look-ahead channel.

**Corrective action**
1. **Ring-disjoint inductive evaluation**: hold out entire rings, with their doctors and patients, for test.
2. **Temporal-emergence scenario**: rings begin after the training cutoff.
3. Build the training graph only from edges with t ≤ cutoff.
4. Give each test claim an inference subgraph containing only historical edges plus its own.
5. Fit all normalization on training data only.

---

### B-11: No real-world validation, and a mislabeled "real-world" artifact

**Severity:** Critical · **Confidence:** High · [VERIFIED]

**Evidence: the "real-world" result is synthetic**
- `logs/real_world_results.json` is labeled "real_world", but it is the synthetic test set ("Synthetic proxy (Kaggle CSVs unavailable)").
- It is produced by `training/train_real_world.py:56-98` and `:255-275`.
- Its `note` field contains a conclusion written before any result ("Directional comparison with GB F1=98.83% is consistent with CMS Medicare benchmark literature").
- The manuscript's statement "We validated our findings are directionally consistent with the CMS fraud literature" (tex:L871-873) has no basis.

**Evidence: the Kaggle pipeline would leak if run**
- It computes provider and patient aggregates on the *full* data before splitting (`train_real_world.py:179-205`).
- It broadcasts provider-level labels to every claim, while providers are shared across train and test. This is identity leakage.
- It tunes with shuffled `StratifiedKFold` on time-ordered data (`:287-291`).

**Corrective action**
1. Delete the manuscript claim and rename the artifact.
2. Fix the real-data pipeline: provider-disjoint grouped splits, train-only aggregates, and provider-level evaluation. Do this before generating any real-data result (E8).

---

### B-12: Citation accuracy

**Severity:** Critical · **Confidence:** Medium–High · [verify] for each item

| Key | Issue |
|---|---|
| `tiong2021` | "Tiong et al., Procedia Computer Science 181 (2021)": the auditor knows of no such paper. The title closely matches **Thornton, Mueller, Schoutsen & van Hillegersberg (2013), "Predicting healthcare fraud in Medicaid: a multidimensional data model and analysis techniques for fraud detection," *Procedia Technology* 9**. That paper is an OLAP / data-model study, not a "tripartite graph". Likely fabricated or garbled. |
| `wang2019spam` (FdGars) | Listed authors "X. Wang, D. Liu, J. Jing, X. Song". FdGars (WWW'19 Companion) is by Jianyu Wang, Rui Wen, Chunming Wu, Yu Huang, Jian Xiong [verify]. |
| `zhang2019` (Player2Vec) | Listed authors are wrong. Actual: Yiming Zhang, Yujie Fan, Yanfang Ye, Liang Zhao, Chuan Shi (CIKM 2019) [verify]. Also described as concerning "games"; the paper concerns underground forums. |
| `akoglu2015` | A survey, cited as demonstrating "Oddball". OddBall is Akoglu, McGlohon & Faloutsos (PAKDD 2010). |
| `liu2021` (PC-GNN) | Described as "transaction fraud", "+8–12 recall points when ring-like topology is present" (tex:L181-184), and "GNN improvements at >50,000 nodes" (tex:L847-848). PC-GNN evaluates on Yelp/Amazon review-fraud graphs. The auditor is not aware of these specific claims in it [verify]. |
| `bauder2017` | "Ensemble F1 0.82–0.91 on the CMS provider fraud dataset" (tex:L166-169). Not confirmed. LEIE-labelled Medicare studies by this group are extremely imbalanced and usually report AUC [verify the exact figures]. |
| `nhcaa2023` | "$10 billion annually" (tex:L93-94). NHCAA's published estimate is "tens of billions" (about 3% of health spending, with some estimates up to 10%) [verify; likely understated]. |
| `pourhabibi2020` | Article number "113278" [verify; auditor recollection is 113303]. |
| `lundberg2017`, `ying2019`, `fey2019` | In the bibliography but never cited. SHAP and PyG are used but not credited in the text. |

**Corrective action**
1. Check every reference against its DOI and publisher record, and add the DOIs.
2. Remove or replace anything that cannot be verified.
3. Cite tools where they are used.

---

### B-13: The abstract and conclusion contradict the paper's own tables

**Severity:** High · **Confidence:** High · [CONTRADICTED]

This is independent of B-01; the text is wrong even taken on its own terms.

- **"F1 98.83% on isolated claim fraud"** (tex:L66-67, L134). 98.83 is the *mixed* F1; Table 3's isolated F1 for GB is 99.10.
- **"Outperforming HGT by approximately 9 pp"** on isolated fraud (tex:L67-68, L810-812). Table 3's isolated gap is 5.3 pp; 8.8 pp is the mixed gap.
- **"<100 ms P99 for HGT"** (tex:L77-78). Table 8 reports a P99 of **140 ms**.
- **"≈18 ms mean latency"** for the hybrid (tex:L790-802, L900-902). With Table 8's P50s and 18% escalation, the expected mean is 21.3 ms if escalated claims run HGT only, or 22.8 ms if GB runs first.
- **"Recall parity with HGT"** for the hybrid. This is logically bounded by min(GB recall at the escalation threshold, HGT recall) and was never measured.
- **"Even 3-layer HGT with 8 attention heads does not close it"** (tex:L812-813). The one-factor ablations never contain that configuration.

**Corrective action:** rewrite the abstract and conclusion only after the new results exist, with each number cross-checked against generated tables.

---

## 4. Detailed audit by dimension

### 4.1 Research contribution and novelty

| # | Finding | Severity | Status / evidence | Action |
|---|---|---|---|---|
| RC-1 | **Problem significance is genuine.** Collusive provider rings are a recognized and costly typology, and the "regime" question is practically important. | — | Sound framing (tex:L91-130) | Keep the question. Strengthen the cost evidence with verifiable sources (NHCAA, DOJ/HHS-OIG takedown reports, GAO) [verify]. |
| RC-2 | **The research gap is overstated.** "No prior work systematically evaluates tabular vs. heterogeneous GNN … across controlled fraud typologies" (tex:L210-217). | High | Several bodies of work already address closely related questions. GADBench (Tang et al., NeurIPS 2023 D&B) systematically compares tree ensembles with GNNs for graph anomaly detection and finds trees with neighbor aggregation highly competitive. Weber et al. (2019, Elliptic) found RF outperforming GCN under a temporal split. Altman et al. (NeurIPS 2023 D&B) generate synthetic AML data with *controlled* injected laundering patterns, and compare GNNs with gradient-boosted trees. | Narrow the claim to what is new: healthcare claims, a heterogeneous patient–provider–facility schema, a controlled provider-ring typology, a feature-vs-structure regime map, and ring-disjoint inductive evaluation. Position explicitly against these works. |
| RC-3 | **"First ablation study for HGT-based healthcare fraud detection"** (tex:L138-141). | Medium | Unverifiable "first" claim; ablations not run (B-05) | Remove, or support with a documented systematic search (databases, query strings, dates). |
| RC-4 | **No methodological contribution.** HGT is used off the shelf and the cascade is standard; "HGT (ours)" (tex:L489, L536) misattributes an existing architecture. | Medium | — | This is acceptable for an empirical or benchmark paper, but then rigor must be exemplary. Rename to "HGT [Hu et al.]". |
| RC-5 | **The "Comparative benchmark … reproducible baseline" contribution** (tex:L142-145) has no code for 4 of its 7 models. | Critical | B-05 | Implement all models, or drop the contribution. |
| RC-6 | **The "deployment characterization" contribution** describes a hybrid that the deployed system does not implement. | High | §4.8 | Measure what is actually served, or drop it. |
| RC-7 | **The "leaky protocol" narrative** says that pre-split doctor fraud rates explain >99.9% (tex:L498-504). The current code has no fraud-rate feature. Our reproduction shows no inflation from leaky aggregates (0.997 vs 0.997), even when a doctor fraud-rate feature is added (0.998 vs 0.998). The historical leaky run (`git:b0636c5:models/comparison.json`) used a different dataset of about 10k claims with no rings, and its F1 values were 0.9946 / 0.9973 / 0.9973, not ">99.90". | High | [CONTRADICTED] | Remove, or re-run as a properly designed leakage case study on the redesigned data. |

### 4.2 Data validity

| # | Aspect | Finding | Severity | Status / evidence | Action |
|---|---|---|---|---|---|
| DV-1 | Provenance | Author-generated synthetic data. Generator, config, and SHA-256 are logged, and the hash matches the committed CSVs. | — | [VERIFIED] `logs/dataset_config.json` | Keep. Add generator version and parameters to every log. |
| DV-2 | Parallel datasets | A second, older dataset of 10,000 claims with no rings and 43 columns (`training/data/processed/*.csv`, from `training/data/generate_synthetic.py`) coexists with the 2,100-claim dataset. Several scripts read `training/data/processed` (e.g., `threshold_analysis.py:22`). Table 2's accuracy values trace to that older dataset (Appendix B). | High | [VERIFIED] | Keep one canonical data tree and delete or archive the stale one. |
| DV-3 | Licensing | No LICENSE file for code or data. | Medium | [VERIFIED] | Add a code license (e.g., MIT or Apache-2.0) and a data license (e.g., CC-BY-4.0). |
| DV-4 | Realism | Every distribution is uniform. Codes are random and unrelated to amount, specialty, or diagnosis. Dates are i.i.d. uniform with no seasonality. Hospitals are assigned to doctors at random. 40% of doctors (20/50) belong to rings. Fraud prevalence is 18%, with 8.5% of claims in rings. Real claim-level fraud labels are rare and noisy, and LEIE-labelled Medicare datasets are far below 1% positive [verify specific rates]. | High | [VERIFIED] | Calibrate to public summary statistics, or use realistic background claims (e.g., Synthea) with injected rings. Sweep prevalence. |
| DV-5 | Sample size | 2,100 claims. The test set has 420 claims, 64 fraud (43 isolated, 21 ring), with 3–6 ring claims per ring. | Critical (with B-06) | [VERIFIED] | Scale up (≥ 50–100k claims, ≥ 1k providers) and use dataset replicates. |
| DV-6 | Feature definitions | The paper's feature list does not match the code (B-09). | Critical | [CONTRADICTED] | Write one feature specification table, generated from code. |
| DV-7 | Label validity | Labels are generator flags: noise-free and static, with no partial ring membership and no delayed labels. `approved` is strongly label-correlated (35% for fraud vs 94% for legitimate); it is a post-adjudication field and is correctly dropped. `doctors.fraud_prone` equals ring membership for 20 of 21 flagged doctors. It is unused today but is a dangerous leakage vector. | Medium | [VERIFIED] | Document label semantics. Add label-noise and partial-membership variants. Blocklist `fraud_prone`. |
| DV-8 | Class imbalance | 18% fraud overall; split rates 19.0 / 17.9 / 15.2%. Accuracy is therefore uninformative. | Medium | [VERIFIED] | Make PR-AUC the primary metric. Report metrics at a fixed alert budget. |
| DV-9 | Duplicates | No duplicate claim rows or IDs. | — | [VERIFIED] | — |
| DV-10 | Temporal structure | Claim dates are drawn independently of label, entity, and ring, so the "temporal split" is distributionally equivalent to a random split. There is no drift, and no ring formation or dissolution over time. Quarterly fraud rate fluctuates randomly between 14.7% and 20.8%. All "temporal correctness" claims are therefore vacuous, and the temporal-vs-random comparison cannot measure anything. | High | [VERIFIED] | Simulate time-varying processes (ring onset, drift, audits), or drop temporal claims. |
| DV-11 | Boundary ties | Claims dated 2025-08-10 straddle the val/test boundary. | Low | [VERIFIED] | Split on date boundaries, with stable tie-breaking. |
| DV-12 | Self-inclusion in training aggregates | Training-split aggregates include the claim itself (`preprocessor.py:111-151`), while val/test use training aggregates only. In training, 36.3% of claims equal their patient's maximum amount; in test, 0.0% do. Median `amount_vs_patient_avg` for fraud is 1.47 in train vs 2.65 in test. This is a train/test feature shift, and a fraud claim inflates its own entity means (a mild target-correlated effect). | High | [VERIFIED] | Compute training aggregates out-of-fold, or from strictly earlier claims (expanding window). |
| DV-13 | Representativeness and generalizability | None established. One synthetic world; no external data. | Critical (with B-11) | [VERIFIED] | E8 (real data), E10 (robustness). |
| DV-14 | Privacy | Synthetic; no PHI. The deployment logs full claim features and client IPs (`backend/main.py` `PredictionRecord`). With real data this would be PHI and require HIPAA/DPDP-grade handling. | Low (paper) | [VERIFIED] | Add an ethics/privacy statement. Scope the deployment claims. |

### 4.3 Leakage and validity

| # | Leakage type | Finding | Severity | Status | Action |
|---|---|---|---|---|---|
| LK-1 | Target leakage | `ring_id` in the features (B-02) | Critical | [VERIFIED] | Allowlist plus test |
| LK-2 | Entity leakage | Same doctors, patients, and rings in train and test; "ring detection" amounts to memorization (B-10) | Critical | [VERIFIED] | Ring-disjoint splits |
| LK-3 | Temporal leakage | Minor today, because time carries no signal (DV-10). `visited` edges and claim nodes from the future are present in the training graph (B-10). | High (once reverse edges exist) | [VERIFIED] | Time-respecting graph |
| LK-4 | Preprocessing contamination | Scaler and encoders are fit on train ✓. Graph claim features are standardized over all nodes ✗ (`graph_builder.py:57-61`). Self-inclusion in training aggregates (DV-12). | Medium | [VERIFIED] | Train-only normalization; out-of-fold aggregates |
| LK-5 | Split / sampling | Not stratified despite the claim; positional split (B-09) | Medium | [VERIFIED] | Fix, or describe honestly |
| LK-6 | Graph-construction leakage | No labels in node features ✓. The transductive graph includes test nodes. Message passing is currently patient→claim only, so test claims do not influence training claims; that changes once reverse edges are added. | High (latent) | [VERIFIED] | Inductive protocol |
| LK-7 | Transductive vs inductive | The paper says transductive; the code defaults to inductive sampling over a full graph; the API claims "inductive inference" (`backend/main.py:309-318`), but the predictor ignores the subgraph and serves RF + rules (§4.8). | High | [CONTRADICTED] | Define and implement a single protocol |
| LK-8 | Information unavailable at deployment | `ring_id` (ground truth). Training-period aggregates for unseen entities. In production, aggregates are hard-coded constants (`backend/production_predictor.py:211-226`: `doctor_total_claims = 20`, `doctor_fraud_rate = 0.05`, …), so serving does not match training. | High | [VERIFIED] | Feature store with point-in-time aggregates, or narrow the claims |
| LK-9 | Leakage via tuning | Tabular models are refit on train+val after tuning ✓ (test untouched). HGT trains on train only. Different data budgets make the comparison unfair. | Medium | [VERIFIED] | Equal data budgets |
| LK-10 | Kaggle pipeline (latent) | Full-data aggregates, provider identity leakage, shuffled CV on temporal data (B-11) | Critical (before use) | [VERIFIED] | Fix before use |

### 4.4 Methodology

| # | Finding | Severity | Evidence | Action |
|---|---|---|---|---|
| ME-1 | HGT information flow and depth collapse | Critical | B-04 | Reverse edges, no `try/except`, flow tests |
| ME-2 | A single `LayerNorm` per layer is shared across all node types, applied after dropout | Low | `gnn.py:62-64, 91` | Per-type norms (`HeteroDictLinear`/`ModuleDict`), conventional ordering |
| ME-3 | `get_embeddings` lacks the fallback and would raise the same `KeyError` | Low | `gnn.py:103-118` | Fixed by ME-1 |
| ME-4 | **Baseline unfairness.** Tabular models get 27 features including the leak. HGT gets 9 claim features plus 4 patient demographics. There is no MLP (no-graph) control on identical inputs. There is **no tabular-plus-graph-features baseline** (neighborhood aggregates, shared-patient overlap, node embeddings). This is the decisive comparator for "when do GNNs outperform tabular"; GADBench shows such baselines are strong. | Critical | — | E5 |
| ME-5 | **Unequal tuning.** Tabular models are grid-searched; HGT uses fixed hyperparameters. The paper presents both as fixed. | High | `baseline.py:100-162` | Equal budgets on validation only (e.g., the same number of Optuna trials per family) |
| ME-6 | Graph schema lacks the relations that carry ring signal: doctor–doctor shared patients, claim–claim same-patient or same-doctor, temporal co-occurrence | High | `graph_builder.py` | Add and ablate them |
| ME-7 | The HAN meta-paths described (tex:L348-350) need reverse edges to exist | Medium | — | Implement with reverse relations |
| ME-8 | **Training protocol.** Early stopping on validation F1 with 75 validation positives is noisy. Cosine `T_max = epochs` interacts with early stopping. There is no torch, numpy, or python seeding. | Medium | `train_gnn.py:161-200` | Early-stop on validation PR-AUC or loss; seed everything; set deterministic flags |
| ME-9 | **Threshold selection.** Every model uses 0.5 / argmax. Class handling differs: LR and RF use balanced weights, GB uses none, HGT uses weighted cross-entropy. Recall at different implicit operating points is not comparable. | High | `baseline.py`, `train_gnn.py:34-45` | Choose thresholds on validation (e.g., fixed precision or a fixed alert budget). Report threshold-free metrics. |
| ME-10 | **Calibration claim is invalid.** "HGT achieves the highest AUC … suggesting better-calibrated probability estimates" (tex:L506-508). AUC is rank-based and says nothing about calibration. No calibration metric was computed. | High | — | Remove, or measure Brier score, ECE, and reliability diagrams |
| ME-11 | **Computational cost.** The paper's HGT training time is 82 s. Our full-batch CPU run averaged 2.4 s. Latencies were never measured. | High | §5.8 | Benchmark (E15) |
| ME-12 | **Hybrid cascade.** Not implemented. Its recall is upper-bounded by GB's recall at the escalation threshold. | High | §4.8 | Implement end-to-end, or drop |
| ME-13 | **HGT equations are inaccurate.** Eq. (2) omits the relation-specific attention matrix W^ATT_φ(e) and the prior μ, uses type-level h_t/h_s instead of node-level h_i/h_j, and does not describe per-head computation or normalization over all neighbors of the target. Eq. (3) does not match HGT's target-specific aggregation with A-Linear and the gated skip (tex:L362-386). | Medium | — | Use the formulation from Hu et al. (2020) and state PyG `HGTConv` specifics |
| ME-14 | **RGCN mischaracterized** as "homogeneous message-passing" (tex:L509-511). RGCN is relational, i.e., typed. | Low | — | Correct |

### 4.5 Evaluation

| # | Finding | Severity | Evidence | Action |
|---|---|---|---|---|
| EV-1 | **The evaluation subsets are ill-defined.** "Claims involving only isolated fraud" / "only ring fraud" (tex:L268-273) never says which negatives are included, so subset F1 and precision are undefined. No code computes subset F1. | High | — | Define, for example: ring-vs-legit uses positives = ring claims, negatives = all legitimate test claims, isolated excluded. Report counts. |
| EV-2 | **Wrong metric emphasis.** Accuracy is prominent; PR-AUC appears in the logs but not in the paper; there are no confusion matrices with counts and no precision@k or recall@fixed-FPR. | High | — | Primary metric PR-AUC, plus recall at a fixed alert budget, plus confusion matrices |
| EV-3 | **Uncertainty is understated.** Only standard deviation over 5 init seeds is reported, and the CI uses z instead of t. | High | B-06 | Dataset replicates; t- or bootstrap CIs |
| EV-4 | **No significance testing.** | Critical | B-06 | Paired tests with multiplicity correction |
| EV-5 | **No real ablations.** Ablations also lack controls (degree-preserving rewiring, feature shuffling, label-free structure). | Critical | B-05 | E9 |
| EV-6 | **No robustness or sensitivity analysis** (prevalence, ring size/count, camouflage, label noise, drift). | High | — | E10 |
| EV-7 | **No external validation.** | Critical | B-11 | E8 |
| EV-8 | **Ring evaluation is misaligned.** | Critical | B-07 | ID-based alignment |
| EV-9 | **The doctor-level ring metric is undefined in the paper.** Fig. 5 claims "HGT identifies all 5 ring-member doctors"; rings have 4 doctors; the logged metric is "≥ 50% of claims flagged". | High | `metrics.py:362-416`; tex:L561-572 | Define provider-level detection metrics formally |
| EV-10 | **Implausible standard deviations.** LR ± 0.62–1.10 across seeds is reported, but the repository's 5-seed LR results are identical (std 0), as is our lbfgs reproduction. | High | `logs/tabular_5seed_results.json` | Report real variability |

### 4.6 Reproducibility

| # | Finding | Severity | Evidence | Action |
|---|---|---|---|---|
| RP-1 | Environment cannot run the GNN; versions are inconsistent | Critical | B-08 | Lock file + container |
| RP-2 | **Execution paths are fragile.** Root wrappers `exec` training modules. Relative paths depend on the working directory: `graph_builder.py:74-83` reads `{processed_dir}/../../data/raw`, which resolves outside the intended tree depending on cwd. Preprocessing and graph construction can read *different* raw datasets. | High | — | One CLI entry point; repo-root-anchored paths |
| RP-3 | **Artifacts are stale and mutually inconsistent.** GB ring recall is 0.7143 in `ring_fraud_evaluation.json`, 1.0 in `tabular_5seed_results.json`, and per-ring 0.5–1.0 in `per_ring_recall.json`. The committed GB has test recall 1.0, which implies ring recall 1.0. The logs cannot be regenerated from the current state: `threshold_analysis.py` needs `training/data/processed/X_test.npy`, which does not exist. | High | [VERIFIED] | Regenerate everything from a clean run; never hand-edit logs |
| RP-4 | No torch seeds; no deterministic flags; per-seed predictions not saved | Medium | grep: no `manual_seed` anywhere | Seed all RNGs; save predictions |
| RP-5 | **Tests encode the desired conclusion.** `training/tests/test_all.py:374-379` asserts that MOCK GNN F1 beats the baselines. No test checks leakage, split integrity, ID alignment, or GNN information flow. | High | — | Delete mock-based tests; add validity tests |
| RP-6 | **CI cannot fail on scientific regressions.** `pytest … \|\| true` (`.github/workflows/deploy.yml:30`); `continue-on-error: true` (`test.yml:40-58`); "skipped" fallbacks in `model-validation.yml`. | Medium | — | Fail on error; add a reproduction job on a small config |
| RP-7 | `verify.py` checks only that files exist and have keys | Low | `verify.py` | Replace with value-level regeneration checks |
| RP-8 | **Data and code availability.** GitHub URL only; no license, no archived DOI, no tagged release matching the paper. | Medium | tex:L1000-1002 | Zenodo DOI for a tagged release; license |
| RP-9 | Leftover venue-specific remnants ("IJISA-scale", "IJISA-ready") | Low | `generate_dataset.py:4`; `metrics.py:209` | Clean up |
| RP-10 | Hardware unspecified or contradictory | Low | tex:L452, Table 8 | Report the actual hardware |

### 4.7 Manuscript quality

| # | Finding | Severity | Evidence | Action |
|---|---|---|---|---|
| MQ-1 | **Argument structure is clear** (question → typology → regime → mechanism → deployment). It is unsupported by evidence. | — | — | Keep the structure, but rebuild the content |
| MQ-2 | **Related work is missing key lines.** Graph anomaly detection benchmarks (GADBench); camouflage- and heterophily-aware fraud GNNs (CARE-GNN, PC-GNN as correctly described, BWGNN, GHRN, H2-FDetector); dense-subgraph and lockstep fraud detection (FRAUDAR, CopyCatch); temporal-split RF vs GCN precedent (Elliptic); synthetic AML pattern benchmarks (Altman et al.); tabular-vs-DL evidence (Grinsztajn et al.; Shwartz-Ziv & Armon); leakage literature (Kaufman et al.; Kapoor & Narayanan); healthcare fraud ML (Herland et al.; Johnson & Khoshgoftaar); healthcare graph fraud [verify specific works]. | High | §9 | Rewrite Related Work |
| MQ-3 | Citation inaccuracies | Critical | B-12 | Verify all |
| MQ-4 | Equations inaccurate | Medium | ME-13 | Fix |
| MQ-5 | **Figures are missing or unreproducible.** None of the 10 referenced image files exist in the repo; only `figures/fig_shap_comparison.pdf` exists, and its HGT panel is hard-coded. 9 of 10 figure labels are never referenced in the text. Fig. 5's caption contradicts the data ("five doctors" vs 4 per ring; "edge opacity indicates HGT attention", but no attention was computed). | High | — | Regenerate every figure from logged results; reference each one |
| MQ-6 | **Table errors.** Table 2 bolds GB's AUC 0.997 although RF has 0.998. "Both RF and GB achieve AUC-ROC above 0.997" (tex:L465-466), but GB equals 0.997. HGT is not the best GNN by F1 (SimpleHGN has 90.54). Table 6's Recall column has no std. Table 7 is incoherent: HGT has a fixed 9-feature claim input, yet the table shows "G1 (7) / G1+G2 (22) / 36". | Medium | — | Fix (moot once regenerated) |
| MQ-7 | **Textual factual errors.** "8× faster" (the P50 ratio in Table 8 is 10.25×). "RF … more stable variance" (0.29 vs 0.31 is not a meaningful difference). "HGT advantage increases with scale, consistent with Liu et al." (untested). "Claims are sorted temporally within the graph to preserve causal ordering" (node order does not affect message passing). | Low–Medium | tex:L855-865, L846-848, L424-425 | Correct or remove |
| MQ-8 | **Limitations are incomplete.** Leakage, entity overlap, static rings, the tiny test set, the single synthetic world, the HGT information-flow restriction, and the lack of significance testing are all missing. | High | tex:L867-882 | Rewrite after the new results |
| MQ-9 | **Declarations are missing.** No Funding, Ethics approval (an explicit "not applicable: synthetic data" statement), Consent, Code availability, or Materials availability. "All authors contributed equally" conflicts with the equal-contribution footnotes on 4 of 5 authors. | Medium | tex:L996-1004 | Add Springer Nature's required declarations; use CRediT roles |
| MQ-10 | **No reporting standard is followed.** | Medium | — | REFORMS checklist (Kapoor et al. 2024, *Science Advances*); dataset datasheet (Gebru et al.); model card (Mitchell et al.); optionally the ML reproducibility checklist (Pineau et al., JMLR 2021) |
| MQ-11 | **Inconsistent framing.** The HEAD version of the paper (committed) presented accuracy values as F1 ("Random Forest: 98.95% F1, Gradient Boosting: 99.05% F1"); the current version swaps in different, unsourced F1 values. | High | `git diff`; Appendix B | Regenerate |

### 4.8 Deployment claims (Section 5 and Appendix A)

| Claim (manuscript) | Code reality | Status |
|---|---|---|
| "We validate all models in a production FastAPI system … 1,000 consecutive prediction requests" (tex:L761-764) | The API serves **RF blended with a hand-written rule score**: `0.65·ML + 0.35·rules + 0.15·ML·rules` (`backend/production_predictor.py:340-350`), with `model_used: "random_forest+rules"`. HGT, GB, LR, RGCN, HAN, and SimpleHGN are not served. No benchmark script exists. | [CONTRADICTED] / [UNSUPPORTED] |
| Table 8 latencies and throughput | No measurement code; GNNs not servable | [UNSUPPORTED] |
| Eight RESTful endpoints (tex:L915-922) | **65** route decorators in `backend/main.py`, with `/health` registered twice (L237 and L1799) | [CONTRADICTED] |
| Risk tiers LOW < 0.45 ≤ MEDIUM ≤ 0.75 < HIGH (tex:L922-923) | 0.4 / 0.7 (`backend/main.py:591`, `backend/predictor.py:186`, `production_predictor.py:369`) | [CONTRADICTED] |
| JWT via API key; SlowAPI 1,000/min single, 100/hour batch | Present (`backend/security.py`; `backend/main.py:296, 377`). Default secrets are `"change-me-in-production"` (`backend/config.py:23-24`). | Supported (with a hygiene caveat) |
| SQLite audit trail | Present (`backend/config.py:19`; `PredictionRecord`) | Supported |
| Drift alerts when KS > 0.15 | 0.15 is a **KL** threshold (`backend/drift_detector.py:28`). KS statistics are computed without any threshold (`:147-153`). | [CONTRADICTED] |
| Streamlit dashboard with six pages | At least **14** pages (`frontend/app.py:190-967`). The dashboard hard-codes HGT metrics. | [CONTRADICTED] |
| Docker Compose; Python 3.11 | Present (`docker/*`, `Dockerfile`: `python:3.11-slim`) | Supported |
| "Reproducible via docker-compose up … after placing trained model artifacts" | No HGT artifacts exist anywhere | [UNSUPPORTED] |
| Inductive inference subgraph | `build_inference_subgraph` result is passed in but ignored by `ProductionFraudPredictor.predict` | [CONTRADICTED] |

**Recommendation:** reduce Appendix A to a short, accurate system description, or remove it. The research contribution should not depend on the deployment system.

---

## 5. Independent verification results

**Diagnostic status.** These results are diagnostics, not publishable results. They were run against committed artifacts from a scratch directory outside the repository. Code is in Appendix A.

### 5.1 Dataset

| Check | Result |
|---|---|
| SHA-256 of the 4 CSVs vs `logs/dataset_config.json` | **Match** |
| Claims / patients / doctors / hospitals | 2,100 / 500 / 50 / 20. Only 492 patients have any claim. |
| Composition | 1,722 legitimate · 200 isolated · 178 ring (5 rings × 4 doctors × 10 patients) |
| `ring_id > 0` ⇔ ring fraud; legitimate claims with `ring_id > 0` | Yes; 0 |
| Legitimate claims filed by ring doctors / ring patients | 679 / 181 |
| Duplicate rows or IDs | 0 / 0 |
| Graph | 2,670 nodes. **6,200** edges: filed_claim 2,100, treated_by 2,100, works_at 50, visited 1,950 |

### 5.2 Separability (fraud generated from shifted uniforms)

| | Legitimate (min / median / max) | Isolated | Ring |
|---|---|---|---|
| `claim_amount` | 128 / 4,187 / 8,491 | 4,818 / 14,190 / 21,895 | 4,504 / 12,634 / 21,995 |
| `num_procedures` | 1 / 3 / 5 | 4 / 8 / 11 | 4 / 7 / 11 |
| `days_in_hospital` | 0 / 3 / 5 | 1 / 8 / 15 | 1 / 9 / 15 |

- Single-feature AUC, ring vs legitimate: amount **0.937**, procedures **0.941**, days 0.861.
- Generator-bound rule (*amount > 8,500 ∨ procedures > 5 ∨ days > 5*): precision **1.000**, recall **0.976**, F1 **0.988**. Isolated recall 0.995; ring recall **0.955**.

### 5.3 Split composition (positional 60/20/20 on date-sorted claims)

| Split | Legitimate | Isolated | Ring | Fraud rate |
|---|---|---|---|---|
| Train (1,260) | 1,021 | 117 | 122 | 19.0% |
| Val (420) | 345 | 40 | 35 | 17.9% |
| Test (420) | 356 | 43 | **21** (per ring: 3/5/4/6/3) | 15.2% |

Entity overlap, test vs train:

| Entity | Share of test claims whose entity appears in train |
|---|---|
| Doctors | 100% |
| Hospitals | 100% |
| Patients | 91.4% |
| Doctors of ring test claims that had ring-fraud claims in train | 100% |

### 5.4 Committed models on committed `X_test.npy`

| Model | F1 | PR-AUC | Ring recall | Isolated recall | FP / FN | Top feature signal |
|---|---|---|---|---|---|---|
| LR (L1, C = 1) | 0.947 | 0.994 | 1.000 | 0.977 | 6 / 1 | Largest coefficient: **`ring_id` (5.55)** |
| RF (100 trees, depth 8) | 1.000 | 1.000 | 1.000 | 1.000 | 0 / 0 | `ring_id` 5th by impurity importance |
| GB (100, 0.05, depth 3) | 1.000 | 1.000 | 1.000 | 1.000 | 0 / 0 | `ring_id` 4th by impurity importance |

### 5.5 Retrained with the paper's hyperparameters

Train only, evaluated on test, mean over seeds {42, 123, 456, 789, 1337}.

| Feature set | LR F1 | RF F1 | GB F1 | GB ring recall |
|---|---|---|---|---|
| 27 (with `ring_id`) | 0.940 | 1.000 | 1.000 | 1.000 |
| 26 (`ring_id` removed) | 0.904 | 0.998 ± 0.004 | **1.000** | **1.000** |
| 3 raw claim features | 0.940 | 0.994 ± 0.004 | **1.000** | **1.000** |

The 2,000-replicate bootstrap 95% CI for GB (26 features) is [1.000, 1.000] for both F1 and ring recall. That is a ceiling effect.

### 5.6 Split and leakage protocols

GB, 12 rebuilt aggregate/ratio features, 5 seeds.

| Protocol | F1 |
|---|---|
| Temporal split, train-only aggregates | 0.997 ± 0.007 |
| Random stratified split, train-only aggregates | 0.988 ± 0.006 |
| "Leaky": aggregates from all claims before the split | 0.997 |
| Leaky + doctor fraud rate / train-only + doctor fraud rate | 0.998 / 0.998 |

The random split does **not** inflate F1, and the leaky protocol does **not** inflate F1. Both contradict Table 9 and the ">99.9%" leaky row.

Mean training-period doctor fraud rate is 0.229 for ring doctors and 0.162 for non-ring doctors. Even this rate alone partially separates ring doctors.

### 5.7 HGT information flow (repository model)

| Check | Result |
|---|---|
| Node types with no incoming relation | `patient` |
| Relations into `claim` | `filed_claim` only |
| PyG 2.8 HGTConv output types | `doctor`, `hospital`, `claim` (no `patient`) |
| Layer-2 behaviour | `KeyError 'patient'`, swallowed; layer-1 output retained |
| Δ claim logits after randomizing doctor + hospital features | **0.0** |
| Δ after deleting `treated_by` / `works_at` / `visited` | **0.0 / 0.0 / 0.0** |
| Δ after randomizing patient features | 0.78 |
| `NeighborLoader` (code default) | ImportError: needs `pyg-lib` or `torch-sparse` |
| `tensor.numpy()` in `.venv` | RuntimeError: "Numpy is not available" |

### 5.8 HGT as coded

5 seeds; full-batch transductive mode; repository `GNNTrainer` with only the `.numpy()` calls bypassed.

| Variant | F1 | Recall | ROC-AUC | PR-AUC | Ring recall | Isolated recall | Train time |
|---|---|---|---|---|---|---|---|
| HGT as coded | 0.933 ± 0.021 | 0.969 ± 0.019 | 0.999 | 0.992 | 0.981 ± 0.043 | 0.963 ± 0.013 | 2.4 s |
| HGT with patient features zeroed (pure MLP) | 0.931 ± 0.016 | 0.981 ± 0.020 | 0.999 | 0.993 | 0.990 ± 0.021 | 0.977 ± 0.033 | 2.5 s |
| GB (§5.5, 26 features) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — |

Conclusions:
- HGT ≡ MLP.
- HGT is no better than GB on any metric.
- The paper's direction (HGT > GB on ring fraud by 14.2 pp) is not reproduced.

### 5.9 Repository ring-recall alignment

Re-sorting `features_raw.csv` exactly as the repository does gives:

| Check | Result |
|---|---|
| Test positions that change claim after the re-sort | **144 / 420** |
| Positions in the "ring" mask | 15 |
| True ring-fraud claims covered by the mask | **14 of 21** |

---

## 6. Arithmetic consistency of the manuscript's own tables

These tests depend only on the paper's numbers and on the test partition. The partition is fixed by the hash-verified data and the stated 60/20/20 split.

### 6.1 Mixed recall must be a weighted average of subset recalls

Mixed positives = isolated positives ∪ ring positives. Therefore *R_mixed = w·R_ring + (1 − w)·R_iso*, where *w* = the ring share of positives. *w* must be identical for every model, because all models share one test set.

| Model | R_iso | R_ring | R_mixed (reported) | Feasible? | Implied *w* | Expected R_mixed at the test *w* = 21/64 |
|---|---|---|---|---|---|---|
| LR | 94.20 | 63.50 | 92.30 | yes | 0.062 | 84.13 |
| RF | 99.45 | 71.20 | 99.12 | yes | 0.012 | 90.18 |
| **GB** | 99.30 | 72.40 | **99.45** | **NO**: exceeds both subsets | −0.006 | 90.47 |
| RGCN | 91.30 | 78.10 | 89.10 | yes | 0.167 | 86.97 |
| HAN | 92.10 | 80.90 | 90.55 | yes | 0.138 | 88.42 |
| SimpleHGN | 93.00 | 82.20 | 91.22 | yes | 0.165 | 89.46 |
| HGT | 94.10 | 86.60 | 88.45 | yes | **0.753** | 91.64 |

- GB's row is mathematically impossible.
- The implied ring share ranges from −0.6% to 75%. The paper's own Table 1 implies 47% (178/378), and the test partition implies 33% (21/64).

### 6.2 GRIM-style feasibility

With a fixed test set, a 5-seed mean recall must equal *k / (5n)* for some integer *k*.

| Subset (n positives) | Reported recalls that cannot occur |
|---|---|
| Isolated (43) | 7 of 7 |
| Ring (21) | 6 of 7 (only RGCN's 78.10 is feasible) |
| Mixed (64) | 7 of 7 |

Table 2 accuracy, n = 420 × 5 seeds:
- LR 94.55, HAN 93.45, SimpleHGN 94.20, and HGT 95.35 cannot occur.
- All of them *can* occur as single runs on n = 2,000. That is the test size of the older 10k-claim dataset, consistent with Appendix B.

### 6.3 Other internal arithmetic

- Hybrid mean latency with Table 8 values: 21.3–22.8 ms, not ≈ 18 ms.
- The abstract's HGT P99 (< 100 ms) vs Table 8 (140 ms).
- The deleted script's GB confusion matrix gives 94.8% accuracy, not the stated 99.05%.

---

## 7. Evidence-to-claim map

| ID | Manuscript claim (location) | Evidence in repository | Status |
|---|---|---|---|
| C-01 | 2,100 claims; 500 patients; 50 doctors; 20 hospitals (Table 1) | `data/raw/*`, `logs/dataset_config.json`, hash verified | **Supported** |
| C-02 | 18% fraud; 200 isolated; 178 ring; 5 rings; 24 months | Data | **Supported** |
| C-03 | 2,670 graph nodes | `hetero_graph.pt` | **Supported** |
| C-04 | 6,300 edges; three relations | 6,200; four relations | **Contradicted** |
| C-05 | Isolated injection U($15k–50k), U(12–20), U(10–25) (L249-256) | `generate_dataset.py:118-127` | **Contradicted** |
| C-06 | Ring claims "individually borderline … only recoverable via graph" (L258-266) | Same distribution as isolated; rule ring recall 0.955; `ring_id` in features | **Contradicted** |
| C-07 | 4 doctors × 10 patients per ring | Generator config | **Supported** |
| C-08 | Evaluation subsets A/B/C (L268-273) | No code; negatives undefined | **Unsupported** |
| C-09 | Temporal 60/20/20 | `preprocessor.py:68-91` | **Supported** |
| C-10 | "Stratified by fraud label within each partition" | Not stratified | **Contradicted** |
| C-11 | Aggregates from training only; fallback for unseen entities; scaler/encoders fit on train (L282-288) | `preprocessor.py`. Self-inclusion caveat (DV-12). Graph normalization uses all nodes. | **Partially supported** |
| C-12 | 5 seeds {42, …, 1337}; paired t-tests (L290-292) | Seeds 0–4; t-test never run | **Contradicted** |
| C-13 | 36 features in groups 7/15/14 (L296-318) | 27 features, different composition | **Contradicted** |
| C-14 | Doctor/hospital fraud-rate features | Excluded in code | **Contradicted** |
| C-15 | Tabular hyperparameters (L334-341; App. B) | Committed models use tuned, different parameters | **Contradicted** |
| C-16 | RGCN, HAN, SimpleHGN configurations (L343-353) | Not implemented | **Unsupported** |
| C-17 | HGT input projections d ∈ {4, 3, 2, 9} | `graph_builder.py`; `gnn.py` | **Supported** |
| C-18 | Two HGTConv layers with residual and LayerNorm | Layer 2 silently skipped; one shared LayerNorm | **Contradicted** |
| C-19 | MLP 128→64→2; AdamW; cosine; patience 15; class weights; clip 1.0 | `gnn.py`; `train_gnn.py` | **Supported** (as code; never successfully run in the repo environment) |
| C-20 | HGT runs in transductive mode | Code default is inductive `NeighborLoader` | **Contradicted** |
| C-21 | Software and hardware (L452-457) | Venv and lock file differ; hardware unknown | **Contradicted / Unknown** |
| C-22 | Table 2: all seven rows | Hard-coded literals; logs contradict; GRIM-infeasible | **Unsupported / Contradicted** |
| C-23 | Leaky RF > 99.90 on all metrics | No run; our reproduction shows no inflation; old 10k run had F1 99.73 | **Unsupported / Contradicted** |
| C-24 | "HGT better-calibrated (highest AUC)" | AUC does not measure calibration; no calibration metric computed | **Invalid reasoning** |
| C-25 | Table 3: regime decomposition | No code; arithmetically impossible | **Unsupported / Contradicted** |
| C-26 | HGT ring recall exceeds GB by +14.2 pp (abstract; L541-548; L828-830; conclusion) | GB ring recall is 1.0 in logs and in our run; HGT ≤ GB | **Contradicted** |
| C-27 | Fig. 5: HGT finds 5/5 ring doctors, GB 1/5; attention opacity | No figure, no HGT run, no attention; rings have 4 doctors | **Unsupported / Contradicted** |
| C-28 | Depth ablation (Table 4) | Impossible: depth ≥ 1 yields identical computation | **Contradicted** |
| C-29 | Heads ablation (Table 5) | No run | **Unsupported** |
| C-30 | Edge ablation (Table 6) | Impossible: Δ = 0.0; values are literals in `shap_comparison.py` | **Contradicted** |
| C-31 | Feature-group ablation for GB and HGT (Table 7) | No run; incoherent for HGT | **Unsupported** |
| C-32 | Ablation heatmap (Fig. 8) | No figure; no data | **Unsupported** |
| C-33 | Random split inflates F1 by 0.7–3.4 pp (Table 9; Fig. 7) | No run; our reproduction shows no inflation (random 0.988 vs temporal 0.997) | **Contradicted** |
| C-34 | SHAP top-5 and percentages (L717-733; Fig. 9) | Logs: RF top feature is `num_procedures`; `doctor_fraud_rate` does not exist; `ring_id` ranks 5th; group percentages computed on wrong index ranges (`shap_comparison.py:57-63`) | **Contradicted** |
| C-35 | Fraud-network figure (Fig. 10) | Only an old figure from the 10k dataset in git history | **Unsupported** |
| C-36 | Table 8 latency and throughput | No benchmark; GNNs not served | **Unsupported** |
| C-37 | Hybrid: 18% escalation, ≈18 ms, HGT-level ring recall | Sweep uses hard-coded latencies and GB scores in place of HGT; arithmetic inconsistent | **Unsupported / Contradicted** |
| C-38 | Abstract: HGT P99 < 100 ms | Table 8: 140 ms | **Contradicted (internal)** |
| C-39 | Abstract: GB F1 98.83 on *isolated* fraud | Table 3: 99.10 isolated; 98.83 mixed | **Contradicted (internal)** |
| C-40 | "Even 3-layer HGT with 8 heads does not close the gap" | Configuration never in any table | **Unsupported** |
| C-41 | "Validated directionally consistent with CMS literature" (L871-873) | Synthetic proxy mislabeled as real-world | **Unsupported** |
| C-42 | Appendix A: 8 endpoints; 0.45/0.75 tiers; KS > 0.15; 6 pages | 65 routes; 0.4/0.7; KL 0.15; ≥ 14 pages | **Contradicted** |
| C-43 | Appendix A: JWT; SlowAPI limits; SQLite audit; Docker on Python 3.11 | Present in code | **Supported** |
| C-44 | Related-work statements about Bauder, Liu, Akoglu, FdGars, Player2Vec, Tiong; NHCAA cost figure | See B-12 | **Inaccurate / Unverified** |
| C-45 | "First ablation study for HGT-based healthcare fraud detection" | No search documented; ablations not run | **Unsupported** |
| C-46 | Data availability: code and generator at the GitHub URL | Repository exists; no license, DOI, or release; results not reproducible | **Partially supported** |

---

## 8. Alternative explanations for apparent performance

Each entry gives the explanation, how strongly the evidence supports it, and how to rule it out in the new study.

| # | Explanation | Evidence | How to rule out |
|---|---|---|---|
| AE-1 | **Separability by construction.** Fraud claims come from shifted uniform distributions, so three thresholds reproduce near-perfect performance. | [VERIFIED] (§5.2) | Matched marginals for ring claims; a δ sweep; always report a rule baseline and a 3-feature baseline |
| AE-2 | **Label leakage via `ring_id`.** | [VERIFIED] | Feature allowlist plus test; re-run without it |
| AE-3 | **Entity memorization.** The same doctors, patients, and rings appear in train and test. | [VERIFIED] | Ring-disjoint and entity-disjoint inductive splits |
| AE-4 | **Self-inclusion in training aggregates.** A fraud claim inflates its own entity means in training, so features shift between train and test. | [VERIFIED] | Out-of-fold or prior-only aggregates |
| AE-5 | **Small-sample variance.** 21 ring positives; 3 claims = 14.2 pp. | [VERIFIED] | Dataset replicates; CIs; paired tests |
| AE-6 | **Operating-point artifacts.** Recall at an implicit 0.5 threshold, under different class weighting per model. | [VERIFIED] | Validation-selected thresholds; PR-AUC; recall at a fixed alert budget |
| AE-7 | **Mislabeled metrics.** Accuracy reported as F1 in the earlier draft; "isolated" vs "mixed" confusion in the abstract. | [VERIFIED] | Generated tables only |
| AE-8 | **HGT ≡ MLP.** Any HGT "advantage" cannot come from structure. | [VERIFIED] | Information-flow tests; MLP control with identical inputs; degree-preserving rewiring control |
| AE-9 | **Misaligned evaluation masks.** Ring metrics computed on the wrong claims. | [VERIFIED] | ID-based alignment |
| AE-10 | **Mixed artifact provenance.** Numbers from the older 10k dataset, mock dictionaries, and literals combined in one table. | [VERIFIED] (Appendix B) | Experiment ledger; clean regeneration |
| AE-11 | **Transductive structural look-ahead.** Once reverse edges exist, future edges and nodes leak structure into training. | [RISK] | Time-respecting graph; inductive inference |
| AE-12 | **Degree and centrality confound.** Ring doctors have more fraud and may differ in degree; a GNN could exploit degree rather than coordination. | [RISK] | Degree-matched negatives; rewiring control; degree-only baseline |
| AE-13 | **Hyperparameter-tuning asymmetry.** | [VERIFIED] (ME-5) | Equal budgets |

---

## 9. Research-gap analysis and competing research

### 9.1 What the literature already covers

These works are relevant and must be engaged with. Items marked [verify] need their bibliographic details checked before citing.

**Tree ensembles vs GNNs for graph anomaly and fraud detection**
- **GADBench**: Tang, Hua, Gao, Wang, Li, *NeurIPS 2023 Datasets & Benchmarks*. Large-scale comparison across graph anomaly detection datasets; tree ensembles with neighborhood aggregation are often the strongest. This directly anticipates "when do GNNs beat tabular", and is the most important missing citation and comparator. [verify authors]
- **Weber et al. 2019**, "Anti-money laundering in Bitcoin: experimenting with graph convolutional networks for financial forensics" (KDD Workshop on Anomaly Detection in Finance). RF outperformed GCN under a temporal split on the Elliptic dataset.

**Synthetic data with controlled fraud typologies**
- **Altman et al., NeurIPS 2023 D&B**, "Realistic synthetic financial transactions for anti-money laundering models". Controlled laundering patterns (fan-in/out, cycles, …) with GNN vs GBT comparisons.
- **Egressy et al., AAAI 2024** (multi-GNN on directed multigraphs), on the same data. [verify]
- **Jesus et al., NeurIPS 2022 D&B**, "Turning the tables: biased, imbalanced, dynamic tabular datasets for ML evaluation" (BAF fraud suite). [verify]

**Fraud GNNs addressing camouflage, imbalance, and heterophily**
- CARE-GNN: Dou et al., CIKM 2020.
- PC-GNN: Liu et al., WWW 2021 (already cited, but mis-described).
- BWGNN: Tang et al., ICML 2022, "Rethinking graph neural networks for anomaly detection".
- GHRN: Gao et al., WWW 2023 (heterophily in graph anomaly detection). [verify]
- H2-FDetector: Shi et al., WWW 2022. [verify]

**Dense-subgraph and lockstep fraud detection** (natural unsupervised baselines for rings)
- FRAUDAR: Hooi et al., KDD 2016.
- CopyCatch: Beutel et al., WWW 2013.
- OddBall: Akoglu, McGlohon, Faloutsos, PAKDD 2010.

**Healthcare fraud ML on Medicare/LEIE data**
- Herland, Khoshgoftaar & Bauder (2018), *J. Big Data*.
- Johnson & Khoshgoftaar (2019), "Medicare fraud detection using neural networks", *J. Big Data*.
- Bauder & Khoshgoftaar (2018), *Health Inf. Sci. Syst.* (class imbalance). [verify]

**Healthcare and insurance graph fraud** [verify all]
- Studies comparing ML and GNNs on Medicare provider graphs, e.g., an IEEE Access 2023 comparative study.
- Attributed heterogeneous information networks with hierarchical attention for health-insurance fraud, e.g., *BMC Medical Informatics and Decision Making* 2023.
- **Action:** locate these through a systematic search and confirm details.

**Tabular vs deep learning**
- Grinsztajn, Oyallon, Varoquaux, NeurIPS 2022 D&B.
- Shwartz-Ziv & Armon, *Information Fusion* 2022.

**Leakage and evaluation pitfalls**
- Kaufman et al., *ACM TKDD* 2012 ("Leakage in data mining").
- Kapoor & Narayanan, *Patterns* 2023.
- Shchur et al. 2018, "Pitfalls of GNN evaluation".
- Errica et al., ICLR 2020.
- Saito & Rehmsmeier, *PLoS ONE* 2015 (PR vs ROC under imbalance).
- Demšar, *JMLR* 2006 (statistical comparison of classifiers).
- Guo et al., ICML 2017 (calibration).

**Inductive and temporal GNNs**
- GraphSAGE: Hamilton et al., NeurIPS 2017.
- Cluster-GCN: Chiang et al., KDD 2019.
- GraphSAINT (already cited).
- TGAT: Xu et al., ICLR 2020.
- TGN: Rossi et al., 2020.

**Explanation caveats**
- Jain & Wallace, NAACL 2019 ("Attention is not explanation").
- GNNExplainer (Ying et al., 2019; already in the bibliography).

### 9.2 Defensible gap (narrowed)

There is no controlled study, for **healthcare claims with a heterogeneous patient–provider–facility schema**, that does all of the following:

1. Varies the **feature-level signal** and the **structural (collusion) signal** of provider-ring fraud independently.
2. Compares relational models against **strong tabular models, including models with engineered graph features and unsupervised dense-subgraph detectors**, under **leakage-safe, ring-disjoint inductive evaluation**.
3. Validates the resulting regime boundaries on **public real-world data**.

This gap is plausible but must be confirmed by a documented systematic search (databases, query strings, dates, inclusion criteria).

### 9.3 What a reviewer will ask

- How does this differ from GADBench's conclusions, and from the AML synthetic-pattern studies?
- Why not use public Medicare data?
- Is the GNN's advantage due to structure, or to degree or identity?
- Is the result robust to ring camouflage and novelty?

---

## 10. Proposed experiments, ranked

The ranking combines scientific value with dependency order. **P0** = mandatory for any submission; **P1** = needed to be competitive at Q1/Q2; **P2** = valuable but optional.

| Rank | ID | Experiment | Key design elements | What it establishes | Priority | Depends on |
|---|---|---|---|---|---|---|
| 1 | E1 | **Integrity reset and reproducible pipeline** | Remove all unsupported numbers. One CLI (`make paper-results`). Config files. Ledger with SHA, data hash, config hash. Per-seed predictions saved. Tables and figures generated from predictions. | An evidentiary basis for every number | P0 | — |
| 2 | E2 | **Leakage-free feature pipeline** | Allowlist; remove `ring_id` and `fraud_prone`; out-of-fold or prior-only aggregates; ID-based alignment; stable sorting; leakage unit tests | Valid tabular baselines | P0 | E1 |
| 3 | E3 | **Generator redesign with controllable regimes, plus dataset replicates** | Parameters: feature shift δ of ring claims (0 = identical to legitimate), ring density, patient-pool overlap, ring size and count, camouflage (ring doctors' legitimate volume), ring onset time, prevalence, label noise. ≥ 10 dataset seeds per setting. Datasheet. | A testable setting for the central hypothesis | P0 | E1 |
| 4 | E4 | **Correct heterogeneous graph and models** | Reverse relations; no `try/except`; information-flow unit tests; time-respecting graph; train-only normalization; optional doctor–doctor shared-patient relation | Relational models that can use structure | P0 | E2 |
| 5 | E5 | **Baseline suite with equal tuning** | LR, RF, GB (plus LightGBM/XGBoost); MLP control on identical inputs; **GB + graph features** (degree, shared-patient Jaccard, 2-hop counts, out-of-fold neighbor label rates, node2vec/metapath2vec embeddings); GraphSAGE (hetero), RGCN, HAN, SimpleHGN, HGT. Equal tuning budgets on validation only. | A fair answer to "when do GNNs outperform tabular" | P0 | E2, E4 |
| 6 | E6 | **Evaluation protocol** | Ring-disjoint inductive split; temporal-emergence split; claim-level and provider-level metrics; PR-AUC primary; recall at a fixed alert budget; validation-chosen thresholds; confusion matrices with counts; t- or bootstrap CIs; paired Wilcoxon or t over dataset replicates; Holm correction | Statistical credibility | P0 | E3–E5 |
| 7 | E7 | **Regime map (phase diagram)** | Grid of δ_feature × structural strength (× camouflage). Plot ΔPR-AUC (best GNN − best tabular+graph-features) with CIs. Identify the boundary. | The paper's core contribution, matching its title | P0 (at least a coarse grid) | E3–E6 |
| 8 | E8 | **Real-world validation** | Public data with an appropriate task. Example A: Kaggle "Healthcare Provider Fraud Detection Analysis" (provider-level labels), with a provider–beneficiary–physician graph, provider-disjoint grouped splits, train-only aggregates, provider-level metrics. Example B: CMS Medicare Part B/D provider data plus LEIE exclusions, optionally with a CMS physician shared-patient graph. Verify licenses, provenance, and label semantics [verify availability]. | External validity | P0 for Q1/Q2 | E2, E4–E6 |
| 9 | E9 | **Mechanism ablations with controls** | Edge types (including reverse relations); depth; heads; degree-preserving rewiring; feature-shuffle; structure-only (no features); attention analysis with caveats | Attribution of gains to coordination structure | P1 | E4–E6 |
| 10 | E10 | **Robustness and sensitivity** | Prevalence 1/5/18%; ring size and count; camouflage; label noise and partial membership; delayed labels; drift; missing edges | Boundaries of the conclusions | P1 | E3, E6 |
| 11 | E11 | **Scale study** | ×10 and ×100 graph sizes; runtime and memory; whether any GNN advantage grows with scale (the paper's claim) | Tests or removes the scale claim | P1 | E3–E6 |
| 12 | E12 | **Unsupervised dense-subgraph baselines** | FRAUDAR-style densest-block detection; CopyCatch-style lockstep detection; shared-patient overlap scores | Whether supervised GNNs are needed at all for rings | P1 | E3 |
| 13 | E13 | **Calibration and cost-sensitive evaluation** | Brier, ECE, reliability diagrams; temperature scaling; investigation-budget curves; expected recovered amount | Operational relevance; replaces the invalid calibration claim | P1 | E6 |
| 14 | E14 | **Explainability with fidelity checks** | SHAP after the leakage fix, with correct groups; GNNExplainer or subgraph explanations with deletion/insertion fidelity | Interpretability claims supported by evidence | P2 | E5 |
| 15 | E15 | **Deployment measurement** | Scripted latency benchmark of the models actually served (warm-up, ≥ 1,000 requests, hardware stated); inductive subgraph construction cost; end-to-end hybrid cascade on test with the threshold chosen on validation | Supports or removes Section 5 | P2 (or drop the section) | E4–E6 |
| 16 | E16 | **Temporal GNNs** (TGN/TGAT) for evolving rings | Event-based graph; ring onset or recruitment | Future work or extension | P2 | E3 (temporal scenario) |

---

## 11. Reproducibility requirements

All of the following are mandatory unless marked optional.

**Environment**
- [ ] One lock file (`uv.lock`, `poetry.lock`, or `pip-compile` output) and one container image, referenced by digest in the paper.
- [ ] torch, NumPy, and PyG ABI compatibility.
- [ ] `pyg-lib` installed if `NeighborLoader` is used.
- [ ] Library versions and hardware printed into every log.

**Determinism**
- [ ] Seed `random`, `numpy`, and `torch`.
- [ ] `torch.use_deterministic_algorithms(True)` where feasible.
- [ ] Record seeds per run.
- [ ] Save per-seed predictions (`claim_id`, `y_true`, `y_score`, split, seed, dataset_seed).

**Single entry point**
- [ ] Generate → preprocess → graph → train → evaluate → tables/figures.
- [ ] Repo-root-anchored paths.
- [ ] One canonical data tree.

**Configuration**
- [ ] YAML or JSON experiment configs for every table and figure.
- [ ] The paper cites the config IDs.

**Provenance**
- [ ] Each output file records git SHA, dataset SHA-256, config hash, and timestamp.
- [ ] Logs are never edited by hand.

**Tests (must fail CI)**
- [ ] Feature blocklist and allowlist.
- [ ] Split disjointness, including entity and ring disjointness where the protocol requires it.
- [ ] Training aggregates computed only from permitted data.
- [ ] ID alignment.
- [ ] GNN information flow (doctor perturbation changes claim logits).
- [ ] No mock metrics reachable by figure or table code.

**CI**
- [ ] Remove `|| true` and `continue-on-error` on the scientific tests.
- [ ] A smoke-reproduction job on a reduced config that checks numeric tolerances.

**Artifacts**
- [ ] Tagged release matching the submission.
- [ ] Zenodo DOI.
- [ ] Code license and data license.
- [ ] Datasheet for the generator and dataset.
- [ ] Model cards.
- [ ] README reproduction steps with expected runtimes.

**Reporting**
- [ ] REFORMS checklist in the supplement.
- [ ] Hyperparameter search spaces and budgets.
- [ ] Compute budget.
- [ ] Failed or negative runs disclosed.

---

## 12. Publication strategy

### 12.1 Genuine contributions available after remediation

1. **A controlled, open benchmark generator** for healthcare claims with tunable provider-ring typology (feature vs structure vs camouflage vs novelty), with a datasheet.
2. **A regime map** identifying where relational models beat strong tabular+graph-feature baselines under leakage-safe inductive evaluation, *or* evidence that they rarely do. A rigorous negative result is publishable.
3. **Validation on public real data** at provider level.
4. **A methodological pitfalls section**: identifier leakage, entity memorization, self-inclusion aggregates, unidirectional heterogeneous edges, and positional misalignment. Each is demonstrated quantitatively on the benchmark. This is a useful secondary contribution, given how common these errors are.

### 12.2 Weaknesses reviewers will press on, even after remediation

- Synthetic-first evidence.
- Real labels at provider level, not claim level.
- Modest methodological novelty.
- Overlap with GADBench and AML-benchmark conclusions.
- Scale.

### 12.3 Venue fit (realistic, conditional on full remediation)

Check the target year's SJR/JCR quartile yourself: rankings change, and no venue's acceptance can be assumed.

| Venue (publisher) | Fit | Requirements to be competitive |
|---|---|---|
| *Journal of Big Data* (Springer) | Has published Medicare-fraud ML work; empirical studies welcome | Real Medicare/LEIE data; scale; rigorous evaluation |
| *Health Information Science and Systems* (Springer) | Health-informatics ML | Real-data validation; clear health-system relevance |
| *Social Network Analysis and Mining* (Springer) | Graph-based fraud analysis | Strong graph methodology; real network data |
| *Applied Intelligence*; *Neural Computing and Applications*; *Knowledge and Information Systems* (Springer) | Applied ML and benchmarks | Comprehensive baselines; statistics; reproducibility |
| *International Journal of Data Science and Analytics* (Springer) | Data-science methodology | Solid protocol; open artifacts |
| *Scientific Reports* (Nature Portfolio) | Judged on soundness, not novelty | Exemplary rigor; real data strongly advised |
| *Data Mining and Knowledge Discovery*; *Machine Learning* (Springer) | High bar | Needs methodological novelty beyond benchmarking, plus multiple real datasets |
| Non-Springer options: *Decision Support Systems*, *Expert Systems with Applications*, *Information Sciences*, *Artificial Intelligence in Medicine*, *JBI*; NeurIPS Datasets & Benchmarks; KDD Applied Data Science | Varying fit | The D&B track needs a substantial, well-documented, community-useful benchmark |

The `sn-jnl` template is appropriate for Springer Nature journals. Keep it, but adapt it to the chosen journal's reference style and declaration requirements.

### 12.4 Minimum bar for a competitive Q1/Q2 submission

- **All P0 experiments** (E1–E8) are complete and reproducible from a tagged release.
- **The primary claims are stated with CIs and paired tests across dataset replicates**, under ring-disjoint inductive evaluation.
- **At least one public real-world dataset** is used, with a leakage-safe protocol.
- **Strong baselines are included**: tabular plus graph features, an MLP control, and dense-subgraph methods.
- **Related work is accurate**, every reference verified with a DOI, and novelty positioned against GADBench and the AML benchmarks.
- **Honest limitations, declarations, and a reporting checklist** are in place.

---

## 13. Staged publication-readiness roadmap

Each stage has exit criteria. Do not advance until the criteria are met.

**Stage 0: Integrity freeze**
- Withdraw all unsupported numbers and figures from the manuscript draft. Mark sections "TBD".
- Brief all co-authors, including the supervisor, on B-01 and B-12.
- Decide the scope: synthetic benchmark plus real-data validation (recommended).
- *Exit:* the manuscript contains no unsupported quantitative claim, and the scope is agreed.

**Stage 1: Engineering validity fixes**
- B-02, B-04, B-07, B-08, B-09 (code side), DV-12, LK-4, RP-2, RP-4, RP-5, RP-6.
- *Exit:* all validity tests pass in CI; the GNN information-flow test passes; the environment rebuilds from the lock file; a smoke run regenerates a tiny results table end to end.

**Stage 2: Benchmark and core experiments**
- E3, E5, E6, E7, with dataset replicates.
- *Exit:* the regime map with CIs and paired tests is generated from the ledger; results are stable across replicates.

**Stage 3: Real-world validation**
- E8, with a provider-disjoint, leakage-safe protocol.
- *Exit:* real-data results with CIs; licenses and provenance documented.

**Stage 4: Mechanism, robustness, and operational analyses**
- E9–E13; optionally E14 and E15.
- *Exit:* the attribution and robustness claims each have generated evidence.

**Stage 5: Manuscript rebuild**
- Rewrite Abstract, Introduction (narrowed gap), Related Work (§9), Methods (from config), Results (generated tables and figures only), Discussion (with the alternative explanations of §8 addressed), Limitations, and Declarations.
- Add the REFORMS checklist, datasheet, and model card.
- Verify every citation.
- *Exit:* every number in the PDF traces to a ledger entry; a co-author not involved in the experiments re-derives three randomly chosen numbers from the artifacts.

**Stage 6: Pre-submission review and venue selection**
- Internal red-team review against this audit.
- Optional preprint.
- Choose a venue (§12.3) and format for it.
- *Exit:* every issue in §15 marked mandatory is closed.

---

## 14. Strongest defensible research direction

**Proposed title direction:** "When does relational structure matter for healthcare fraud? A controlled benchmark and real-world validation of heterogeneous GNNs versus graph-augmented tabular models."

**Core thesis, to be tested rather than assumed:** a relational model's advantage over a strong tabular model equipped with graph-derived features is governed by:

1. The ratio of feature-level to structural signal in the fraud.
2. Ring novelty, i.e., whether the ring's entities were seen during training.
3. Camouflage, i.e., how much legitimate activity ring members generate.

The prediction is that GNNs help mainly when rings are *novel and feature-camouflaged*. When fraud is feature-separable, or rings are already known entities, GB with graph features is equal or better at a fraction of the cost.

**Why this is defensible**
- It turns the paper's original question into a falsifiable, parameterized experiment.
- It is robust to either outcome.
- It engages directly with GADBench and the AML benchmark literature.
- It exploits the one real advantage of synthetic data: controllable ground truth.

**Required ingredients:** E1–E8 (mandatory), with E9, E10, and E12 strongly recommended.

**Secondary contribution:** a quantified catalogue of evaluation pitfalls observed in this very codebase, each shown to inflate or distort results on the corrected benchmark: identifier leakage, entity memorization, self-inclusion aggregates, unidirectional heterogeneous edges, and positional misalignment.

**Not recommended:**
- Re-submitting the current story ("HGT +14.2 pp on rings") after merely fixing numbers. The repository's own data contradict it.
- Framing the paper around the deployment system.

---

## 15. Complete issue register

### 15.1 Mandatory validity fixes (must be resolved before submission)

| ID | Issue | Severity | Ref |
|---|---|---|---|
| M-01 | Remove all results without generating evidence; build the experiment ledger | Critical | B-01 |
| M-02 | Remove `ring_id`, `fraud_prone`, and other ground-truth columns; allowlist plus test | Critical | B-02 |
| M-03 | Redesign the generator so the hypothesis is testable; regime parameters; replicates; datasheet | Critical | B-03, E3 |
| M-04 | Fix HGT information flow (reverse relations; no `try/except`; flow tests) | Critical | B-04 |
| M-05 | Implement or remove RGCN, HAN, SimpleHGN, all ablations, split and leaky comparisons, latency, hybrid | Critical | B-05 |
| M-06 | Statistical protocol: replicates, correct CIs, paired tests, multiplicity correction, effect sizes | Critical | B-06 |
| M-07 | ID-based alignment in all evaluation code; stable sorting | Critical | B-07 |
| M-08 | Working pinned environment and container; versions logged | Critical | B-08 |
| M-09 | Align Methods with code (features, split, hyperparameters, tuning, seeds, edges, mode, hardware) | Critical | B-09 |
| M-10 | Ring-disjoint inductive and temporal-emergence evaluation; time-respecting graph; train-only normalization | Critical | B-10 |
| M-11 | Remove the "validated vs CMS literature" claim; rename the mislabeled artifact; fix the Kaggle pipeline before use | Critical | B-11 |
| M-12 | Verify and replace every citation; add DOIs; cite tools used; remove uncited entries | Critical | B-12 |
| M-13 | Fix abstract and conclusion contradictions | High | B-13 |
| M-14 | Define evaluation subsets (including negatives) and report counts | High | EV-1 |
| M-15 | PR-AUC primary; recall at a fixed alert budget; confusion matrices | High | EV-2 |
| M-16 | Validation-selected thresholds; consistent class handling | High | ME-9 |
| M-17 | Remove the calibration claim or measure calibration | High | ME-10 |
| M-18 | MLP control and tabular+graph-feature baselines | Critical (merged into M-05 for counting) | ME-4 |
| M-19 | Equal tuning budgets and equal data budgets across families | High | ME-5, LK-9 |
| M-20 | Out-of-fold or prior-only training aggregates (no self-inclusion) | High | DV-12 |
| M-21 | Real-world validation on at least one public dataset (provider-level, leakage-safe) | Critical (merged into M-11 for counting) | E8 |
| M-22 | Rewrite Related Work; narrow novelty; remove "first" claims; position against GADBench and AML benchmarks | High | RC-2, RC-3, MQ-2 |
| M-23 | Correct the HGT equations; remove "(ours)" | Medium | ME-13, RC-4 |
| M-24 | Regenerate all figures from logged results; commit them; reference each one | High | MQ-5 |
| M-25 | Section 5 and Appendix A: measure what is served, or remove; match the code (endpoints, tiers, drift, pages) | High | §4.8 |
| M-26 | Remove mock-asserting tests; add validity tests; make CI fail on error | High | RP-5, RP-6 |
| M-27 | Seed all RNGs; deterministic flags; save per-seed predictions | Medium | RP-4, ME-8 |
| M-28 | Single canonical data tree; repo-anchored paths; remove the stale 10k dataset from experiment paths | High | DV-2, RP-2 |
| M-29 | Regenerate stale and mutually inconsistent logs from a clean run | High | RP-3 |
| M-30 | Address temporal realism, or drop the temporal-correctness and random-split claims | High | DV-10 |
| M-31 | Fix tie handling at date boundaries | Low | DV-11 |
| M-32 | Rewrite Limitations | High | MQ-8 |
| M-33 | Add declarations (Funding, Ethics N/A, Consent N/A, Code/Data availability with license and DOI; CRediT) | Medium | MQ-9, RP-8 |
| M-34 | Add a code license and data license | Medium | DV-3 |
| M-35 | Fix table errors (bolding, "above 0.997", Table 7 coherence) | Medium | MQ-6 |
| M-36 | Fix textual factual errors (RGCN "homogeneous", "8×", scale claim, node-order "causality") | Medium | MQ-7, ME-14 |
| M-37 | Define provider-level ring metrics formally | High | EV-9 |
| M-38 | Report real seed and replicate variability (no implausible standard deviations) | High | EV-10 |
| M-39 | Remove or substantiate the leaky-protocol narrative | High | RC-7 |
| M-40 | Remove the hybrid "recall parity" claim unless measured end to end | High | ME-12, C-37 |
| M-41 | Remove the scale claim unless E11 is run | Medium | MQ-7 |
| M-42 | Report actual training times and hardware | Medium | ME-11, RP-10 |
| M-43 | Graph schema: justify relations; add ring-bearing relations, or explain their absence | High | ME-6 |
| M-44 | Implement HAN meta-paths properly (requires reverse relations) | Medium | ME-7 |
| M-45 | Train-only normalization of graph node features | Medium | LK-4 |
| M-46 | Single, implemented GNN protocol (transductive or inductive), described accurately | High | LK-7 |
| M-47 | Remove or blocklist the `doctors.fraud_prone` / `patients.fraud_prone` leakage vectors | Medium | DV-7 |
| M-48 | Correct the leakage statement in the evaluation note (tex:L152-159) to match what is actually computed | Medium | C-14 |
| M-49 | Clean "IJISA" remnants and venue-specific strings | Low | RP-9 |
| M-50 | Adopt the REFORMS checklist | Medium | MQ-10 |

### 15.2 Optional improvements (strengthen the paper; not validity blockers)

| ID | Improvement | Severity if omitted | Ref |
|---|---|---|---|
| O-01 | Fine-grained phase diagram (beyond the mandatory coarse grid) | Medium | E7 |
| O-02 | Scale study (×10, ×100) | Medium | E11 |
| O-03 | Unsupervised dense-subgraph baselines (FRAUDAR / CopyCatch-style) | Medium | E12 |
| O-04 | Camouflage and adversarial robustness | Medium | E10 |
| O-05 | Label noise, partial membership, delayed labels | Medium | E10 |
| O-06 | Calibration methods and cost-sensitive curves | Low | E13 |
| O-07 | GNN explanations with fidelity metrics | Low | E14 |
| O-08 | Second real-world dataset | Medium | E8 |
| O-09 | Realistic background claims (e.g., Synthea) | Low | DV-4 |
| O-10 | Temporal GNNs | Low | E16 |
| O-11 | Model cards | Low | MQ-10 |
| O-12 | Trim Appendix A to a short system note | Low | §4.8 |
| O-13 | Replace default secrets; deployment hygiene | Low | §4.8 |

### 15.3 Severity tally

Merged items (M-18 into M-05; M-21 into M-11) are counted once, under the item they merge into.

| Severity | Mandatory | Optional | Total | IDs |
|---|---|---|---|---|
| Critical | 12 | 0 | 12 | M-01 to M-12 |
| High | 21 | 0 | 21 | M-13 to M-17, M-19, M-20, M-22, M-24 to M-26, M-28 to M-30, M-32, M-37 to M-40, M-43, M-46 |
| Medium | 13 | 6 | 19 | M-23, M-27, M-33 to M-36, M-41, M-42, M-44, M-45, M-47, M-48, M-50; O-01 to O-05, O-08 |
| Low | 2 | 7 | 9 | M-31, M-49; O-06, O-07, O-09 to O-13 |
| **Total** | **48** | **13** | **61** | |

Several items are cross-referenced to the same underlying blocker. Ready to submit means **every M-item is closed**.

---

## Appendix A: Reproduction code for the audit checks

Run from any directory with the repository's `.venv` interpreter, passing the repository root as `REPO`. Every check is read-only.

### A.1 Generator-bound rule and `ring_id` leakage

```python
import json, numpy as np, pandas as pd
REPO = "/path/to/FraudGuard"
c = pd.read_csv(f"{REPO}/data/raw/claims.csv")
rule = (c.claim_amount > 8500) | (c.num_procedures > 5) | (c.days_in_hospital > 5)
tp = (rule & (c.fraud_label == 1)).sum(); fp = (rule & (c.fraud_label == 0)).sum(); fn = (~rule & (c.fraud_label == 1)).sum()
print("rule F1", 2*tp/(2*tp+fp+fn), "FP", fp, "ring recall", (rule & (c.ring_id > 0)).sum() / (c.ring_id > 0).sum())
fn_ = json.load(open(f"{REPO}/data/processed/feature_names.json")); print("ring_id index:", fn_.index("ring_id"))
```

### A.2 HGT information flow

```python
import sys, torch; sys.path.insert(0, f"{REPO}/training")
from src.models.gnn import build_gnn_model
d = torch.load(f"{REPO}/data/processed/hetero_graph.pt", weights_only=False)
m = build_gnn_model(d).eval()
with torch.no_grad():
    base = m(d.x_dict, d.edge_index_dict)
    x = {k: v.clone() for k, v in d.x_dict.items()}; x["doctor"] = torch.randn_like(x["doctor"]) * 100
    print("Δ doctor perturbation:", (m(x, d.edge_index_dict) - base).abs().max().item())          # 0.0
    ei = {k: v for k, v in d.edge_index_dict.items() if k[1] != "treated_by"}
    print("Δ remove treated_by:", (m(d.x_dict, ei) - base).abs().max().item())                   # 0.0
```

### A.3 Positional misalignment in the repository's evaluation

```python
fr = pd.read_csv(f"{REPO}/data/processed/features_raw.csv"); fr["claim_date"] = pd.to_datetime(fr.claim_date)
s = fr.sort_values("claim_date").reset_index(drop=True)          # as in baseline._load_test_claims
print((s.claim_id.values[-420:] != fr.claim_id.values[-420:]).sum(), "of 420 test positions misaligned")  # 144
```

### A.4 Mixed-recall feasibility

```python
r_iso, r_ring, r_mix = 99.30, 72.40, 99.45   # Table 3, Gradient Boosting
print("feasible:", min(r_iso, r_ring) <= r_mix <= max(r_iso, r_ring))   # False
```

The full diagnostic scripts used for Section 5 (data audit, model retraining, graph audit, HGT runs, consistency checks) can be added to the repository under `paper/audit_scripts/` on approval.

---

## Appendix B: Provenance timeline of the manuscript's numbers

| Date | Commit | Event |
|---|---|---|
| 2026-04-07 | `b0636c5` | Initial commit. 10,000-claim dataset (`generate_synthetic.py`, no rings). `models/comparison.json` (leaky): LR / RF / GB F1 0.9946 / 0.9973 / 0.9973, `gnn_hgt: {}`. `MOCK_RESULTS` added with HGT acc .9571 / recall .8843 / F1 .8971 / AUC .9801. |
| 2026-04-16 | `8c61b22` | `models/comparison.json` updated: LR acc **0.9455** (F1 0.797), RF acc **0.9895** (F1 0.9572), GB acc **0.9905** (F1 0.9615). HGT entry acc 0.9935 / precision "0.9910" / F1 0.9678, formatted as if by hand. The frontend hard-codes the same values. |
| 2026-04-16 | `aacb30e` | File moved to `training/models/comparison.json`. Still present. |
| 2026-04-20 | `a8ba35f` | `fraud_detection_paper.tex` first contains the Table 2 row "Gradient Boosting & 99.05 & 98.22 & 99.45 & 98.83 & 0.997". The accuracy values match the `8c61b22` artifact (older 10k dataset); no artifact contains the F1, precision, recall, or AUC values. |
| 2026-04-20 | `26425e5` → `432518b` | `generate_paper_figures.py` hard-codes all Table 2 values, a 99.9% "before fix", and an "estimated" confusion matrix. Deleted the same day. |
| 2026-05-22 | `cdd91c5` | 2,100-claim ring generator, logs (tabular F1 = 1.0; HGT "not run"), and `shap_comparison.py` with hard-coded ablation deltas and "+14.2pp" narrative. |
| 2026-05-23 | `4d4f1a0` | "Ready for Latex Rewrite". |
| 2026-10-19 | working tree | Current manuscript. 7-model tables, ablations, latency, hybrid. None generated by repository code. |

*End of audit. No remediation will be undertaken until the authors review and explicitly approve this audit.*
