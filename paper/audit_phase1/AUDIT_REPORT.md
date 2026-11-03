# HISS manuscript readiness audit, Phase 1 (2026-10-28)

**Manuscript:** "When Does Relational Structure Help Healthcare Fraud Detection? A Controlled Benchmark of
Heterogeneous Graph Neural Networks and Graph-Augmented Tabular Models."

**What was audited.** `fraud_detection_paper.tex` and `supplementary_material.tex` at commit `7b154fe` (tag
`hiss-submission-v1`, on the public GitHub remote), the code under `research/`, every run record under `results/`,
the generated tables and figures, and the review documents under `paper/hiss_review/`.

**Scope caveat.** The brief names `fraud_detection_paper(5).pdf`. That file is not on this machine: I searched the
repository, `~/Downloads` and Spotlight. The two-column `paper/hiss_submission/fraud_detection_paper_compiled.pdf`
matches the audited source (its `.tex` differs only in layout commands). Two tracked PDFs are stale and must not be
circulated: `paper/fraud_detection_paper.pdf` (compiled 27 Oct; says "Fourteen equally tuned methods"; has no
BWGNN, CARE-GNN or ring shapes) and `paper/supplementary_material.pdf` (10 pages, against 16 for the current
`ESM_1.pdf`). If `(5).pdf` is not the current source, re-check this report against it.

**Correction (2026-11-01).** The first version of this report said test labels are never used for model selection.
That is true within a rotation, but not for the shared public-data tuning; the executive summary and Section F now
say so. Plan-file history and the confirmatory Wilcoxon test were also checked and added (B13, C11).

**Correction (2026-11-01, after the fifth round).** M2 called `claim_closure` "the decisive feature". The pre-declared
closure experiment does not support that word. Giving a GNN the two closure columns recovered 82–90% of SAGE's gap and
52–84% of HGT's, but GB-CAG without them kept 79–98% of its structure gain. The columns are sufficient for most of the
gap and not necessary for GB-CAG. The expressivity point of M2 stands; "decisive" has been removed from it.

**Correction (2026-11-02): this audit missed a public dataset.** M5 described Ma et al. (IJCAI 2023) as using "real
insurer data", and the report accepted the manuscript's statement that public data with labelled collusion do not exist.
Both were wrong. Ma et al.'s claim table sums to 558,211 claims (40,474 inpatient, 517,737 outpatient), exactly this
study's public dataset, and they cite the same Kaggle page. Their expert group-fraud flags are public at
<https://github.com/finint/RDPGL> and cover 100 providers. I had read only the abstract. Reading the dataset section, in
order to check a table cell, is what found this. The statement is corrected in the manuscript, and a pre-declared
sixth round analyses the flags (`paper/hiss_review/analysis_plan_round6.md`). The lesson for an audit: a claim of
absence ("no public data exist") needs a search aimed at disproving it, and a closest work needs its data section read.

**What was changed.** Nothing in the manuscript, code or `results/`. This folder (`paper/audit_phase1/`) is new
and untracked. About five CPU-minutes of re-training were run, with outputs written to temporary directories only.

**Status legend used below**

| Code | Meaning |
|---|---|
| **V-IND** | Recomputed independently from raw run records or raw data with my own code (`scripts/recompute.py` and ad hoc checks), without `analyze.py` |
| **V-REPRO** | Re-trained with the HEAD code; test scores identical to the stored predictions (max \|Δ\| = 0) |
| **V-CHK** | Covered by the repository's `paper/check_manuscript_numbers.py` (520/520 pass, re-run today) or `sync_manuscript --check`; I did not recompute it independently |
| **PARTIAL** | Numbers correct; the interpretation or scope goes beyond the evidence |
| **UNSUPPORTED** | No presented evidence supports the statement |
| **CONTRADICTED** | The records show otherwise |

---

## Executive summary

* **No BLOCKER.** Every number I recomputed matched the raw records: 522 table cells (Tables 1, 2, 3, 4 and 7), all
  public-data descriptives, the abstract's derived ranges and the confirmatory test. 37 method-runs re-trained with the
  current code reproduced their stored test scores exactly. These include runs from the first round, runs from source
  states that were never committed, and real-data runs. I found no label leakage, and the provider splits are
  disjoint. No synthetic test period and no rotation's test fold is used to train, early-stop or threshold a model.
  One qualification: the shared public-data hyperparameters were chosen on a separate split of the same providers,
  so about 80% of each test fold's providers had their labels used during that tuning (D5, L6). The nested analysis
  removes this, but for the first repetition only.
* **Five MAJOR issues**, all fixable before submission:
  1. **M1, tuning adequacy.** Each GNN sees 10 of 144 configurations, and every later "re-tuning" re-selects from the
     **same** 10 candidates. The central claim that GNNs do not beat GB-CAG is conditional on this.
  2. **M2, scope of "message passing alone".** The claim-closure feature essentially counts 4-cycles, which
     message-passing GNNs cannot count in general (Chen et al. 2020, uncited). The claim holds for the GNN class tested, not for GNNs
     in general.
  3. **M3, unsupported claim.** "A re-analysis of a small fixed benchmark shows how such data can suggest the opposite
     conclusion": the re-analysis runs no GNN.
  4. **M4, provenance sentence contradicted.** The text says every later-round run came from committed, unmodified
     code. 33 run files come from source states that match no commit. Sampled reruns are bit-exact, so the results
     stand, but the sentence is false as written.
  5. **M5, novelty positioning.** Three uncited studies of graph-based and collusive health-insurance fraud: an
     IJCAI 2023 GNN for organised medical-insurance fraud, a TVCG 2023 system for collusive fraud, and a deployed
     graph-analysis system in AI Magazine 2016.
* **Recommendation: GO for manuscript revision, NO-GO for submission as it stands** (Section F).

---

## A. Claim-to-evidence matrix

Section numbers follow the compiled article (1 Introduction … 11 Conclusion). Table and figure numbers follow source
order: Table 1 `tab:regime_ringap`, 2 `tab:pooled`, 3 `tab:regime_isoap`, 4 `tab:lowvolume`, 5 `tab:ablation`,
6 `tab:protocol`, 7 `tab:kaggle`, 8 `tab:map`, A1 `tab:generator`; Fig. 1 protocol, 2 schema, 3 regime, 4 Δ-heatmaps,
5 dose-response, 6 ring graph, 7 public-data AP.

### A.1 Abstract and contributions

| # | Location | Claim | Source artifact | Status | Discrepancy / note |
|---|---|---|---|---|---|
| A1 | Abstract, Methods | "Sixteen equally tuned methods" | `research/experiments/tune.py`; `results/tuning/best_hyperparameters.json` | **CONTRADICTED (wording)** | 13 methods were tuned with up to 10 configurations. The two hybrids inherit their base configuration (no budget). The overlap score is not tuned. → m1, M1 |
| A2 | Abstract, Methods | Nine regimes × 12 seeds; four ring volumes; five ring shapes; public data | `results/synthetic` (108 worlds, 16 methods each); `synthetic_volume8/lowvolume/volume2`; `synthetic_shapes` (48 worlds) | V-IND | Volumes 8, 4 and 2 and the shapes use seeds 1–6 only (stated in the body, not the abstract) |
| A3 | Abstract, Results | Aggregates detect most rings at default volume (AP 0.89–0.92) | `synthetic` runs, GB-CA at δ = 0 | V-IND | 0.886–0.919 |
| A4 | Abstract, Results | Graph features add up to 0.10 for dense rings | `synthetic_ring_ap_comparisons.csv` | V-CHK; Table 1 V-IND | +0.104 at δ = 0, s = 0.9 |
| A5 | Abstract, Results | At 4–8 claims/month: aggregates 0.05–0.42; graph features 0.41–0.92 for denser rings; pure GNNs trail by 0.17–0.57 | `synthetic_volume8`, `synthetic_lowvolume` | V-IND | 0.049–0.422; 0.409–0.923; 0.169–0.573. Six seeds, frozen hyperparameters. Depends on GNN tuning (M1) |
| A6 | Abstract, Results | No pure GNN beats GB-CAG over regimes (upper 95% limits ≤ 0.006) | Table 2 | V-IND | Per-comparison limit. The Bonferroni-simultaneous limit over the 7 pure GNNs is +0.009. Restricting to δ = 0 lowers the GNNs further (SAGE −0.005, upper limit +0.003), so the claim is not an artifact of the ceiling regimes. → m8 |
| A7 | Abstract, Results | … "or for any other ring shape" | `shapes_comparisons.csv` | V-CHK | — |
| A8 | Abstract, Results | Hybrids gained 0.01 | Table 2 (+0.010, +0.009) | V-IND | At δ = 0 (seeds 1–6), averaging over three training seeds shrinks SAGE+AG's gain from +0.013 to +0.009, not significant (Online Resource S6). The confirmatory test used training seed 0 only. → m10 |
| A9 | Abstract, Results | GNNs 0.09–0.23 worse on isolated fraud | Table 2 | V-IND | −0.093 to −0.229 |
| A10 | Abstract, Results | Recurring members inflate performance by up to 0.60 | `protocol_gap.csv` (MLP-C +0.604) | V-CHK | Persistent and emergent worlds of a seed do not share their background. → m5 |
| A11 | Abstract, Results | Public data: GB on provider aggregates best (0.704) | `results/kaggle` | V-IND; V-REPRO | Folds 0 and 3 re-trained bit-exact |
| A12 | Abstract, Conclusion | A few engineered features captured structure "better than message passing alone" | — | **PARTIAL** | True for the message-passing GNNs tested with a 10-candidate search. → M1, M2 |
| A13 | §1, contribution 1 | Map of when relational structure helps | Tables 1, 4 and 8; dose-response; shapes | V-IND (Tables 1, 4) / V-CHK | Scope as in A12 |
| A14 | §1, contribution 2 | Leakage-tested benchmark; automated tests enforce disjointness and period isolation; code public | `research/tests` (34 pass); CI run 40730861070 on `7b154fe` (research-validity job: success); repository returns HTTP 200 unauthenticated; tag on remote | V | — |
| A15 | §1, contribution 3a | Recurring members inflate by up to 0.60 AP "and can reorder methods" | Table 6 | V-IND (from table values) | At s = 0.1, HGT moves from last to second and GB-CA passes GB-CAG |
| A16 | §1, contribution 3b | Pooled tests that treat worlds as independent overstate significance | `synthetic_pooled.csv` (naive vs. clustered p); `test_statistics.py` | V-CHK | — |
| A17 | §1, contribution 3c; §11 | "A re-analysis of a small fixed benchmark shows how such data can suggest the opposite conclusion" | `results/legacy/legacy_reanalysis.json` | **UNSUPPORTED** | The re-analysis fits only GB on three raw claim attributes (ring AP 1.000). No GNN or graph model is run. The "opposite" result exists only in `archive/manuscript/…_2026-10-19.tex` (HGT ring F1 87.0%, +11.2 points over GB), which the research README says must not be cited. → M3 |
| A18 | §1, contribution 4 | GNNs detect dense rings by counting shared counterparties | Table 5; Online Resource S10, S11 | V-CHK; interpretation **PARTIAL** | The degree-encoding dependence of HGT is shown. The cycle-counting limit of message passing is not discussed. → M2 |
| A19 | §1, contribution 5 | Provider-disjoint, equal budgets, repeated CV, per-rotation selection, corrected tests | `kaggle`, `kaggle_repeated` (25 rotations × 15 methods), `kaggle_nested` (5 × 15) | V-IND | Labels are provider-disjoint, but the graph is transductive (stated in §5.3). About 80% of each test fold's providers were labelled in the shared-tuning split. → m11 |

### A.2 Benchmark, methods and statistics (§§3–4)

| # | Location | Claim | Source artifact | Status | Discrepancy / note |
|---|---|---|---|---|---|
| B1 | §3.1, Table A1 | Generator sizes, rates, ring shape and volume | `research/synth/generator.py` defaults | V-IND (code) | — |
| B2 | §3.1 | The regimes of a seed share all claims except ring claims | generator draw order; `test_statistics.py`; my check (s = 0.1 vs. 0.9 legitimate claims identical) | V-IND | — |
| B3 | §3.1 | Ring patients 9.3 vs. 1.6 claims; ring doctors 38 vs. 16; low volume 3.7 and 20 | `volume_signal.csv` | V-CHK | — |
| B4 | §3.1 | Persistent and emergent protocols have identical claim and ring-claim volumes per period | generator | V (expected volumes) | Backgrounds differ: seed 1 has 17,908 vs. 17,943 legitimate claims. → m5 |
| B5 | §3.2 | A claim's features and graph use only claims of the same period | `features.py`, `graph.py`; tests (ii), (viii) | V-IND (code) with one exception | `amt_vs_specialty` uses the specialty means of training periods 1–2 for every period. This is label-free and never uses future information for validation or test. → m4 |
| B6 | §4.3 | Shared projection, residual + LayerNorm and head; BWGNN and CARE-GNN keep their published structure | `models/gnn.py` | V (code) | — |
| B7 | §4.3 | Hybrids reuse their base configuration and get no separate budget | `tune.py --merge`; hyperparameters recorded in every run match the tuning file (4,800+ method-runs checked) | V-IND | `results/tuning/trials_*.json` also holds 10 unused SAGE+AG trials. → m16 |
| B8 | §4.3 | Neural models: AdamW, class-weighted BCE, early stopping (patience 20, **at most 200 epochs**) | `models/train.py` | **CONTRADICTED (minor)** | `fit_mlp` allows up to 300 epochs and does not clip gradients; the GNNs clip at 1.0. → m2 |
| B9 | §4.3 | One training seed; identical initial weights across worlds and tuning | `train.py` (`set_seed(0)`) | V (code) | Disclosed. → L1 |
| B10 | §4.4 | Equal budget of up to 10 configurations on development worlds 901 and 902 (δ = 0.5, s = 0.5) | `tune.py`; trial files | V-IND | The budget covers 6.9% of each GNN space, 9.3% for GB and 100% for LR. Candidate lists are fixed by sampling seed 0 and reused verbatim in every re-tuning. → M1 |
| B11 | §4.3 (implicit) | GB = `HistGradientBoostingClassifier` with the tuned `max_iter` | `train.py` | **PARTIAL** | scikit-learn's `early_stopping='auto'` switches on above 10,000 training rows. That happens for seeds 2, 4, 5 and 6 (10,005–10,130 claims) but not 1 or 3 (9,921 and 9,888). It holds out 10% of the training claims internally. Not a leak, but undisclosed. → m3 |
| B12 | §4.5 | Routing stages reproduce stored predictions exactly (GNNs) and within 3e-8 (GB-CAG) | `results/cascade/runs/*` reproduction fields; `verify.py` | V-CHK | That reproduction ran at commit `a232dcd` (23 Oct), not the tagged release. → M4 |
| B13 | §4.6 | Seeds 7–12 run under a plan committed before any run | plan `6c6a0c2` at 10:32 UTC on 26 Oct; first round-2 run at 10:56 UTC | V-IND | Rounds 3 and 4 likewise: plans `2eb368c` and `98638c4` precede their runs. Each plan file has exactly one commit, so none was edited after its runs began |
| B14 | §4.6 | Holm adjustment "within each pre-specified family" | `analyze.py` | **PARTIAL** | Round-1 families first appear in commit `cfbc4f1` together with the round-1 results. They cannot be shown to be pre-specified; rounds 2–4 can. → m9 |
| B15 | §4.6 | Seed-level t-tests; 95% t-CIs | `metrics.clustered_comparison` | V-IND (Table 2 recomputed, Holm p included) | — |

### A.3 Results on the controlled benchmark (§§6–7)

| # | Location | Claim | Source artifact | Status | Discrepancy / note |
|---|---|---|---|---|---|
| C1 | §6 intro | 4,943 test claims; 384 ring; 133 isolated; chance 0.080 and 0.029; recall cap 0.479 | runs | V-CHK | — |
| C2 | Table 1 | Ring AP, 16 methods × 9 regimes | `synthetic` runs → `synthetic_ring_ap.tex` | **V-IND (144/144 cells)** | — |
| C3 | Fig. 3 | Ring AP with 95% intervals | `figures.py` → `fig_regime_ring_ap.pdf` | V (file identical to `Fig3.pdf`; regenerated in `0094fab`) | Not regenerated here, because that would overwrite `results/figures` |
| C4 | §6.1–6.3 | GB-C/MLP-C near chance; GB-CA 0.886–0.919; structure gain +0.005/+0.074/+0.104; pooled gains over GB-CA | comparisons CSVs | V-CHK (Table 1 V-IND) | — |
| C5 | Table 2 | Pooled Δ vs. GB-CAG with half-width and Holm p (ring, isolated, overall) | `synthetic` runs | **V-IND (30/30 cells)** | — |
| C6 | §6.4 | One significant per-regime GNN win (RGCN, δ = 2, s = 0.1, +0.005) | `synthetic_ring_ap_comparisons.csv` | V-CHK | — |
| C7 | §6.4 | Hybrid gain; training-seed replication; per-regime re-tuning changes nothing | `synthetic_modelseeds`, `synthetic_regimetuned` | V-CHK | "Re-tuning" re-selected from the same 10 candidates. → M1 |
| C8 | Fig. 4 | Δ heatmaps | `fig_delta_heatmaps.pdf` | V (identical to `Fig4.pdf`) | — |
| C9 | Table 3; §6.5 | Isolated AP; every GNN worse; MLP within 0.025 | `synthetic` runs | **V-IND (144/144)**; interpretation **PARTIAL** | "Message passing dilutes" is presented as established. The MLP control differs from the GNNs in more than message passing. → m13 |
| C10 | Table 4 | Low-volume ring AP: default, frozen, re-tuned | `synthetic`, `synthetic_lowvolume`, `…_retuned` | **V-IND (144/144)**; V-REPRO (re-tuned s = 0.1, seed 1: GB-CA, GB-CAG, SAGE+AG) | — |
| C11 | §6.6 | Confirmatory test: SAGE+AG − GB-CAG = +0.024 [0.005, 0.043], p = 0.016, 9/12 seeds | `synthetic_hybrid_confirm` | **V-IND** | Matches plan C of `analysis_plan_round2.md` (two-sided paired t, α = 0.05, seeds 7–18, two methods). The planned Wilcoxon test is reported in the Online Resource (p = 0.009) and matches `hybrid_confirmation.json`. → m10 |
| C12 | §6.6; Fig. 5 | Dose-response over volumes 16, 8, 4 and 2 | `dose_response_*`; `fig_dose_response.pdf` | V-CHK; figure identical to `Fig5.pdf` | Fig. 5 omits the fraud-specialised GNNs (the caption does not claim them) |
| C13 | §6.6 | δ > 0 at volume 4: SAGE, HAN and RGCN beat GB-CAG for sparse rings; LR does too | `lowvolume_shift_comparisons.csv` | V-CHK | — |
| C14 | §6.7 | Shapes: no pure GNN above GB-CAG; 32 of 56 comparisons significantly below; SAGE+AG above for the distributed ring | `shapes_comparisons.csv` | V-CHK | — |
| C15 | §6.8 | Routing ≈ GB-CAG; fused +0.006 | `cascade_*` | V-CHK | — |
| C16 | §6.9 | Training and inference cost | `synthetic_timing.csv`, `latency.json` | V-CHK | Times measured under parallel load (disclosed). → L7 |
| C17 | §7.1; Fig. 6 | GB-CAG flags 81/96 and 91/96 ring claims; HGT 62/96 and 94/96 | `results/cascade/predictions` (seed 1) | **V-IND** | Not covered by the repository's checker |
| C18 | Table 5; §7.2 | Ablations (rewiring, depth, relations, degree encodings) | `ablation`, `ablation_arch` | V-CHK (table synced) | Rewiring is applied to all periods and the model is retrained |
| C19 | §7.3 | Ring-id leak → 1.000; no reverse relations −0.177 | Table 5 | V-CHK | — |
| C20 | Table 6; §7.4 | New vs. recurring rings | `synthetic_persistent` | V-CHK | → m5 |

### A.4 Public data (§§5, 8)

| # | Location | Claim | Source artifact | Status | Discrepancy / note |
|---|---|---|---|---|---|
| D1 | §5.1 | 558,211 claims (40,474 IP, 517,737 OP); Nov 2008–Dec 2009; 5,410 providers (506 = 9.4%); 138,556 beneficiaries; 100,737 physicians | raw CSVs (SHA-256 match `research/README.md`) | **V-IND** | Repository probes found CMS synthetic-file conventions in the beneficiary file (inconclusive). The manuscript does not mention this. → m12 |
| D2 | §5.2 | P = 23, G = 8, L = 4 features; 363,300 *serves* edges; 7,685 shared physicians (16,287 edges); 22 beneficiary attributes | `realdata/kaggle.py` | **V-IND** | — |
| D3 | §5.2 | L uses only training labels and never a provider's own | `label_features` (diagonal removed; `known` restricted to training indices) | V-IND (code) | — |
| D4 | §5.3 | Five stratified rotations (k test, k+1 validation, 3 training); every provider tested once | `kaggle_exp.splits` | **V-IND** | 3,246 / 1,082 / 1,082 providers; fraud 303–304 / 101–102 / 101–102; disjoint and exhaustive |
| D5 | §5.3 | Hyperparameters selected once on a separate 60/20/20 split (seed 1000), budget 6 | tuning trials; run records | V-IND | "Separate" means a different split of the **same** providers. In each test fold, 199–242 providers (18–22%) were in the tuning-validation set and 633–666 (59–62%) in the tuning-training set. Disclosed qualitatively. → m11, L6 |
| D6 | §5.3 | Nested re-selection inside each rotation using only that rotation's validation fold | `kaggle_nested/runs` (75 rows) | **V-IND** | Selection, early stopping and threshold use the validation fold; the test fold is untouched. Choices are listed in C.4 |
| D7 | Table 7 | AP, ROC-AUC, F1, Recall@10% (mean ± s.d. over 5 folds) | `kaggle` runs | **V-IND (60/60 cells)**; V-REPRO for 6 tabular methods (fold 0) and 3 (fold 3) | — |
| D8 | §8 | None of 28 corrected differences significant (smallest Holm p = 0.15); a naive test would flag SAGE+G | `kaggle_comparisons.csv` | V-CHK | The 28 count GB-P vs. GB-PGL twice (once per reference). → m7 |
| D9 | §8 | Bootstrap places every GNN, hybrid and GB-PG below GB-P, "so the shortfall does not depend on which providers happened to be tested" | `kaggle_bootstrap.csv` | **PARTIAL** | The reference was chosen as the best method on the same test folds, and the 14 intervals are unadjusted. → m6 |
| D10 | §8 | 25 rotations: GB-P 0.693; only RGCN significant; nested: GB-P 0.706 best | `kaggle_repeated`, `kaggle_nested` | V-CHK; structure V-IND (25 rotations × 15 methods, no duplicates) | V-REPRO for 6 tabular methods + SAGE (repeat 3, fold 1) and 3 tabular methods (repeat 4, fold 4) |
| D11 | §8 | Descriptives: 99.5% share beneficiaries; median 30; 23.9% vs. 25.3%; 19.1% vs. 14.2%; 73%; median claims 156 vs. 27 | `kaggle_descriptives.json` | V-CHK | — |
| D12 | Fig. 7 | Public-data AP with corrected CIs | `fig_kaggle_ap.pdf` | V (identical to `Fig7.pdf`; contains BWGNN and CARE-GNN) | — |

### A.5 Discussion, reproducibility and declarations (§§9–11, back matter)

| # | Location | Claim | Source artifact | Status | Discrepancy / note |
|---|---|---|---|---|---|
| E1 | Table 8 | Map of conditions | numbers drawn from Tables 1 and 4 and the Online Resource | V-CHK | Scope as in A12 |
| E2 | §9.2 | GNNs cannot count, hence the engineered features win | Table 5 | **PARTIAL** | → M2 |
| E3 | §9.3 | The small fixed benchmark "cannot tell whether relational structure helps" (anomalous ring claims, recurring rings, 21 test ring claims) | `legacy_reanalysis.json` | V-IND | Supported. Only the "opposite conclusion" wording elsewhere is unsupported (A17) |
| E4 | §9.6 | Limitations list | — | V | Missing from the list: M1 (candidate set) and M2 (expressivity) |
| E5 | §10 | Tests (i)–(xi) run in CI | `research/tests`; CI | V | 34 passed locally in 8.6 s; CI research-validity job passed on `7b154fe` |
| E6 | §10 | Every table equals the generated file | `sync_manuscript --check` | V | "tables differing: none" |
| E7 | §10 | "Every run of the later rounds was made from a committed, unmodified version of the code" | run provenance (`logs/provenance.log`, `logs/source_hashes.log`) | **CONTRADICTED** | 33 run files have source hashes matching no commit. Many more are flagged `dirty` (non-`.py` changes). → M4 |
| E8 | §10 | Retraining GB-CAG, SAGE and HGT "with the released code" reproduced stored predictions | cascade runs at `a232dcd` | **PARTIAL** | That reproduction used an earlier commit, not the release. My reruns with the tagged code are bit-exact for 10 synthetic methods (including GB-CAG and SAGE; not HGT) and 7 public-data methods |
| E9 | §10 | REFORMS checklist and datasheet in the repository | `paper/reforms_checklist.md`, `research/DATASHEET.md` | V (exist) | — |
| E10 | Data availability | Checksums listed | `research/README.md` | V-IND | All four match |
| E11 | Code availability | Public repository, tag `hiss-submission-v1` | `git ls-remote`; HTTP 200 | V | DOI promised on acceptance |
| E12 | Supplementary info | Predictions (~30 MB) not tracked; the code regenerates them | `.gitignore` | V | The bootstrap and Fig. 6 need them; archive them with the DOI |

---

## B. Prioritized issue register

### MAJOR

| ID | Issue | Evidence | Recommended fix | New experiment? |
|---|---|---|---|---|
| **M1** | **GNN tuning adequacy is untested, and "re-tuning" is re-selection.** Claims that no pure GNN beats GB-CAG, and that pure GNNs trail by 0.17–0.57 at low volume, are conditional on 10 random configurations per GNN. | `tune.candidate_configs(seed=0)` yields the same 10 candidates every time. Low-volume and per-regime "re-tuning" reused them verbatim: 13/13, 11/11 and 11/11 identical lists (`results/tuning*/trials_*.json`). Budget fraction: GNNs 10/144 = 6.9%; GB 10/108; LR 6/6; RF 10/18; public data 6/144 for GNNs. Several GNNs chose boundary values: layers = 3 (maximum) for SAGE, HAN, HGT and Simple-HGN; hidden = 32 (minimum) for SAGE, HAN and HGT. Validation-AP spread across candidates is 0.029–0.055 for the five general-purpose GNNs (BWGNN 0.008, CARE-GNN 0.019) against 0.007 for GB-CAG, so their performance is configuration-sensitive. The manuscript says "equally tuned" and "re-tuning". The repository's own `reproducibility_report.md` notes "same candidates (sampling seed 0)"; the manuscript does not. | **Text (required):** state the budget as "up to 10 configurations drawn once from each method's grid (6.9% of a GNN grid)"; say that later tunings re-selected among the same candidates; add this to Limitations; replace "equally tuned". **Experiment (recommended):** a pre-declared expanded search for SAGE (sum), RGCN and HGT with fresh candidates (new sampling seed, ≥40 configurations, layers up to 4, hidden up to 128), selected on low-volume development worlds and evaluated at volume 4, δ = 0, s ∈ {0.5, 0.9}, and at default volume δ = 0, s = 0.9 (seeds 1–6). Give GB-CAG the same expanded budget. | **Yes, recommended.** Resolves whether the large low-volume gap survives better tuning. Rough cost from observed run times: 3–6 CPU-hours (estimate, not measured) |
| **M2** | **"Message passing alone" over-generalises; the mechanism misses the expressivity result.** | The `claim_closure` feature is computed as `M = B @ D` in `features.py`. It sums, over the patient's doctors, the patients each shares with the claim's doctor. Apart from a degree term (the patient itself), that is a count of 4-cycles p–d′–p′–d through the claim's patient–doctor edge. Chen et al. (NeurIPS 2020) show that message-passing networks cannot, in general, count such substructures: they cannot induced-subgraph-count any connected pattern of three or more nodes. All seven pure GNNs tested are message-passing networks. The paper's mechanism (§§7.2, 9.2) cites only degree counting (Xu et al. 2019). The hybrids are message-passing networks plus these cycle counts. | **Text (required):** cite Chen et al. (2020); restrict claims to "the message-passing GNNs tested"; reframe §9.2 to separate degree counting (fixed by degree encodings or sum aggregation) from cycle counting (not fixable by message passing). Note that a substructure-aware GNN was not tested. **Experiment (optional):** one substructure-aware or identity-aware GNN variant at volume 4, s ∈ {0.5, 0.9}. | **Optional.** Required only if the authors want to keep a general claim about GNNs |
| **M3** | **Contribution 3 and the Conclusion claim a re-analysis "shows how such data can suggest the opposite conclusion"; no presented result shows that.** | `legacy_reanalysis.json` contains only GB on three raw claim attributes (ring AP 1.000, recall 21/21). No GNN or graph model is run. The only "opposite" result is the 19 Oct archived manuscript (HGT ring F1 87.0%, +11.2 points over GB), whose outputs are labelled unverified. §9.3 and S14 themselves say only that the dataset "cannot tell". | **Text (minimum):** replace with "…cannot separate relational learning from claim-level anomaly detection or from recognising known members". **Or** run GB-CAG, HGT and SAGE on the 2,100-claim dataset with the current pipeline and report the ranking with its 21-claim uncertainty. | **Optional** (minutes of compute) |
| **M4** | **The reproducibility statement is contradicted by the run records.** | §10: "every run of the later rounds was made from a committed, unmodified version of the code". 33 run files have `research_sources_sha256` matching no commit (`logs/source_hashes.log`): 4 round-2 main-grid worlds (δ = 0: s = 0.1 seeds 11 and 12; s = 0.5 seeds 7 and 8); 5 low-volume re-tuned worlds (s = 0.1 seed 1; s = 0.5 seeds 1 and 2; s = 0.9 seeds 1 and 2); and 24 round-2 public-data files (15 nested GNN files for HAN, RGCN and SAGE/SAGE+G × 5 folds; 9 repeated rotations, r3 folds 1–4 and r4 folds 0–4). Many more are flagged `dirty` while their `.py` tree matches a commit, including all 12 confirmatory-test runs (tree = `8c3434a`/`62ef9fa`). §10 also says the routing reproduction used "the released code"; it ran at `a232dcd`. **Mitigation verified:** HEAD reruns are bit-exact for s = 0.1 seed 11 (8 methods), low-volume re-tuned s = 0.1 seed 1 (3 methods), r3 fold 1 (6 tabular methods + SAGE) and r4 fold 4 (3 tabular methods). | **Text (required):** state that the 33 files listed come from uncommitted source states and that other later-round runs had uncommitted non-code changes; replace "released code" with the commit used. **Re-runs:** re-train every method in those 33 files with the tagged code and log the comparison. 4 of the 33 have been partly re-run here (21 method-runs, all bit-exact). | **Re-runs only, no new science.** Roughly 1–3 CPU-hours (estimate), mostly the 15 nested GNN files |
| **M5** | **Related work omits collusion-specific and graph-based healthcare-fraud studies.** | Uncited, verified to exist (abstracts only): Ma et al., IJCAI 2023 (RDPGL: heterogeneous GAT plus community-aware risk diffusion against organised medical-insurance fraud gangs; on this study's own public dataset with added expert group-fraud flags, see the correction above); Zhou et al., IEEE TVCG 29(6) 2023 (FraudAuditor: co-visit network and community detection for collusive health-insurance fraud); Liu et al., AI Magazine 37(2) 2016 (graph analysis of patient–doctor–pharmacy networks for fraud, waste and abuse, deployed). The Related Work says relational information "enters, if at all, through hand-engineered features" and that prior labels "mark providers, patients or activities rather than collusive groups". RDPGL targets organised groups. | **Text (required):** add these, position the paper as a controlled benchmark rather than a detector, and soften the two sentences above. "We found no comparable controlled study" likely survives. | No |

### MINOR

| ID | Issue | Evidence | Recommended fix |
|---|---|---|---|
| m1 | "Sixteen equally tuned methods" (abstract) | B7, B10 | "16 methods: 13 tuned with an equal budget of up to 10 configurations, two hybrids inheriting their base configuration, one unsupervised score" |
| m2 | MLP training differs from the stated protocol | `fit_mlp`: max 300 epochs, no gradient clipping; GNNs: 200 and clipping at 1.0 | State the MLP settings |
| m3 | Undisclosed internal early stopping in GB | `HistGradientBoostingClassifier(early_stopping='auto')` is on above 10,000 rows: seeds 2, 4, 5 and 6 but not 1 and 3, with 10% of the training claims held out internally | Disclose, or fix `early_stopping` explicitly in a future run (would change results; not needed for this revision) |
| m4 | "Built only from claims of the same period" is not exact | `amt_vs_specialty` uses training-period specialty means in every period | Add the exception (label-free, no future information for evaluation) |
| m5 | Persistent–emergent pairing does not hold the background fixed | Seed 1: 17,908 vs. 17,943 legitimate claims; δ/s regimes do share | State it in the Table 6 caption |
| m6 | Bootstrap reference chosen post hoc; unadjusted | `analyze.kaggle_bootstrap` fixes GB-P, the best on the same test folds | Call it descriptive, or drop "does not depend on which providers…" |
| m7 | The Holm family of 28 contains one duplicate pair | GB-P vs. GB-PGL under both references | Say 27 distinct comparisons, or remove the duplicate |
| m8 | "The data exclude a pure-GNN advantage larger than 0.006" | Per-comparison bounds; Bonferroni-simultaneous over 7 GNNs: +0.009 | "…larger than about 0.01 (simultaneous bound +0.009)" |
| m9 | "Pre-specified families" for round 1 | Families and round-1 results first committed together (`cfbc4f1`) | "pre-specified for rounds 2–4" |
| m10 | The confirmatory test confirms over generator seeds only | Training seed 0, as in tuning; the S6 replication found the default-volume hybrid gain non-significant | Say "fresh generator seeds, one training seed". Optionally rerun the 12 confirm worlds with training seeds 1 and 2 (minutes) |
| m11 | "Provider-disjoint" is under-specified | D5: about 80% of each test fold's providers were labelled in the shared-tuning split. All 25 rotations use the shared configurations; nested selection covers the first repetition only | "label-disjoint providers on a transductive graph; tuning used a separate split of the same providers (nested check in §8)" |
| m12 | Real-data provenance is presented more firmly than the evidence allows | Abstract: "public provider-labelled Medicare claims"; the repository's probe found CMS synthetic-file conventions (all dates of birth on the 1st; codes 1/2), inconclusive | "claims described by their publisher as Medicare data"; mention the probe in Limitations |
| m13 | "Message passing dilutes claim-level anomalies" stated as established (section title; "because") | The MLP control differs in degree encodings and architecture; early stopping uses overall AP | "consistent with dilution" |
| m14 | Selection uses overall AP while ring AP is primary | `fit_graph` and tuning use overall validation AP | Disclose as a design choice |
| m15 | Stale artifacts and an unverifiable exclusion | Stale tracked PDFs (scope caveat); `(5).pdf` not found; `analyze.py` says earlier runs are "kept on disk but excluded", but none exist and none were ever committed | Replace the stale PDFs (needs approval); fix the comment; say in the README whether any earlier runs were discarded |
| m16 | Unused hybrid trials in `results/tuning` | 10 SAGE+AG trials; the merge overwrites them with SAGE's configuration | One line in the README |
| m17 | No dense-subgraph or community baseline beyond the authors' own overlap score | FRAUDAR and CopyCatch cited, not run; community detection as in FraudAuditor | Note as a limitation; optional baseline |

### ACCEPTABLE LIMITATIONS (disclose; no experiment needed)

| ID | Limitation | Status in manuscript |
|---|---|---|
| L1 | One training seed per world; identical initial weights across worlds and tuning | Disclosed; S6 replication |
| L2 | Secondary analyses (volumes, shapes, ablations, persistent, routing) on six seeds | Disclosed |
| L3 | Engineered features designed with knowledge of the typology | Disclosed; shapes partly mitigate |
| L4 | Simple generator; 8% ring prevalence; static rings; no adaptive fraud | Disclosed |
| L5 | Real data: provider labels, no rings, undocumented origin, one year, transductive graph | Disclosed (add m12) |
| L6 | Shared tuning split overlaps the test providers | Disclosed; the nested check addresses it (quantify per m11) |
| L7 | Timings under parallel load; fraud-GNN timings not comparable | Disclosed |
| L8 | No public data with labelled collusion | Disclosed; cannot be resolved |
| L9 | Low power of the corrected tests (5 or 25 rotations) | Disclosed |
| L10 | PC-GNN and other fraud GNNs not compared | Disclosed |

---

## C. Reproducibility report

### C.1 Environment

* macOS (Darwin 25.6.0), Intel 4-core laptop. `.venv-research`: Python 3.12.13; numpy 1.26.4, pandas 2.2.3,
  scipy 1.13.1, scikit-learn 1.5.2, torch 2.2.2, torch-geometric 2.6.1. These match `research/requirements.txt`.
* Repository at `7b154fe` (= tag `hiss-submission-v1` = `origin/main`), clean working tree.
* Public data: the four `Train*.csv` files match the SHA-256 values in `research/README.md`.
* GitHub checks were made with the **akshay-hudev** account. Two earlier read-only `gh run` calls in this session
  ran under the then-active akshay-goAI account; the active account was switched and the CI check repeated as
  akshay-hudev.

### C.2 Commands run (from the repository root)

| # | Command | Output checked | Result |
|---|---|---|---|
| 1 | `.venv-research/bin/python -m pytest research/tests -q` | `logs/pytest.log` | **34 passed** (8.6 s) |
| 2 | `.venv-research/bin/python verify.py` | `logs/verify.log` | All artifacts present with provenance |
| 3 | `.venv-research/bin/python -m research.experiments.sync_manuscript --check` | stdout | "tables differing from results/tables: none" |
| 4 | `.venv-research/bin/python paper/check_manuscript_numbers.py` | `logs/check_manuscript_numbers.log` | **520/520** |
| 5 | `shasum -a 256 data/external/kaggle_provider_fraud/Train*.csv` | stdout | 4/4 match |
| 6 | `PYTHONPATH=. .venv-research/bin/python paper/audit_phase1/scripts/provenance.py` | `logs/provenance.log` | Commit, dirty flag and source hash for every run set |
| 7 | `.venv-research/bin/python paper/audit_phase1/scripts/source_hashes.py` | `logs/source_hashes.log` | 3 source states match no commit (M4) |
| 8 | `.venv-research/bin/python paper/audit_phase1/scripts/recompute.py` | `logs/recompute.log` | Tables 1, 2, 3, 4 and 7: **522 cells, 0 mismatches**; confirmatory test reproduced |
| 9 | `PYTHONPATH=. RESEARCH_THREADS=2 .venv-research/bin/python paper/audit_phase1/scripts/repro_synth.py synthetic results/tuning/best_hyperparameters.json emergent_d0.0_s0.1_g11_m0 GB-C GB-CA GB-CAG LR-CA RF-CA MLP-C SAGE SAGE+AG` | `logs/repro_synth.log` | 8/8 bit-exact (uncommitted-source world) |
| 10 | `… repro_synth.py synthetic results/tuning/best_hyperparameters.json emergent_d0.0_s0.9_g1_m0 GB-CA LR-CA RF-CA MLP-C RGCN SAGE+AG HGT+AG` | same log | 7/7 bit-exact (first-round world, no source hash) |
| 11 | `… repro_synth.py synthetic_lowvolume_retuned results/tuning_lowvolume/best_hyperparameters.json emergent_d0.0_s0.1_v4_g1_m0 GB-CA GB-CAG SAGE+AG` | same log | 3/3 bit-exact (uncommitted-source world) |
| 12 | `PYTHONPATH=. RESEARCH_THREADS=2 .venv-research/bin/python paper/audit_phase1/scripts/repro_kaggle.py` | `logs/repro_kaggle_tabular.log` | 18/18 bit-exact (repeat 0 folds 0 and 3; r3 fold 1; r4 fold 4) |
| 13 | `… repro_kaggle_gnn.py 3 1 kaggle_repeated r3_fold1_main SAGE` | `logs/repro_kaggle_sage.log` | Bit-exact; AP 0.6957; best epoch 7 (uncommitted-source rotation) |
| 14 | `gh run view 40730861070 --repo akshay-hudev/Fraud-Guard --json jobs` | stdout | Research-validity CI job: success on `7b154fe` |

Ad hoc checks, run inline and summarised here: hyperparameters recorded in every run file against the tuning
files (0 mismatches over 4,800+ method-runs); identical candidate lists across tunings; persistent vs. emergent
backgrounds; GB training-set sizes; public-data counts and split sizes; Fig. 6 counts; pooled bounds by δ subset;
plan-commit vs. first-run timestamps; git history of deleted run files (none).

### C.3 Failures

* **No pipeline failure.**
* Two failures were bugs in my audit scripts: a wrong prediction-file name (`fold0_main` → `fold0_fast`) and a LaTeX
  table parser. Both were fixed and the scripts re-run; the copies here are the fixed versions.

### C.4 Real-data model selection, per split

Reported rotations (repeat 0): every fold has 3,246 training, 1,082 validation and 1,082 test providers, pairwise
disjoint and exhaustive. Test folds contain 101–102 fraudulent providers. Test-fold overlap with the shared-tuning
split: 199–242 providers with the tuning-validation set and 633–666 with the tuning-training set.

**Shared selection** (one configuration per method, used in all 25 rotations; recorded in every run file, 0
mismatches):

| Method | Configuration |
|---|---|
| LR-P | C = 1 |
| RF-P | depth 16, max_features 0.3, min_leaf 10, 300 trees |
| GB-P, GB-PG, GB-PGL | lr 0.03, max_iter 200, 15 leaves, min_leaf 50, l2 0 (identical, from the same candidate list) |
| MLP-P | 128 hidden, 2 layers, dropout 0.3, lr 1e-3, wd 1e-5 |
| SAGE, SAGE+G | mean, 3 layers, 64 hidden, dropout 0.3, lr 3e-3, wd 1e-4 |
| RGCN | sum, 2 layers, 64, dropout 0.3, lr 1e-2, wd 1e-3 |
| HAN | 2 heads, 2 layers, 64, dropout 0.1, lr 3e-3, wd 1e-4 |
| Simple-HGN, HGT, HGT+G | 4 heads, 2 layers, 64, dropout 0.1, lr 1e-2, wd 1e-3 |
| BWGNN | order 3, 64, dropout 0.3, lr 3e-3, wd 1e-3 |
| CARE-GNN | 2 layers, 64, dropout 0.1, lr 1e-3, wd 1e-4 |

**Nested selection** (repeat 0; selected on the rotation's validation fold, test fold untouched). Number of
rotations, out of 5, in which the shared configuration was re-selected:

| Re-selected in | Methods |
|---|---|
| 5/5 | SAGE, SAGE+G |
| 2/5 | BWGNN, GB-PG, GB-PGL |
| 1/5 | HAN, HGT, HGT+G, MLP-P, Simple-HGN |
| 0/5 | GB-P (always min_leaf 10 instead of 50), LR-P, RF-P, RGCN, CARE-GNN |

The full per-fold configurations are in `results/kaggle_nested/runs/*.json` (field `hp`).

### C.5 Not verified

| Item | Why not verified |
|---|---|
| Whether `fraud_detection_paper(5).pdf` equals the audited source | File not found |
| Full regeneration of tables and figures with `analyze.py` and `figures.py` | They overwrite `results/`, which was out of bounds. Tables were recomputed independently instead; figures were checked for identity with the submission copies only |
| Reruns of the other 29 uncommitted-source files, and of the untested methods in the 4 sampled ones | Compute; M4 remediation |
| Any rerun of the cascade, ablation, latency, shapes, modelseeds or regime-tuned sets | Not sampled. The reported numbers for these rest on V-CHK only |
| Content of cited papers | Descriptions come from `paper/hiss_review/novelty_comparison.md` (checked against Crossref and abstracts on 2026-10-25) and were not re-read here |
| Content of the uncited papers in M5 and D | Existence and abstracts only (web search, 2026-10-28) |
| Origin of the public dataset (possible CMS synthetic files) | Inconclusive; publisher documentation is silent |

---

## D. Novelty-gap matrix

Descriptions of cited works follow `paper/hiss_review/novelty_comparison.md`. Uncited works were verified to exist
today, from abstracts only.

| Study | Cited? | Data / labels | Protocol | Graph-augmented tabular baseline? | Finding | Overlap with this paper | Unresolved comparison |
|---|---|---|---|---|---|---|---|
| Yoo, Shin, Kyeong 2023, IEEE Access | Yes | Same public provider dataset | Not verified | Yes (ML + centrality) | ML with centrality > GNNs; GNNs 250–300× slower | **Real-data finding replicated, not new** | Different metrics; no direct numeric comparison. "Replicates" is fair if phrased as "consistent with" |
| Mardani & Moradi 2024, IEEE Access | Yes | Same dataset | Not verified | Partial | GTN F1 0.53 vs. RF 0.52 | Same null result on real data | — |
| Mardani et al. 2026, Complex Intell Syst | Yes | 10% claim sample of the same dataset | 5-fold; disjointness not described | No | Graph embeddings ≈ trees (AUC 0.87–0.88) | Same question on real data | Not comparable (sampling) |
| Lu 2023; Hong 2024; Muhammad 2025 | Yes | Single-payer data; patient, activity or accept/reject labels | Disjointness unclear | No | Heterogeneous GNNs ≥ graph-only or no-graph baselines | Motivation | — |
| **Ma et al. 2023, IJCAI (RDPGL)** | **No** | Real medical-insurance data; **organised fraud** claims | Not verified | Not known | Heterogeneous GAT + community-aware risk diffusion beats state-of-the-art baselines for organised fraud | **Closest in typology** (collusive groups) | Contradicts the impression that prior work ignores collusive groups. Their real-data claim that a GNN helps for organised fraud versus this paper's controlled claim that engineered features suffice should be discussed |
| **Zhou et al. 2023, IEEE TVCG (FraudAuditor)** | **No** | Real health-insurance visit records | Case studies | n/a | Co-visit network + community detection finds collusive groups | Same structural signal as the overlap score | A community-detection baseline (m17) |
| **Liu et al. 2016, AI Magazine** | **No** | Patient–doctor–pharmacy networks (government and commercial) | Deployment | n/a | Graph analysis finds suspicious relationships | Early heterogeneous healthcare graph | Prior art for "relational structure in claims" |
| Vandervorst et al. 2026, DMKD | Yes | Insurance fraud (open + proprietary) | Not verified | **Yes (graph GBM)** | On par or better than state of the art | Closest methodologically | — |
| GADBench 2023; GraphLand 2025; Weber 2019 | Yes | Non-healthcare graphs | Varied | Yes | Trees + neighbourhood features strong | Same pattern | — |
| Altman et al. 2023, NeurIPS D&B | Yes | Synthetic AML with typologies | Synthetic | — | Typology-controlled data | Closest design analogue | — |
| **BOND, Liu et al. 2022, NeurIPS D&B** | **No** | Injected vs. organic anomalies | Benchmark | — | Most deep methods fail on organic outliers; no method handles both structural and contextual outliers | Validity of injected-typology benchmarks | Discuss as a limitation of synthetic injection |
| **Chen et al. 2020, NeurIPS** | **No** | Theory | — | — | Message-passing networks cannot count substructures of ≥ 3 nodes | **Explains the main result (M2)** | A substructure-aware GNN was not tested |
| CARE-GNN; BWGNN; PC-GNN | Yes | — | — | — | Fraud-specialised GNNs | Two compared | PC-GNN not compared (disclosed) |
| FRAUDAR; CopyCatch | Yes | — | — | — | Dense-block detectors | Cited only | No unsupervised dense-block baseline (m17) |

**Verdict per contribution**

| Contribution | Verdict |
|---|---|
| 1. Map of when structure helps | **Novel and supported** for healthcare claims under controlled typologies. Scope it to the message-passing GNNs tested and the tuning used (M1, M2) |
| 2. Leakage-tested benchmark | **Supported.** Tests pass locally and in CI; the code is public. "Reusable" is fair; realism is not validated (L4) |
| 3. Evaluation pitfalls | Memorisation (+0.60, reordering) **supported and useful**. The seed-as-unit point is correct but standard clustered-data practice; present it as such. "Opposite conclusion" is **unsupported** (M3) |
| 4. Mechanism | Degree-encoding dependence **supported**. The account of why engineered features win is **incomplete** without the expressivity result (M2) |
| 5. Public-data check | **Supported as a replication** of Yoo et al. (2023) under a stricter, verified protocol. Not an independent discovery; the manuscript already says so |

---

## E. Remediation plan

The plan is finite. Text tasks need no compute. Compute tasks are listed with what they resolve; none should start
without the authors' approval and, following the project's practice, a committed plan.

| # | Task | Resolves | Done when | Type / effort |
|---|---|---|---|---|
| T0 | Confirm which PDF is "(5)"; replace or remove the stale tracked PDFs | Scope caveat; m15 | The authors confirm (5).pdf = `7b154fe` source, or supply it for a diff; stale PDFs regenerated or removed (needs approval) | Author action |
| T1 | Correct the §10 provenance sentences; rerun every method in the 33 uncommitted-source files with the tagged code and log the comparison | M4 | §10 matches the records; a committed log shows bit-exact (or explained) results for all 33 files | Text + ~1–3 CPU-h (estimate) |
| T2 | Disclose the tuning design; decide on the expanded-tuning check | M1, m1 | Methods and Limitations state budget fraction and reused candidates; "equally tuned" removed. If run: a pre-declared plan, plus a table of GNN − GB-CAG under expanded tuning (volume 4, s ∈ {0.5, 0.9}; default δ = 0, s = 0.9) with CIs, and claims scoped to the result | Text; experiment ~3–6 CPU-h (estimate; **recommended**) |
| T3 | Add the expressivity framing; scope GNN claims | M2, m13 | Chen et al. (2020) cited; §§7.2, 9.2, 11 and the abstract say "the message-passing GNNs tested"; dilution phrased as "consistent with" | Text |
| T4 | Fix the "opposite conclusion" claim | M3 | Removed, or backed by a reported GB-CAG/HGT/SAGE comparison on the 2,100-claim dataset | Text (or minutes of compute) |
| T5 | Extend Related Work | M5, m17 | RDPGL, FraudAuditor, Liu et al. (2016) and BOND cited and positioned; the "hand-engineered features" and "rather than collusive groups" sentences revised | Text |
| T6 | Statistics wording bundle | m6, m7, m8, m9, m10, m11 | Bootstrap reframed; Holm count 27; simultaneous bound reported; "pre-specified (rounds 2–4)"; confirmatory test described as one training seed; provider overlap quantified | Text |
| T7 | Methods documentation bundle | m2, m3, m4, m5, m12, m14, m16 | MLP settings, GB internal early stopping, `amt_vs_specialty` exception, persistent backgrounds, data provenance, overall-AP selection and hybrid trials all stated | Text |
| T8 | *(Optional)* Confirmatory test with training seeds 1 and 2 | m10 | Same plan-C test reported for each training seed | Minutes of compute |
| T9 | Re-run the full check suite after revision | Regression | pytest, `verify.py`, `sync_manuscript --check` and `check_manuscript_numbers.py` all pass, with checks added for new numbers | ~5 min |

Mandatory before submission: T0, T1 (text; reruns strongly advised), T2 (text), T3, T4 (text), T5, T6, T7, T9.
Recommended: the T2 experiment. Optional: T4 experiment, T8, the M2 experiment.

---

## F. GO / NO-GO

**GO for manuscript revision (Phase 2). NO-GO for submission as the manuscript currently stands.**

**Why GO:**

* The quantitative evidence is sound wherever I could test it. 522 table cells, every public-data descriptive, the
  abstract's derived ranges and the confirmatory test match the raw records. 37 method-runs re-trained with the
  tagged code reproduce bit-exactly, across first-round, uncommitted-source and public-data runs.
* The leakage controls hold. Features and graphs are period-local and label-free; emergent rings are disjoint;
  provider splits are disjoint; within a rotation, test labels are not used for training, early stopping or
  thresholds. The exception is the shared public-data tuning, which used the labels of providers that later appear
  in test folds. The manuscript discloses this, and GB-P stayed best under nested selection (D5, L6).
* The pre-registered rounds were run after their plans were committed. The confirmatory test followed its plan
  exactly.
* None of the five MAJOR issues invalidates a reported number. Each is a scope, wording, documentation or
  positioning problem with a defined fix.

**Why not submit yet:**

* §10 makes a provenance statement that the records contradict (M4).
* One contribution claim has no supporting result (M3).
* The headline claim about GNNs is stated more generally than the evidence allows (M1, M2).
* The related work misses the closest collusion-specific studies (M5).
* The identity of the PDF named in the brief is unconfirmed, and stale PDFs sit in the repository (T0).

A reviewer who checks the run records or knows the expressivity literature would find M1, M2 and M4 quickly.

**Decision needed from the authors:** whether to run the expanded-tuning check (T2) before revision, or to scope the
GNN claims to "under a 10-configuration budget" and list it as a limitation. I recommend running it: it is the one
experiment that resolves a material uncertainty in the central claim.
