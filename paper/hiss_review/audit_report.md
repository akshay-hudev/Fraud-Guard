# Scientific audit report (2026-10-25)

Branch `hiss-submission-audit`, starting from `main` at `7f18336`.

## Scope and method

The whole repository was inspected. Every reported finding was traced to its evidence: generator, features,
graphs, models, training, metrics, analysis, run records, tables and figures. This was an independent audit;
earlier audit documents (`paper/research_audit.md`, `paper/remediation_log.md`, `paper/revision_2026-10-23.md`)
were read but not assumed correct.

**Baseline state (all checks passed before any change):**

* 24/24 research tests.
* `verify.py`: all artifacts present with provenance.
* `sync_manuscript --check`: every manuscript table equals its generated file.
* `analyze.py` regenerated every table bit-for-bit from the run records; only provenance metadata in
  `summary.json` differed.
* The committed manuscript compiled to a PDF of the same size as the committed one (Tectonic 0.17.0,
  `sn-jnl.cls` v12).

**Severity scale:**

* **Critical**: would invalidate a central conclusion.
* **Major**: a reviewer would likely require a change, or a reported claim is not supported as stated.
* **Moderate**: a weakness that should be acknowledged or fixed.
* **Minor**: presentation or precision.

## 1. What is scientifically sound

* **Leakage controls in the synthetic benchmark.** Ground truth is kept separate from claims; features are
  invariant to label permutation; each audit period's features and graphs use only that period's claims; emergent
  rings are entity-disjoint across splits; at δ = 0 ring claims are distributed like legitimate claims; and
  information flow from doctors to claims is tested for every GNN. All are enforced by tests and re-run here.
* **Model selection.** Tuning uses separate development worlds (seeds 901/902, δ = 0.5), and thresholds, early
  stopping and routing fractions use validation periods only. No path from test labels to model selection was found.
* **Provenance.** Every number traces to a run record (commit, config hash, data hashes). Retrained GB-CAG, SAGE
  and HGT reproduce the stored test predictions on all 54 worlds.
* **Real-data protocol.** Provider-disjoint rotations; label-neighbourhood features use training labels only; the
  report stays at provider level and explicitly declines to claim ring detection.
* **Mechanism analyses** (rewiring, HGT single-factor ablations, reliance) and the **memorisation analysis**
  (persistent vs. emergent rings) are correctly designed and honestly scoped.

## 2. Issues found

| ID | Severity | Issue | Evidence | Action | Status |
|---|---|---|---|---|---|
| A1 | **Major** | Pooled "54-world" tests treated the nine regimes of a seed as independent. The generator draws entities, legitimate claims, isolated fraud and ring members before applying δ and s, so for a given seed all nine regimes share 18,445 identical non-ring claims and the same ring doctors. Between-seed share of residual variance was 0.10–0.45 (≈0.11 expected under independence). | Session check script; new test `test_regimes_of_a_seed_share_legitimate_claims_and_ring_members` | **Re-analysed** with the seed as the unit (seed means, df = 5). The pooled table, routing table, figures and text were updated; naive values are kept in the CSVs as an audit trail. | Done |
| A2 | **Major** (consequence of A1) | Claims that do not survive A1: hybrids "+0.008, Holm p = 0.013/0.019" (now p = 0.12/0.29); GB-CAG better than **every** pure GNN on overall AP (now significant for RGCN, HAN and Simple-HGN only); HAN and Simple-HGN significantly worse on ring AP (now p = 0.063/0.23); HGT and SAGE alone in the routing table (p < 0.001/0.018 → 0.10/0.60). Survives: GNNs worse on isolated fraud (Holm p ≤ 0.009); graph-aware > GB-CA (p ≤ 0.021); fused GB-CAG→SAGE +0.006 (p = 0.042). | `results/tables/synthetic_pooled.csv`, `cascade_comparisons.csv` (columns `t_p_holm` vs. `naive_t_p_holm`) | **Claims narrowed** in the abstract, introduction, Results 6.3–6.7, discussion, recommendations and conclusion | Done |
| A3 | **Major** | Ring-volume confound. At default volume a ring patient files 9.3 claims per test period against 1.6 for other patients (99th percentile ≈ 5); ring doctors 38 vs. 16. "Per-entity aggregates detect most rings (AP 0.89–0.93)" was presented as a finding, but it follows from one generator constant. The manuscript acknowledged a volume signal but not its size. | `results/tables/volume_signal.csv` | **New experiment** (low-volume stress test, 4 instead of 16 ring claims per month, δ = 0, all 14 methods, six seeds, frozen and re-tuned hyperparameters). Aggregates collapse to near chance (0.05); GB-CAG keeps 0.41–0.60 at s ≥ 0.5; pure GNNs reach 0.06–0.16 in both settings; the hybrid SAGE+AG matches GB-CAG on average and beats it only for sparse rings after re-tuning (near chance). The central claim is **strengthened** for pure GNNs and **qualified** for hybrids; the aggregates claim is **narrowed** to the default volume. | Done |
| A4 | **Major** | Missing closest prior work: Yoo et al. 2023 (*IEEE Access*) found, on public Medicare provider data (by its description, the same dataset), that ML with graph-centrality features beat GNNs at 250–300× less training time. Also missing: Mardani & Moradi 2024, Mardani et al. 2026 (same dataset), Lu et al. 2023, Hong et al. 2024 and Muhammad et al. 2025 (heterogeneous GNNs on insurer data), the Curtis et al. 2025 review, GraphLand 2025, Vandervorst et al. 2026, Platonov et al. 2023. The real-data result is a replication, not a new finding. | `novelty_comparison.md`, `reference_verification.md` | **Twelve verified references added** (with Dal Pozzolo 2018 and Nadeau & Bengio 2003). New related-work paragraph; contribution 4 restated as a replication under a stricter protocol; research gap narrowed to controlled conditions, mechanism and memorisation | Done |
| A5 | **Major** | Real-data tests: paired t-tests over five rotations whose training sets overlap (each pair shares 2 of 3 training folds) are anti-conservative. The reported "SAGE+G significantly worse, Holm p = 0.021" does not survive the Nadeau–Bengio correction (Holm p = 0.13). | `results/tables/kaggle_comparisons.csv` | **Re-analysed**: corrected resampled t-test, plus a provider-level bootstrap with the trained models held fixed (every GNN and hybrid below GB-P). Text and figure intervals updated | Done |
| A6 | Moderate | GNN hyperparameters on the real data were selected on a split overlapping the test folds; only tabular and MLP models were re-selected per rotation. | `kaggle_exp.py`; `results/kaggle_nested/` | **Narrowed and explained**: the overlap is expected to favour the GNNs, so their shortfall is if anything understated. Nested GNN tuning **run in the second round**: re-selection changed GNN AP by −0.008 to +0.036, none significant, so the expected bias was **not** observed and the sentence claiming it was removed; with every method selected per rotation GB-P stays best | Done (2026-10-26) |
| A7 | Moderate | Designer advantage: the engineered G features (shared-patient Jaccard, triadic closure) were designed knowing the injected typology (dense doctor–patient blocks). This favours GB-CAG over learned message passing and limits external validity for unknown typologies. | Code inspection of `synth/features.py`, `synth/generator.py` | **Acknowledged** in discussion 9.2 and limitations | Done (writing) |
| A8 | Moderate | Real-data provenance: the documentation describes neither the source of the claims nor how labels were obtained. The beneficiary file uses CMS synthetic-file conventions (all birth dates on the 1st; chronic conditions coded 1/2), but no published source confirms a synthetic origin. Provider–beneficiary links are geographically concentrated (median 90% of a large provider's beneficiaries in one state, against 8.7% at random), so links are not random. | Session check on the CSVs; web search found no provenance documentation | **Narrowed**: section renamed "Public Provider-Labelled Data"; "real-world" and "de-identified" replaced by what is verifiable (publisher's description; pseudonymous codes); result not to be read as performance on current Medicare claims | Done (writing) |
| A9 | Moderate | A single training seed per world (model seed 0) in every experiment, not only the ablations as the limitations implied. | `synthetic.py`, `cascade.py` | **Acknowledged** in Methods 4.3 and limitations | Done |
| A10 | Moderate | Numerical error: "SAGE+AG +0.043, unadjusted p = 0.011, Holm p = 0.61" (δ = 0, s = 0.1). The correct Holm p is 0.095; 0.61 is rounded from HGT+AG's 0.645. Conclusion unchanged (not significant). | Original `synthetic_ring_ap_comparisons.csv` (backup) | **Corrected** | Done |
| A11 | Moderate | Code availability: the public repository (`origin/main` at `76ee3e9`) lacks 12 local commits containing the routing, HGT-ablation, re-analysis, nested-tuning and cost code and results. The routing records cite commit `a232dcd`, which is not public. | `git log origin/main..main` | **Author action** (push and tag); not done, because pushing is outward-facing. The branch now has further unpushed commits | Open |
| A12 | Minor | "The MLP control receives exactly the claim inputs of the GNNs." In fact it sees claim plus static entity attributes, without degree encodings or messages. | `synth/features.py` vs. `synth/graph.py` | **Corrected** wording | Done |
| A13 | Minor | "Standardisation statistics are fitted on training data". Static entity attributes are standardised over all entities (label-free, time-invariant). | `synth/graph.py::entity_features` | **Corrected** wording | Done |
| A14 | Minor | Low-volume table caption draft said ring doctors carry "no excess volume" (true for seed 1 only; 1.3× on average). | `volume_signal.csv` | **Corrected** before publication | Done |
| A15 | Minor | ROC-AUC and CI not defined at first use (HISS: abbreviations must be defined). | Manuscript | **Defined** | Done |
| A16 | Minor | The real-data figure label was clipped; routing-panel labels touched the frame. | Rendered figures | **Fixed** and regenerated | Done |
| A17 | Minor | References lacked DOIs (HISS: "please always include DOIs"). | — | **Verified DOIs added** to 38 of 52 entries | Done |
| A18 | Minor | Figure files not named Fig\<N\> (HISS: mandatory). | — | `paper/make_hiss_bundle.py` renames them in the upload bundle | Done |
| A20 | Minor | "Supplementary information" said the repository contains "the test predictions underlying every table and figure"; `results/**/predictions/` is git-ignored, so the public repository does not contain them. | `.gitignore` | **Corrected** (statement now accurate); archiving them proposed as U-11 | Done |
| A21 | Minor | Budgeted recall (top 5% of claims) is at its ceiling of 0.476 for every method with aggregate or graph inputs (0.43–0.48), because fraud prevalence (10.5%) exceeds the budget. It cannot separate the methods, which the text did not say. | `synthetic_recall_at_5pct.csv` | **Stated** in the Results introduction; AP remains the comparison metric | Done |
| A19 | Integrity | Section 9.3 re-analyses the authors' own earlier 2,100-claim dataset. If an earlier version of the manuscript (headline "HGT +14.2 pp ring recall") was submitted, posted or presented, it may need to be disclosed in the submission form (no cover letter will be sent). | `paper/revision_2026-10-23.md` | Wording now says "an earlier, simpler generator of ours"; the corresponding author confirmed on 2026-10-27 that the earlier version was not submitted anywhere and that the manuscript is not under consideration elsewhere | Done |

## 3. Classification

* **Already sound, unchanged:** leakage controls; tuning protocol; provenance; mechanism, memorisation and pitfall
  analyses; within-regime statistics; real-data protocol design.
* **Insufficiently supported as written (now narrowed or re-analysed):** A1, A2, A3 (aggregates claim), A5, A8, A10.
* **Inconsistent or incorrect:** A10, A12, A13, A14.
* **Required new experiments:**
  * A3, done; extended in the second round to volumes 8 and 2.
  * A6 (nested GNN tuning), done in the second round.
  * δ > 0 at low volume, model-seed replication and per-regime tuning, not run.
* **Fixable by writing alone:** A4, A7, A8, A9, A12, A13, A15, A19.
* **Must stay unchanged until validated:** any claim about real collusive rings (no public labelled data). The
  claim that message passing adds information beyond engineered features is now supported, but only as a small
  effect (+0.009 to +0.010 AP at default volume; +0.024 in the confirmatory low-volume test).

## 4. Second-round update (2026-10-26)

The authors asked for the remaining weaknesses to be addressed. The pre-declared second round
(`analysis_plan_round2.md`) changed these items:

| Item | Change |
|---|---|
| A1/A2 (power) | Twelve seeds (11 df). The hybrids' gain over GB-CAG, not significant with six seeds, is significant with twelve (+0.010 [0.005, 0.015], Holm p = 0.006; +0.009 [0.004, 0.015], p = 0.016). One pure-GNN advantage became significant (RGCN, δ = 2, s = 0.1, +0.005, every method > 0.98), so "no GNN significantly better in any regime" was withdrawn. GB-CAG now beats every pure GNN on overall AP |
| A3 (volume) | Dose-response over 16, 8, 4 and 2 claims per month: the value of structure peaks at 8 (+0.42) and vanishes at 2 |
| Exploratory hybrid result (low volume) | Confirmed on fresh seeds 7–18 by the single pre-specified test: +0.024 [0.005, 0.043], p = 0.016 |
| A5 (real-data power) | 25 rotations: intervals narrower, still no difference significant after correction |
| A6 (GNN tuning overlap) | Done; see the table above |
| A21 | With twelve seeds the recall ceiling is 0.479 and fraud prevalence 10.5% |

**Errors introduced while drafting the second round and caught by the checks before commit:** stale pooled and
regime tables in the article, a deleted table, a pooled-over-densities range quoted as a per-cell range, two
rounding errors and a mis-stated p-value bound. **An older error found in this round:** the legacy comparison
table (now in the Online Resource) counted the 54 worlds as "independent datasets", which contradicts A1; it now
reports 54 worlds from 6 independent seeds. All are listed in `change_log.md`.

## 5. Third-round update (2026-10-27)

| Item | Change |
|---|---|
| A9 (training once per world) | Training seeds 1 and 2 added for five methods (E-3). Pure-GNN and isolated-fraud deficits unchanged; the hybrid gain shrinks (+0.013 → +0.009 pooled; sparse rings +0.043 → +0.030, no longer significant with six seeds). The text now calls the gain "of the order of the variation between training runs" |
| A22 (new, found in this round) | **Correctness.** The Methods said that variation between worlds includes initialisation variance. With a fixed training seed every world, and the tuning worlds, start from identical GNN weights (`check_initial_weights.py`). Corrected; E-3 quantifies the effect. **Done** |
| Single tuning regime | Re-tuning at δ = 0, s = 0.1 and 0.9 (E-4) changed no result significantly. **Done** |
| Low volume only at δ = 0 | δ = 1 and 2 at volume 4 (E-2). New condition in which pure GNNs beat GB-CAG (sparse rings, shifted claims), shared with logistic regression; reported in the abstract-level findings, Section 6.6, Table 8 and the recommendations. **Done** |
| A5 (public data, three GNNs only in the repeated rotations) | All methods over 25 rotations (E-7); RGCN significantly below GB-P. **Done** |
