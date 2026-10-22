# Remediation log

This log tracks how each issue in `paper/research_audit.md` (2026-10-19) was handled. The audit itself is
unchanged. "Done" means implemented and verified; the evidence column names the check.

Scope decisions agreed with the authors (2026-10-20):
* The 1,110-line `fraud_detection_paper.tex` audited on 2026-10-19 is the version being remediated.
* Real-world validation uses the Kaggle "Healthcare Provider Fraud Detection Analysis" dataset (CC0).

## Architecture of the fix

The audit showed that the paper's numbers could not be traced to code. Rather than patch the legacy pipeline
(which also serves the demo app), a new, self-contained `research/` package now produces every manuscript
number, table and figure, with a provenance record per run (`results/**/runs/*.json`: git commit, config hash,
dataset hashes, library versions). The legacy `training/` pipeline received targeted correctness fixes so the
demo app no longer serves leaky models. Unverified legacy outputs were moved to `archive/legacy_unverified/`.

## Blockers

| ID | Issue | Resolution | Status | Evidence |
|---|---|---|---|---|
| B-01 | Results without generating evidence | All manuscript numbers regenerated from `research/` runs; `MOCK_RESULTS` removed; hard-coded ablation literals removed from `shap_comparison.py`; hard-coded HGT metrics removed from `frontend/app.py` and `backend/main.py`; unverified logs archived with a README | Done (manuscript numbers: see "Manuscript" below) | `verify.py`; `results/summary.json` |
| B-02 | `ring_id` label leakage | Legacy: `LABEL_DERIVED_COLUMNS` blocklist + assertion in `preprocessor.py`; demo models retrained (26 features). Research: ground truth kept in a separate `truth` frame; features provably label-invariant | Done | `training/tests/test_all.py::test_no_label_derived_features`; `research/tests::test_features_invariant_to_labels` |
| B-03 | Synthetic data could not test the hypothesis | New generator with independent feature shift δ and structural density s; ring claims at δ = 0 are distributed like legitimate claims (KS-tested) | Done | `research/synth/generator.py`; `test_zero_shift_ring_claims_match_legit_distribution` |
| B-04 | HGT blind to graph structure | Legacy and research graphs now carry reverse relations; the silent `try/except` was replaced by an explicit error; information-flow tests for every GNN | Done | legacy HGT Δlogit after doctor perturbation 0.158 (was 0.0); `test_doctor_information_reaches_claims[*]` |
| B-05 | Missing baselines / experiments | Implemented SAGE, RGCN, HAN, Simple-HGN, HGT, MLP control, GB with graph features, hybrids, unsupervised overlap score; ablations (rewiring, relation/degree removal, depth); known-vs-new-ring protocol | Done | `research/models/gnn.py`; `research/experiments/*` |
| B-06 | No statistics; underpowered | 6 independent worlds per regime (54 total; pooled analyses over all 54); paired t-tests (Holm within pre-specified families), Wilcoxon as robustness, t-based 95% CIs; test periods have ~384 ring claims (vs 21 before) | Done | `research/metrics.py`; `research/experiments/analyze.py` |
| B-07 | Positional misalignment of ring labels | Legacy: `split_claim_ids.json` + `load_split_claims` (ID-based), stable sorts. Research: alignment asserted in every run | Done | `test_split_ids_align_with_arrays`; `test_split_alignment` |
| B-08 | Environment could not run the GNN | `.venv-research` with pinned NumPy-1.x stack (`research/requirements*.txt`); full-batch training (no `pyg-lib` needed); CI job installs CPU torch and runs validity tests | Done | `research/README.md`; `.github/workflows/test.yml` (`research-validity`) |
| B-09 | Paper–code mismatches | Methods section rewritten from the code/configs; every constant traceable to `research/` | Done with manuscript rewrite | manuscript Sections 3–5 |
| B-10 | Entity leakage; transductive overlap | Emergent protocol: test rings formed by unseen doctors/patients; each audit period analysed separately (no future claims in features or graphs); training-only standardisation | Done | `test_emergent_rings_are_entity_disjoint`; `test_period_features_ignore_other_periods`; `test_graphs_contain_only_their_period` |
| B-11 | No real-world validation; mislabeled artifact | Kaggle provider-level pipeline (`research/realdata/kaggle.py`): provider-level task, provider-disjoint 5-fold rotation, label-neighbourhood features from training labels only; legacy `train_real_world.py` (synthetic fallback, full-data aggregates) replaced by a forwarder | Done | `research/experiments/kaggle_exp.py` |
| B-12 | Citation accuracy | Bibliography rebuilt; `tiong2021` removed (replaced by Thornton et al. 2013); FdGars and Player2Vec author lists corrected; OddBall cited to its own paper; PC-GNN described accurately; NHCAA figure restated; uncited entries removed or cited | Done in manuscript; items marked ⚠ below still need checking against the publisher record | manuscript bibliography |
| B-13 | Abstract/conclusion contradictions | Abstract and conclusion rewritten from generated results only | Done with manuscript rewrite | — |

## Other mandatory items (selection; full mapping in the audit's §15)

| ID | Resolution | Status |
|---|---|---|
| M-14/M-15 | Ring AP / isolated AP with explicit negatives (legitimate claims); PR-AUC primary; recall at 5% budget | Done |
| M-16 | Thresholds selected on validation (max F1) | Done |
| M-17 | Calibration claim removed (not measured) | Done in manuscript |
| M-18 | MLP control and GB with graph features included | Done |
| M-19 | Equal tuning budget (≤10 configs) on held-out development worlds; equal data budgets (all models train on periods 1–2) | Done |
| M-20 | Self-inclusion removed by construction: every split uses identical within-period aggregate definitions | Done |
| M-22 | Related work rewritten; novelty narrowed; positioned against GADBench, Elliptic, AML synthetic benchmarks | Done in manuscript |
| M-25 | Deployment section and latency table removed from the paper (not measured); demo app described only in Data/Code availability | Done |
| M-26 | Mock-asserting tests removed; validity tests added; research tests required in CI; API tests labelled as needing a live server | Done |
| M-27 | Seeds set for Python/NumPy/torch; deterministic algorithms (warn-only); per-run predictions saved | Done |
| M-28 | Canonical paths (`ledger.REPO_ROOT`); legacy graph builder no longer depends on the working directory | Done |
| M-29 | Legacy logs regenerated by the corrected pipeline (demo only, labelled in `logs/README.md`) | Done |
| M-30 | Equal-length audit periods replace the vacuous "temporal split" | Done |
| M-33/M-34 | Declarations completed with the authors' answers (2026-10-22): no funding; MIT licence (`LICENSE`); author order H M Akshay (corresponding, lead), Gaurav S B, Eshita Narayana, Harshita Srivastava, Malini M Patil; equal-contribution marks removed; CRediT roles updated accordingly. Archived DOI to be created on acceptance | Done (DOI pending) |
| M-50 | REFORMS checklist (`paper/reforms_checklist.md`) and generator datasheet (`research/DATASHEET.md`) | Done |

## Known residual items (author action)

* On acceptance, archive a tagged release on Zenodo and replace "An archived release with a DOI will be provided upon acceptance" with the DOI.
* References flagged for verification were checked on 2026-10-22 (Crossref/publisher pages): Pourhabibi et al. DSS 133:113303 (DOI 10.1016/j.dss.2020.113303); BWGNN PMLR 162:21076–21089; REFORMS author list; NHCAA wording ("tens of billions of dollars each year").
* Revoke the Kaggle API token shared during remediation.
* The deployed demo API still has a duplicate `/health` route and 0.4/0.7 risk tiers; these are app issues and
  are no longer described in the paper.

## Deviations from the planned design (and why)

* **Grid size.** Planned 12 regimes × 10 worlds; run 9 regimes (δ ∈ {0, 1, 2} × s ∈ {0.1, 0.5, 0.9}) × 6 worlds, because
  the full grid would have taken > 30 h of CPU on the available laptop. δ = 0.5 was dropped (it is the tuning regime).
  Runs started under the larger plan are kept on disk but excluded by `analyze.py` so every cell is balanced.
* **Hybrids inherit tuning.** SAGE+AG / HGT+AG (and SAGE+G / HGT+G on real data) reuse their base model's tuned
  configuration instead of a separate budget (conservative for the hybrids; stated in the manuscript).
* **Secondary experiments trimmed.** Known-ring protocol: 6 methods × 2 regimes × 6 worlds; ablations: δ = 0, s = 0.9, 6 worlds.
* **Real-data tuning split** overlaps the evaluation folds (same providers, different split); disclosed as a limitation.
* **Timings** were measured with several jobs in parallel (two threads each) and are reported as indicative only.

## Outcome (all numbers generated by `research/experiments/analyze.py`)

* Synthetic, unseen rings: claim-only models at chance (ring AP ≈ 0.09) when δ = 0; per-entity aggregates 0.89–0.93;
  graph information adds up to +0.10 only for moderately/densely linked rings.
* Pooled over 54 worlds, no GNN beats GB with engineered graph features (SAGE −0.0001, HGT −0.011, HAN −0.024);
  hybrids +0.008 (Holm p ≈ 0.01–0.02). GNNs are 0.085–0.147 AP worse on isolated fraud.
* Ablations: degree-preserving rewiring collapses all graph-aware methods to ≈ 0.76; HGT without degree encodings −0.38;
  legacy one-directional graph −0.18; legacy `ring_id` leak inflates AP to 1.000.
* Recurring (persistent) rings inflate an attribute-only MLP from 0.09 to 0.59–0.69.
* Real data (5,410 providers): GB on provider aggregates best (AP 0.704 ± 0.045); graph/label features and all GNNs do
  not improve it; labels are not assortative through shared beneficiaries.

The original headline claim (HGT +14.2 pp ring recall over GB) is **not supported**; the manuscript was rewritten around
the evidence above. Title changed accordingly.

## Manuscript

`fraud_detection_paper.tex` (1,200 lines, Springer `sn-jnl`, single file) compiles cleanly with the official class
(Tectonic 0.15; no overfull boxes, undefined references or unused bibliography entries). Figures are read from
`results/figures/` (`\graphicspath`). Every table is generated by `analyze.py`; figures by `figures.py`.

## Repository cleanup before merging to `main` (2026-10-22)

Removed (unused, stale or misleading): `training/data/processed/` (legacy 10k-claim data; the API preferred its stale
graph mappings), `training/models/baseline/*.json` (pre-fix feature list containing `ring_id` and leaky metrics),
`training/data/generate_synthetic.py` (superseded generator), `training/notebooks/exploration.ipynb` (broken paths),
`paper/temporal_gnn_additions.txt` (notes for the old manuscript), `requirements_frozen.txt` (pins matching no
environment), root shims `baseline.py` and `metrics.py` (unused), and the CI workflows `model-validation.yml`
(could not fail), `drift-retraining.yml` (all steps commented out) and `deploy.yml` (invalid `secrets` in step `if:`;
duplicated the Docker build in `test.yml`).

Fixed: API/config paths now point to `data/processed` and `models` (SQLite default `data/fraud_detection.db`, which
exists in a fresh clone); `docker/Dockerfile.api` and `docker-compose.yml` commands; `training/README.md` no longer
shows the unsupported "HGT 99.35% — best overall" table; README Kaggle command updated.
Verified on a fresh export of the committed tree: legacy tests 32 passed, research tests 17 passed, `verify.py` OK,
API `/health` and `/stats` return 200.
Not changed (file access blocked): `.env.example` still lists `MODEL_DIR=training/models` and
`PROCESSED_DATA_DIR=training/data/processed`; change them to `models` and `data/processed`.
