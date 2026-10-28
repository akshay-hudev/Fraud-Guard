# Experiment reproducibility report (2026-10-25)

**Machine:** Intel Core i7-1068NG7 (4 cores, 8 threads), 16 GB RAM, macOS 26 (Darwin 25.6).

**Environment:** `.venv-research` (Python 3.12.13, numpy 1.26.4, pandas 2.2.3, scipy 1.13.1, scikit-learn 1.5.2,
torch 2.2.2, torch-geometric 2.6.1, pinned in `research/requirements*.txt`).

**LaTeX:** Tectonic 0.17.0, official release binary, SHA-256
`7c90ef5b6ddb1eb1937e4337add5237b79338e4b9676459fa91187d24d6cdf80` verified against GitHub's published digest.
Homebrew could not build it on this Intel Mac.

## 1. Reproduction of the existing results (before any change)

| Check | Command | Outcome |
|---|---|---|
| Validity tests | `.venv-research/bin/python -m pytest research/tests -q` | 24 passed |
| Artifact provenance | `.venv-research/bin/python verify.py` | All 324 synthetic, 12 persistent, 6 + 6 ablation, 20 Kaggle, 5 nested and 54 routing run files present with provenance; all 54 routing records reproduce the stored predictions (max diff ≤ 3e-8) |
| Tables from run records | `.venv-research/bin/python -m research.experiments.analyze`, then `git status` | Every table under `results/tables/` regenerated **bit-for-bit**; only `summary.json` provenance (time, commit) differed. Restored afterwards |
| Manuscript tables | `.venv-research/bin/python -m research.experiments.sync_manuscript --check` | No table differs |
| Manuscript build | Tectonic on the committed `.tex` + `sn-jnl.cls` + figures | Compiles; PDF size identical to the committed `paper/fraud_detection_paper.pdf` (363,757 bytes); no undefined references or overfull boxes |
| Public data | `shasum -a 256 data/external/kaggle_provider_fraud/Train*.csv` | All four checksums match `research/README.md` |

Not re-run (compute): the original 54-world grid, the tuning, the ablations, the Kaggle evaluation and the cost
benchmark. They are covered by the bit-for-bit table regeneration and the earlier reproduction check of the
routing runs.

## 2. Independent re-analyses (no retraining)

| Analysis | How | Result |
|---|---|---|
| Shared background across regimes | Generated seed 1 at all nine regimes and compared the non-ring claims | 18,445 legitimate and isolated claims identical in all nine regimes; ring doctors identical. Now a test |
| Seed-clustered pooled tests | Seed means over the nine regimes, t-test with 5 df; exact sign-flip as a check | See `results/tables/synthetic_pooled.csv` (`t_p_holm` vs. `naive_t_p_holm`) and audit item A2 |
| Corrected tests on public data | Nadeau–Bengio corrected resampled t-test, n₂/n₁ = 1/3 | No difference significant after Holm (smallest p = 0.13) |
| Provider bootstrap | 1,000 stratified resamples within fold and label, trained models fixed | Every GNN and hybrid below GB-P (`kaggle_bootstrap.csv`) |
| Volume signal | 36 worlds (2 volumes × 3 densities × 6 seeds), test period | Default: ring patients 9.3 vs. 1.6 claims (5.7×), ring doctors 38 vs. 16 (2.5×), prevalence 8.0%. Low: 3.7 vs. 1.6, 20 vs. 16, 2.1% (`volume_signal.csv`) |
| Data provenance probes | Beneficiary file: DOB day-of-month, chronic-condition codes, geographic spread of providers' beneficiaries | All DOBs on the 1st; codes 1/2 (CMS synthetic-file conventions); median 90% of a large provider's beneficiaries in its modal state (random assignment would give 8.7%). Inconclusive about origin; no claim made |
| Public-data counts (Section 5) | Loaded the CSVs with `research.realdata.kaggle` | All match: 558,211 claims (40,474 inpatient, 517,737 outpatient), claims dated 2008-11-27 to 2009-12-31, 5,410 providers (506 fraudulent, 9.35%), 138,556 beneficiaries, 100,737 physicians, 23 P and 8 G features, 363,300 provider–beneficiary edges, 7,685 shared physicians with 16,287 edges, 22 beneficiary attributes |
| Numbers in the text | 117 numbers in sections not rewritten (data counts, chance levels, cost, ablations, reliance, memorisation, routing, descriptives) recomputed from artifacts | 116/117 matched; the heads range (−0.028 → −0.037) was corrected. One further error was found while rewriting (SAGE+AG Holm p 0.61 → 0.095) |

## 3. New experiments

| Experiment | Command | Seeds and configuration | Wall clock | Outcome |
|---|---|---|---|---|
| Low-volume stress test, frozen hyperparameters | `PY -m research.experiments.synthetic --shifts 0.0 --densities <s> --seeds 1 2 3 4 5 6 --ring-claims-per-month 4 --out results/synthetic_lowvolume` (three processes, `RESEARCH_THREADS=2`) | Generator seeds 1–6, training seed 0, `results/tuning/best_hyperparameters.json` | ≈25 min | 18/18 worlds, no errors (commit `12d89ff`, clean) |
| Low-volume re-tuning | `PY -m research.experiments.tune --ring-claims-per-month 4 --out results/tuning_lowvolume --tag <group> --methods …`, then `--merge` | Development seeds 901/902, δ = 0.5, s = 0.5, volume 4; budget 10; same candidates as the main tuning (sampling seed 0) | ≈40 min | Configurations changed for 10 of 13 entries; HAN, RGCN and Simple-HGN unchanged (commit `7ea9c1f`, clean). Simple-HGN ran in its own process; the HGT process was stopped after writing HGT's trials, before it started its duplicate Simple-HGN pass |
| Low-volume stress test, re-tuned | as the frozen run, plus `--hps results/tuning_lowvolume/best_hyperparameters.json --out results/synthetic_lowvolume_retuned` | as above | ≈75 min (interrupted once) | 18/18 worlds, no errors. The first launch was killed when the agent session ended after 5 worlds; the relaunch skipped the finished worlds. HAN, RGCN and Simple-HGN, whose configuration re-tuning left unchanged, reproduce their frozen-run results exactly (deterministic training). See the provenance note below |

**Provenance note (re-tuned runs).** The 18 records carry `dirty: true`. When they started, `research/` held
uncommitted edits, so the flag was set, but none of the edits touched code the experiment imports:

* **13 runs at `9f615d7`.** The recorded `research_sources_sha256` (`4a0bc11f20e34678`) equals the hash of the
  committed `.py` sources at `9f615d7`; only `research/README.md` and `research/DATASHEET.md` were uncommitted.
* **5 runs at `8c97426`.** The recorded hash (`c601208ff2218c10`) was reproduced exactly by taking `8c97426` and
  applying the caption-only edits to `research/experiments/analyze.py` that existed at launch. `synthetic.py` does
  not import `analyze.py`.

**Not run in the first round:** nested GNN tuning (E-1) and further volume levels (E-2), both run in the second
round (Section 5); δ > 0 at low volume, training-seed replication and per-regime tuning (still not run).

## 4. Low-volume results (δ = 0, ring AP; chance 0.021)

Full table: manuscript Table 4 (`results/tables/lowvolume.tex`). Tests: `results/tables/lowvolume_comparisons.csv`
(paired over six seeds per density, Holm within each method's three densities; pooled over densities with the
seed as unit).

| Method | Frozen s = 0.1 / 0.5 / 0.9 | Re-tuned s = 0.1 / 0.5 / 0.9 |
|---|---|---|
| GB-CA (aggregates, no graph) | 0.049 / 0.052 / 0.055 | 0.047 / 0.052 / 0.051 |
| GB-CAG | 0.066 / 0.409 / 0.570 | 0.078 / 0.445 / 0.600 |
| Pure GNNs (range) | 0.046–0.074 / 0.065–0.133 / 0.058–0.151 | 0.050–0.074 / 0.065–0.133 / 0.058–0.162 |
| SAGE+AG | 0.103 / 0.379 / 0.566 | 0.129 / 0.420 / 0.600 |
| HGT+AG | 0.074 / 0.324 / 0.523 | 0.074 / 0.301 / 0.477 |
| Overlap score (unsupervised) | 0.057 / 0.194 / 0.281 | same |

**Pooled over densities:**

* Pure GNNs are 0.23–0.29 (frozen) and 0.26–0.31 (re-tuned) below GB-CAG (Holm p ≤ 0.002 and ≤ 0.001).
* SAGE+AG matches GB-CAG in both settings (+0.001 and +0.009).
* The only significant advantage over GB-CAG in either setting: SAGE+AG at s = 0.1 after re-tuning (+0.051,
  Holm p = 0.035), where every method is near chance.

## 5. Second round (2026-10-26)

**Plan.** `paper/hiss_review/analysis_plan_round2.md`, committed at `6c6a0c2` (16:02:51 IST) before any run.

**Runner.** `PARALLEL=4 bash paper/hiss_review/run_round2.sh` (each job with `RESEARCH_THREADS=2 OMP_NUM_THREADS=2`).
Started 16:02:55, finished 23:47:43 IST (7 h 45 min): 136 jobs, 136 `done`, 0 `FAILED`
(`results/round2_logs/queue.log`; per-job logs git-ignored).

| Group | Command (per job) | Jobs | Outcome |
|---|---|---|---|
| A. Main grid, seeds 7–12 | `PY -m research.experiments.synthetic --shifts <δ> --densities <s> --seeds <g> --group all` | 54 | 54 worlds; the main grid has all 108 worlds (`verify.py`) |
| B. Ring volumes 8 and 2 | as A with `--shifts 0.0 --ring-claims-per-month {8,2} --out results/synthetic_volume{8,2}` | 36 | 18 + 18 worlds |
| C. Confirmatory test | `PY -m research.experiments.synthetic --shifts 0.0 --densities 0.1 --seeds 7 … 18 --ring-claims-per-month 4 --hps results/tuning_lowvolume/best_hyperparameters.json --methods GB-CAG SAGE+AG --out results/synthetic_hybrid_confirm` | 1 | 12 worlds |
| E1. Nested HGT, Simple-HGN | `PY -m research.experiments.kaggle_exp --stage nested --nested-gnns --folds <k> --methods HGT HGT+G` (and `SimpleHGN`) | 10 | 5 + 5 rotation files |
| D. Repeated cross-validation | `PY -m research.experiments.kaggle_exp --stage eval --repeat <1–4> --folds <k> --methods LR-P RF-P GB-P GB-PG GB-PGL MLP-P SAGE HAN HGT` | 20 | 20 rotation files (25 rotations with the reported split) |
| E2. Nested RGCN, SAGE, HAN | as E1 with `RGCN`, `SAGE SAGE+G`, `HAN` | 15 | 15 rotation files |

Then `PY -m research.experiments.analyze`, `PY -m research.experiments.figures` (dose-response figure),
`PY -m research.experiments.sync_manuscript`, `PY verify.py` (all artifacts present),
`PY paper/check_manuscript_numbers.py` (345/345), `PY paper/order_references.py --check`, and
`PY paper/check_cross_references.py <build dir>` (27/27) after compiling both documents with Tectonic.
`pytest research/tests` passes (27/27).

**Provenance of the round-2 runs.** The git state is recorded when a process starts. Experiment code, meaning
everything `synthetic.py` and `kaggle_exp.py` import, is identical in every state below; only `analyze.py`,
`figures.py` and `sync_manuscript.py` changed, and no experiment imports them.

| Recorded state | Runs | Explanation |
|---|---|---|
| `6c6a0c2`, clean, sources `eb03b4697c4919c4` | 4 synthetic | Equals the committed sources at `6c6a0c2` |
| `6c6a0c2`, dirty, `1e7bfee25a2f3b9e` | 4 synthetic | Started while `analyze.py`/`figures.py` were being edited for `8c3434a` |
| `6c6a0c2` or `8c3434a`, dirty, `14e8bec1584dcbeb` | 1 + 93 synthetic, 4 nested | Equals the committed sources at `8c3434a`; dirty only through `research/README.md` |
| `5581e2e`, clean, `df47d889bd584376` | 6 nested, 11 repeated | Equals the committed sources at `430e014`/`5581e2e` |
| `5581e2e`, dirty, `dfb870223106e371` | 15 nested, 9 repeated | Reproduced exactly by applying, to `5581e2e`, the one uncommitted edit of `analyze.py` that existed at launch (the extended nested table) |

**Observations.**

* Nested SAGE selected the shared configuration in all five rotations (validation AP 0.683 against 0.671 for the
  runner-up in the shared tuning), so with deterministic training its nested and shared results are identical.
* Round-2 runs shared the CPU with a different mix of concurrent jobs than round 1, so their wall-clock training
  times are not comparable; the timing table uses round-1 seeds only.
* The single-column article compiles to 34 pages, the Online Resource to 10, and the two-column bundle to 27 pages,
  without errors, undefined references or overfull boxes. The bundle was unzipped into an empty directory and
  compiled on its own.

**Still not run at the end of round 2:** training-seed replication (E-3), per-regime tuning (E-4), δ > 0 at reduced
volume, and the repeated rotations for RGCN, Simple-HGN and the hybrids. All four were run in round 3 (Section 6).

## 6. Third round (2026-10-27)

**Plan.** `paper/hiss_review/analysis_plan_round3.md`, committed at `2eb368c` (13:05:21 IST) before any run.

**Runner.** `PARALLEL=4 bash paper/hiss_review/run_round3.sh` (each job with `RESEARCH_THREADS=2 OMP_NUM_THREADS=2`).
Started 13:05:28, finished 21:55:06 IST (8 h 50 min): 116 jobs, 116 `done`, 0 `FAILED`
(`results/round3_logs/queue.log`, git-ignored). The E-4 evaluation jobs (`round3_regime_eval.sh`) waited for their
regime's tuning to be merged.

| Experiment | Jobs | Run files |
|---|---|---|
| E-4 tuning (two regimes × six method groups) | 12 | `results/tuning_regime_d0.0_s{0.1,0.9}/` (trials and merged configurations) |
| E-7 repeated rotations | 20 | `results/kaggle_repeated/runs/r<1–4>_fold<0–4>_extra.json` |
| E-2 δ = 1, 2 at volume 4 | 36 | 36 |
| E-3 training seeds 1, 2 | 36 | 36 |
| E-4 evaluation | 12 | 12 |

**Provenance.** Every round-3 record (84 synthetic, 20 public-data, 14 tuning files) carries commit `2eb368c`,
`dirty: false` and research-sources hash `3cb55752dad0700c`. No file under `research/` was edited during the runs;
the analysis code was drafted in the scratchpad and moved into `analyze.py` after the last run.

**Checks after integration.** `analyze` (only the repeated-rotation tables changed among existing tables, as
expected), `sync_manuscript --check` (no differences), `verify.py` (all artifacts present),
`check_manuscript_numbers.py` (458/458), `check_cross_references.py` (33/33) after compiling both documents with
Tectonic (article 35 pages, Online Resource 14 pages, no overfull boxes), `pytest research/tests` (27/27).

**Shared initial weights.** `paper/hiss_review/check_initial_weights.py` builds SAGE and HGT on a development world
(seed 901) and a test world (seed 3) after seeding with 0: all 52,161 and 65,325 initial parameters are identical.
Every reported GNN run therefore starts from the weights used when its configuration was selected; E-3 measures
the consequence.

## 7. Fourth round (2026-10-27/28)

**Plan.** `paper/hiss_review/analysis_plan_round4.md`, committed at `98638c4` on branch `round4-dev` before any run.

**First launch (lost).** The round was first run from a git worktree of `round4-dev` in the session's temporary
directory (`/private/tmp/...`):

* The first launch on 27 October failed immediately because the worktree had no Python environment
  (`.venv-research/bin/python: No such file`). Only that runner was stopped; the log was kept as
  `queue_failed_launch_no_venv.log`.
* The relaunch at 19:15:53 completed the tuning, the merges, the nested public-data runs and the main-grid and
  low-volume runs of the two fraud-specialised GNNs (35 of 107 jobs by 22:23, none failed).
* The computer then restarted overnight. macOS empties `/private/tmp` on restart, so the worktree and every
  uncommitted fourth-round result were lost. None of them had been used in the manuscript. Six main-grid regimes had
  been looked at informally; see "Observations made before the results were complete" below.

**Second launch.** `PARALLEL=4 bash paper/hiss_review/run_round4.sh` from the repository's own working tree, which
survives restarts, at commit `78072e8` of `hiss-submission-audit`. That branch contains `98638c4` merged; the
experiment code (`research/models`, `research/synth`, `synthetic.py`, `kaggle_exp.py`, `tune.py`, `realdata`,
`metrics.py`, `ledger.py`) is identical to `98638c4` (`git diff 98638c4 78072e8` on these paths is empty). The
research-sources hash in the records differs from that of `98638c4` because `analyze.py`, `figures.py` and
`sync_manuscript.py`, which no experiment imports, had changed. Started 11:19:22 IST on 28 October. No file under
`research/` is edited while it runs.

**Observations made before the results were complete.** From the first launch, the mean ring AP of BWGNN and
CARE-GNN against GB-CAG in the six finished main-grid regimes (δ = 0 and 1) was looked at once, without tests,
to report progress to the authors. The design was not changed afterwards; the plan already fixed every setting,
seed and method.
