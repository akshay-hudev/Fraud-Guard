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

**Not run:**

* Nested GNN tuning on the public data (≈25 CPU-hours). The command is in `unresolved_issues.md` (E-1).
* Further volume levels, δ > 0 at low volume, training-seed replication and per-regime tuning (E-2 to E-4).

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
