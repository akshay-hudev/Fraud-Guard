# Analysis plan for the third experimental round

Written 2026-10-27 and committed **before** any run of this round started; the commit time is the record. The
authors asked for the remaining optional experiments of `unresolved_issues.md` to be run before submission.

## Scope

Four experiments, E-2, E-3, E-4 and E-7, are run.

Two cannot be run, and this is stated in the manuscript's limitations:

* **E-5** needs real claims data with labelled collusive schemes, which are not publicly available.
* **E-6** needs a deployed system with incremental feature computation.

Every outcome below is reported in the manuscript or its Online Resource, whatever its direction. Any conclusion
that changes is changed. No further seeds or settings will be added after the results are seen.

## E-2. Feature shift at reduced ring volume

* **Purpose.** At volume 4 only δ = 0 was run. This experiment tests whether a claim-level signal (δ > 0) takes
  over when the volume signal is absent, and whether the conclusions about structure and GNNs hold beyond δ = 0.
* **Design.**
  * δ ∈ {1, 2}, s ∈ {0.1, 0.5, 0.9}, ring volume 4 claims per month, seeds 1–6.
  * All 14 methods, main hyperparameters (frozen), output `results/synthetic_lowvolume_shift`.
  * For a seed, these worlds share their legitimate claims, isolated fraud and ring members with the δ = 0 and
    volume-16 worlds.
* **Analyses** (as for the low-volume test, seed as unit). For each δ:
  * ring AP of every method;
  * GB-CAG − GB-CA (structure gain);
  * each pure GNN and hybrid − GB-CAG, per density (Holm within each method's three densities) and pooled over
    densities (Holm over methods).
* **Expected value.** It completes the map of conditions (article Table 8) along the δ axis at reduced volume. It
  can show that the map's δ = 2 row ("structure redundant") also holds when members are not conspicuous.

## E-3. Training-seed replication

* **Purpose.** Every reported run trains each model once per world (training seed 0). The hybrid gain over GB-CAG
  is small (+0.01 AP), so initialisation variance could matter. This experiment separates training-seed variance
  from world variance.
* **Design.**
  * Training seeds 1 and 2, in addition to the existing seed 0.
  * δ = 0, s ∈ {0.1, 0.5, 0.9}, default volume, generator seeds 1–6.
  * Methods GB-CAG, SAGE, HGT, SAGE+AG, HGT+AG, with the main hyperparameters.
  * Output `results/synthetic_modelseeds`.
* **Analyses.**
  * For each world and method, mean ring AP and isolated AP over training seeds 0–2.
  * Hybrid − GB-CAG and pure GNN − GB-CAG on these means, per density (paired over six seeds, Holm within each
    method's three densities) and pooled over densities (seed as unit, Holm over methods).
  * The same comparisons with training seed 0 only, on the same worlds, for reference.
  * Standard deviation across training seeds within a world, against standard deviation across worlds.
* **Expected value.** It shows whether the hybrid gain and the GNN–GB-CAG differences survive averaging over
  initialisations, and how much of the reported variance comes from training.

## E-4. Per-regime tuning

* **Purpose.** All methods were tuned once, on development worlds at δ = 0.5, s = 0.5. If that regime suits one
  method family, per-regime differences could reflect tuning rather than the method. This experiment re-tunes
  in the two regimes where the comparisons matter most:
  * δ = 0, s = 0.1: sparse rings, where the hybrids gain most and structure least;
  * δ = 0, s = 0.9: dense rings, where structure matters most.
* **Design.**
  * Same protocol as the main tuning: budget of ten configurations per method, development seeds 901 and 902,
    validation AP; hybrids inherit their base configuration.
  * Development worlds at the target regime and default volume (`tune.py --ring-shift --ring-density`).
  * Outputs `results/tuning_regime_d0.0_s0.1/` and `results/tuning_regime_d0.0_s0.9/`.
  * Then all 14 methods are evaluated on seeds 1–6 in each regime with these configurations; output
    `results/synthetic_regimetuned`.
* **Analyses**, for each regime:
  * each method re-tuned − main tuning (paired over seeds 1–6, the same worlds; Holm over methods);
  * each pure GNN and hybrid − GB-CAG with every method re-tuned (Holm over methods).
* **Expected value.** It tests whether the regime-level conclusions depend on the single tuning regime. Pure GNNs
  might gain most at s = 0.9.

## E-7. Repeated rotations for the remaining GNNs on the public data

* **Purpose.** The 25-rotation analysis covered SAGE, HAN and HGT only.
* **Design.**
  * Repeats 1–4 of the five-fold rotation, with the same reshuffled folds as round 2.
  * Methods RGCN, Simple-HGN, SAGE+G and HGT+G, with the shared hyperparameters.
  * Output `results/kaggle_repeated/runs/r<rep>_fold<k>_extra.json`.
* **Analysis.**
  * Corrected repeated cross-validation test against GB-P (J = 25, factor 1/25 + 1/3, 24 df).
  * The Holm family becomes all twelve comparisons against GB-P. The eight comparisons reported in round 2 are
    re-adjusted within it.
* **Expected value.** It completes the public-data comparison over 25 rotations for every method.

## Cost and execution

* 116 jobs in all: 12 tuning jobs, 20 Kaggle jobs, 36 + 36 synthetic jobs and 12 evaluation jobs.
* Four workers with two threads each; estimated 9–11 hours.
* Run by `paper/hiss_review/run_round3.sh`. Each E-4 evaluation job waits until the tuning of its regime has
  finished and been merged.
* During the runs, no file under `research/` is edited, so the run records carry a clean provenance. The analysis
  code for this round is written outside `research/` and moved in after the last run.
