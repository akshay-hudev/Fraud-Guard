# Analysis plan for the second experimental round

Written 2026-10-26 and committed **before** any run of this round started; the commit time is the record.

## Why this round exists

It addresses the weaknesses listed after the first revision:

* low power: six seeds give five degrees of freedom;
* a single reduced ring volume;
* an exploratory hybrid result;
* five overlapping real-data rotations;
* shared GNN tuning on the real data.

The extension of the main grid was decided **after** the six-seed results were seen. To limit the resulting
risk of optional stopping:

* the final number of seeds is fixed here;
* every outcome is reported, whatever its direction;
* the six-seed results remain in git history.

## A. Main grid: seeds 7–12 (final N = 12)

* **Design.**
  * Same generator, regimes and methods as the main experiment: δ ∈ {0, 1, 2}, s ∈ {0.1, 0.5, 0.9}, all 14
    methods.
  * Hyperparameters from `results/tuning/best_hyperparameters.json`; training seed 0; output `results/synthetic`.
* **Stopping rule.** Twelve seeds is final. No further seeds will be added, whatever the results.
* **Analyses.** Definitions are unchanged; only the seed set changes.
  * Within-regime paired t-tests over 12 seeds, Holm within each nine-regime family.
  * Pooled comparisons over the 12 seed means (t-test, df = 11), Holm within metric and reference.
* **Kept at seeds 1–6** (the cost of extending them is not justified):
  * persistent-ring comparison;
  * routing;
  * ablations and reliance;
  * every low-volume and dose-response analysis (B, C).

## B. Ring-volume dose-response (δ = 0)

* **Design.**
  * Ring volume ∈ {16, 8, 4, 2} claims per month. 16 is the main grid; 4 already exists; 8 and 2 are new.
  * s ∈ {0.1, 0.5, 0.9}; seeds 1–6; all 14 methods; frozen hyperparameters.
  * For a given seed, all volume levels share their legitimate claims, isolated fraud and ring members, so the
    levels are compared on otherwise identical worlds.
* **Reported for every level** (descriptive plus paired tests, seed as unit):
  * ring AP of each method;
  * structure gain (GB-CAG − GB-CA);
  * pure GNN − GB-CAG;
  * hybrid − GB-CAG.

## C. Confirmatory test of the hybrid result

* **Exploratory finding.** After low-volume re-tuning, SAGE+AG > GB-CAG at δ = 0, s = 0.1 (seeds 1–6;
  +0.051, Holm p = 0.035).
* **Hypothesis.** At δ = 0, s = 0.1, volume 4, with the low-volume re-tuned configurations, SAGE+AG and GB-CAG
  differ in ring AP.
* **Data.** Fresh generator seeds 7–18 (n = 12), never used at low volume. Only these two methods are run.
* **Test.** One pre-specified two-sided paired t-test, α = 0.05, no multiplicity adjustment. The 95% CI and a
  Wilcoxon signed-rank test are reported alongside.
* **Reporting rule.**
  * *Confirmed* if p < 0.05 and the difference is positive.
  * Otherwise *not confirmed*; the seed 1–6 result is then reported as exploratory only.

## D. Repeated cross-validation on the public data

* **Design.**
  * Repeats 1–4 of the five-fold provider rotation (stratified folds reshuffled with seed = repeat), together with
    the reported repeat 0: 25 rotations.
  * Methods: LR-P, RF-P, GB-P, GB-PG, GB-PGL, MLP-P, SAGE, HAN, HGT. RGCN, Simple-HGN and the hybrids are omitted
    for compute.
  * Hyperparameters from the shared tuning, as in the main evaluation.
* **Analysis.**
  * Each method vs. GB-P, corrected repeated cross-validation t-test: variance factor 1/25 + 1/3, df = 24
    (Nadeau & Bengio 2003; Bouckaert & Frank 2004).
  * Holm over the eight comparisons.

## E. Nested GNN tuning on the public data

* **Design.**
  * Per-rotation selection on each rotation's validation fold, for SAGE, RGCN, HAN, Simple-HGN and HGT.
  * Same budget of six configurations; SAGE+G and HGT+G inherit their base model's per-rotation choice.
  * Rotations of repeat 0. Output `results/kaggle_nested/runs/fold<k>_gnn_<model>.json`.
* **Analysis.** Nested vs. shared selection per method (corrected test, as for the tabular models), and nested
  GNNs vs. nested GB-P.

## Reporting commitments

* All outcomes of A–E are reported in the manuscript or its supplement and in `results/`.
* Any conclusion in the manuscript that changes because of this round will be changed.
