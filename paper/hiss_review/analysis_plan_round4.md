# Analysis plan for the fourth experimental round

Written 2026-10-27 and committed **before** any run of this round started; the commit time is the record. The
authors asked for the two weaknesses a HISS reviewer would most likely raise to be addressed before submission.

## Why this round exists

1. **No fraud-specialised GNN.** The five compared GNNs are general heterogeneous architectures. The related work
   cites GNNs designed for fraud (CARE-GNN, PC-GNN, BWGNN), but none was compared.
2. **Designer advantage.** The engineered graph features (shared-patient Jaccard, triadic closure) were designed
   knowing the injected ring shape, a dense doctor–patient block. Their advantage could be specific to that shape.

Every outcome below is reported in the manuscript or its Online Resource, whatever its direction.

* If a fraud-specialised GNN, or any GNN under a new ring shape, significantly outperforms GB-CAG, the manuscript's
  conclusions are changed accordingly.
* The abstract, the summary table of conditions and the recommendations are revised to match.
* No settings, seeds or methods will be added after the results are seen.

Two observations were made before this plan was written, while the code was being checked:

* one smoke-test world per ring shape, with untuned fraud GNNs;
* a reproduction check showing that the existing models' predictions are unchanged.

They are not part of the results.

## Code changes (all covered by tests)

* **Generator.** `ring_shape` ∈ {`block` (default, unchanged), `hub`, `pairs`}:
  * `hub`: one hub doctor linked to every ring patient, and each patient to one spoke doctor;
  * `pairs`: each patient linked to exactly two ring doctors drawn at random.
* **Unchanged defaults.** The default reproduces the recorded worlds exactly: claims, truth and configuration hashes
  are identical, and a test pins this.
* **BWGNN** (Tang et al. 2022, heterogeneous variant):
  * Beta-wavelet filters of order 2 or 3 on each claim–claim relation (shared patient, shared doctor), summed over
    relations;
  * the published structure (two-layer input transform, one filter bank).
* **CARE-GNN** (Dou et al. 2020):
  * label-aware similarity, and top-p neighbour selection per relation;
  * thresholds start at 0.5 and move by ±0.02 after each epoch according to the mean distance of the kept
    neighbours; they stop moving once the last ten moves cancel;
  * threshold-weighted aggregation and the auxiliary similarity loss (weight 2).
  * Differences from the original:
    * full-batch training with class-weighted loss instead of mini-batches with under-sampling, as for every
      other model here;
    * the relation weights are the thresholds, as in the original's default aggregator.
* **Shared design.** Both models run on the claim-only relation graphs used by HAN, whose claim inputs include the
  patient, doctor and hospital attributes. They share the input projection and classification head with the other
  GNNs.
* **Unchanged training.** The training loop gains CARE-GNN's auxiliary loss and threshold update (zero and a no-op
  for every other model). SAGE and SAGE+AG retrained on one world reproduce their stored predictions exactly
  (maximum difference 0).

## A. Fraud-specialised GNNs

* **Tuning.** Same protocol and budget as every other method:
  * ten configurations, on the development worlds (seeds 901 and 902, δ = 0.5, s = 0.5), by validation AP;
  * BWGNN: hidden 32/64, order 2/3, dropout, learning rate, weight decay;
  * CARE-GNN: the GNN space with one or two layers;
  * re-tuned in the same way on the low-volume development worlds.
  * The existing selected configurations are unchanged by the merge; this is checked.
* **Synthetic runs.**
  * The main grid: nine regimes × seeds 1–12.
  * The low-volume test, frozen and re-tuned: δ = 0, three densities, seeds 1–6.
  * Ring volumes 8 and 2: δ = 0, three densities, seeds 1–6.
* **Public data.** Shared tuning on the separate split, the five rotations, per-rotation (nested) selection, and
  the four repeated rotations.
* **Analyses.** The existing analyses with the two methods added to the GNN family. The Holm families grow
  accordingly, so the adjusted p-values of the other methods are recomputed within the larger families. Primary
  question: does either fraud-specialised GNN outperform GB-CAG on ring AP:
  * pooled over regimes, with the seed as unit;
  * within any regime (Holm within each method's nine regimes);
  * at reduced ring volume.
* **Not extended to them:** the persistent-ring comparison, routing, ablations, training-seed replication and
  per-regime tuning.

## B. Ring shapes and sizes

* **Design.**
  * δ = 0, seeds 1–6, all 16 methods (the 14 of the article plus the two fraud-specialised GNNs), main
    configurations.
  * Four shapes, each at two volumes, with per-member volume held at the default (16 claims per month for 4
    doctors and 12 patients) or halved:
    * **hub**: 4 doctors (1 hub, 3 spokes), 12 patients; 16 and 8 claims per month;
    * **distributed** (`pairs`): 8 doctors, 24 patients; 32 and 16 claims per month;
    * **small block**: 2 doctors, 6 patients, s = 0.5; 8 and 4 claims per month;
    * **large block**: 8 doctors, 24 patients, s = 0.5; 32 and 16 claims per month.
  * 48 worlds in all, output `results/synthetic_shapes`.
  * For the hub and distributed shapes, s is not used; the runs carry s = 0.5.
* **Analyses**, in each shape × volume cell:
  * ring AP of every method;
  * GB-CAG − GB-CA (structure gain);
  * each GNN and hybrid − GB-CAG, paired over the six seeds, Holm over methods within the cell;
  * the overlap score's ring AP, to show how strongly the engineered closure statistic responds to each shape.
* **Primary question.** In which shapes, if any, does message passing outperform the engineered features?
* **Expected value.** It tests directly whether the paper's central conclusion depends on the shape the features
  were designed for.

## Cost and execution

* About 30 CPU-hours. Run by `paper/hiss_review/run_round4.sh` from a worktree of branch `round4-dev`, while round
  3 finishes in the main working tree. Records cite the branch's commits, which are merged into the submission
  branch.
* Evaluation jobs wait for their tuning to be merged, as in round 3.
* Synthetic jobs run one world or one regime per job and skip finished worlds.
