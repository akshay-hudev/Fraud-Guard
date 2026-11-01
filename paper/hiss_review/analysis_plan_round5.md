# Analysis plan for the fifth experimental round

Written 2026-11-01 and committed **before** any run of this round started; the commit time is the record. It answers
three issues raised by the Phase 1 readiness audit (`paper/audit_phase1/AUDIT_REPORT.md`): M1, M2 and m10.

## Why this round exists

1. **M1, tuning adequacy.** Each GNN saw 10 of its 144 configurations, and every later "re-tuning" re-selected among
   the same 10 candidates. So no earlier check can show whether the GNNs' deficit comes from the candidate list.
2. **M2, the mechanism.** The claim closure is, apart from a degree term, a count of four-cycles through the claim's
   patient–doctor edge. Message passing cannot count such substructures in general (Chen et al. 2020). If that is what
   the GNNs lack, giving a GNN those columns alone should recover most of the gap.
3. **m10, the confirmatory test.** It confirmed the hybrid's sparse-ring gain over fresh generator seeds but with one
   training seed, the seed used for tuning.

Every outcome below is reported in the manuscript or its Online Resource, whatever its direction. No settings, seeds
or methods will be added after the results are seen.

Observations made before this plan was committed, none of them part of the results:

* a timing probe of four large configurations on development world 901 (volume 4, s = 0.9);
* a check that stored runs never reached the 200-epoch cap (largest best epoch 128);
* a reproduction check of one stored world after the `fit_graph` change (bit-exact for GB-CAG, SAGE and HGT).

## A. Wide hyperparameter search

* **Methods.** SAGE, RGCN, Simple-HGN and HGT, the four strongest general-purpose GNNs, and GB-CAG with the same budget.
  HAN, BWGNN and CARE-GNN are not included: they trailed these four on the claim-only graphs.
* **Candidates.** 30 per method: the main configuration, plus 29 draws (sampling seed 1) from a wider grid, none of
  which any earlier tuning offered.
  * GNNs: hidden {16, 32, 64, 128}, layers {2, 3, 4}, dropout {0, 0.1, 0.3}, learning rate {3e-4, 1e-3, 3e-3, 1e-2},
    weight decay {1e-5, 1e-4, 1e-3}, and aggregation {sum, mean} or heads {2, 4, 8}.
  * GB-CAG: learning rate {0.01, 0.03, 0.1, 0.2}, leaves {7, 15, 31, 63, 127}, minimum leaf {5, 10, 20, 50},
    l2 {0, 1, 10}, iterations {200, 400, 800}.
* **Selection.** Validation **ring AP**, the primary metric, instead of overall AP, and early stopping on the same
  quantity (at most 300 epochs, patience 30; main protocol 200 and 20). All 30 candidates run on development world 901.
  The six best also run on world 902. The candidate with the best mean over the two is selected.
* **Development worlds** are drawn at the evaluation regime (δ = 0 and the cell's s and volume), the most favourable
  choice the design allows. Seeds 901 and 902 are never used for evaluation.
* **Cells.** δ = 0: volume 4 with s = 0.5 and s = 0.9, and default volume with s = 0.9; each tuned separately.
* **Evaluation.** Seeds 1–6, training seed 0, ring AP on the test period. The training label is unchanged (any fraud).
* **Comparisons.** In each cell: each GNN against GB-CAG, both searched widely; and each method's wide result against
  its main result on the same seeds. Paired t over six seeds with 95% CIs, Holm-adjusted over the methods of the cell.

**Decision rules**

* *Volume 4.* The claim that pure GNNs trail GB-CAG stands for a cell only if every GNN's mean stays below GB-CAG's
  and the upper 95% limit of its difference is below zero. Otherwise the abstract, Section 6.6 and the conclusion are
  changed to report the wide-search result for that cell.
* *Default volume.* If any GNN is significantly above GB-CAG, the statement that no pure GNN outperformed it is
  qualified accordingly.
* The narrower result is never reported alone. If the wide search changes a conclusion, the wide result takes
  precedence in the abstract.

## B. Closure columns

* **Variants** (δ = 0, volumes 8 and 4, s = 0.5 and 0.9, seeds 1–6, main hyperparameters; GB variants use GB-CAG's):
  * SAGE+Cl and HGT+Cl: the GNN with `claim_closure` and `claim_closure_norm` on claim nodes;
  * GB-CA+Cl: GB-CA with the two closure columns;
  * GB-CAG−Cl: GB-CAG without them, its other five graph features kept.
* **Prediction.** SAGE+Cl and HGT+Cl recover at least half of the gap between the pure GNN and GB-CAG, and GB-CAG−Cl
  loses most of the structure gain.
* **Comparisons.** Paired t over six seeds with 95% CIs, Holm-adjusted over the four cells of each comparison:
  variant against its base, and variant against GB-CAG.
* **Reporting rule.** If the prediction fails, the manuscript does not offer four-cycle counting as the explanation;
  it states only that the tested GNNs cannot count such patterns in general and that the columns did not close the gap.

## C. Confirmatory test over training seeds

* **Data.** As in plan C of round 2: δ = 0, s = 0.1, volume 4, seeds 7–18, the low-volume configurations, GB-CAG and
  SAGE+AG. Training seeds 1 and 2 are added to the original training seed 0.
* **Test.** Two-sided paired t, α = 0.05, on each world's mean over the three training seeds (primary), and for each
  training seed separately (descriptive).
* **Reporting rule.** *Robust to the training seed* if the primary test has p < 0.05 and a positive difference.
  Otherwise the manuscript states that the confirmation held for one training seed but not over three.

## Execution

`PARALLEL=4 bash paper/hiss_review/run_round5.sh`, from a committed tree, with `research/` unmodified until the queue
ends. Run records carry the commit and source hash.
