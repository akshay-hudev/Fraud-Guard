# Analysis plan for the sixth experimental round

Written 2026-11-02 and committed **before** any model score, feature or graph statistic was evaluated against the
group-fraud labels; the commit time is the record.

## Why this round exists

The manuscript said that public claims data with labelled collusion do not exist. That is false. Ma et al. (IJCAI 2023)
added expert-annotated group-fraud labels to the public provider-labelled dataset that this study already uses, and
published them at <https://github.com/finint/RDPGL>. The Phase 1 audit missed this, and described that study's data
loosely as "real insurer data". The statement will be corrected whatever this round finds.

The labels also allow a check that was thought impossible: whether the controlled conclusions hold on labels that mark
organised fraud rather than any fraud. The main uncertainty this resolves is external: until now, every conclusion
about rings rested on our own simulator.

Every outcome below is reported in the manuscript or its Online Resource, whatever its direction. No settings, methods
or labels will be added after the results are seen.

## Data

* `claims.csv` and `bene.csv` from `finint/RDPGL` at commit `6e0151d40ea5e9def878959d9863bb6f2a8cf8f4` (git blobs
  `0f440bd27f3f537c379a46fd7bea10486e417701` and `00b5fd2ec98f3086813139d0b10ee7cb41a278b5`, both verified). The repository
  is GPL-3.0; the files are downloaded by documented command, not redistributed.
* Provider identifiers there are integers. The mapping to Kaggle identifiers is by rank (the i-th sorted Kaggle provider
  is 1000 + i). It must be confirmed for all 5,410 providers on claim count, inpatient count, total reimbursement and
  total deductible, and on the fraud label. If any provider fails, this round stops and the text says only that the
  labels exist.

## Observations made before this plan was committed

None of them involves a model, a feature or a graph.

* The README and the paper's dataset section. The annotation criterion is not documented: the paper says only that
  anti-fraud experts of China's National Healthcare Security Administration judged each claim from domain expertise.
  Only a binary flag is published, with no group identifiers.
* 558,211 claims, 5,410 providers, 506 fraud providers, 100 group-fraud providers. The group flag is constant within
  a provider and is a subset of the provider fraud label.
* The 100 group-fraud providers hold 109,683 of the 212,796 claims of fraud providers, so they are unusually large.
* The rank mapping matched the 4,339 providers with a unique fingerprint, with full label agreement.

## A. Primary analysis: stored predictions, re-scored

* **Models and folds.** Unchanged: the 15 methods and 25 rotations of the public-data analysis, trained on the provider
  fraud label, with their stored test predictions. Nothing is retrained. This mirrors the synthetic protocol, where
  models are trained on any fraud and scored on rings.
* **Primary metric, group AP.** In each test fold: group-fraud providers against non-fraud providers, the other fraud
  providers excluded. It is the analogue of ring AP.
* **Secondary metric, individual AP.** Fraud providers without the group flag against non-fraud providers, group-fraud
  providers excluded.
* **Test.** Corrected repeated cross-validation test, as in the rest of the public-data analysis (variance factor
  1/25 + 1/3, 24 df).
* **Families.** Holm-adjusted within each.
  * *G1, structure:* GB-PG − GB-P and GB-PGL − GB-P.
  * *G2, graph models:* each of SAGE, RGCN, HAN, Simple-HGN, HGT, BWGNN, CARE-GNN, SAGE+G and HGT+G against GB-PG.
  * *G3, one test:* whether GB-PG − GB-P is larger on group AP than on individual AP.

## B. Secondary analysis: retraining on the group label

Models in A were not trained to find groups, so a graph model might do better if asked to.

* Target: the group flag. The other fraud providers are removed from training, validation and test.
* Hyperparameters: unchanged, those of the public-data analysis.
* GB-P and GB-PG on all 25 rotations; SAGE, RGCN and HGT on the five rotations of the first repetition.
* Tests: GB-PG − GB-P with the test of A; each GNN against GB-PG on the five rotations with the corrected resampled
  test (variance factor 1/5 + 1/3, 4 df), Holm over the three GNNs. With five rotations this has little power, and the
  manuscript says so.

## C. Description of the labels

Group-fraud, other fraud and non-fraud providers are compared on median claims per provider, and on the medians of
the label-free graph features `bene_max_jaccard`, `bene_shared_share` and `bene_log_n_coproviders`. The article's
assortativity statistic is repeated for the group flag: the mean share of a provider's co-providers that carry the
flag, for flagged and unflagged providers, through shared beneficiaries and shared physicians.

## Predictions

The 100 providers are already known to be unusually large, so Table 8 of the article makes two predictions.

1. **P1.** Aggregates find most of them and graph features add little: GB-PG − GB-P is not significantly positive
   (G1, Holm p ≥ 0.05).
2. **P2.** No pure GNN is significantly above GB-PG on group AP (G2, Holm p < 0.05 with a positive difference).

## Reporting rules

* The abstract, the public-data results and the conclusion report the group AP result, whichever way it falls.
* **If P1 fails**, the manuscript states that graph features added to detection of group-fraud providers on these
  labels, with the caveat below on circularity.
* **If P2 fails**, the statement that no pure GNN outperformed gradient boosting with graph features is qualified in
  the abstract and conclusion.
* **Circularity.** The annotation criterion is undocumented. If the experts used shared patients, a structure gain
  would be partly built in, and the text must say so whichever way P1 falls.
* **What this is not.** These are not confirmed fraud cases, and the claims show signs of being synthetic
  (Section 5.1 of the article). The manuscript may call the result consistent or inconsistent with the map. It may not
  call it a validation of the benchmark.
* **Dense-ring regime.** The result bears on that regime only if C shows the flagged providers to be more densely linked
  than others of similar standing. Otherwise the manuscript says that these data sit in the regime of unusually active
  members and do not test dense, low-volume rings.

## Execution

From a committed tree. The code and this plan sit on branch `hiss-revision-round6`, in a separate worktree, so that
`HEAD` and `research/` of the main tree stay unchanged while round 5 runs. The branch is merged after round 5 ends.
Run records carry the commit and source hash.
