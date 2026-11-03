# HISS submission review: summary and readiness assessment (2026-10-25; updated 2026-11-03)

This review covers the target journal *Health Information Science and Systems* (Springer Nature), the manuscript
`fraud_detection_paper.tex` with its Online Resource `supplementary_material.tex`, and branch
`hiss-submission-audit`. It was carried out as an independent audit: scientific validity first, then novelty,
reproducibility, journal compliance and writing. No experiment was invented and no declaration was changed without
the authors.

| Document | Content |
|---|---|
| [audit_report.md](audit_report.md) | Issues by severity, with evidence, action and status |
| [novelty_comparison.md](novelty_comparison.md) | Verified literature comparison; what is and is not novel |
| [reference_verification.md](reference_verification.md) | How each reference and attributed claim was verified |
| [hiss_compliance.md](hiss_compliance.md) | Journal checklist from the official guidelines (mandatory vs. recommended) |
| [reproducibility_report.md](reproducibility_report.md) | Commands, seeds, environments, outcomes, failures, provenance of every round |
| [change_log.md](change_log.md) | Every material change to the paper and code |
| [unresolved_issues.md](unresolved_issues.md) | Author confirmations, decisions, experiments not run |
| [submission_package.md](submission_package.md) | Files to upload and the pre-upload checklist |
| [anticipated_reviewer_questions.md](anticipated_reviewer_questions.md) | Likely reviewer questions with prepared answers |
| [analysis_plan_round2.md](analysis_plan_round2.md) to [round6](analysis_plan_round6.md) | The five later experimental rounds, each fixed before its runs (commits `6c6a0c2`, `2eb368c`, `98638c4`, `3a7631a`, `e59f69f`) |
| `run_round2.sh` to `run_round5.sh` | The jobs of rounds 2–5 (136, 116, 107 and 23); round 6 re-scores stored predictions (`research/README.md`, step 16) |
| [../audit_phase1/AUDIT_REPORT.md](../audit_phase1/AUDIT_REPORT.md) | The Phase 1 readiness audit (2026-10-28), with its corrections, which led to rounds 5 and 6 |

**Rounds after the first audit.** The authors asked for the weaknesses to be addressed before submission. Three
pre-declared rounds followed:

* **Round 2** (2026-10-26): seeds 7–12 of the main grid, ring volumes 8 and 2, a confirmatory test of the hybrid
  result, repeated cross-validation and nested GNN tuning on the public data; the manuscript was split into an
  article and an Online Resource.
* **Round 3** (2026-10-27): feature shift at reduced volume, training-seed replication, per-regime tuning, and
  repeated rotations for every method.
* **Round 4** (2026-10-27/28): two fraud-specialised GNNs (BWGNN, CARE-GNN) and four further ring shapes and sizes.
* **Round 5** (2026-11-01/02), after the Phase 1 audit: a three-times wider hyperparameter search, the closure-column
  test of the mechanism, and the confirmatory test over two further training seeds.
* **Round 6** (2026-11-02): published expert group-fraud flags for the public data, which the manuscript had wrongly
  said did not exist, used to re-score the public-data analysis.

Every outcome is reported, including those that weakened earlier statements.

## Submission-readiness assessment

### Scientifically validated (evidence re-checked in this review)

* **Leakage controls**: label invariance, period isolation, entity-disjoint test rings, information flow,
  alignment, ring-shape construction. Automated tests: 44/44 pass.
* **Regeneration**: every table regenerates from the run records; retrained GB-CAG, SAGE and HGT reproduce the
  stored predictions on all 54 routing worlds; 663 numbers quoted in the article and Online Resource are recomputed
  from the artifacts (`paper/check_manuscript_numbers.py`, all pass). An earlier version of this file said that every
  run of rounds 2–4 came from a clean commit. That was false: 33 run files came from source states that match no
  commit. All 33 were re-run from the tagged code, and every AP was reproduced exactly (227 method runs). Rounds 5 and
  6 ran from clean commits.
* **No pure GNN beats gradient boosting with engineered graph features (GB-CAG)**: averaged over regimes (twelve
  seeds, seed as unit; upper 95% limits ≤ +0.006 AP), for any of four other ring shapes and sizes, and with
  per-regime re-tuning, and after a 30-candidate search selected on ring AP in three cells, where the low-volume
  deficit stayed at 0.26–0.56. One per-regime exception: RGCN at δ = 2, s = 0.1 (+0.005), where every method exceeds 0.98.
  The two fraud-specialised GNNs are below GB-CAG (−0.025 and −0.037, Holm p < 0.001).
* **One condition where some pure GNNs win**: sparse rings at reduced volume whose claims are also shifted (δ = 1,
  2): SAGE and HAN (and RGCN at δ = 2) beat GB-CAG by 0.07–0.18, but so does logistic regression without graph
  inputs; the manuscript attributes this to learning a weak claim-level shift, not to detecting coordination.
* **Hybrids gain a small amount**: +0.010 and +0.009 averaged over regimes (Holm p ≤ 0.016), most for sparse rings
  and for the distributed ring shape; confirmed at low volume on fresh seeds (+0.024, p = 0.016) and on the mean of
  three training seeds (+0.032, p = 0.005). With three
  training seeds the gain shrinks (+0.013 → +0.009 at δ = 0) and is of the order of training-seed variation; the
  article says so.
* **Isolated fraud**: every GNN is 0.093–0.229 AP worse than GB-CAG (Holm p < 0.001).
* **Value of structure**: +0.074 and +0.104 for moderately and densely linked rings at default volume; over ring
  volumes 16/8/4/2 the gain is 0.06/0.42/0.30/none; graph features also capture hub rings and larger and smaller
  blocks.
* **Memorisation**: recurring ring members inflate an attribute-only model by up to 0.60 AP.
* **Public data**: GB-P has the highest mean AP over 5 and 25 rotations and with per-rotation selection; graph
  features change it by at most +0.006/+0.007 (upper 95% limits, 25 rotations); only RGCN's deficit is significant.
  On the 100 providers that experts flagged as group fraud, graph features add nothing (−0.001) and every graph model
  is below GB-PG, as predicted before scoring; those providers are conspicuous by volume.

### Improved but not fully validated

* **GNN shortfall on the public data**: consistent in direction but not significant over rotations after
  correction, except RGCN.
* **Mechanism at low volume**: two closure columns recover most of the GNNs' gap, but GB-CAG does not need them, so
  four-cycle counting is not shown to be the explanation (the pre-declared prediction failed in one of its two parts).
* **Secondary analyses** (low volume, dose-response, shapes, mechanism, training seeds): six seeds; shapes at δ = 0
  only; mechanism in one regime.
* **Fraud-specialised GNNs**: our implementations on HAN's claim-only relation graphs; not in the persistent-ring,
  ablation, routing, training-seed or per-regime analyses.
* **Cost**: indicative (parallel load); excludes feature upkeep in an online system.
* **External validity**: synthetic benchmark; real data with labelled collusion do not exist publicly.

### Requires author confirmation

In `unresolved_issues.md`:

* U-3: funding, competing-interest and ethics statements;
* U-4: AI-use wording (extended on 2026-10-28 to name help with experiment design);
* circulating the final PDF to all authors (U-2 was approved before rounds 3 and 4 changed the results).

## Final decision criteria

**1. Does the work fit HISS's scope?**
Plausibly, but not centrally. HISS covers data mining and knowledge discovery on health information resources
for decision making, and applications for health information systems. Fraud screening of payer claims data fits
that description. Insurance and fraud are not named in the scope, however, and no HISS article on claims fraud
was found. The manuscript makes the health-information-systems relevance explicit. The authors decided (2026-10-27)
to submit directly, without a pre-submission enquiry or a cover letter.

**2. What is the precise, defensible contribution?**
A map, from a controlled and leakage-tested benchmark, of when relational structure helps detect collusive
provider rings in claim graphs (Table 8 of the article):

* Structure adds value when ring claims are not anomalous on their own and the ring's members are linked densely
  enough to stand out from ordinary shared care. Its value peaks when ring members are only moderately conspicuous
  by volume, and vanishes when rings file about one claim per member.
* Gradient boosting with a few engineered shared-patient features captures that value as well as five heterogeneous
  and two fraud-specialised GNNs, for five ring shapes and sizes, and far better at reduced volume; message passing
  on top of those features adds about 0.01 AP.
* GNNs dilute isolated-fraud signals.
* Two evaluation pitfalls are quantified: recurring ring members (up to 0.60 AP) and pooled tests over worlds that
  share data.

The analysis of the public provider-labelled data is a stricter replication of prior work, not a new finding.

**3. Is the contribution sufficiently distinct from the closest studies?**
For the controlled conditions, the volume dose-response, the shapes, the mechanism and the pitfalls: yes. We found
no study that varies the feature-level, structural, volume and shape signals of rings independently for
patient–provider graphs or tests on unseen rings. For the headline "graph-feature tabular models match or beat
GNNs": no. Yoo et al. (2023) showed this on the same public data, and GADBench and GraphLand outside healthcare. The
manuscript says so.

**4. Are the core results reproducible?**
Yes:

* every table regenerates from the run records, and the manuscript tables equal the generated files;
* retrained models reproduce the stored predictions;
* 44/44 research tests pass; 663/663 quoted numbers match; 44/44 cross-document references agree;
* both documents and the two-column upload bundle compile without errors, undefined references or overfull boxes;
* the code version is tagged `hiss-submission-v2` (U-1; `hiss-submission-v1` remains for the first version).

**5. Are the statistical comparisons defensible?**
Yes, with stated limits:

* the generator seed is the unit of pooled synthetic tests (twelve seeds, 11 df);
* rounds 2–6 were each fixed in a committed plan before their runs, every outcome is reported, including a failed
  prediction in round 5, and the article discloses that they were added after the first results were seen;
* the hybrid result was confirmed by pre-specified tests on fresh seeds and over three training seeds;
* public-data comparisons use the corrected resampled test (5 rotations) and the corrected repeated
  cross-validation test (25 rotations);
* Holm adjustment within pre-specified families, re-adjusted when the families grew; intervals are reported, and
  non-significance is not called equivalence.

**6. Does the real-world evaluation support the claims?**
Only the narrow claim made: on one public dataset with provider-level labels of undocumented provenance, no
graph-based method improved on provider aggregates, over 25 rotations and with per-rotation selection, and labels
show no assortativity through shared beneficiaries. On published expert group-fraud flags the same holds, as
predicted, but those providers stand out by volume. It cannot validate ring detection or the dense-ring regime, and
the manuscript says so.

**7. The three biggest reasons a HISS reviewer might reject the paper**

1. **External validity.** The central conclusions rest on the authors' own simulator. The public data now include
   expert group-fraud flags, but those providers are large, so they do not test dense rings of unremarkable members.
2. **Incremental novelty.** "Tree ensembles with graph features beat GNNs" is known in general and on this
   dataset. The paper's own claim is the map of conditions and the pitfalls, which a reviewer may still judge
   modest.
3. **Scope.** Insurance fraud is peripheral to HISS's mostly clinical scope.

**8. Changes completed to address them**

* (1) Group-fraud flags on the public data (round 6); a wider hyperparameter search and a mechanism test (round 5);
  low-volume stress test, ring-volume dose-response, feature shift at reduced volume, four further ring shapes
  and sizes (the designer-advantage concern), fraud-specialised GNNs, training-seed and per-regime-tuning checks,
  narrowed real-data claims with provenance stated honestly; public-data analyses strengthened.
* (2) Contributions built around the map of conditions and the quantified pitfalls; verified closest works cited;
  real-data result framed as a replication; novelty table.
* (3) Health-information-systems framing; HISS compliance (structured abstract of 239 words, 246 if ranges count as
  two, DOIs, figure naming
  and Arial lettering, Online Resource as PDF, declarations).

**9. What remains unresolved?**

* Author confirmations U-3 and U-4, and circulating the final PDF to every author.
* Optional: ORCID iDs (U-6); revoking an old Kaggle API token (U-10); archiving the test predictions with a DOI on
  acceptance (U-11).
* Experiments that cannot be run: real data with labelled collusion (E-5); latency of a deployed system (E-6).

**10. Submit now?**
**The scientific work for this submission is complete, and the package is technically ready** once the authors
have confirmed U-3 and U-4, seen the final PDF, and pushed the tag (U-1). That does not make acceptance likely or certain: the three
reasons in point 7 remain, and only reviewers can judge them. Improved writing alone is not a reason to call the
manuscript ready; the conditions above are.
