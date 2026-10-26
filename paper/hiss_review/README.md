# HISS submission review: summary and readiness assessment (2026-10-25)

This review covers the target journal *Health Information Science and Systems* (Springer Nature), the manuscript
`fraud_detection_paper.tex`, and branch `hiss-submission-audit`. It was carried out as an independent audit:
scientific validity first, then novelty, reproducibility, journal compliance and writing. No experiment was
invented and no declaration was changed without the authors.

| Document | Content |
|---|---|
| [audit_report.md](audit_report.md) | Issues by severity, with evidence, action and status |
| [novelty_comparison.md](novelty_comparison.md) | Verified literature comparison; what is and is not novel |
| [reference_verification.md](reference_verification.md) | How each reference and attributed claim was verified |
| [hiss_compliance.md](hiss_compliance.md) | Journal checklist from the official guidelines (mandatory vs. recommended) |
| [reproducibility_report.md](reproducibility_report.md) | Commands, seeds, environments, outcomes, failures |
| [change_log.md](change_log.md) | Every material change to the paper and code |
| [unresolved_issues.md](unresolved_issues.md) | Blockers, author decisions, experiments not run (with commands) |
| [submission_package.md](submission_package.md) | Files to upload and the pre-upload checklist |

## Submission-readiness assessment

### Scientifically validated (evidence re-checked in this review)

* **Leakage controls**: label invariance, period isolation, entity-disjoint test rings, information flow,
  alignment. Ten automated tests, re-run here.
* **Bit-for-bit regeneration** of every table from the run records. The retrained GB-CAG, SAGE and HGT reproduce
  the stored predictions on all 54 worlds.
* **Isolated fraud**: every GNN is worse than gradient boosting with engineered graph features (GB-CAG) on
  isolated fraud, by 0.085–0.147 AP (Holm p ≤ 0.009 with the seed as unit).
* **Value of structure at default volume**: graph-aware methods beat gradient boosting without graph features
  (Holm p ≤ 0.021, except HAN). Structure adds up to 0.10 AP for dense rings at default volume.
* **No GNN better than GB-CAG** in any regime, nor at low volume.
* **Low ring volume**:
  * Aggregates fall to near chance.
  * GB-CAG reaches 0.41–0.57 for s ≥ 0.5.
  * Pure GNNs are 0.23–0.29 AP lower (Holm p ≤ 0.002).
  * Hybrids match GB-CAG.
  * Re-tuning on low-volume development worlds: see `reproducibility_report.md` §4 and manuscript §6.6.
* **Memorisation**: recurring ring members inflate an attribute-only model by up to 0.60 AP.
* **Public data**: GB-P has the highest mean AP; no graph-based method has a higher mean.

### Improved but not fully validated

* **Hybrids' small ring-AP gain (+0.008)**: positive in 5 of 6 seeds, not significant.
* **GNN shortfall on the public data**:
  * Conditional on the trained models, the provider bootstrap puts every GNN below GB-P.
  * Over rotations, the corrected test is not significant.
  * GNN hyperparameters still share the overlapping tuning split; the bias direction favours the GNNs.
* **Low-volume test**: one reduced volume, δ = 0 only, one training seed per world.
* **Mechanism analyses**: one regime, one training seed per world.
* **Cost**: indicative (parallel load); excludes feature upkeep in an online system.

### Requires author confirmation

U-1 to U-4 and U-7 in `unresolved_issues.md`:

* public code version (blocker);
* approval of all authors and of the CRediT roles;
* funding, competing-interest and ethics statements;
* AI-use wording;
* disclosure of any earlier version of the manuscript.

### Requires additional experiments (optional for submission)

E-1 to E-6 in `unresolved_issues.md`:

* nested GNN tuning on the public data;
* further volume levels;
* training-seed replication;
* per-regime tuning;
* labelled-collusion data;
* online latency.

## Final decision criteria

**1. Does the work fit HISS's scope?**
Plausibly, but not centrally. HISS covers data mining and knowledge discovery on health information resources
for decision making, and applications for health information systems. Fraud screening of payer claims data fits
that description. Insurance and fraud are not named in the scope, however, and no HISS article on claims fraud
was found. The revision makes the health-information-systems relevance explicit. A pre-submission enquiry is
advisable.

**2. What is the precise, defensible contribution?**
A controlled, leakage-tested benchmark for collusive provider rings in heterogeneous claim graphs, and the
conditions it reveals:

* Relational structure adds value only for moderately or densely linked rings, and matters most when ring members
  are not otherwise unusual (low volume).
* Gradient boosting with a few engineered shared-patient features captures that value at least as well as five
  heterogeneous GNNs, and far better at low volume.
* GNNs dilute isolated-fraud signals.
* Recurring ring members inflate evaluations.

The analysis of the public provider-labelled data is a stricter replication of prior work, not a new finding.

**3. Is the contribution sufficiently distinct from the closest studies?**
For the controlled conditions, mechanism and memorisation analyses: yes. We found no study that varies
feature-level and structural ring signals independently for patient–provider graphs or tests on unseen rings.
For the headline "graph-feature tabular models match or beat GNNs": no. Yoo et al. (2023) showed this on the same
public data, and GADBench and GraphLand outside healthcare. The manuscript now says so.

**4. Are the core results reproducible?**
Yes, from the repository on this machine:

* every table regenerates bit-for-bit from the run records;
* retrained models reproduce the stored predictions;
* 27/27 research tests and 53 legacy tests pass;
* the manuscript compiles without errors.

They are **not yet reproducible by others**: the public repository lacks the code and results of the last 12
commits (U-1).

**5. Are the statistical comparisons defensible?**
Now yes, with stated limits:

* the seed is the unit of pooled synthetic tests;
* comparisons over the overlapping real-data rotations use the corrected resampled t-test;
* Holm adjustment within pre-specified families;
* intervals are reported, and non-significance is not called equivalence.

Before this revision they were not: several "significant" pooled results came from treating 54 correlated worlds
as independent. Power remains low (5 df; five rotations).

**6. Does the real-world evaluation support the claims?**
Only the narrow claim now made: on one public dataset with provider-level labels of undocumented provenance, no
graph-based method improved on provider aggregates, and labels show no assortativity through shared
beneficiaries. It cannot validate ring detection or the dense-ring regime, and the manuscript says so.

**7. The three biggest reasons a HISS reviewer might reject the paper**

1. **External validity.** The central conclusions rest on the authors' own simulator, whose engineered features
   were designed for its typology, and the only real data cannot test ring detection.
2. **Incremental novelty.** "Tree ensembles with graph features beat GNNs" is known in general and on this
   dataset (Yoo et al. 2023, previously uncited).
3. **Scope and form.** Insurance fraud is peripheral to HISS's mostly clinical scope; the manuscript is long
   (≈36 pages) and dense.

**8. Changes completed to address them**

* (1) Low-volume stress test with frozen and re-tuned hyperparameters, which removes the strongest confound
  (ring volume). Designer-advantage caveat and narrowed real-data claims, with provenance stated honestly.
* (2) Ten verified closely related works added, contributions restated around conditions, mechanism and
  memorisation, real-data result framed as a replication, novelty table.
* (3) Health-information-systems framing (claims data, audit budgets, batch computation) and HISS compliance
  fixes (DOIs, abbreviations, figure naming, declarations).
* Statistical re-analysis throughout. Two numerical errors corrected.

**9. What remains unresolved?**

* Public code version (U-1).
* Author confirmations (U-2 to U-4, U-7).
* Scope enquiry (U-8).
* Optional: two-column format, ORCID iDs, figure lettering.
* Experiments not run: nested GNN tuning (E-1), further volume levels (E-2), training-seed replication (E-3),
  per-regime tuning (E-4), labelled collusion data (E-5), online latency (E-6).

**10. Submit now, or another revision cycle first?**
**Another short cycle first; not a scientific overhaul.** The manuscript is now scientifically defensible, but
it should not be submitted until:

* the code version it describes is public and tagged (U-1; otherwise the Code availability statement is untrue);
* all five authors have approved the substantially changed results and confirmed the declarations
  (U-2 to U-4, U-7).

These are days of work, not months. Optionally, before submitting:

* make a pre-submission enquiry about scope;
* consider moving the cost details and the small-benchmark re-analysis (§9.3, Table 9) to supplementary material
  to shorten the paper.

Nested GNN tuning (E-1) would remove the last tuning caveat but cannot reverse the conclusion.
