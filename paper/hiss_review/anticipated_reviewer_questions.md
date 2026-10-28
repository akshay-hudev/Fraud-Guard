# Questions reviewers are likely to ask, with prepared answers

For the authors, to prepare for the review. Each answer says what the manuscript already contains and where, and
states the weakness plainly where there is one. Section and table numbers refer to the compiled article
(single-column) and to Online Resource 1 (S-numbers). Do not paste these answers into a response letter unchanged:
answer the questions the reviewers actually ask.

## Scope and contribution

**1. Why is insurance-fraud detection in scope for *Health Information Science and Systems*?**
Payer claims are a health information resource, and fraud screening is a decision-support function of payers'
claims-processing and audit systems. The work is data mining on that resource for decision support, and it
evaluates the analytics components such systems would use. The Introduction and the Discussion (paragraph after
the recommendations) state this. Weakness: fraud is not named in the journal's aims and scope; the authors chose
to submit without an enquiry.

**2. What is new compared with Yoo et al. (2023), who found graph features beat GNNs on Medicare data?**
Yoo et al. compared methods on one provider-labelled dataset, which cannot say *when* structure helps or *why*.
This study varies the properties of collusive rings independently (feature shift, density, volume and, in the
fourth round, shape and size), tests on rings of unseen members, explains the result mechanistically (ablations,
rewiring, input reliance) and quantifies two evaluation pitfalls. On the public data it replicates Yoo et al.'s
direction under stricter conditions (provider-disjoint folds, equal tuning, corrected tests, nested selection).
See contribution list (Section 1) and `paper/hiss_review/novelty_comparison.md`.

**3. Is a negative result ("GNNs do not beat engineered features") publishable?**
The contribution is the map of conditions (Table 8), not a single negative result: structure helps for dense
rings and at reduced volume; GNNs match or trail engineered features where structure helps; some GNNs win for
sparse rings with mildly anomalous claims at reduced volume, where a linear model also wins. Recommendations for
payers' systems follow from it (Section 9).

## Benchmark validity

**4. Synthetic data: why should results transfer to real claims?**
They may not, and the manuscript says so (Limitations; Conclusion). Real claims with labelled collusive schemes
are not publicly available (E-5 could not be run). The synthetic benchmark is used because only there can the
typology and signal strength be varied and the ground truth be known. The public provider-labelled data are a
consistency check, not a validation of the dense-ring regime (Section 9, public-data paragraph).

**5. The engineered graph features were designed knowing the injected ring shape. Isn't the comparison rigged?**
Acknowledged in the Discussion ("One caveat favours the engineered features"). The fourth round tested four other
shapes and sizes (hub-and-spoke, distributed pairs, small and large blocks) with every method.
*[To complete after round 4: outcome, with Table reference.]*

**6. Ring prevalence (8% of non-isolated claims) is unrealistically high.**
Stated in the Limitations. The ring-volume experiments lower it to 1.1–4.2% (volumes 8, 4, 2) and change the
conclusions in the direction reported (aggregates lose power; structure dominates for dense rings; nothing
detects rings that file one claim per member).

**7. Rings are static, of one size, and fraudsters do not adapt.**
Stated in the Limitations. Ring size and shape were varied in the fourth round; temporal dynamics and adaptive
behaviour were not modelled.

## Methods compared and tuning

**8. Why no fraud-specialised GNNs (CARE-GNN, PC-GNN, BWGNN, GAGA)?**
BWGNN and CARE-GNN were added in the fourth round, tuned with the same budget, and run in the main grid, at
reduced volume and on the public data. PC-GNN and GAGA were not run.
*[To complete after round 4: outcome.]*
Implementation differences from the original papers are stated in Section 4.3 (claim-only relation graphs as for
HAN; CARE-GNN trained full-batch with the class-weighted loss used for every model instead of under-sampling).

**9. Ten configurations per method is a small tuning budget; GNNs may need more.**
Every method had the same budget, chosen on development worlds not used for evaluation. Three checks: re-tuning
at low volume (Table 4), per-regime re-tuning at δ = 0, s = 0.1 and 0.9 (S7: no method changed significantly),
and per-rotation (nested) selection on the public data (S12: GNNs changed by −0.008 to +0.036, none
significantly). A larger budget was not tried; say so if asked.

**10. Each model was trained once per world. Are the differences robust to initialisation?**
The third round retrained five methods with two further training seeds (S6). GNN deficits are unchanged; the
hybrids' small gain shrinks (+0.013 to +0.009 pooled) and is of the order of the training-seed variation, which
the article now states. A related subtlety, found and corrected during the revision: with a fixed training seed
every world, including the tuning worlds, starts from the same initial weights (Section 4.3).

**11. Full-batch training on ~11,000-node graphs. Does this scale?**
It does not need to for this benchmark; larger graphs would need sampling-based training (Limitations). Cost and
latency measurements are in Section 6.8 and S8, with the caveat that feature upkeep in an online system was not
measured (E-6 could not be run without a deployment).

## Statistics

**12. Why is the seed, not the world, the unit of replication?**
The nine regimes of a seed share legitimate claims, isolated fraud and ring members (by design, test (x) in
Section 10), so pooled tests average each seed's nine paired differences and test the twelve seed means (11 df).
Treating 108 worlds as independent overstates significance; this is one of the quantified pitfalls.

**13. Multiple comparisons?**
Holm adjustment within pre-specified families (Section 4.5); the families are stated with each table. The
fourth round enlarged the method families, and the p-values of the original methods were re-adjusted.

**14. Non-significant differences are not evidence of equivalence.**
Agreed and stated (Section 4.5); the manuscript reports confidence intervals and upper limits (for example, every
pure GNN's upper 95% limit against GB-CAG is at most +0.006 AP averaged over regimes).

**15. Were analyses chosen after seeing results?**
Seeds 7–12 and three further rounds were added after the first results were seen, each under a plan committed
before its runs (`analysis_plan_round2.md` to `round4.md`; commit times are the record). The one result singled
out after exploration (hybrid advantage at low volume) was retested on fresh seeds with a single pre-declared
test (S4). Stated in Section 4.5 and the Limitations.

**16. Why average precision and not recall at an investigation budget?**
Recall at the top 5% of claims is reported, but every method with aggregates reaches 0.43–0.48 of a ceiling of
0.479 because fraud prevalence exceeds the budget, so it cannot separate methods (Section 6, first paragraph).
AP covers the whole ranking. Recall at the top 10% of providers is reported for the public data.

## Public data

**17. The Kaggle dataset's provenance and labels are undocumented.**
Stated (Sections 5 and 9). It is used because it is the most widely used public provider-labelled set, which
allows comparison with earlier work, not as a model of current Medicare claims.

**18. Why corrected resampled tests?**
Rotations share training data, so a plain paired t-test overstates significance; over 25 rotations it would have
called every difference significant, whereas the corrected test (Nadeau & Bengio; Bouckaert & Frank) finds only
RGCN's deficit significant (S12).

## Reproducibility and declarations

**19. Is the code available?**
Yes, publicly, at the tagged version `hiss-submission-v1` (Code availability). A DOI archive is planned on
acceptance; do not claim it exists before then. Test predictions (about 30 MB) are regenerated by the code.

**20. Use of AI tools.**
Declared in Section 4.6. Be ready to describe the authors' own verification of the AI-assisted analysis.

**21. Ethics.**
Synthetic data and a public de-identified dataset; no human participants. "Not applicable", as confirmed by the
authors.
