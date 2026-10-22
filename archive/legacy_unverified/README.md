# Legacy artifacts — unverified, do not cite

These files were produced (or partly hand-written) by the pre-remediation pipeline and are kept only for
traceability. The research audit (`paper/research_audit.md`) found them unreliable:

| File | Problem |
|---|---|
| `logs/tabular_5seed_results.json`, `logs/full_results_table.json`, `logs/model_results_table.*` | Models trained with the label-derived `ring_id` feature (audit B-02); RF/GB F1 = 1.0 |
| `logs/ring_fraud_evaluation.json`, `logs/per_ring_recall.json`, `logs/threshold_sweep.json` | Ring metrics computed on positionally misaligned claims (audit B-07); HGT reported as 0.0 because it never ran; sweep substituted GB scores for HGT and used hard-coded latencies |
| `logs/shap_comparison.json`, `figures/fig_shap_comparison.pdf` | HGT edge-ablation deltas (−8.4/−4.7/−1.2/−10.6) and the "+14.2pp" narrative were string literals, not results (audit B-01) |
| `logs/statistical_tests.json` | Test never ran ("HGT not run") |
| `logs/real_world_results.json` | Labelled "real world" but computed on the synthetic data (audit B-11) |
| `logs/tabular_best_params.json` | Degenerate tuning (validation F1 = 1.0 ties) on leaky features |
| `logs/dataset_config.json` | Valid generator config/hash for the legacy 2,100-claim dataset (kept for reference) |
| `training_models/comparison.json` | Older 10,000-claim dataset; HGT row contains hand-formatted values with no generating run |

All results reported in the manuscript are now produced by the `research/` pipeline and recorded in `results/`.

## Audited manuscript

`archive/manuscript/fraud_detection_paper_audited_2026-10-19.tex` is the exact 1,110-line manuscript audited on
2026-10-19 (SHA-256 `61becf493e5775529a0d193e4fd13d08fc2b0ac7600cd812e06fad52543c022f`, recovered from editor local
history because it had not been committed). Its quantitative claims are unsupported (see the audit); it is kept only
for traceability.
