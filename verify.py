import os, json

print("=== FILE CHECK ===")
expected = {
    "logs/full_results_table.json":    ["logistic_regression","random_forest","gradient_boosting"],
    "logs/statistical_tests.json":     ["t_statistic","p_value","cohens_d"],
    "logs/per_ring_recall.json":       ["per_ring_recall"],
    "logs/ring_fraud_evaluation.json": ["ring_claim_recall","ring_doctor_recall"],
    "logs/tabular_best_params.json":   ["GB","RF","LR"],
    "logs/dataset_config.json":        ["dataset_sha256","n_rings"],
    "logs/real_world_results.json":    ["models"],
    "figures/threshold_sensitivity.png": None
}

all_pass = True
for path, required_keys in expected.items():
    if not os.path.exists(path):
        print(f"MISSING   {path}"); all_pass = False; continue
    size = os.path.getsize(path)
    if size < 10:
        print(f"EMPTY     {path}"); all_pass = False; continue
    if required_keys and path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"PARTIAL   {path} missing keys: {missing}")
            all_pass = False; continue
    print(f"PASS  {path}  ({size} bytes)")

print()
print("=== KEY METRICS SPOT CHECK ===")

with open("logs/statistical_tests.json") as f:
    st = json.load(f)
hgt_seeds = st.get("hgt_ring_recall_per_seed", [])
gb_seeds  = st.get("gb_ring_recall_per_seed", [])
if not hgt_seeds:
    print("  NOTE: HGT skipped (torch-sparse not installed) - t-test needs HGT seeds")
    print(f"  GB ring recall per seed: {gb_seeds}")
else:
    print(f"  t={st.get('t_statistic')}, p={st.get('p_value')}, d={st.get('cohens_d')}, sig={st.get('significant_at_0.05')}")

with open("logs/per_ring_recall.json") as f:
    pr = json.load(f)
rings = pr.get("per_ring_recall", {})
if isinstance(rings, dict):
    sample = {k: v for k, v in list(rings.items())[:3]}
    print(f"  per-ring recalls (sample): {sample}")
else:
    print(f"  per_ring_recall type: {type(rings)} value: {rings}")

with open("logs/full_results_table.json") as f:
    rt = json.load(f)
print(f"  Models in results table: {list(rt.keys())}")
for m in ["gradient_boosting","logistic_regression"]:
    if m in rt:
        row = rt[m]
        print(f"  {m}: f1={row.get('f1_mean') or row.get('f1')}, auc_pr={row.get('auc_pr_mean') or row.get('auc_pr')}")

with open("logs/real_world_results.json") as f:
    rw = json.load(f)
print(f"  real_world dataset: {rw.get('dataset')}")
print(f"  real_world models: {list(rw.get('models',{}).keys())}")

print()
if all_pass:
    print("FINAL: 8/8 FILES READY")
    if not hgt_seeds:
        print("NOTE: Install torch-sparse to enable HGT + t-test results")
        print("      pip install torch-sparse (or pyg-lib) then re-run pipeline")
else:
    print("FINAL: INCOMPLETE - check above")
