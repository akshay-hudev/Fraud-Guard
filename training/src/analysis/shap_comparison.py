"""
Tabular feature attribution (TreeSHAP) for the RF and GB baselines.
Produces logs/shap_comparison.json and figures/fig_shap_comparison.{png,pdf}.

Feature groups are assigned by feature name (not by column index), and only
quantities computed here are written out — no GNN numbers are included, since
GNN attributions require their own experiment.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, pickle
import shap

os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

CLAIM_LEVEL = {"claim_amount", "num_procedures", "days_in_hospital", "claim_month", "claim_dayofweek", "claim_day"}
RATIO_PREFIX = "amount_vs_"
GROUP_COLORS = {"claim-level": "#4472C4", "entity/aggregate": "#ED7D31", "cross-entity ratio": "#C00000"}


def feature_group(name: str) -> str:
    if name in CLAIM_LEVEL:
        return "claim-level"
    if name.startswith(RATIO_PREFIX):
        return "cross-entity ratio"
    return "entity/aggregate"


with open("data/processed/splits.pkl", "rb") as f:
    splits = pickle.load(f)
X_test = splits["X_test"]
with open("data/processed/feature_names.json") as f:
    feature_names = list(json.load(f))
assert len(feature_names) == X_test.shape[1], "feature_names.json does not match X_test"

shap_results = {}
for model_name in ["random_forest", "gradient_boosting"]:
    model_path = f"models/baseline/{model_name}.pkl"
    if not os.path.exists(model_path):
        continue
    model = joblib.load(model_path)
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    mean_abs = np.abs(sv).mean(axis=0)
    totals = {g: 0.0 for g in GROUP_COLORS}
    for name, value in zip(feature_names, mean_abs):
        totals[feature_group(name)] += float(value)
    grand = sum(totals.values()) or 1.0
    shap_results[model_name] = {
        "top_features": [
            {"name": feature_names[i], "group": feature_group(feature_names[i]),
             "mean_abs_shap": round(float(mean_abs[i]), 4)}
            for i in np.argsort(mean_abs)[::-1][:10]
        ],
        "group_importance_pct": {g: round(100 * v / grand, 1) for g, v in totals.items()},
    }

with open("logs/shap_comparison.json", "w") as f:
    json.dump({"tabular_shap": shap_results}, f, indent=2)
print("Saved logs/shap_comparison.json")

fig, axes = plt.subplots(1, len(shap_results), figsize=(7 * len(shap_results), 6), squeeze=False)
for ax, (model_name, res) in zip(axes[0], shap_results.items()):
    top = res["top_features"]
    ax.barh([t["name"] for t in top], [t["mean_abs_shap"] for t in top],
            color=[GROUP_COLORS[t["group"]] for t in top])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(model_name.replace("_", " ").title())
    ax.invert_yaxis()
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()],
           loc="lower center", ncol=3)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("figures/fig_shap_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig_shap_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close()
print("Saved figures/fig_shap_comparison.png and .pdf")
