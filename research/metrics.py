"""Evaluation metrics and statistical comparisons.

Primary metric: average precision (area under the precision-recall curve).
Operating-point metrics use a threshold chosen on the validation split
(maximum F1), never on the test split.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def _ap(y, s):
    return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def _roc(y, s):
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def val_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Threshold maximising F1 on the validation split."""
    p, r, t = precision_recall_curve(y_val, s_val)
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12), 0)
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def recall_at_budget(y: np.ndarray, s: np.ndarray, budget: float) -> float:
    """Recall when the top ``budget`` fraction of claims (by score) is investigated."""
    k = max(1, int(round(budget * len(s))))
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def evaluate_scores(y_val, s_val, y_test, s_test, fraud_type_test=None, budget: float = 0.05) -> dict:
    thr = val_threshold(y_val, s_val)
    pred = (s_test >= thr).astype(int)
    out = {
        "ap": _ap(y_test, s_test),
        "roc_auc": _roc(y_test, s_test),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        f"recall_at_{int(budget * 100)}pct": recall_at_budget(y_test, s_test, budget),
        "threshold": thr,
        "n_test": int(len(y_test)),
        "n_pos": int(np.sum(y_test)),
    }
    if fraud_type_test is not None:
        ft = np.asarray(fraud_type_test)
        for kind in ["ring", "isolated"]:
            keep = (ft == "legit") | (ft == kind)
            if (ft == kind).any():
                yk = (ft[keep] == kind).astype(int)
                out[f"{kind}_ap"] = _ap(yk, s_test[keep])
                out[f"{kind}_recall"] = float(pred[ft == kind].mean())
                out[f"n_{kind}"] = int((ft == kind).sum())
    return out


def t_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (float(x.mean() - h), float(x.mean() + h))


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of a - b across replicates: mean, t-CI, paired t and Wilcoxon p-values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    out = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else float("nan")}
    out["ci_low"], out["ci_high"] = t_ci(d)
    if len(d) >= 2 and d.std(ddof=1) > 0:
        out["t_p"] = float(stats.ttest_rel(a[ok], b[ok]).pvalue)
        out["cohen_dz"] = float(d.mean() / d.std(ddof=1))
    else:
        out["t_p"], out["cohen_dz"] = float("nan"), float("nan")
    out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 and np.any(d != 0) else float("nan")
    return out


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are kept as NaN)."""
    p = np.asarray(pvalues, float)
    idx = np.flatnonzero(np.isfinite(p))
    adj = np.full_like(p, np.nan)
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
