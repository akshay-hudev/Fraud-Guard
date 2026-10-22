"""Training loops for graph models, the MLP control and tabular models."""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..ledger import set_seed
from .gnn import MLP, GraphModel

torch.set_num_threads(int(os.environ.get("RESEARCH_THREADS", "4")))


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    pos = float(y.sum())
    return torch.tensor((len(y) - pos) / max(pos, 1.0))


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def _as_list(g):
    return g if isinstance(g, (list, tuple)) else [g]


def _scores(model, graphs, target) -> np.ndarray:
    return np.concatenate([torch.sigmoid(model(g)[g[target].target_mask]).numpy() for g in graphs])


def fit_graph(kind: str, graphs: dict, hp: dict, seed: int, max_epochs: int = 200, patience: int = 20,
              target: str = "claim") -> dict:
    """Train on the 'train' graphs, early-stop on 'val' AP, score the 'val' and 'test' target claims.

    Each split may hold several graphs (one per audit period); the training loss
    is the mean over all training target claims.
    """
    set_seed(seed)
    g_tr, g_va, g_te = (_as_list(graphs[k]) for k in ["train", "val", "test"])
    model = GraphModel(kind, g_tr[0], hidden=hp["hidden"], layers=hp["layers"], heads=hp.get("heads", 2),
                       dropout=hp["dropout"], aggr=hp.get("aggr", "mean"), target=target)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    y_tr = [g[target].y[g[target].target_mask].float() for g in g_tr]
    pw = _pos_weight(torch.cat(y_tr))
    n_tr = sum(len(y) for y in y_tr)
    y_va = np.concatenate([g[target].y[g[target].target_mask].numpy() for g in g_va])
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        for g, y in zip(g_tr, y_tr):
            loss = F.binary_cross_entropy_with_logits(model(g)[g[target].target_mask], y, pos_weight=pw,
                                                      reduction="sum") / n_tr
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            ap = _ap(y_va, _scores(model, g_va, target))
        if ap > best + 1e-6:
            best, best_state, best_epoch, bad = ap, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s_va = _scores(model, g_va, target)
        t1 = time.time()
        s_te = _scores(model, g_te, target)
        infer_seconds = time.time() - t1
    return {"val_scores": s_va, "test_scores": s_te, "best_val_ap": best, "best_epoch": best_epoch,
            "train_seconds": train_seconds, "infer_seconds": infer_seconds, "model": model}


def fit_mlp(X: dict, y: dict, hp: dict, seed: int, max_epochs: int = 300, patience: int = 20) -> dict:
    set_seed(seed)
    scaler = StandardScaler().fit(X["train"])
    T = {k: torch.tensor(scaler.transform(v), dtype=torch.float32) for k, v in X.items()}
    y_tr = torch.tensor(y["train"], dtype=torch.float32)
    model = MLP(T["train"].size(1), hp["hidden"], hp["layers"], hp["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    pw = _pos_weight(y_tr)
    best, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(T["train"]), y_tr, pos_weight=pw).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            ap = _ap(y["val"], model(T["val"]).numpy())
        if ap > best + 1e-6:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s_va = torch.sigmoid(model(T["val"])).numpy()
        t1 = time.time()
        s_te = torch.sigmoid(model(T["test"])).numpy()
        infer_seconds = time.time() - t1
    return {"val_scores": s_va, "test_scores": s_te, "best_val_ap": best,
            "train_seconds": train_seconds, "infer_seconds": infer_seconds}


def make_tabular(family: str, hp: dict, seed: int):
    if family == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000))
    if family == "rf":
        return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                      min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
                                      class_weight="balanced_subsample", n_jobs=4, random_state=seed)
    if family == "gb":
        return HistGradientBoostingClassifier(learning_rate=hp["learning_rate"], max_leaf_nodes=hp["max_leaf_nodes"],
                                              min_samples_leaf=hp["min_samples_leaf"], l2_regularization=hp["l2"],
                                              max_iter=hp["max_iter"], class_weight="balanced", random_state=seed)
    raise ValueError(family)


def fit_tabular(family: str, X: dict, y: dict, hp: dict, seed: int) -> dict:
    model = make_tabular(family, hp, seed)
    t0 = time.time()
    model.fit(X["train"], y["train"])
    train_seconds = time.time() - t0
    s_va = model.predict_proba(X["val"])[:, 1]
    t1 = time.time()
    s_te = model.predict_proba(X["test"])[:, 1]
    return {"val_scores": s_va, "test_scores": s_te, "best_val_ap": _ap(y["val"], s_va),
            "train_seconds": train_seconds, "infer_seconds": time.time() - t1, "model": model}
