"""Fifth round: wide search candidates, early stopping on ring AP, closure columns, unchanged defaults."""

import json

import numpy as np
from sklearn.metrics import average_precision_score

from research.experiments.expanded import (BUDGET, CLOSURE, GNN_WIDE, WIDE_SPACES, expanded_candidates, fit_method,
                                           ring_val_keep)
from research.experiments.synthetic import augment_claims, prepare
from research.experiments.tune import BUDGET as MAIN_BUDGET, candidate_configs
from research.models.train import fit_graph
from research.synth.generator import GeneratorConfig, generate

MAIN = {"hidden": 32, "layers": 3, "dropout": 0.3, "lr": 0.01, "weight_decay": 1e-5, "heads": 4}
HP = {"hidden": 16, "layers": 2, "dropout": 0.1, "lr": 0.01, "weight_decay": 1e-4, "aggr": "sum"}


def small_world():
    world = generate(GeneratorConfig(n_patients=600, n_doctors=80, n_hospitals=10, rings_per_period=2, ring_shift=0.0,
                                     ring_density=0.9, seed=3))
    return (world, *prepare(world))


def test_main_tuning_candidates_are_unchanged():
    # The earlier rounds' candidate lists must stay reproducible: same length, same default first, same draws.
    a, b = candidate_configs("HGT"), candidate_configs("HGT")
    assert a == b and len(a) == MAIN_BUDGET
    assert a[0]["hidden"] == 64 and a[0]["layers"] == 2 and a[0]["lr"] == 3e-3


def test_wide_candidates_start_with_main_config_and_are_otherwise_new():
    cands = expanded_candidates("HGT", MAIN)
    assert len(cands) == BUDGET and cands[0] == MAIN
    old = [json.dumps(c, sort_keys=True) for c in candidate_configs("HGT")]
    new = [json.dumps(c, sort_keys=True) for c in cands[1:]]
    assert len(set(new)) == BUDGET - 1 and not set(new) & set(old)
    for k, values in WIDE_SPACES["HGT"].items():
        assert all(c[k] in values for c in cands[1:])


def test_wide_grid_is_compatible_with_attention_heads():
    assert all(h % k == 0 for h in GNN_WIDE["hidden"] for k in WIDE_SPACES["HGT"]["heads"])


def test_early_stopping_targets_ring_ap_and_default_is_unaffected():
    world, tab, cols, graphs = small_world()
    plain = fit_graph("sage", graphs, HP, 0, max_epochs=5, patience=5)
    all_true = fit_graph("sage", graphs, HP, 0, max_epochs=5, patience=5, val_keep=np.ones(len(plain["val_scores"]), bool))
    assert plain["best_val_ap"] == all_true["best_val_ap"]
    assert np.array_equal(plain["test_scores"], all_true["test_scores"])
    keep = ring_val_keep(world)
    assert (~keep).sum() > 0
    res = fit_method("SAGE", world, tab, cols, graphs, HP)
    y = world.y[world.split == "val"]
    assert abs(res["best_val_ap"] - average_precision_score(y[keep], res["val_scores"][keep])) < 1e-9


def test_closure_columns_extend_claim_nodes_only():
    world, tab, cols, graphs = small_world()
    extra = augment_claims(graphs, tab, {"A": CLOSURE, "G": []})
    for split in graphs:
        for g, g2 in zip(graphs[split], extra[split]):
            assert g2["claim"].x.size(1) == g["claim"].x.size(1) + len(CLOSURE)
            assert g2["doctor"].x.size(1) == g["doctor"].x.size(1)


def test_closure_is_a_degree_term_plus_four_cycles():
    # Online Resource: closure = deg(p) - 1 + the four-cycles through the claim's patient-doctor edge.
    world, tab, _, _ = small_world()
    rows = np.flatnonzero(world.split == "test")
    claims = world.claims.iloc[rows]
    patients = claims.groupby("patient_id")["doctor_id"].agg(set).to_dict()
    doctors = claims.groupby("doctor_id")["patient_id"].agg(set).to_dict()
    closure = tab["test"]["claim_closure"].to_numpy()
    for i, (p, d) in enumerate(zip(claims["patient_id"], claims["doctor_id"])):
        cycles = sum(len((doctors[other] & doctors[d]) - {p}) for other in patients[p] - {d})
        assert closure[i] == len(patients[p]) - 1 + cycles

