"""Tests for the routing experiment, the HGT single-factor ablations and the cost benchmark.

Run with:  .venv-research/bin/python -m pytest research/tests -q
"""

import numpy as np
import pytest
import torch

from research.experiments.ablation import drop_node_type, n_static_columns, rewire, zero_inputs
from research.experiments.cascade import ESCALATION, route, score_metrics, select_escalation
from research.experiments.latency import computation_subgraph
from research.experiments.synthetic import prepare
from research.metrics import _ap
from research.models.gnn import GraphModel
from research.synth.features import build_tabular
from research.synth.generator import GeneratorConfig, generate

SMALL = GeneratorConfig(n_patients=1500, n_doctors=120, n_hospitals=15, rings_per_period=2, seed=7)


@pytest.fixture(scope="module")
def world():
    return generate(SMALL)


@pytest.fixture(scope="module")
def prepared(world):
    return prepare(world)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y_val, y_test = rng.random(400) < 0.1, rng.random(300) < 0.1
    g_val, h_val = rng.random(400) + 0.3 * y_val, rng.random(400) + 0.5 * y_val
    g_test, h_test = rng.random(300) + 0.3 * y_test, rng.random(300) + 0.5 * y_test
    return y_val.astype(int), y_test.astype(int), g_val / 1.5, h_val / 1.5, g_test / 1.5, h_test / 1.5


@pytest.mark.parametrize("mode", ["rerank", "fuse"])
def test_routing_is_per_claim_and_uses_validation_only(scores, mode):
    y_val, y_test, g_val, h_val, g_test, h_test = scores
    e, _ = select_escalation(y_val, g_val, h_val, mode)
    assert e in ESCALATION
    s, esc = route(g_test, h_test, g_val, h_val, e, mode)
    perm = np.random.default_rng(1).permutation(len(g_test))
    s_perm, _ = route(g_test[perm], h_test[perm], g_val, h_val, e, mode)
    assert np.allclose(s_perm, s[perm]), "a claim's score must not depend on other test claims"
    if esc.any() and (~esc).any():
        assert s[esc].min() > s[~esc].max(), "escalated claims must rank above non-escalated ones"


def test_full_escalation_nests_the_single_models(scores):
    y_val, y_test, g_val, h_val, g_test, h_test = scores
    rerank, esc = route(g_test, h_test, g_val, h_val, 1.0, "rerank")
    assert esc.all() and _ap(y_test, rerank) == pytest.approx(_ap(y_test, h_test))
    fused, _ = route(g_test, h_test, g_val, h_val, 1.0, "fuse")
    gp = np.searchsorted(np.sort(g_val), g_test, side="right")
    hp = np.searchsorted(np.sort(h_val), h_test, side="right")
    assert np.allclose(fused, 1 + (gp + hp) / (2 * len(g_val)))


def test_score_metrics_budget_and_ring_level():
    y = np.zeros(100, int)
    y[:10] = 1
    ft = np.where(y == 1, "ring", "legit")
    ring_id = np.zeros(100, int)
    ring_id[:10] = [1, 2] * 5
    ring_id[8:10] = 3                    # ring 3 has two claims, both ranked last among positives
    s = y + np.linspace(0.01, 0, 100)
    m = score_metrics(y, s, ft, ring_id, budget=0.08)
    assert m["precision_at_8pct"] == 1.0 and m["ring_recall_at_8pct"] == pytest.approx(0.8)
    assert m["rings_with_any_claim_in_budget"] == pytest.approx(2 / 3)


def test_test_period_rewiring_leaves_other_periods_unchanged(world):
    test_period = int(world.period[world.split == "test"][0])
    w = rewire(world, seed=0, periods=[test_period])
    a, b = build_tabular(world), build_tabular(w)
    for split in ["train", "val"]:
        assert a[split].equals(b[split])
    assert not a["test"].equals(b["test"])
    rows = world.period == test_period
    assert (world.claims.loc[rows, "doctor_id"].value_counts().sort_index()
            .equals(w.claims.loc[rows, "doctor_id"].value_counts().sort_index())), "doctor claim volumes must be kept"


def test_arch_transforms_keep_originals_and_message_flow(world, prepared):
    _, _, graphs = prepared
    g = graphs["train"][0]
    x_claim, x_doc = g["claim"].x.clone(), g["doctor"].x.clone()
    n_static = n_static_columns(world)
    no_h = drop_node_type(graphs, "hospital")["train"][0]
    assert "hospital" not in no_h.node_types and "hospital" in g.node_types
    assert {et[2] for et in no_h.edge_types} == set(no_h.node_types)
    torch.manual_seed(0)
    GraphModel("hgt", no_h, hidden=16, layers=2, heads=2, dropout=0.0).eval()(no_h)
    deg_only = zero_inputs(graphs, n_static, claim=True, static=True)["train"][0]
    assert torch.count_nonzero(deg_only["claim"].x) == 0
    k = n_static["doctor"]
    assert torch.count_nonzero(deg_only["doctor"].x[:, :k]) == 0
    assert torch.equal(deg_only["doctor"].x[:, k:], x_doc[:, k:])
    assert torch.equal(g["claim"].x, x_claim) and torch.equal(g["doctor"].x, x_doc), "originals modified"


def test_computation_subgraph_reproduces_full_graph_scores(prepared):
    _, _, graphs = prepared
    g = graphs["test"][0]
    torch.manual_seed(0)
    model = GraphModel("hgt", g, hidden=16, layers=3, heads=2, dropout=0.0).eval()
    with torch.no_grad():
        full = model(g)
        for c in [0, 17, g["claim"].num_nodes - 1]:
            sub, pos = computation_subgraph(g, c, hops=3)
            assert torch.allclose(model(sub)[pos], full[c], atol=1e-5)
        claims = np.array([3, 40, 41])
        sub, pos = computation_subgraph(g, claims, hops=3)
        assert torch.allclose(model(sub)[pos], full[claims], atol=1e-5)
