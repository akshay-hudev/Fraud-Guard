"""Validity tests for the research pipeline (leakage, alignment, information flow).

Run with:  .venv-research/bin/python -m pytest research/tests -q
"""

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from research.experiments.synthetic import augment_claims, feature_matrix, prepare
from research.metrics import holm, recall_at_budget, val_threshold
from research.models.gnn import GraphModel
from research.synth.features import build_tabular
from research.synth.generator import GeneratorConfig, generate
from research.synth.graph import build_graphs, build_metapath_graphs
from research.synth.features import claim_level

SMALL = GeneratorConfig(n_patients=1500, n_doctors=120, n_hospitals=15, rings_per_period=2, seed=7)
LABEL_COLUMNS = {"fraud_label", "fraud_type", "ring_id", "fraud_prone", "approved"}


@pytest.fixture(scope="module")
def world():
    return generate(SMALL)


@pytest.fixture(scope="module")
def prepared(world):
    return prepare(world)


def test_ground_truth_not_in_claims(world):
    assert not LABEL_COLUMNS & set(world.claims.columns)


def test_features_contain_no_label_columns(prepared):
    tab, cols, _ = prepared
    used = {c for g in cols.values() for c in g}
    assert not LABEL_COLUMNS & used
    for frame in tab.values():
        assert not LABEL_COLUMNS & set(frame.columns)


def test_features_invariant_to_labels(world):
    """Permuting ground truth must not change a single feature value."""
    shuffled = copy.copy(world)
    shuffled.truth = world.truth.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a, b = build_tabular(world), build_tabular(shuffled)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_period_features_ignore_other_periods(world):
    """A test-period claim's aggregate/graph features depend only on test-period claims."""
    base = build_tabular(world)["test"]
    perturbed = copy.copy(world)
    claims = world.claims.copy()
    other = world.split != "test"
    claims.loc[other, "claim_amount"] *= 3.0
    claims.loc[other, "doctor_id"] = np.random.default_rng(0).integers(0, SMALL.n_doctors, other.sum())
    perturbed.claims = claims
    after = build_tabular(perturbed)["test"]
    keep = [c for c in base.columns if c != "amt_vs_specialty"]   # specialty means come from training claims by design
    pd.testing.assert_frame_equal(base[keep], after[keep])


def test_emergent_rings_are_entity_disjoint(world):
    c, t = world.claims, world.truth
    ring = t["ring_id"] > 0
    members = {}
    for split in ["train", "val", "test"]:
        m = ring.to_numpy() & (world.split == split)
        members[split] = (set(c.loc[m, "doctor_id"]), set(c.loc[m, "patient_id"]))
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not members[a][0] & members[b][0], "ring doctors reused across splits"
        assert not members[a][1] & members[b][1], "ring patients reused across splits"


def test_zero_shift_ring_claims_match_legit_distribution():
    """At delta = 0 ring claims are drawn like legitimate claims of the same specialty."""
    w = generate(GeneratorConfig(ring_shift=0.0, seed=3))
    cl = claim_level(w)
    spec = w.doctors.set_index("doctor_id").loc[w.claims["doctor_id"], "specialty"].to_numpy()
    resid = cl["log_amount"].to_numpy() - pd.Series(cl["log_amount"].to_numpy()).groupby(spec).transform("median").to_numpy()
    ft = w.truth["fraud_type"].to_numpy()
    assert stats.ks_2samp(resid[ft == "ring"], resid[ft == "legit"]).pvalue > 0.01
    shifted = generate(GeneratorConfig(ring_shift=2.0, seed=3))
    assert claim_level(shifted)["log_amount"][shifted.truth["fraud_type"] == "ring"].mean() > \
        cl["log_amount"][ft == "ring"].mean() + 0.5


def test_split_alignment(world, prepared):
    tab, _, graphs = prepared
    for split in ["train", "val", "test"]:
        ids = world.claims["claim_id"].to_numpy()[world.split == split]
        assert (tab[split]["claim_id"].to_numpy() == ids).all()
        y = torch.cat([g["claim"].y for g in graphs[split]]).numpy()
        assert (y == world.y[world.split == split]).all()
        rows = torch.cat([g["claim"].rows for g in graphs[split]]).numpy()
        assert (world.claims["claim_id"].to_numpy()[rows] == ids).all()


def test_graphs_contain_only_their_period(world, prepared):
    _, _, graphs = prepared
    for split, glist in graphs.items():
        for g in glist:
            periods = np.unique(world.period[g["claim"].rows.numpy()])
            assert len(periods) == 1 and world.split[g["claim"].rows.numpy()[0]] == split


def test_every_node_type_receives_messages(prepared):
    _, _, graphs = prepared
    g = graphs["train"][0]
    assert {et[2] for et in g.edge_types} == set(g.node_types)


@pytest.mark.parametrize("kind", ["sage", "rgcn", "simplehgn", "hgt"])
def test_doctor_information_reaches_claims(prepared, kind):
    _, _, graphs = prepared
    g = graphs["train"][0]
    torch.manual_seed(0)
    model = GraphModel(kind, g, hidden=16, layers=2, heads=2, dropout=0.0).eval()
    with torch.no_grad():
        base = model(g)
        g2 = copy.copy(g)
        g2["doctor"].x = g["doctor"].x + 5 * torch.randn_like(g["doctor"].x)
        moved = (model(g2) - base).abs().max().item()
    assert moved > 1e-4, f"{kind}: claim logits ignore doctor features"


def test_han_uses_metapath_neighbours(world, prepared):
    _, _, graphs = prepared
    meta = build_metapath_graphs(world, {"train": graphs["train"][:1]})["train"][0]
    torch.manual_seed(0)
    model = GraphModel("han", meta, hidden=16, layers=1, heads=2, dropout=0.0).eval()
    src, dst = meta["claim", "cdc", "claim"].edge_index
    target, neighbour = int(dst[0]), int(src[0])
    with torch.no_grad():
        base = model(meta)[target].item()
        m2 = copy.copy(meta)
        x = meta["claim"].x.clone()
        x[neighbour] += 5
        m2["claim"].x = x
        assert abs(model(m2)[target].item() - base) > 1e-5


def test_hybrid_graphs_keep_alignment(prepared):
    tab, cols, graphs = prepared
    hyb = augment_claims(graphs, tab, cols)
    n_extra = feature_matrix(tab, cols, "AG")["train"].shape[1]
    for split in graphs:
        for g, h in zip(graphs[split], hyb[split]):
            assert h["claim"].x.shape[1] == g["claim"].x.shape[1] + n_extra
            assert torch.equal(h["claim"].x[:, :g["claim"].x.shape[1]], g["claim"].x)


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, float("nan")])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_recall_at_budget_and_threshold():
    y = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    s = np.array([.1, .2, .9, .8, .3, .1, .1, .2, .1, .05])
    assert recall_at_budget(y, s, 0.2) == pytest.approx(2 / 3)
    thr = val_threshold(y, s)
    assert 0.05 < thr <= 0.8
