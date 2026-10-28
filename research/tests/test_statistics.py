"""Statistical units: seed-level pooling over regimes and the corrected test over cross-validation rotations.

Run with:  .venv-research/bin/python -m pytest research/tests -q
"""

import numpy as np
import pytest
from scipy import stats

from research.metrics import clustered_comparison, corrected_resampled_comparison
from research.synth.generator import GeneratorConfig, generate

SMALL = dict(n_patients=1500, n_doctors=120, n_hospitals=15, rings_per_period=2, seed=7)


def test_regimes_of_a_seed_share_legitimate_claims_and_ring_members():
    """Why pooled comparisons treat the seed, not the regime, as the unit of replication."""
    a = generate(GeneratorConfig(ring_shift=0.0, ring_density=0.1, **SMALL))
    b = generate(GeneratorConfig(ring_shift=2.0, ring_density=0.9, **SMALL))

    def background(w):
        keep = (w.truth["fraud_type"] != "ring").to_numpy()
        c = w.claims.loc[keep].drop(columns="claim_id")
        return c.sort_values(list(c.columns)).to_numpy()

    def ring_doctors(w):
        return set(w.claims.loc[(w.truth["fraud_type"] == "ring").to_numpy(), "doctor_id"])

    assert np.array_equal(background(a), background(b))
    assert ring_doctors(a) == ring_doctors(b)


def test_clustered_comparison_tests_the_cluster_means():
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(6), 9)
    a = rng.normal(size=54) + np.repeat(rng.normal(size=6), 9)    # a shared effect within each cluster
    b = rng.normal(size=54)
    res = clustered_comparison(a, b, cluster)
    d = (a - b).reshape(6, 9).mean(1)
    assert res["n"] == 6 and res["n_clusters"] == 6
    assert res["mean_diff"] == pytest.approx(d.mean())
    assert res["t_p"] == pytest.approx(stats.ttest_1samp(d, 0.0).pvalue)


def test_corrected_resampled_comparison_widens_the_naive_interval():
    a = np.array([0.70, 0.66, 0.74, 0.69, 0.72])
    b = np.array([0.67, 0.65, 0.70, 0.66, 0.70])
    res = corrected_resampled_comparison(a, b, test_train_ratio=1 / 3)
    d = a - b
    se = np.sqrt(1 / 5 + 1 / 3) * d.std(ddof=1)
    assert res["mean_diff"] == pytest.approx(d.mean())
    assert res["ci_high"] - res["mean_diff"] == pytest.approx(stats.t.ppf(0.975, 4) * se)
    assert res["t_p"] > res["naive_t_p"]
