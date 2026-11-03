"""Sixth round: the provider mapping, the two AP definitions and the checks that stop a bad linkage."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from research.experiments.group_labels import FIRST_ID, fingerprint, group_ap, group_flags, rank_ids


def test_rank_ids_follow_the_sorted_order():
    assert rank_ids(["PRV3", "PRV1", "PRV2", "PRV1"]) == {"PRV1": FIRST_ID, "PRV2": FIRST_ID + 1, "PRV3": FIRST_ID + 2}


def test_fingerprint_treats_their_minus_one_like_our_missing_value():
    base = {"Provider": ["a", "a"], "flag": [0, 1], "InscClaimAmtReimbursed": [10.0, 20.0]}
    theirs = pd.DataFrame({**base, "DeductibleAmtPaid": [-1.0, 5.0]})
    ours = pd.DataFrame({**base, "DeductibleAmtPaid": [np.nan, 5.0]})
    assert fingerprint(theirs, "Provider", "flag").equals(fingerprint(ours, "Provider", "flag"))


def test_group_ap_excludes_the_other_class():
    y_fraud = np.array([1, 1, 1, 0, 0, 0])
    y_group = np.array([1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.95, 0.5, 0.1, 0.05])
    group, individual = group_ap(y_fraud, y_group, scores)
    keep_group = np.array([True, True, False, True, True, True])
    assert group == average_precision_score(y_group[keep_group], scores[keep_group])
    keep_other = np.array([False, False, True, True, True, True])
    assert individual == average_precision_score([1, 0, 0, 0], scores[keep_other])


def test_a_broken_mapping_stops_the_round(tmp_path, monkeypatch):
    # One claim's amount differs between the two sources: the linkage must fail rather than pass silently.
    lab = pd.DataFrame({"Provider": ["PRV1", "PRV2"], "PotentialFraud": ["Yes", "No"]})
    claims = pd.DataFrame({"Provider": ["PRV1", "PRV2"], "is_inpatient": [0, 1], "InscClaimAmtReimbursed": [10.0, 20.0],
                           "DeductibleAmtPaid": [0.0, 0.0]})
    monkeypatch.setattr("research.experiments.group_labels.load_raw", lambda d: (lab, None, claims))
    rd = pd.DataFrame({"Provider": [FIRST_ID, FIRST_ID + 1], "inpatent": [0, 1], "InscClaimAmtReimbursed": [10.0, 99.0],
                       "DeductibleAmtPaid": [0.0, 0.0], "PotentialFraud": [1, 0], "PotentialGroupFraud": [1, 0]})
    rd.to_csv(tmp_path / "claims.csv", index=False)
    with pytest.raises(ValueError, match="do not map"):
        group_flags(tmp_path, tmp_path)
