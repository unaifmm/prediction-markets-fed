"""From outcome prices to the expected rate change."""
import numpy as np
import pandas as pd

from src.probs import rate_change_bp, summarise


def test_outcome_labels_map_to_rate_changes():
    cases = {"Cut >25bps": -50, "Cut 25bps": -25, "Fed maintains rate": 0, "Hike 0bps": 0,
             "No cut/hike": 0, "Hike 25bps": 25, "Hike >25bps": 50, "Hike >50bps": 75,
             "50+ bps decrease": -50, "25 bps decrease after May meeting?": -25, "No Change": 0,
             "0 bps increase?": 0, "25+ bps increase": 25, "75+ bps decrease": -75}
    for label, bp in cases.items():
        assert rate_change_bp(label) == bp, label
    assert np.isnan(rate_change_bp("Other"))


def test_expected_change_is_probability_weighted_and_normalised():
    # 85% hold, 15% hike 25 -> +3.75 bp; prices adding up to 1.02 are scaled back to 1
    p = pd.DataFrame({0.0: [0.85, 0.867], 25.0: [0.15, 0.153]})
    out = summarise(p)
    assert abs(out["exp_bp"].iloc[0] - 3.75) < 1e-9
    assert abs(out["exp_bp"].iloc[1] - 0.153 * 25 / 1.02) < 1e-9


def test_unquoted_tails_count_as_zero_only_when_the_rest_is_complete():
    p = pd.DataFrame({-25.0: [0.02, 0.02], 0.0: [0.97, 0.50], 25.0: [0.01, np.nan], 50.0: [np.nan, np.nan]})
    out = summarise(p)
    assert np.isfinite(out["exp_bp"].iloc[0])        # 1.00 quoted: the missing 50 bp tail is ~0
    assert np.isnan(out["exp_bp"].iloc[1])           # only 0.52 quoted: not usable
