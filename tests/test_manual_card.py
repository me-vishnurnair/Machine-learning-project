"""Verify the manual form against genuine saved training features and weights."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from card_core import FEATURES, build_features, load_card_bundle, score_payment
from fraud_core import DataValidationError
from manual_card_core import features_from_details

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_card_bundle(ROOT / "card_artifacts")


def details():
    return dict(currency="USD", timestamp="2020-12-12 14:00:31", amount=100.0,
        category="shopping_net", history_available=True, prior_mean_amount=75.0,
        minutes_since_previous=120.0, transactions_1h=0, transactions_24h=3,
        merchant_seen=True, category_seen=True)


def test_all_saved_examples_have_identical_manual_features_and_predictions(bundle):
    cases = pd.read_csv(bundle.directory / "demo_cases.csv", float_precision="round_trip")
    assert len(cases) == 16
    for _, row in cases.iterrows():
        entered = {**row.to_dict(), "currency": "USD"}
        for field in ("history_available", "merchant_seen", "category_seen"):
            entered[field] = bool(entered[field])
        actual = features_from_details(entered, bundle.metadata["categories"])
        expected = cases.loc[[row.name], list(FEATURES)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[list(FEATURES)], expected, check_dtype=False, atol=1e-12, rtol=1e-12)
        score, flagged = score_payment(actual, bundle)
        assert score == pytest.approx(row.model_score, abs=1e-12)
        assert flagged == bool(row.flagged)


def test_missing_history_matches_original_feature_engineering(bundle):
    entered = {**details(), "history_available": False}
    actual = features_from_details(entered, bundle.metadata["categories"])
    raw = pd.DataFrame([dict(account_id="test", timestamp=entered["timestamp"], amount=entered["amount"],
        merchant="test shop", category=entered["category"], transaction_id="one")])
    expected = build_features(raw)[list(FEATURES)]
    pd.testing.assert_frame_equal(actual[list(FEATURES)], expected, check_dtype=False)
    assert np.isfinite(score_payment(actual, bundle)[0])


@pytest.mark.parametrize("changes,match", [
    ({"currency": "INR"}, "USD"),
    ({"amount": 0}, "positive"),
    ({"amount": np.nan}, "finite"),
    ({"timestamp": "invalid"}, "date/time"),
    ({"category": "unsupported"}, "category"),
    ({"history_available": "false"}, "earlier card activity"),
    ({"prior_mean_amount": 0}, "positive earlier average"),
    ({"minutes_since_previous": 0}, "positive earlier average"),
    ({"transactions_24h": 1.5}, "whole-number"),
    ({"transactions_24h": np.inf}, "whole-number"),
    ({"transactions_1h": 5}, "cannot exceed"),
    ({"transactions_1h": 1}, "previous-hour"),
    ({"minutes_since_previous": 15}, "previous-hour"),
    ({"transactions_24h": 0}, "previous-24-hours"),
    ({"minutes_since_previous": 1500}, "previous-24-hours"),
    ({"merchant_seen": "yes"}, "merchant"),
])
def test_inconsistent_or_unsupported_details_are_rejected(bundle, changes, match):
    with pytest.raises(DataValidationError, match=match):
        features_from_details({**details(), **changes}, bundle.metadata["categories"])


@pytest.mark.parametrize("gap,hour,day", [(60,1,1), (1440,0,1), (1440.1,0,0)])
def test_window_boundaries_match_training_contract(bundle, gap, hour, day):
    features_from_details({**details(), "minutes_since_previous":gap,
        "transactions_1h":hour, "transactions_24h":day}, bundle.metadata["categories"])


def test_no_labels_or_unrelated_balances_enter_the_features(bundle):
    entered = details()
    expected = features_from_details(entered, bundle.metadata["categories"])
    actual = features_from_details({**entered, "label":1, "receiver_balance":99999}, bundle.metadata["categories"])
    pd.testing.assert_frame_equal(actual, expected)
