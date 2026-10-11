"""Adapt human-entered card-payment summaries to the saved feature contract.

No PCA values, invented transaction history, labels, or model retraining are
involved. Summaries have exactly the same meanings as card_core.build_features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from card_core import FEATURES
from fraud_core import DataValidationError


def features_from_details(details: dict, categories: list[str]) -> pd.DataFrame:
    """Validate supplied facts and derive the original model's 13 inputs."""
    if details.get("currency") != "USD":
        raise DataValidationError("This model uses USD. Enter payment and average amounts in US dollars.")
    if details.get("category") not in categories:
        raise DataValidationError("Choose a supported purchase category.")
    try:
        at = pd.Timestamp(details["timestamp"])
        if pd.isna(at):
            raise ValueError("Missing timestamp")
        amount = float(details["amount"])
        if not np.isfinite(amount) or amount <= 0:
            raise ValueError("Invalid amount")
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError("Provide a valid payment date/time and a positive, finite amount.") from exc
    available = details.get("history_available")
    if not isinstance(available, (bool, np.bool_)):
        raise DataValidationError("Specify whether earlier card activity is available.")
    average = gap = np.nan
    count_hour = count_day = merchant_seen = category_seen = 0
    if available:
        try:
            average = float(details["prior_mean_amount"])
            gap = float(details["minutes_since_previous"])
            count_hour = float(details["transactions_1h"])
            count_day = float(details["transactions_24h"])
            if not np.isfinite([average, gap, count_hour, count_day]).all() or average <= 0 or gap <= 0:
                raise ValueError("Invalid history")
            if any(value < 0 or value != int(value) for value in (count_hour, count_day)):
                raise ValueError("Invalid count")
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise DataValidationError("Use a positive earlier average and time gap, with nonnegative whole-number payment counts.") from exc
        if count_hour > count_day:
            raise DataValidationError("Payments in the previous hour cannot exceed payments in the previous 24 hours.")
        # Training uses [time - window, time): include a payment exactly at the
        # lower boundary and exclude payments simultaneous with the new one.
        if (gap <= 60) != (count_hour >= 1):
            raise DataValidationError("The previous-hour count must agree with the time since the last payment.")
        if (gap <= 1440) != (count_day >= 1):
            raise DataValidationError("The previous-24-hours count must agree with the time since the last payment.")
        for name in ("merchant_seen", "category_seen"):
            if not isinstance(details.get(name), (bool, np.bool_)):
                raise DataValidationError("Specify whether this merchant and purchase category were used before.")
        merchant_seen = int(details["merchant_seen"])
        category_seen = int(details["category_seen"])
    hour = at.hour + at.minute / 60  # Same minute resolution as training.
    row = {
        "amount": amount, "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24), "weekday": float(at.dayofweek),
        "category": details["category"], "prior_mean_amount": average,
        "amount_vs_mean": amount / max(average, .01) if available else np.nan,
        "transactions_1h": count_hour, "transactions_24h": count_day,
        "minutes_since_previous": gap, "merchant_seen": float(merchant_seen),
        "category_seen": float(category_seen), "history_available": float(available),
    }
    if available and not np.isfinite(row["amount_vs_mean"]):
        raise DataValidationError("The payment-to-average ratio is too large. Check both amounts.")
    result = pd.DataFrame([row], columns=list(FEATURES))
    result["timestamp"] = at  # Display context; excluded from estimator inputs.
    return result
