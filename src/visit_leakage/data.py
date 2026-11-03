"""Load the RAND Health Insurance Experiment extract and build person-proxy groups."""

from __future__ import annotations

import pandas as pd
from statsmodels.datasets import randhie

TARGET_SOURCE = "mdvis"
TARGET = "any_visit"
FEATURES = [
    "lncoins", "idp", "lpi", "fmde", "physlm", "disea", "hlthg", "hlthf", "hlthp",
]


def load_visits() -> tuple[pd.DataFrame, pd.Series]:
    """Return features and the binary target ``any_visit`` (1 if >= 1 MD visit).

    Raises ``ValueError`` if the bundled data has an unexpected shape, so a
    silently changed dataset cannot produce numbers that look plausible.
    """
    raw = randhie.load_pandas().data
    missing = [c for c in [TARGET_SOURCE, *FEATURES] if c not in raw.columns]
    if missing:
        raise ValueError(f"expected columns missing from dataset: {missing}")
    if raw[[TARGET_SOURCE, *FEATURES]].isna().any().any():
        raise ValueError("dataset contains missing values; cleaning step needed")
    X = raw[FEATURES].copy()
    y = (raw[TARGET_SOURCE] > 0).astype(int).rename(TARGET)
    return X, y


def make_groups(X: pd.DataFrame, decimals: int = 6) -> pd.Series:
    """Give rows with identical feature values the same integer group id.

    The extract has no person identifier, but one person observed over several
    years repeats the same insurance plan and (mostly) the same health profile.
    Identical feature rows are therefore a proxy for "same person". The proxy is
    imperfect in both directions; see DESIGN_NOTES.md.
    """
    if X.empty:
        raise ValueError("cannot build groups from an empty frame")
    rounded = X.round(decimals)
    ids = rounded.groupby(list(rounded.columns), sort=False).ngroup()
    return ids.rename("group")
