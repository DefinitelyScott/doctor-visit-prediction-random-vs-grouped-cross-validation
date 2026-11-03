"""Cross-validation under two schemes, plus a direct measure of group leakage."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

SCHEMES = ("random", "grouped")


def iter_splits(
    scheme: str,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, fold, train_idx, test_idx)``.

    ``random`` ignores groups, so a person's rows can land on both sides.
    ``grouped`` keeps every row of a group on one side.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    dummy_X = np.zeros((len(y), 1))
    for repeat in range(n_repeats):
        if scheme == "random":
            splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed + repeat)
            parts = splitter.split(dummy_X, y)
        else:
            splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + repeat)
            parts = splitter.split(dummy_X, y, groups)
        for fold, (train, test) in enumerate(parts):
            yield repeat, fold, train, test


def leaked_share(groups: pd.Series, train: np.ndarray, test: np.ndarray) -> float:
    """Fraction of test rows whose group also appears in the training rows."""
    if len(test) == 0:
        raise ValueError("test set is empty")
    train_groups = set(groups.iloc[train])
    return float(np.mean([g in train_groups for g in groups.iloc[test]]))


def score_fold(
    model: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Fit a fresh clone and score predicted probabilities on the test fold."""
    fitted = clone(model).fit(X_train, y_train)
    p = fitted.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "log_loss": float(log_loss(y_test, p, labels=[0, 1])),
    }


def run_cv(
    models: dict[str, ClassifierMixin],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 20261101,
) -> pd.DataFrame:
    """Score every model on every fold of both schemes; one row per model-fold."""
    rows = []
    for scheme in SCHEMES:
        for repeat, fold, train, test in iter_splits(scheme, y, groups, n_splits, n_repeats, seed):
            leak = leaked_share(groups, train, test)
            for name, model in models.items():
                metrics = score_fold(
                    model, X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]
                )
                rows.append(
                    {"scheme": scheme, "model": name, "repeat": repeat, "fold": fold,
                     "leaked_share": leak, **metrics}
                )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of each metric per scheme and model."""
    grouped = results.groupby(["model", "scheme"])[["auc", "brier", "log_loss"]]
    out = grouped.agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def leakage_gap(summary: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    """Random-minus-grouped mean of ``metric`` for each model."""
    wide = summary.pivot(index="model", columns="scheme", values=f"{metric}_mean")
    wide["gap"] = wide["random"] - wide["grouped"]
    return wide[["random", "grouped", "gap"]]


def paired_gain(
    results: pd.DataFrame, scheme: str, model_a: str, model_b: str, metric: str = "auc"
) -> dict[str, float]:
    """Fold-by-fold ``model_a - model_b`` on identical folds within one scheme.

    Both models are scored on the same splits, so the difference removes the
    fold-to-fold difficulty that both share. Returns the mean, the SD, and the
    number of folds where ``model_a`` was higher.
    """
    sub = results[results["scheme"] == scheme]
    a = sub[sub["model"] == model_a].set_index(["repeat", "fold"])[metric]
    b = sub[sub["model"] == model_b].set_index(["repeat", "fold"])[metric]
    if a.empty or b.empty:
        raise ValueError(f"no rows for {model_a!r} or {model_b!r} under scheme {scheme!r}")
    diff = a - b
    return {"mean": float(diff.mean()), "sd": float(diff.std()),
            "wins": int((diff > 0).sum()), "folds": int(len(diff))}
