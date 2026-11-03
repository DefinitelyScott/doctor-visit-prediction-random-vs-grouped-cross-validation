import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from visit_leakage.evaluate import (
    iter_splits, leakage_gap, leaked_share, paired_gain, run_cv, score_fold, summarise,
)
from visit_leakage.models import BASELINE, build_models


def test_random_splits_partition_rows_each_repeat(toy):
    _, y, g = toy
    seen = {}
    for repeat, _, train, test in iter_splits("random", y, g, 5, 2, seed=1):
        assert not set(train) & set(test)
        seen.setdefault(repeat, []).extend(test)
    for rows in seen.values():
        assert sorted(rows) == list(range(len(y)))


def test_grouped_splits_never_share_a_group(toy):
    _, y, g = toy
    for _, _, train, test in iter_splits("grouped", y, g, 5, 2, seed=1):
        assert not set(g.iloc[train]) & set(g.iloc[test])


def test_random_splits_do_leak_groups(toy):
    _, y, g = toy
    shares = [leaked_share(g, tr, te) for _, _, tr, te in iter_splits("random", y, g, 5, 1, 1)]
    assert min(shares) > 0.5


def test_leaked_share_exact():
    g = pd.Series([0, 0, 1, 2])
    assert leaked_share(g, np.array([0, 2]), np.array([1, 3])) == 0.5
    assert leaked_share(g, np.array([0]), np.array([1])) == 1.0


def test_leaked_share_empty_test_raises():
    with pytest.raises(ValueError):
        leaked_share(pd.Series([0]), np.array([0]), np.array([], dtype=int))


def test_unknown_scheme_raises(toy):
    _, y, g = toy
    with pytest.raises(ValueError, match="scheme"):
        list(iter_splits("loo", y, g, 5, 1, 1))


def test_splits_are_deterministic(toy):
    _, y, g = toy
    a = [te.tolist() for *_, te in iter_splits("grouped", y, g, 5, 1, seed=7)]
    b = [te.tolist() for *_, te in iter_splits("grouped", y, g, 5, 1, seed=7)]
    assert a == b


def test_score_fold_baseline_has_chance_auc(toy):
    X, y, _ = toy
    m = build_models(0)[BASELINE]
    s = score_fold(m, X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:])
    assert s["auc"] == pytest.approx(0.5)


def test_memorising_model_wins_only_under_random_cv(toy):
    X, y, g = toy
    models = {"rf": RandomForestClassifier(n_estimators=50, random_state=0)}
    res = run_cv(models, X, y, g, n_splits=5, n_repeats=1, seed=3)
    auc = res.groupby("scheme")["auc"].mean()
    assert auc["random"] > 0.9
    assert auc["grouped"] < 0.65


def test_run_cv_shape_and_columns(toy):
    X, y, g = toy
    models = {k: v for k, v in build_models(0).items() if k in (BASELINE, "logistic_regression")}
    res = run_cv(models, X, y, g, n_splits=4, n_repeats=2, seed=0)
    assert len(res) == 2 * 2 * 4 * 2  # schemes x models x repeats x folds
    assert {"scheme", "model", "repeat", "fold", "auc", "brier", "log_loss", "leaked_share"} <= set(res.columns)
    assert res.loc[res.scheme == "grouped", "leaked_share"].eq(0).all()


def test_summarise_and_gap_arithmetic():
    res = pd.DataFrame({
        "model": ["m"] * 4,
        "scheme": ["random", "random", "grouped", "grouped"],
        "auc": [0.9, 0.8, 0.6, 0.5],
        "brier": [0.1] * 4,
        "log_loss": [0.3] * 4,
    })
    summary = summarise(res)
    gap = leakage_gap(summary, "auc")
    assert gap.loc["m", "random"] == pytest.approx(0.85)
    assert gap.loc["m", "grouped"] == pytest.approx(0.55)
    assert gap.loc["m", "gap"] == pytest.approx(0.30)


def test_paired_gain_uses_matching_folds():
    res = pd.DataFrame({
        "scheme": ["grouped"] * 4,
        "model": ["a", "a", "b", "b"],
        "repeat": [0, 0, 0, 0],
        "fold": [0, 1, 0, 1],
        "auc": [0.70, 0.60, 0.65, 0.65],
    })
    out = paired_gain(res, "grouped", "a", "b")
    assert out["mean"] == pytest.approx(0.0)
    assert out["wins"] == 1 and out["folds"] == 2


def test_paired_gain_unknown_model_raises():
    res = pd.DataFrame({"scheme": ["grouped"], "model": ["a"], "repeat": [0], "fold": [0], "auc": [0.5]})
    with pytest.raises(ValueError):
        paired_gain(res, "grouped", "a", "zzz")
