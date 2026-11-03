import pandas as pd
import pytest

from visit_leakage import data
from visit_leakage.data import FEATURES, load_visits, make_groups


def test_load_visits_shape_and_binary_target():
    X, y = load_visits()
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 20190
    assert set(y.unique()) == {0, 1}
    assert 0.68 < y.mean() < 0.70


def test_load_visits_rejects_missing_values(monkeypatch):
    class Fake:
        data = pd.DataFrame({c: [1.0, None] for c in ["mdvis", *FEATURES]})

    monkeypatch.setattr(data.randhie, "load_pandas", lambda: Fake)
    with pytest.raises(ValueError, match="missing values"):
        load_visits()


def test_load_visits_rejects_missing_columns(monkeypatch):
    class Fake:
        data = pd.DataFrame({"mdvis": [1, 2]})

    monkeypatch.setattr(data.randhie, "load_pandas", lambda: Fake)
    with pytest.raises(ValueError, match="columns missing"):
        load_visits()


def test_make_groups_identical_rows_share_id():
    X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 1.0], "b": [0, 0, 0, 1]})
    g = make_groups(X)
    assert g.iloc[0] == g.iloc[1]
    assert len({g.iloc[0], g.iloc[2], g.iloc[3]}) == 3


def test_make_groups_tolerates_float_noise():
    X = pd.DataFrame({"a": [1.0, 1.0 + 1e-9]})
    assert make_groups(X).nunique() == 1


def test_make_groups_empty_raises():
    with pytest.raises(ValueError):
        make_groups(pd.DataFrame({"a": []}))


def test_real_data_has_far_fewer_groups_than_rows():
    X, _ = load_visits()
    assert make_groups(X).nunique() == 2760
