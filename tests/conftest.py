import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy():
    """120 'people' x 4 rows each. Their feature rows repeat; the label is noise
    tied to the person, so only memorising the person helps."""
    rng = np.random.default_rng(0)
    n_people = 120
    person_feats = rng.normal(size=(n_people, 3))
    person_label = rng.integers(0, 2, size=n_people)
    X = pd.DataFrame(np.repeat(person_feats, 4, axis=0), columns=["f1", "f2", "f3"])
    y = pd.Series(np.repeat(person_label, 4), name="y")
    groups = pd.Series(np.repeat(np.arange(n_people), 4), name="group")
    return X, y, groups
