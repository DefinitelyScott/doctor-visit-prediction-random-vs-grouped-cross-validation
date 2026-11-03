"""The model ladder: a no-skill baseline, a linear model, and two tree ensembles."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINE = "baseline_prior"


def build_models(seed: int) -> dict[str, ClassifierMixin]:
    """Return fresh, seeded estimators keyed by name.

    The random forest is deliberately left deep (``min_samples_leaf=1``): that is
    the setting that can memorise repeated rows, which is the effect under test.
    """
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, random_state=seed, n_jobs=1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
