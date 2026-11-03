from visit_leakage.models import BASELINE, build_models


def test_expected_model_ladder():
    assert list(build_models(0)) == [
        BASELINE, "logistic_regression", "random_forest", "hist_gradient_boosting",
    ]


def test_models_are_fresh_objects_each_call():
    a, b = build_models(0), build_models(0)
    assert a["random_forest"] is not b["random_forest"]


def test_seed_reaches_the_stochastic_models():
    assert build_models(5)["random_forest"].random_state == 5
    assert build_models(5)["hist_gradient_boosting"].random_state == 5
