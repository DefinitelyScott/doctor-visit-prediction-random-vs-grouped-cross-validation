"""End-to-end on a small subsample so the suite stays fast."""

from visit_leakage.data import load_visits, make_groups
from visit_leakage.evaluate import leakage_gap, run_cv, summarise
from visit_leakage.models import build_models
from visit_leakage.report import format_summary, plot_auc


def test_pipeline_runs_and_writes_figure(tmp_path):
    X, y = load_visits()
    X, y = X.iloc[:2000], y.iloc[:2000]
    groups = make_groups(X)
    res = run_cv(build_models(1), X, y, groups, n_splits=3, n_repeats=1, seed=1)
    summary = summarise(res)
    gap = leakage_gap(summary)
    leak = res.groupby("scheme")["leaked_share"].mean()
    text = format_summary(summary, gap, leak)
    assert "random_forest" in text and "grouped" in text
    out = tmp_path / "fig.png"
    plot_auc(res, out)
    assert out.stat().st_size > 1000
