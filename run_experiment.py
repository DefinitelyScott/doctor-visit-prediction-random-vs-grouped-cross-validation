"""Run the full experiment: prints the tables, writes outputs/."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from visit_leakage.data import load_visits, make_groups
from visit_leakage.evaluate import leakage_gap, paired_gain, run_cv, summarise
from visit_leakage.models import build_models
from visit_leakage.report import format_summary, plot_auc

# scipy emits a harmless option warning from inside scikit-learn's logistic solver.
warnings.filterwarnings("ignore", message="Unknown solver options")

SEED = 20261101
OUT = Path(__file__).parent / "outputs"


def main() -> None:
    X, y = load_visits()
    groups = make_groups(X)
    print(f"rows={len(X)}  distinct person-proxy groups={groups.nunique()}  "
          f"P(any visit)={y.mean():.3f}\n")
    results = run_cv(build_models(SEED), X, y, groups, n_splits=5, n_repeats=3, seed=SEED)
    summary = summarise(results)
    gap = leakage_gap(summary, "auc")
    leak = results.groupby("scheme")["leaked_share"].mean()
    print(format_summary(summary, gap, leak))
    print("\nPaired fold-by-fold AUC difference (identical folds within each scheme):")
    for scheme in ("random", "grouped"):
        for a, b in [("hist_gradient_boosting", "logistic_regression"),
                     ("random_forest", "logistic_regression"),
                     ("random_forest", "hist_gradient_boosting")]:
            g = paired_gain(results, scheme, a, b)
            print(f"  {scheme:<8} {a} - {b}: {g['mean']:+.3f} ± {g['sd']:.3f}  "
                  f"(first model higher in {g['wins']}/{g['folds']} folds)")
    OUT.mkdir(exist_ok=True)
    results.to_csv(OUT / "fold_results.csv", index=False)
    plot_auc(results, OUT / "auc_random_vs_grouped.png")
    print(f"\nwrote {OUT / 'fold_results.csv'} and {OUT / 'auc_random_vs_grouped.png'}")


if __name__ == "__main__":
    main()
