"""Plain-text tables and the one figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import SCHEMES  # noqa: E402

COLORS = {"random": "#c0504d", "grouped": "#2f5d8a"}


def format_summary(summary: pd.DataFrame, gap: pd.DataFrame, leak: pd.DataFrame) -> str:
    """Render the headline tables as text."""
    lines = ["Test rows whose person-proxy group also sits in the training set:"]
    for scheme, share in leak.items():
        lines.append(f"  {scheme:<8} {share:6.1%}")
    lines += ["", f"{'model':<24}{'scheme':<9}{'AUC':>14}{'Brier':>16}{'log loss':>16}"]
    for _, r in summary.sort_values(["model", "scheme"]).iterrows():
        lines.append(
            f"{r['model']:<24}{r['scheme']:<9}"
            f"{r['auc_mean']:>8.3f}±{r['auc_std']:.3f}"
            f"{r['brier_mean']:>10.4f}±{r['brier_std']:.4f}"
            f"{r['log_loss_mean']:>10.4f}±{r['log_loss_std']:.4f}"
        )
    lines += ["", "AUC, random minus grouped:", gap.round(3).to_string()]
    return "\n".join(lines)


def plot_auc(results: pd.DataFrame, path: Path) -> None:
    """Per-fold AUC for each model under both schemes."""
    models = list(results["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    for i, scheme in enumerate(SCHEMES):
        offset = (i - 0.5) * width
        for j, model in enumerate(models):
            vals = results[(results["scheme"] == scheme) & (results["model"] == model)]["auc"]
            ax.scatter([j + offset] * len(vals), vals, s=14, alpha=0.5, color=COLORS[scheme])
            ax.hlines(vals.mean(), j + offset - 0.12, j + offset + 0.12, color=COLORS[scheme], lw=2.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models])
    ax.set_ylabel("ROC AUC (one dot per fold; bar = mean)")
    ax.set_title("Any-doctor-visit prediction: random vs person-grouped CV")
    handles = [plt.Line2D([], [], color=COLORS[s], lw=3, label=f"{s} CV") for s in SCHEMES]
    ax.legend(handles=handles, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
