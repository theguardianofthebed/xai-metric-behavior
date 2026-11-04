"""Three plain figures whose axes match the EXP-01 estimands."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_results(
    output: Path,
    population_rows: list[dict[str, Any]],
    enumeration_rows: list[dict[str, Any]],
    analysis: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = output / "figures"
    directory.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"exact": "#333333", "A": "#2166ac", "B": "#b2182b"}
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for name in ("exact", "A", "B"):
        rows = [r for r in population_rows if r["input_id"] == 0 and r["explanation"] == name]
        checks = [r for r in enumeration_rows if r["fixture"] == "first_current_input_d20" and r["explanation"] == name]
        ax.plot([r["k"] for r in rows], [r["population_FC"] for r in rows], color=colors[name], label=f"{name}: analytical")
        ax.scatter([r["k"] for r in checks], [r["enumerated_FC"] for r in checks], color=colors[name],
                   marker="x", s=55, label=f"{name}: exact enumeration")
    ax.set(xlabel="Subset size k (features per draw)", ylabel="Population Pearson FC",
           title="Population check: first sampled input (d=20)", ylim=(-1.05, 1.05), xticks=config["subset_sizes"])
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(directory / "population_vs_subset_size.png", dpi=160)
    plt.close(fig)

    # Corrupted candidates expose sampling variability; the exact anchor is retained
    # in CSVs and the population figure, without disguising its degenerate-noise role.
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    for col, name in enumerate(("A", "B")):
        for k in config["subset_sizes"]:
            rows = sorted((r for r in analysis["score_summary"] if r["explanation"] == name and r["k"] == k),
                          key=lambda row: row["budget"])
            budgets = [r["budget"] for r in rows]
            axes[0, col].plot(budgets, [r["bias_given_eligible"] for r in rows], marker="o", label=f"k={k}")
            axes[0, col].fill_between(budgets, [r["bias_ci_low"] for r in rows], [r["bias_ci_high"] for r in rows], alpha=0.12)
            axes[1, col].plot(budgets, [r["rmse_given_eligible"] for r in rows], marker="o", label=f"k={k}")
        axes[0, col].axhline(0, color="#555555", linewidth=0.8)
        axes[0, col].set(title=f"Explanation {name}", ylabel="Bias given eligible estimates")
        axes[1, col].set(xlabel="Budget B (subset draws per score)", ylabel="RMSE given eligible estimates", ylim=(0, None))
        for row in range(2):
            axes[row, col].set_xscale("log")
            axes[row, col].set_xticks(config["budgets"], labels=[str(b) for b in config["budgets"]])
            axes[row, col].legend(fontsize=8)
    fig.suptitle("Analysis-eligible estimator error: equal-weight inputs; pointwise input-cluster intervals")
    fig.savefig(directory / "estimator_error_vs_budget.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), constrained_layout=True, sharey=True)
    for row, (count_key, label) in enumerate((
        ("n_undefined", "Undefined / all attempts"),
        ("n_numerical_instability", "Numerically unstable / all attempts"),
    )):
        for col, name in enumerate(("exact", "A", "B")):
            ax = axes[row, col]
            for k in config["subset_sizes"]:
                rows = sorted((r for r in analysis["score_summary"] if r["explanation"] == name and r["k"] == k),
                              key=lambda item: item["budget"])
                fractions = [r[count_key] / r["n_attempted"] for r in rows]
                ax.plot([r["budget"] for r in rows], fractions, marker="o", label=f"k={k}")
            ax.set(xlabel="Budget B", title=name, ylim=(-0.02, 1.02), xscale="log")
            ax.set_xticks(config["budgets"], labels=[str(b) for b in config["budgets"]])
            ax.legend(fontsize=8)
        axes[row, 0].set_ylabel(label)
    fig.suptitle("Analysis-ineligible outcomes: undefined and upstream near-constant warnings")
    fig.savefig(directory / "analysis_ineligible_fraction.png", dpi=160)
    plt.close(fig)
