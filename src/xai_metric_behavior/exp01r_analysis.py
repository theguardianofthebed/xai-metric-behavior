"""EXP-01-R summaries and figures that distinguish a target from sampled scores.

The established paired analysis is reused with IG in its first explanation slot.
No analytical value replaces a Quantus return. The budget-comparison summaries
use the same eligible repetitions across all compared budgets, and weight inputs
equally. Their uncertainty is descriptive, not held-out confirmation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze_scores, mean_or_nan, point_judgment


REAL_EXPLANATION_NAMES = ("IG", "A", "B")


def analyze_real_scores(
    scores: np.ndarray, reasons: np.ndarray, population: np.ndarray,
    subset_sizes: list[int], budgets: list[int], *, row_ids: np.ndarray,
    tie_tolerance: float, bootstrap_resamples: int, bootstrap_confidence: float,
    analysis_seed: int, analysis_eligible: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    """Reuse the verified paired estimator analysis; relabel slot zero as IG."""
    result = analyze_scores(
        scores, reasons, population, subset_sizes, budgets,
        tie_tolerance=tie_tolerance, bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence, analysis_seed=analysis_seed,
        analysis_eligible=analysis_eligible,
    )
    for rows in result.values():
        for row in rows:
            if row.get("explanation") == "exact":
                row["explanation"] = "IG"
            for key in ("contrast_id", "description"):
                if key in row:
                    row[key] = row[key].replace("Delta_exact_", "Delta_IG_").replace("exact:", "IG:")
            if "input_id" in row:
                row["source_test_row_id"] = int(row_ids[int(row["input_id"])])

    n, repetitions, _, _, _ = scores.shape
    result["budget_error_matched"] = []
    # Each k/candidate uses the same eligible repetitions at every compared B.
    for ki, k in enumerate(subset_sizes):
        for ei, name in enumerate(REAL_EXPLANATION_NAMES):
            joint = analysis_eligible[:, :, ki, :, ei].all(axis=2)
            for bi, budget in enumerate(budgets):
                input_mse, input_bias = [], []
                for i in range(n):
                    errors = scores[i, joint[i], ki, bi, ei] - population[i, ei]
                    input_mse.append(mean_or_nan(errors ** 2))
                    input_bias.append(mean_or_nan(errors))
                result["budget_error_matched"].append({
                    "k": k, "budget": budget, "explanation": name,
                    "rmse_given_common_budget_eligibility": float(np.sqrt(mean_or_nan(input_mse))),
                    "bias_given_common_budget_eligibility": mean_or_nan(input_bias),
                    "n_inputs_attempted": n,
                    "n_inputs_with_reference_and_common_repeats": int(np.isfinite(input_mse).sum()),
                    "n_input_repetitions_attempted": n * repetitions,
                    "n_input_repetitions_joint_all_budgets": int(joint.sum()),
                    "n_input_repetitions_excluded": int(joint.size - joint.sum()),
                    "comparison_budgets": ",".join(str(b) for b in budgets),
                    "aggregation": "within-input common-repeat MSE, equal input weights, then square root",
                })

    # One directly readable RQ table: the same input/repetition is retained for
    # every k, every B, and BOTH nonexact controls. This cannot improve an error
    # curve simply by using different eligible samples at different budgets.
    all_settings_joint = analysis_eligible[..., 1:3].all(axis=(2, 3, 4))
    result["rq_budget_summary"] = []
    for bi, budget in enumerate(budgets):
        input_mse = []
        for i in range(n):
            values = scores[i, :, :, bi, 1:3][all_settings_joint[i]]
            error = values - population[i, None, None, 1:3]
            input_mse.append(mean_or_nan(error.reshape(-1) ** 2))
        result["rq_budget_summary"].append({
            "budget": budget,
            "rmse_nonexact_controls_given_joint_all_settings": float(np.sqrt(mean_or_nan(input_mse))),
            "n_inputs_attempted": n,
            "n_inputs_with_reference_and_joint_repeats": int(np.isfinite(input_mse).sum()),
            "n_input_repetitions_attempted": n * repetitions,
            "n_input_repetitions_joint_all_settings": int(all_settings_joint.sum()),
            "n_input_repetitions_excluded": int(all_settings_joint.size - all_settings_joint.sum()),
            "joint_scope": "A and B; every declared subset size and budget",
            "aggregation": "mean squared error over paired repeats/k/candidates per input; equal input weights; square root",
        })

    result["population_margins"] = [{
        "input_id": i, "source_test_row_id": int(row_ids[i]),
        "population_margin_A_minus_B": float(population[i, 1] - population[i, 2]),
        "population_judgment": point_judgment(float(population[i, 1] - population[i, 2]), tie_tolerance),
        "practical_tie_tolerance": tie_tolerance,
        "interpretation": "comparison of these two coordinate-shift controls; no assumed A>B ordering",
    } for i in range(n)]
    result["reference_rank_summary"] = []
    for k in subset_sizes:
        for budget in budgets:
            rows = [r for r in result["pairwise_margins"] if r["k"] == k and r["budget"] == budget]
            joint = [r for r in rows if r["joint_analysis_eligible"]]
            decisive = [r for r in joint if r["population_judgment"] in ("A_preferred", "B_preferred")]
            reversals = sum(bool(r["strict_reversal_from_population"]) for r in decisive)
            input_rates = []
            for i in range(n):
                selected = [r for r in decisive if r["input_id"] == i]
                if selected:
                    input_rates.append(sum(bool(r["strict_reversal_from_population"]) for r in selected) / len(selected))
            result["reference_rank_summary"].append({
                "k": k, "budget": budget, "n_pair_attempts": len(rows),
                "n_joint_eligible": len(joint), "n_ineligible": len(rows) - len(joint),
                "n_joint_eligible_with_decisive_population_margin": len(decisive),
                "n_joint_eligible_with_population_tie_or_undefined": len(joint) - len(decisive),
                "strict_reversal_count": reversals,
                "reversal_rate_given_decisive_joint_eligible_equal_inputs": mean_or_nan(input_rates),
                "reversal_fraction_all_pair_attempts": reversals / len(rows),
                "finite_tie_count_given_joint_eligible": sum(r["point_judgment"] in ("exact_numerical_tie", "practical_tie") for r in joint),
                "practical_tie_tolerance": tie_tolerance,
                "interpretation": "conditional diagnostic; a near-zero population margin is not a decisive ranking",
            })
    return result


def make_rq_answer(
    manifest: dict[str, Any], population: np.ndarray,
    enumeration_rows: list[dict[str, Any]], analysis: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], str]:
    """Produce a direct answer with numerical evidence and its scope attached."""
    valid_enum = [r for r in enumeration_rows if r["reference_valid"] and r["enumeration_valid"]]
    spans = []
    for name in REAL_EXPLANATION_NAMES:
        values = [r["enumerated_FC"] for r in valid_enum if r["explanation"] == name]
        if values:
            spans.append(float(np.ptp(values)))
    counts = manifest["counts"]
    budgets = analysis["rq_budget_summary"]
    rank = analysis["reference_rank_summary"]
    answer = {
        "question": "On fixed Adult explanations, does subset size k change population FC, or only its finite estimate from B sampled groups?",
        "population_answer": "Under the stated additive log-odds and uniform-subset contract, the population value is k-invariant (derived). Enumeration tests that identity on the first selected input at allowed k only.",
        "enumeration_scope": "first selected input of this phase; allowed k only; no reserved confirmation setting",
        "enumeration_n_valid_comparisons": len(valid_enum),
        "enumeration_n_attempts": len(enumeration_rows),
        "enumeration_max_absolute_error": max((r["absolute_error"] for r in valid_enum), default=None),
        "enumeration_max_k_span": max(spans, default=None),
        "finite_answer": "Quantus scores are finite-sample estimates. Their errors and any reversals below describe this run; a larger B need not improve every individual score.",
        "matched_budget_error": budgets,
        "fc_counts": {key: counts[key] for key in (
            "n_attempted_scores", "n_official_finite_scores", "n_analysis_eligible_scores",
            "n_undefined_scores", "n_numerically_unstable_scores", "n_exception_scores")},
        "reference_ranking": rank,
        "IG_population_max_distance_from_one": float(np.max(np.abs(population[np.isfinite(population[:, 0]), 0] - 1))) if np.isfinite(population[:, 0]).any() else None,
        "status": "observed_smoke_not_confirmation" if manifest["phase"] == "smoke" else "observed_development_not_confirmation",
        "limits": "One fitted additive model; encoded-coordinate interventions can be off the data support. Labels are income outcomes, not explanation ground truth. No universal metric claim or held-out result.",
    }
    err = answer["enumeration_max_absolute_error"]
    span = answer["enumeration_max_k_span"]
    lines = [
        f"EXP-01-R / {manifest['phase'].upper()} — direct RQ answer",
        answer["question"], "",
        "POPULATION (derived, with independent enumeration):",
        answer["population_answer"],
        f"  Enumeration comparisons with defined references: {len(valid_enum)}/{len(enumeration_rows)}.",
        f"  Maximum enumeration/reference error: {err:.3g}." if err is not None else "  No defined enumeration/reference error is available.",
        f"  Maximum enumerated population spread across k: {span:.3g}." if span is not None else "  Population spread is unavailable.",
        "", "FINITE QUANTUS ESTIMATES (observed):", answer["finite_answer"],
        "  B | RMSE of controls A/B | joint input-repeats across every k/B/control",
    ]
    for row in budgets:
        lines.append(f"  {row['budget']:>3} | {row['rmse_nonexact_controls_given_joint_all_settings']:.6f} | "
                     f"{row['n_input_repetitions_joint_all_settings']}/{row['n_input_repetitions_attempted']}")
    lines.extend([
        "  RMSE averages paired repetitions/settings within each input, then gives inputs equal weight.",
        f"  Official FC: {counts['n_attempted_scores']} attempts; {counts['n_analysis_eligible_scores']} eligible; "
        f"{counts['n_undefined_scores']} undefined; {counts['n_numerically_unstable_scores']} near-constant; "
        f"{counts['n_exception_scores']} exceptions.",
        "  Undefined and near-constant counts are reason incidences and can overlap; raw returns are retained.",
        "", "PAIRWISE A/B DIAGNOSTIC:",
        "  k / B | strict reversals / jointly eligible decisive-margin pairs | all attempted pairs",
    ])
    for row in rank:
        lines.append(f"  {row['k']:>2} / {row['budget']:>3} | {row['strict_reversal_count']}/"
                     f"{row['n_joint_eligible_with_decisive_population_margin']} | {row['n_pair_attempts']}")
    lines.extend(["  Population ties and finite ties remain separate in the CSVs; A>B was never assumed.",
                  "", "SCOPE: " + answer["status"], answer["limits"], ""])
    return answer, "\n".join(lines)


def plot_real_results(
    output: Path, scores: np.ndarray, eligible: np.ndarray, population: np.ndarray,
    row_ids: np.ndarray, enumeration_rows: list[dict[str, Any]],
    analysis: dict[str, list[dict[str, Any]]], config: dict[str, Any],
) -> None:
    """Draw declared-input examples, matched errors, and all-attempt failures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    colors = {"IG": "#64748b", "A": "#2563eb", "B": "#c2410c"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "axes.facecolor": "white"})
    ks, budgets = config["subset_sizes"], config["budgets"]
    shown = min(len(row_ids), 4)  # Fixed before observing outcomes; never search for a reversal.
    fig, axes = plt.subplots(shown, len(budgets), figsize=(12.5, 2.65 * shown + 1.0),
                             sharex=True, sharey=True, squeeze=False)
    for i in range(shown):
        for bi, budget in enumerate(budgets):
            ax = axes[i, bi]
            for ei, name in enumerate(REAL_EXPLANATION_NAMES):
                col = colors[name]
                ax.axhline(population[i, ei], color=col, linestyle="--", linewidth=1.1, alpha=.85)
                values = np.where(eligible[i, :, :, bi, ei], scores[i, :, :, bi, ei], np.nan)
                means = [mean_or_nan(values[:, ki]) for ki in range(len(ks))]
                for ki, k in enumerate(ks):
                    finite = values[np.isfinite(values[:, ki]), ki]
                    if len(finite):
                        jitter = np.linspace(-.11, .11, len(finite))
                        ax.scatter(k + jitter, finite, color=col, s=11, alpha=.33, linewidths=0)
                ax.plot(ks, means, color=col, marker="o", markersize=4, linewidth=1.45)
            n_eligible = int(eligible[i, :, :, bi, :].sum())
            attempts = eligible[i, :, :, bi, :].size
            ax.set_title(f"Test row {int(row_ids[i])}  |  B={budget}\neligible {n_eligible}/{attempts}", fontsize=10)
            ax.set(xticks=ks, ylim=(-1.06, 1.06))
            ax.grid(axis="y", alpha=.18)
            if bi == 0:
                ax.set_ylabel("Pearson FC (higher score)")
            if i == shown - 1:
                ax.set_xlabel("k: features replaced per draw")
    legend = [Line2D([0], [0], color=colors[n], linewidth=2, label=n) for n in REAL_EXPLANATION_NAMES]
    legend.extend([Line2D([0], [0], color="#334155", linestyle="--", label="Population reference (derived)"),
                   Line2D([0], [0], color="#334155", marker="o", label="Eligible-repeat mean"),
                   Line2D([0], [0], color="#334155", linestyle="", marker=".", alpha=.45, label="Individual eligible score")])
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Does k change the target FC, or just its finite estimate?\n"
                 "Fixed Adult inputs, fitted log-odds model and Captum explanations; first declared inputs", fontsize=14)
    fig.subplots_adjust(top=.88, bottom=.115, left=.07, right=.985, wspace=.14, hspace=.34)
    fig.savefig(directory / "rq_population_vs_estimates.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for ei, name in enumerate(REAL_EXPLANATION_NAMES):
        rows = [r for r in enumeration_rows if r["explanation"] == name]
        ax.plot(ks, [population[0, ei]] * len(ks), color=colors[name], label=f"{name}: analytical reference")
        valid = [r for r in rows if r["enumeration_valid"]]
        ax.scatter([r["k"] for r in valid], [r["enumerated_FC"] for r in valid],
                   marker="x", s=65, color=colors[name], zorder=5, label=f"{name}: enumerate actual replacements")
    ax.set(xlabel="k: features replaced per subset", ylabel="Population Pearson FC", xticks=ks,
           ylim=(-1.06, 1.06), title=f"Independent population check: first declared test row {int(row_ids[0])}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(directory / "rq_population_enumeration.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), constrained_layout=True)
    for ei, name in enumerate(("A", "B")):
        ax = axes[ei]
        for k in ks:
            rows = [r for r in analysis["budget_error_matched"] if r["k"] == k and r["explanation"] == name]
            count = rows[0]["n_input_repetitions_joint_all_budgets"]
            total = rows[0]["n_input_repetitions_attempted"]
            ax.plot([r["budget"] for r in rows], [r["rmse_given_common_budget_eligibility"] for r in rows],
                    marker="o", label=f"k={k}; paired {count}/{total}")
        ax.set(xlabel="B: subset draws per Quantus score", ylabel="RMSE from population reference",
               title=f"Control {name}: unchanged across evaluation settings", ylim=(0, None), xscale="log")
        ax.set_xticks(budgets, labels=[str(b) for b in budgets])
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
    fig.suptitle("Finite-sampling error: same eligible repetitions across budgets, equal input weights\n"
                 "Observed smoke/development summaries; no guaranteed monotone decrease", fontsize=12)
    fig.savefig(directory / "rq_error_vs_budget.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), constrained_layout=True, sharey=True)
    for ax, key, label in zip(axes,
                              ("n_undefined", "n_numerical_instability", "n_exception"),
                              ("Undefined", "Near-constant warning", "Public/adapter exception")):
        for k in ks:
            values = []
            for budget in budgets:
                rows = [r for r in analysis["score_summary"] if r["k"] == k and r["budget"] == budget]
                values.append(sum(r[key] for r in rows) / sum(r["n_attempted"] for r in rows))
            ax.plot(budgets, values, marker="o", label=f"k={k}")
        ax.set(xlabel="B: draws per score", title=label, ylim=(-.025, 1.025), xscale="log")
        ax.set_xticks(budgets, labels=[str(b) for b in budgets])
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Reason count / all attempted FC calls")
    fig.suptitle("Read sampling-error results together with exclusions\n"
                 "All three candidates included; reason incidences may overlap", fontsize=12)
    fig.savefig(directory / "rq_failure_fractions.png", dpi=160)
    plt.close(fig)
