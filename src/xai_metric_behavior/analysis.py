"""Condition comparisons that preserve input and evaluator-repetition pairing.

The finite-score estimand is conditional on joint validity. Subsets are never
treated as independent input observations. Intervals are descriptive, pointwise
input-cluster bootstrap intervals, not a multiple-testing or equivalence procedure.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .core import EXPLANATION_NAMES


def mean_or_nan(values: np.ndarray | list[float]) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else float("nan")


def point_judgment(margin: float, tolerance: float) -> str:
    if not np.isfinite(margin):
        return "undefined"
    if margin == 0.0:
        return "exact_numerical_tie"
    if abs(margin) <= tolerance:
        return "practical_tie"
    return "A_preferred" if margin > 0 else "B_preferred"


def interval_judgment(low: float, high: float, tolerance: float) -> str:
    if not np.isfinite(low) or not np.isfinite(high):
        return "uncertainty_unresolved"
    if low > tolerance:
        return "A_preferred_with_interval"
    if high < -tolerance:
        return "B_preferred_with_interval"
    if low >= -tolerance and high <= tolerance:
        return "interval_inside_practical_band_not_equivalence_test"
    return "uncertainty_unresolved"


def cluster_interval(
    input_values: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    confidence: float,
) -> tuple[float, float]:
    """Resample complete eligible input summaries, preserving paired contrasts."""
    values = np.asarray(input_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or resamples < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def analyze_scores(
    scores: np.ndarray,
    reasons: np.ndarray,
    population: np.ndarray,
    subset_sizes: list[int],
    budgets: list[int],
    *,
    tie_tolerance: float,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
    analysis_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Summarize scores[N,R,K,B,E], with E ordered exact,A,B.

    Per-arm valid-only summaries are descriptive. Primary contrasts each use
    their own common eligible input/repetition set, exported by repetition ID.
    """
    n_inputs, repetitions, nk, nb, ne = scores.shape
    if (nk, nb, ne) != (len(subset_sizes), len(budgets), len(EXPLANATION_NAMES)):
        raise ValueError("score shape does not match the declared setting/explanation axes")
    rng = np.random.Generator(np.random.PCG64(analysis_seed))
    result: dict[str, list[dict[str, Any]]] = {
        "input_summary": [], "score_summary": [], "pairwise_margins": [],
        "contrast_records": [], "contrast_summary": [],
    }

    for ki, k in enumerate(subset_sizes):
        for bi, budget in enumerate(budgets):
            for ei, explanation in enumerate(EXPLANATION_NAMES):
                data = scores[:, :, ki, bi, ei]
                mean_values, biases, squared_errors, variances = [], [], [], []
                valid_counts, exceptions = [], []
                for i in range(n_inputs):
                    vals = data[i, np.isfinite(data[i])]
                    n_valid = len(vals)
                    pop = population[i, ei]
                    errors = vals - pop
                    mean_score = mean_or_nan(vals)
                    bias = mean_or_nan(errors)
                    mse = mean_or_nan(errors ** 2)
                    mc_var = float(np.var(vals, ddof=1)) if n_valid >= 2 else float("nan")
                    n_exceptions = sum(str(v).startswith("exception:") for v in reasons[i, :, ki, bi, ei])
                    result["input_summary"].append({
                        "input_id": i, "k": k, "budget": budget, "explanation": explanation,
                        "n_attempted": repetitions, "n_valid": n_valid,
                        "n_nonfinite": repetitions - n_valid - n_exceptions,
                        "n_exception": n_exceptions, "population_score": pop,
                        "mean_score_given_valid": mean_score,
                        "bias_given_valid": bias,
                        "rmse_given_valid": float(np.sqrt(mse)),
                        "mc_variance_given_valid": mc_var,
                    })
                    mean_values.append(mean_score)
                    biases.append(bias)
                    squared_errors.append(mse)
                    variances.append(mc_var)
                    valid_counts.append(n_valid)
                    exceptions.append(n_exceptions)
                ci_low, ci_high = cluster_interval(np.array(mean_values), rng, bootstrap_resamples, bootstrap_confidence)
                bias_low, bias_high = cluster_interval(np.array(biases), rng, bootstrap_resamples, bootstrap_confidence)
                total_valid, total_exceptions = sum(valid_counts), sum(exceptions)
                result["score_summary"].append({
                    "k": k, "budget": budget, "explanation": explanation,
                    "aggregation": "equal_weight_input_means_given_valid",
                    "n_inputs_attempted": n_inputs,
                    "n_inputs_with_valid_repeats": sum(v > 0 for v in valid_counts),
                    "n_attempted": n_inputs * repetitions,
                    "n_valid": total_valid,
                    "n_nonfinite": n_inputs * repetitions - total_valid - total_exceptions,
                    "n_exception": total_exceptions,
                    "validity_probability": total_valid / (n_inputs * repetitions),
                    "mean_score_given_valid": mean_or_nan(mean_values),
                    "score_ci_low": ci_low, "score_ci_high": ci_high,
                    "bias_given_valid": mean_or_nan(biases),
                    "bias_ci_low": bias_low, "bias_ci_high": bias_high,
                    "rmse_given_valid": float(np.sqrt(mean_or_nan(squared_errors))),
                    "mean_within_input_mc_variance_given_valid": mean_or_nan(variances),
                    "ci_unit": "input_cluster", "ci_confidence": bootstrap_confidence,
                })

    population_margin = population[:, 1] - population[:, 2]
    for ki, k in enumerate(subset_sizes):
        for bi, budget in enumerate(budgets):
            for i in range(n_inputs):
                for rep in range(repetitions):
                    a, b = scores[i, rep, ki, bi, 1:3]
                    margin = float(a - b) if np.isfinite(a) and np.isfinite(b) else float("nan")
                    actual = point_judgment(margin, tie_tolerance)
                    reference = point_judgment(float(population_margin[i]), tie_tolerance)
                    reversal = ((actual, reference) in {
                        ("A_preferred", "B_preferred"), ("B_preferred", "A_preferred")
                    })
                    result["pairwise_margins"].append({
                        "input_id": i, "repetition": rep, "k": k, "budget": budget,
                        "joint_valid": bool(np.isfinite(margin)), "margin_A_minus_B": margin,
                        "point_judgment": actual, "population_margin": population_margin[i],
                        "population_judgment": reference,
                        "strict_reversal_from_population": reversal,
                    })

    def add_contrast(
        contrast_id: str,
        kind: str,
        description: str,
        arms: list[np.ndarray],
        coefficients: list[int],
        reference: np.ndarray,
        *,
        rank_change: bool = False,
    ) -> None:
        """All arms use identical eligible repetitions before any averaging."""
        joint = np.logical_and.reduce([np.isfinite(arm) for arm in arms])
        values = np.full((n_inputs, repetitions), np.nan)
        combined = sum(c * arm for c, arm in zip(coefficients, arms))
        values[joint] = combined[joint]
        input_effects, input_reversal_rates, total_reversals = [], [], 0
        total_exact_ties, total_practical_ties, total_same_preferences = 0, 0, 0
        for i in range(n_inputs):
            eligible = np.flatnonzero(joint[i])
            mean_effect = mean_or_nan(values[i, eligible])
            row: dict[str, Any] = {
                "contrast_id": contrast_id, "kind": kind, "input_id": i,
                "n_attempted": repetitions, "n_joint_valid": len(eligible),
                "joint_valid_repetitions": json.dumps(eligible.tolist(), separators=(",", ":")),
                "mean_difference_given_joint_valid": mean_effect,
                "population_difference": reference[i],
            }
            input_effects.append(mean_effect)
            if rank_change:
                # Arms are A_current,B_current,A_reference,B_reference.
                counts = {"strict_reversal": 0, "exact_numerical_tie": 0, "practical_tie": 0,
                          "same_preference": 0}
                for rep in eligible:
                    current = point_judgment(float(arms[0][i, rep] - arms[1][i, rep]), tie_tolerance)
                    previous = point_judgment(float(arms[2][i, rep] - arms[3][i, rep]), tie_tolerance)
                    if (current, previous) in {("A_preferred", "B_preferred"), ("B_preferred", "A_preferred")}:
                        counts["strict_reversal"] += 1
                    elif "exact_numerical_tie" in (current, previous):
                        counts["exact_numerical_tie"] += 1
                    elif "practical_tie" in (current, previous):
                        counts["practical_tie"] += 1
                    else:
                        counts["same_preference"] += 1
                rate = counts["strict_reversal"] / len(eligible) if len(eligible) else float("nan")
                row.update({**counts, "n_invalid": repetitions - len(eligible),
                            "reversal_rate_given_joint_valid": rate})
                input_reversal_rates.append(rate)
                total_reversals += counts["strict_reversal"]
                total_exact_ties += counts["exact_numerical_tie"]
                total_practical_ties += counts["practical_tie"]
                total_same_preferences += counts["same_preference"]
            result["contrast_records"].append(row)

        low, high = cluster_interval(np.array(input_effects), rng, bootstrap_resamples, bootstrap_confidence)
        eligible_inputs = np.isfinite(input_effects)
        joint_count = int(joint.sum())
        summary: dict[str, Any] = {
            "contrast_id": contrast_id, "kind": kind, "description": description,
            "estimand": "equal_weight_input_paired_difference_given_joint_valid",
            "n_inputs_attempted": n_inputs, "n_inputs_eligible": int(eligible_inputs.sum()),
            "n_attempted": n_inputs * repetitions, "n_joint_valid": joint_count,
            "n_invalid": n_inputs * repetitions - joint_count,
            "joint_validity_probability": joint_count / (n_inputs * repetitions),
            "mean_difference": mean_or_nan(input_effects), "ci_low": low, "ci_high": high,
            "population_difference_same_eligible_inputs": mean_or_nan(reference[eligible_inputs]),
            "ci_unit": "input_cluster", "ci_confidence": bootstrap_confidence,
            "selection_warning": ("all_attempts_joint_valid" if joint_count == n_inputs * repetitions
                                  else "conditional_only_broader_judgment_unresolved"),
        }
        if kind == "pairwise_D":
            summary["point_judgment"] = point_judgment(summary["mean_difference"], tie_tolerance)
            summary["uncertainty_judgment"] = interval_judgment(low, high, tie_tolerance)
        if rank_change:
            rate_low, rate_high = cluster_interval(np.array(input_reversal_rates), rng,
                                                  bootstrap_resamples, bootstrap_confidence)
            summary.update({
                "strict_reversal_count": total_reversals,
                "exact_numerical_tie_count": total_exact_ties,
                "practical_tie_count": total_practical_ties,
                "same_preference_count": total_same_preferences,
                "reversal_rate_given_joint_valid": mean_or_nan(input_reversal_rates),
                "reversal_rate_ci_low": rate_low, "reversal_rate_ci_high": rate_high,
                "reversal_fraction_all_attempts": total_reversals / (n_inputs * repetitions),
            })
        result["contrast_summary"].append(summary)

    # Pairwise margins at each condition: positive means A receives the larger FC.
    for ki, k in enumerate(subset_sizes):
        for bi, budget in enumerate(budgets):
            add_contrast(f"D_k{k}_B{budget}", "pairwise_D", f"A minus B at k={k}, budget={budget}",
                         [scores[:, :, ki, bi, 1], scores[:, :, ki, bi, 2]], [1, -1], population_margin)

    # Prespecified comparisons: k against the first declared k at fixed B;
    # B against the largest declared budget at fixed k. No post-hoc pair search.
    comparisons: list[tuple[str, int, int, int, int]] = []
    for ki in range(1, nk):
        for bi in range(nb):
            comparisons.append(("subset_size", ki, bi, 0, bi))
    reference_bi = int(np.argmax(budgets))
    for ki in range(nk):
        for bi in range(nb):
            if bi != reference_bi:
                comparisons.append(("budget", ki, bi, ki, reference_bi))
    for factor, ki, bi, rki, rbi in comparisons:
        label = f"k{subset_sizes[ki]}_B{budgets[bi]}_minus_k{subset_sizes[rki]}_B{budgets[rbi]}"
        for ei, name in enumerate(EXPLANATION_NAMES):
            add_contrast(f"Delta_{name}_{label}", "score_change", f"{name}: {factor}; {label}",
                         [scores[:, :, ki, bi, ei], scores[:, :, rki, rbi, ei]], [1, -1],
                         np.zeros(n_inputs))
        add_contrast(f"rank_{label}", "rank_change", f"A/B judgment change: {factor}; {label}",
                     [scores[:, :, ki, bi, 1], scores[:, :, ki, bi, 2],
                      scores[:, :, rki, rbi, 1], scores[:, :, rki, rbi, 2]],
                     [1, -1, -1, 1], np.zeros(n_inputs), rank_change=True)
    return result
