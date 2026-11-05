"""Condition comparisons that preserve input and evaluator-repetition pairing.

The finite-score estimand is conditional on joint analysis eligibility. Subsets are never
treated as independent input observations. Intervals are descriptive, pointwise
input-cluster bootstrap intervals, not a multiple-testing or equivalence procedure.
Official finite scores with an upstream NearConstantInputWarning remain recorded
but do not enter conditional estimation or ranking summaries.
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
    analysis_eligible: np.ndarray | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Summarize scores[N,R,K,B,E], with E ordered exact,A,B.

    Per-arm eligible-only summaries are descriptive. Primary contrasts each use
    their own common eligible input/repetition set, exported by repetition ID.
    ``scores`` contains unmodified official returns, including warned finite values.
    """
    n_inputs, repetitions, nk, nb, ne = scores.shape
    if (nk, nb, ne) != (len(subset_sizes), len(budgets), len(EXPLANATION_NAMES)):
        raise ValueError("score shape does not match the declared setting/explanation axes")
    if reasons.shape != scores.shape:
        raise ValueError("reason shape does not match scores")
    official_finite = np.isfinite(scores)
    exceptions = np.char.startswith(reasons.astype(str), "exception:")
    numerical_instability = np.char.startswith(reasons.astype(str), "near_constant_") & ~exceptions
    undefined = ~official_finite & ~exceptions
    expected_eligible = official_finite & ~exceptions & ~numerical_instability
    if analysis_eligible is not None and not np.array_equal(analysis_eligible, expected_eligible):
        raise ValueError("analysis eligibility disagrees with the finite/no-exception/no-near-constant policy")
    eligible_mask = expected_eligible
    # Mask only this analysis copy; the caller and raw CSV retain official values.
    scores = np.where(eligible_mask, scores, np.nan)
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
                eligible_counts = []
                for i in range(n_inputs):
                    vals = data[i, np.isfinite(data[i])]
                    n_eligible = len(vals)
                    pop = population[i, ei]
                    errors = vals - pop
                    mean_score = mean_or_nan(vals)
                    bias = mean_or_nan(errors)
                    mse = mean_or_nan(errors ** 2)
                    mc_var = float(np.var(vals, ddof=1)) if n_eligible >= 2 else float("nan")
                    result["input_summary"].append({
                        "input_id": i, "k": k, "budget": budget, "explanation": explanation,
                        "n_attempted": repetitions, "n_official_finite": int(official_finite[i, :, ki, bi, ei].sum()),
                        "n_analysis_eligible": n_eligible, "n_ineligible": repetitions - n_eligible,
                        "n_undefined": int(undefined[i, :, ki, bi, ei].sum()),
                        "n_numerical_instability": int(numerical_instability[i, :, ki, bi, ei].sum()),
                        "n_exception": int(exceptions[i, :, ki, bi, ei].sum()), "population_score": pop,
                        "mean_score_given_eligible": mean_score,
                        "bias_given_eligible": bias,
                        "rmse_given_eligible": float(np.sqrt(mse)),
                        "mc_variance_given_eligible": mc_var,
                    })
                    mean_values.append(mean_score)
                    biases.append(bias)
                    squared_errors.append(mse)
                    variances.append(mc_var)
                    eligible_counts.append(n_eligible)
                ci_low, ci_high = cluster_interval(np.array(mean_values), rng, bootstrap_resamples, bootstrap_confidence)
                bias_low, bias_high = cluster_interval(np.array(biases), rng, bootstrap_resamples, bootstrap_confidence)
                total_eligible = sum(eligible_counts)
                total_finite = int(official_finite[:, :, ki, bi, ei].sum())
                result["score_summary"].append({
                    "k": k, "budget": budget, "explanation": explanation,
                    "aggregation": "equal_weight_input_means_given_eligible",
                    "n_inputs_attempted": n_inputs,
                    "n_inputs_with_eligible_repeats": sum(v > 0 for v in eligible_counts),
                    "n_attempted": n_inputs * repetitions,
                    "n_official_finite": total_finite,
                    "n_analysis_eligible": total_eligible,
                    "n_ineligible": n_inputs * repetitions - total_eligible,
                    "n_undefined": int(undefined[:, :, ki, bi, ei].sum()),
                    "n_numerical_instability": int(numerical_instability[:, :, ki, bi, ei].sum()),
                    "n_exception": int(exceptions[:, :, ki, bi, ei].sum()),
                    "official_finite_probability": total_finite / (n_inputs * repetitions),
                    "analysis_eligibility_probability": total_eligible / (n_inputs * repetitions),
                    "counting_note": "all_attempts_denominator; undefined_and_numerical_warning_counts_can_overlap",
                    "mean_score_given_eligible": mean_or_nan(mean_values),
                    "score_ci_low": ci_low, "score_ci_high": ci_high,
                    "bias_given_eligible": mean_or_nan(biases),
                    "bias_ci_low": bias_low, "bias_ci_high": bias_high,
                    "rmse_given_eligible": float(np.sqrt(mean_or_nan(squared_errors))),
                    "mean_within_input_mc_variance_given_eligible": mean_or_nan(variances),
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
                        "joint_official_finite": bool(official_finite[i, rep, ki, bi, 1:3].all()),
                        "joint_analysis_eligible": bool(np.isfinite(margin)),
                        "has_undefined": bool(undefined[i, rep, ki, bi, 1:3].any()),
                        "has_numerical_instability": bool(numerical_instability[i, rep, ki, bi, 1:3].any()),
                        "has_exception": bool(exceptions[i, rep, ki, bi, 1:3].any()),
                        "margin_A_minus_B_given_joint_eligible": margin,
                        "point_judgment": actual if np.isfinite(margin) else "analysis_ineligible",
                        "population_margin": population_margin[i],
                        "population_judgment": reference,
                        "strict_reversal_from_population": reversal if np.isfinite(margin) else None,
                    })

    def add_contrast(
        contrast_id: str,
        kind: str,
        description: str,
        arm_indices: list[tuple[int, int, int]],
        coefficients: list[int],
        reference: np.ndarray,
        *,
        rank_change: bool = False,
    ) -> None:
        """All arms use identical eligible repetitions before any averaging."""
        arms = [scores[:, :, ki, bi, ei] for ki, bi, ei in arm_indices]
        joint = np.logical_and.reduce([np.isfinite(arm) for arm in arms])
        joint_finite = np.logical_and.reduce([official_finite[:, :, ki, bi, ei] for ki, bi, ei in arm_indices])
        has_undefined = np.logical_or.reduce([undefined[:, :, ki, bi, ei] for ki, bi, ei in arm_indices])
        has_instability = np.logical_or.reduce([numerical_instability[:, :, ki, bi, ei] for ki, bi, ei in arm_indices])
        has_exception = np.logical_or.reduce([exceptions[:, :, ki, bi, ei] for ki, bi, ei in arm_indices])
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
                "n_attempted": repetitions, "n_joint_eligible": len(eligible),
                "n_joint_official_finite": int(joint_finite[i].sum()),
                "n_ineligible": repetitions - len(eligible),
                "n_with_undefined": int(has_undefined[i].sum()),
                "n_with_numerical_instability": int(has_instability[i].sum()),
                "n_with_exception": int(has_exception[i].sum()),
                "joint_eligible_repetitions": json.dumps(eligible.tolist(), separators=(",", ":")),
                "mean_difference_given_joint_eligible": mean_effect,
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
                row.update({**counts, "reversal_rate_given_joint_eligible": rate})
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
            "estimand": "equal_weight_input_paired_difference_given_joint_eligible",
            "n_inputs_attempted": n_inputs, "n_inputs_eligible": int(eligible_inputs.sum()),
            "n_attempted": n_inputs * repetitions, "n_joint_eligible": joint_count,
            "n_joint_official_finite": int(joint_finite.sum()),
            "n_ineligible": n_inputs * repetitions - joint_count,
            "n_with_undefined": int(has_undefined.sum()),
            "n_with_numerical_instability": int(has_instability.sum()),
            "n_with_exception": int(has_exception.sum()),
            "counting_note": "all_attempts_denominator; reason_incidence_counts_can_overlap_across_arms",
            "joint_analysis_eligibility_probability": joint_count / (n_inputs * repetitions),
            "mean_difference": mean_or_nan(input_effects), "ci_low": low, "ci_high": high,
            "population_difference_same_eligible_inputs": mean_or_nan(reference[eligible_inputs]),
            "ci_unit": "input_cluster", "ci_confidence": bootstrap_confidence,
            "selection_warning": ("all_attempts_joint_eligible" if joint_count == n_inputs * repetitions
                                  else "conditional_only_broader_judgment_unresolved"),
        }
        if kind == "pairwise_D":
            summary["point_judgment"] = (point_judgment(summary["mean_difference"], tie_tolerance)
                                         if np.isfinite(summary["mean_difference"]) else "no_analysis_eligible_estimate")
            summary["uncertainty_judgment"] = interval_judgment(low, high, tie_tolerance)
        if rank_change:
            rate_low, rate_high = cluster_interval(np.array(input_reversal_rates), rng,
                                                  bootstrap_resamples, bootstrap_confidence)
            summary.update({
                "strict_reversal_count": total_reversals,
                "exact_numerical_tie_count": total_exact_ties,
                "practical_tie_count": total_practical_ties,
                "same_preference_count": total_same_preferences,
                "reversal_rate_given_joint_eligible": mean_or_nan(input_reversal_rates),
                "reversal_rate_ci_low": rate_low, "reversal_rate_ci_high": rate_high,
                "reversal_fraction_all_attempts": total_reversals / (n_inputs * repetitions),
            })
        result["contrast_summary"].append(summary)

    # Pairwise margins at each condition: positive means A receives the larger FC.
    for ki, k in enumerate(subset_sizes):
        for bi, budget in enumerate(budgets):
            add_contrast(f"D_k{k}_B{budget}", "pairwise_D", f"A minus B at k={k}, budget={budget}",
                         [(ki, bi, 1), (ki, bi, 2)], [1, -1], population_margin)

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
                         [(ki, bi, ei), (rki, rbi, ei)], [1, -1],
                         np.zeros(n_inputs))
        add_contrast(f"rank_{label}", "rank_change", f"A/B judgment change: {factor}; {label}",
                     [(ki, bi, 1), (ki, bi, 2), (rki, rbi, 1), (rki, rbi, 2)],
                     [1, -1, -1, 1], np.zeros(n_inputs), rank_change=True)
    return result
