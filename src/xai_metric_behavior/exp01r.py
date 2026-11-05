"""EXP-01-R: fixed Adult explanations measured by public Quantus FC/FE.

Smoke and development are implemented. Reserved confirmation remains blocked:
it needs a later freeze that binds the fitted pipeline as well as the protocol.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
from math import comb
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import yaml

from .core import checked_pearson, enumerate_fc
from .experiment import (canonical_hash, file_hash, implementation_hash, write_csv,
                         _git_provenance, _runtime, _write_json)
from .exp01r_analysis import (REAL_EXPLANATION_NAMES, analyze_real_scores,
                             make_rq_answer, plot_real_results)
from .exp01r_data import (DATASET_ID, IG_PARAMETERS, MODEL_PARAMETERS, PreparedAdult,
                         prepare_adult)
from .quantus_backend import (PublicScore, ScalarLogitModel, backend_provenance,
                             evaluator_seed_words, public_score)


REAL_METRIC_VARIANT = "exp01r-quantus-public-0.6.0-v1"
REAL_PHASE_IDS = {"smoke": 10, "development": 11}
ALLOWED_SIZES = {1, 5, 10, 12}
RESERVED_SIZES = {2, 7, 11}
FAILURE_FIELDS = ["stage", "input_id", "source_test_row_id", "repetition", "k", "budget",
                  "explanation", "category", "reason", "exception_stage"]


def load_exp01r_config(path: str | Path) -> dict[str, Any]:
    """Reject silent contract changes and reserved conditions before any work."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("experiment_id") != "EXP-01-R":
        raise ValueError("configuration must identify experiment_id: EXP-01-R")
    if config.get("phase") == "confirmation":
        raise ValueError("EXP-01-R confirmation is blocked until a model/pipeline-bound freeze is implemented after development")
    if config.get("phase") not in REAL_PHASE_IDS:
        raise ValueError("EXP-01-R phase must be smoke or development")
    fields = {"experiment_id", "phase", "dimension", "baseline", "n_inputs", "repetitions",
              "subset_sizes", "budgets", "seeds", "data", "model", "ig", "analysis",
              "enumeration", "write_traces"}
    if set(config) != fields:
        raise ValueError(f"unexpected/missing configuration fields: {sorted(set(config) ^ fields)}")
    nested = {
        "seeds": {"split", "model", "evaluator", "analysis"},
        "data": {"dataset", "preprocessing", "confirmation_size"},
        "model": set(MODEL_PARAMETERS), "ig": set(IG_PARAMETERS),
        "analysis": {"numerical_tolerance", "practical_tie_tolerance", "bootstrap_resamples", "confidence"},
        "enumeration": {"enabled", "n_inputs", "max_subsets"},
    }
    for key, required in nested.items():
        if not isinstance(config.get(key), dict) or set(config[key]) != required:
            raise ValueError(f"{key} must contain exactly {sorted(required)}; unsupported settings are never ignored")
    if config["dimension"] != 13 or config["baseline"] != 0.0:
        raise ValueError("EXP-01-R fixes 13 encoded coordinates and literal baseline zero")
    if config["data"] != {"dataset": DATASET_ID, "preprocessing": "standard_scaler_train_only", "confirmation_size": 512}:
        raise ValueError("the declared Adult source, train-only preprocessing and 512-record reservation are fixed")
    if config["model"] != MODEL_PARAMETERS or config["ig"] != IG_PARAMETERS:
        raise ValueError("model and IG parameters/gates must match the declared pre-run protocol")
    for key in ("n_inputs", "repetitions"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("subset_sizes", "budgets"):
        values = config[key]
        if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
            raise ValueError(f"{key} must be a nonempty integer list")
        if sorted(set(values)) != values:
            raise ValueError(f"{key} must be sorted without duplicates")
    if set(config["subset_sizes"]) & RESERVED_SIZES:
        raise ValueError("reserved confirmation subset sizes must not be consumed by smoke/development")
    if not set(config["subset_sizes"]) <= ALLOWED_SIZES:
        raise ValueError("smoke/development subset sizes must come from [1, 5, 10, 12]")
    if min(config["budgets"]) < 2:
        raise ValueError("Pearson needs at least two subset draws")
    if any(type(seed) is not int or seed < 0 for seed in config["seeds"].values()):
        raise ValueError("declare nonnegative integer seeds")
    if len(set(config["seeds"].values())) != 4:
        raise ValueError("split, model, evaluator and analysis seed roots must be distinct")
    analysis = config["analysis"]
    if not 0 < analysis["numerical_tolerance"] < .001:
        raise ValueError("a small positive numerical tolerance is required")
    if not 0 < analysis["practical_tie_tolerance"] < 1:
        raise ValueError("the practical tie tolerance must lie between zero and one")
    if type(analysis["bootstrap_resamples"]) is not int or analysis["bootstrap_resamples"] < 2:
        raise ValueError("at least two descriptive bootstrap resamples are required")
    if not 0 < analysis["confidence"] < 1:
        raise ValueError("confidence must lie between zero and one")
    enumeration = config["enumeration"]
    if enumeration["enabled"] is not True or enumeration["n_inputs"] != 1:
        raise ValueError("enumerate the first selected input at allowed subset sizes")
    if type(enumeration["max_subsets"]) is not int or enumeration["max_subsets"] < max(comb(13, k) for k in config["subset_sizes"]):
        raise ValueError("enumeration cap is too small for the declared 13-feature subsets")
    if config["write_traces"] is not True:
        raise ValueError("EXP-01-R requires actual public-call traces")
    return config


def _enumerate_first_input(config: dict[str, Any], prepared: PreparedAdult) -> list[dict[str, Any]]:
    """Enumerate actual fitted-model replacements, not sums asserted by the theorem."""
    model = ScalarLogitModel(prepared.weights, prepared.bias)

    def predict(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        is_vector = values.ndim == 1
        with torch.no_grad():
            out = model(torch.tensor(values.reshape(-1, len(prepared.weights)), dtype=torch.float64))
        result = out.detach().cpu().numpy().reshape(-1)
        return result[0] if is_vector else result

    x = prepared.x[0]
    explanations = {name: prepared.explanations[name][0] for name in REAL_EXPLANATION_NAMES}
    rows = []
    tolerance = config["analysis"]["numerical_tolerance"]
    for k in config["subset_sizes"]:
        enumerated = enumerate_fc(predict, x, np.zeros_like(x), explanations, k,
                                  max_subsets=config["enumeration"]["max_subsets"])
        for name in REAL_EXPLANATION_NAMES:
            reference = checked_pearson(explanations[name], prepared.reference[0])
            score = enumerated[name]
            error = abs(score.score - reference.score) if reference.valid and score.valid else float("nan")
            agrees = reference.valid == score.valid
            passed = bool(agrees and (not reference.valid or error <= tolerance))
            rows.append({
                "input_id": 0, "source_test_row_id": int(prepared.row_ids[0]),
                "dimension": 13, "k": k, "n_subsets": comb(13, k), "explanation": name,
                "coordinate_prediction": reference.score, "enumerated_FC": score.score,
                "reference_valid": reference.valid, "enumeration_valid": score.valid,
                "reference_reason": reference.reason, "enumeration_reason": score.reason,
                "absolute_error": error, "numerical_tolerance": tolerance, "passed": passed,
                "status": "numerical_check" if reference.valid and score.valid else "undefined_population_recorded",
                "reference_engine": "exhaustive_replacements_through_fitted_scalar_pytorch_model",
                "scope": "first_selected_input_at_allowed_k_only",
            })
    return rows


def _public_validation_rejection(result: PublicScore) -> bool:
    """An upstream validator rejection is an outcome, not a license to patch it."""
    return (result.exception_stage == "official_call" and result.exception is not None
            and result.exception.startswith("AssertionError:"))


def run_exp01r(config_path: str | Path, output_path: str | Path,
               data_dir: str | Path) -> dict[str, Any]:
    """Fit/load the frozen pipeline, run declared public calls, and retain evidence."""
    config = load_exp01r_config(config_path)
    backend = backend_provenance()
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    n, repetitions = config["n_inputs"], config["repetitions"]
    sizes, budgets = config["subset_sizes"], config["budgets"]
    tolerance = config["analysis"]["numerical_tolerance"]
    snapshot = Path(__file__).resolve().parents[2] / "docs" / "atlas_exp01r_protocol_2026-11-05.json"
    runtime = _runtime()
    for name in ("captum", "scikit-learn"):
        runtime["packages"][name] = importlib.metadata.version(name)
    manifest: dict[str, Any] = {
        "experiment_id": "EXP-01-R", "phase": config["phase"], "status": "running",
        "run_id": canonical_hash({"config": config, "started": started})[:16],
        "started_at_utc": started, "config_sha256": canonical_hash(config),
        "implementation_sha256": implementation_hash(), "git": _git_provenance(), "runtime": runtime,
        "metric_variant": REAL_METRIC_VARIANT, "shared_backend_variant": backend["metric_variant"],
        "library_execution": "Official Quantus 0.6.0 public FC/FE returns. Captum generates actual IG; NumPy is reference/trace verification only.",
        "backend_execution_sha256": backend["execution_sha256"], "quantus_backend": backend,
        "protocol_source": {"path": "docs/atlas_exp01r_protocol_2026-11-05.json",
                            "sha256": file_hash(snapshot) if snapshot.exists() else None},
        "contract": {
            "target": "fixed fitted high-income log-odds: beta'x + intercept; not a probability or predicted-class output",
            "baseline": 0.0, "signed": True, "abs": False, "normalise": False, "softmax": False,
            "input_coordinates": "13 original encoded columns, each standardized using training statistics only",
            "baseline_interpretation": "train means in encoded raw coordinates; categorical paths/replacements may be off support",
            "selection": "uniform size-k subsets without replacement within each draw; independent draws",
            "comparator": "Pearson", "direction": "higher",
            "aggregation": "eligible repetitions within input; equal input weights; paired eligibility before contrasts",
            "analysis_eligibility": "finite official return, no exception and no upstream NearConstantInputWarning",
            "exception_policy": "retain upstream AssertionError validation rejections; halt after retaining structural/parity/infrastructure failures",
            "controls": {"IG": "actual Captum output after coordinate/completeness gates", "A": "cyclic shift IG by +1", "B": "cyclic shift IG by +2"},
            "ranking_reference": "per-input Corr(A,v)-Corr(B,v); either sign and ties retained; no presumed A>B",
            "confirmation": "512 reserved test row IDs and k={2,7,11}; no confirmation runner until model/pipeline-bound freeze",
        },
        "rng": {"seed_roots": config["seeds"], "phase_id": REAL_PHASE_IDS[config["phase"]],
                "evaluator_generator": "Quantus native legacy MT19937",
                "coupling": "reset identical per-input/repetition seeds; check actual common k/budget/candidate mask prefixes",
                "global_state": "shared Quantus adapter restores NumPy RNG and Torch default dtype"},
        "counts": {"n_inputs": n, "repetitions_per_input": repetitions,
                   "n_planned_scores": n * repetitions * len(sizes) * len(budgets) * 3,
                   "n_attempted_scores": 0, "n_official_finite_scores": 0, "n_analysis_eligible_scores": 0,
                   "n_numerically_unstable_scores": 0, "n_undefined_scores": 0, "n_exception_scores": 0,
                   "n_public_validation_exceptions": 0, "n_fatal_errors": 0,
                   "n_public_FC_calls": 0, "n_public_FE_calls": 0, "n_FE_official_finite": 0,
                   "n_FE_analysis_eligible": 0, "n_FE_numerically_unstable": 0, "n_FE_undefined": 0,
                   "n_FE_exceptions": 0, "n_trace_checks_passed": 0, "n_pairing_checks_passed": 0,
                   "n_score_parity_passed": 0, "n_score_parity_deferred_near_constant": 0,
                   "n_warning_records": 0, "n_failures_logged": 0},
        "validation": {"reserved_records_evaluated": 0, "reserved_subset_sizes_evaluated": []},
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(output / "manifest.json", manifest)  # Provenance exists before preparation/measurement.
    failures: list[dict[str, Any]] = []
    call_index, max_trace_error, max_stable_error = 0, 0.0, 0.0
    prepared: PreparedAdult | None = None

    def invoke(kind: str, i: int, name: str, seed: np.ndarray,
               k: int | None = None, budget: int | None = None) -> PublicScore:
        manifest["counts"][f"n_public_{kind}_calls"] += 1
        try:
            return public_score(kind, prepared.x[i], prepared.weights, prepared.explanations[name][i], seed,
                                subset_size=k, budget=budget, bias=prepared.bias, tolerance=tolerance)
        except Exception as error:
            return PublicScore(float("nan"), False, f"exception:{type(error).__name__}",
                               exception=f"{type(error).__name__}: {error}", exception_stage="adapter_setup")

    def retain(kind: str, i: int, name: str, result: PublicScore, seed: np.ndarray, *,
               repetition: int | None = None, k: int | None = None, budget: int | None = None,
               pairing_error: str | None = None, pairing_verified: bool = False) -> str | None:
        nonlocal call_index, max_trace_error, max_stable_error
        call_index += 1
        exception = pairing_error or result.exception
        exception_stage = "trace_pairing" if pairing_error else result.exception_stage
        trace_file = f"traces/input_{i:04d}.npz"
        if kind == "FE" or exception:
            trace_file = f"traces/{kind}_call_{call_index:07d}.npz"
            np.savez_compressed(output / trace_file, **{key: np.asarray(value) for key, value in result.trace.items()
                                                       if value is not None})
        finite = bool(np.isfinite(result.score))
        unstable = any(w["category"] == "NearConstantInputWarning" for w in result.warnings)
        eligible = bool(result.analysis_eligible and exception is None)
        if result.checks:
            manifest["counts"]["n_trace_checks_passed"] += 1
            parity = result.checks.get("score_parity_status")
            if parity == "passed":
                manifest["counts"]["n_score_parity_passed"] += 1
            elif parity == "not_asserted_near_constant":
                manifest["counts"]["n_score_parity_deferred_near_constant"] += 1
            error = result.checks.get("trace_replay_absolute_error")
            if error is not None:
                max_trace_error = max(max_trace_error, float(error))
                if parity == "passed":
                    max_stable_error = max(max_stable_error, float(error))
        manifest["counts"]["n_pairing_checks_passed"] += int(pairing_verified)
        manifest["counts"]["n_warning_records"] += len(result.warnings)
        if _public_validation_rejection(result):
            manifest["counts"]["n_public_validation_exceptions"] += 1
        record = {
            "call_id": call_index, "metric": kind,
            "public_api": f"quantus.Faithfulness{'Correlation' if kind == 'FC' else 'Estimate'}.__call__",
            "input_id": i, "source_test_row_id": int(prepared.row_ids[i]), "explanation": name,
            "repetition": repetition, "k": k, "budget": budget, "seed_words": seed.tolist(),
            "score": float(result.score) if finite else None, "raw_score_repr": repr(float(result.score)),
            "valid": finite, "analysis_eligible": eligible, "numerical_instability": unstable,
            "reason": result.reason if not pairing_error else "exception:trace_pairing",
            "exception": exception, "exception_stage": exception_stage,
            "checks": result.checks, "pairing_verified": pairing_verified,
            "mask_sha256": result.trace.get("mask_sha256"), "uv_sha256": result.trace.get("uv_sha256"),
            "model_input_sha256": result.trace.get("model_input_sha256"), "trace_file": trace_file,
            "trace_locator": None if kind == "FE" or exception else {
                "repetition_index": repetition, "subset_size_index": sizes.index(k),
                "candidate_index": REAL_EXPLANATION_NAMES.index(name), "draw_prefix_length": budget},
            "warnings": result.warnings,
        }
        with (output / "metric_calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        location = {"stage": f"public_{kind}", "input_id": i, "source_test_row_id": int(prepared.row_ids[i]),
                    "repetition": repetition, "k": k, "budget": budget, "explanation": name}
        if exception:
            failures.append({**location, "category": "public_validation_exception" if _public_validation_rejection(result) and not pairing_error else "fatal_guard_or_api_exception",
                             "reason": exception, "exception_stage": exception_stage})
        else:
            if not finite:
                failures.append({**location, "category": "undefined", "reason": result.reason})
            if unstable:
                failures.append({**location, "category": "numerical_instability", "reason": result.reason})
        return exception

    try:
        prepared = prepare_adult(config, data_dir, output)
        if prepared.x.shape != (n, 13) or set(prepared.explanations) != set(REAL_EXPLANATION_NAMES):
            raise RuntimeError("prepared inputs/candidates do not match the declared experiment axes")
        if prepared.reference.shape != prepared.x.shape or any(prepared.explanations[name].shape != prepared.x.shape for name in REAL_EXPLANATION_NAMES):
            raise RuntimeError("prepared reference/explanations do not preserve the input feature coordinates")
        manifest["preparation"] = prepared.provenance
        manifest["model_sha256"] = prepared.provenance["model_sha256"]
        manifest["partition_sha256"] = prepared.provenance["partition_sha256"]
        manifest["contract"]["model_intercept"] = float(prepared.bias)
        _write_json(output / "manifest.json", manifest)
        population = np.full((n, 3), np.nan)
        population_rows = []
        max_fe_error = 0.0
        for i in range(n):
            references = [checked_pearson(prepared.explanations[name][i], prepared.reference[i]) for name in REAL_EXPLANATION_NAMES]
            population[i] = [reference.score for reference in references]
            for ei, name in enumerate(REAL_EXPLANATION_NAMES):
                reference = references[ei]
                seed = evaluator_seed_words(config["seeds"]["evaluator"], REAL_PHASE_IDS[config["phase"]], int(prepared.row_ids[i]), 0)
                fe = invoke("FE", i, name, seed)
                exception = retain("FE", i, name, fe, seed, k=1, budget=13)
                finite = bool(np.isfinite(fe.score))
                unstable = any(w["category"] == "NearConstantInputWarning" for w in fe.warnings)
                eligible = bool(fe.analysis_eligible and exception is None)
                counts = manifest["counts"]
                counts["n_FE_official_finite"] += int(finite)
                counts["n_FE_analysis_eligible"] += int(eligible)
                counts["n_FE_numerically_unstable"] += int(unstable)
                counts["n_FE_undefined"] += int(not finite and exception is None)
                counts["n_FE_exceptions"] += int(exception is not None)
                if reference.valid and eligible:
                    max_fe_error = max(max_fe_error, abs(reference.score - fe.score))
                denominator = float(np.abs(prepared.reference[i]).sum())
                l1_error = float(np.abs(prepared.explanations[name][i] - prepared.reference[i]).sum() / denominator) if denominator else float("nan")
                for k in sizes:
                    population_rows.append({
                        "input_id": i, "source_test_row_id": int(prepared.row_ids[i]), "k": k, "explanation": name,
                        "population_FC": reference.score, "reference_type": "conditional_coordinate_correlation",
                        "population_valid": reference.valid, "population_reason": reference.reason,
                        "singleton_FE": fe.score, "singleton_FE_analysis_eligible": eligible,
                        "singleton_FE_exception": exception or "", "singleton_FE_reason": fe.reason,
                        "singleton_FE_source": "quantus.FaithfulnessEstimate.__call__",
                        "population_margin_A_minus_B": float(population[i, 1] - population[i, 2]),
                        "reference_error_L1": l1_error, "reference_error_denominator": denominator,
                    })
                write_csv(output / "population_scores.csv", population_rows)
                if exception and not _public_validation_rejection(fe):
                    raise RuntimeError(f"FE/adapter failure retained before stopping: {exception}")
        enumeration_rows = _enumerate_first_input(config, prepared)
        write_csv(output / "exact_enumeration.csv", enumeration_rows)
        valid_errors = [r["absolute_error"] for r in enumeration_rows if np.isfinite(r["absolute_error"])]
        manifest["validation"].update({
            "enumeration_comparisons": len(enumeration_rows),
            "enumeration_valid_comparisons": len(valid_errors),
            "enumeration_all_checked_consistent": all(r["passed"] for r in enumeration_rows),
            "max_enumeration_absolute_error": max(valid_errors, default=None),
            "max_singleton_FE_coordinate_error_given_eligible": max_fe_error,
            "enumeration_scope": "first selected record in this phase, allowed k only",
            "numerical_tolerance": tolerance,
        })
        if not all(r["passed"] for r in enumeration_rows) or max_fe_error > tolerance:
            raise RuntimeError("population/reference gate failed; examine saved enumeration and singleton FE evidence")

        shape = (n, repetitions, len(sizes), len(budgets), 3)
        scores = np.full(shape, np.nan)
        eligible = np.zeros(shape, dtype=bool)
        reasons = np.full(shape, "not_evaluated", dtype="U64")
        raw_fields = ["input_id", "source_test_row_id", "repetition", "k", "budget", "explanation",
                      "score", "population_score", "error", "valid", "analysis_eligible", "numerical_instability",
                      "reason", "exception", "exception_stage", "score_source"]
        max_budget = max(budgets)
        with (output / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=raw_fields)
            writer.writeheader()
            for i in range(n):
                masks = {k: np.full((repetitions, max_budget, k), -1, dtype=np.int16) for k in sizes}
                u = np.full((repetitions, len(sizes), max_budget, 3), np.nan)
                v = np.full((repetitions, len(sizes), max_budget), np.nan)
                perturbed = np.full_like(v, np.nan)
                mask_coverage = np.zeros((repetitions, len(sizes)), dtype=np.int64)
                u_coverage = np.zeros((repetitions, len(sizes), 3), dtype=np.int64)
                original = np.full(repetitions, np.nan)
                try:
                    for rep in range(repetitions):
                        seed = evaluator_seed_words(config["seeds"]["evaluator"], REAL_PHASE_IDS[config["phase"]], int(prepared.row_ids[i]), rep)
                        for ki in reversed(range(len(sizes))):
                            k = sizes[ki]
                            for bi in reversed(range(len(budgets))):
                                budget = budgets[bi]
                                for ei, name in enumerate(REAL_EXPLANATION_NAMES):
                                    counts = manifest["counts"]
                                    counts["n_attempted_scores"] += 1
                                    result = invoke("FC", i, name, seed, k, budget)
                                    pairing_error, paired = None, False
                                    if result.exception is None:
                                        try:
                                            trace = result.trace
                                            if np.isnan(original[rep]):
                                                original[rep] = float(trace["original_output"])
                                            elif original[rep] != float(trace["original_output"]):
                                                raise RuntimeError("original prediction changed across paired calls")
                                            # Compare every observed common k/B prefix. A failed call
                                            # never becomes a mask reference or an invented trace.
                                            for previous_ki, previous_k in enumerate(sizes):
                                                common_b = min(budget, int(mask_coverage[rep, previous_ki]))
                                                common_k = min(k, previous_k)
                                                if common_b and not np.array_equal(trace["subsets"][:common_b, :common_k], masks[previous_k][rep, :common_b, :common_k]):
                                                    raise RuntimeError("actual Quantus masks violate paired k/B/candidate prefixes")
                                            common_b = min(budget, int(mask_coverage[rep, ki]))
                                            if common_b and (not np.array_equal(trace["V"][:common_b], v[rep, ki, :common_b]) or not np.array_equal(trace["perturbed_outputs"][:common_b], perturbed[rep, ki, :common_b])):
                                                raise RuntimeError("paired model responses differ across candidates/budgets")
                                            common_u = min(budget, int(u_coverage[rep, ki, ei]))
                                            if common_u and not np.array_equal(trace["U"][:common_u], u[rep, ki, :common_u, ei]):
                                                raise RuntimeError("attribution sums violate the recorded budget prefix")
                                            masks[k][rep, :budget] = trace["subsets"]
                                            u[rep, ki, :budget, ei] = trace["U"]
                                            v[rep, ki, :budget], perturbed[rep, ki, :budget] = trace["V"], trace["perturbed_outputs"]
                                            mask_coverage[rep, ki] = max(mask_coverage[rep, ki], budget)
                                            u_coverage[rep, ki, ei] = max(u_coverage[rep, ki, ei], budget)
                                            paired = True
                                        except Exception as error:
                                            pairing_error = f"{type(error).__name__}: {error}"
                                    exception = retain("FC", i, name, result, seed, repetition=rep, k=k, budget=budget,
                                                       pairing_error=pairing_error, pairing_verified=paired)
                                    finite = bool(np.isfinite(result.score))
                                    usable = bool(result.analysis_eligible and exception is None)
                                    unstable = any(w["category"] == "NearConstantInputWarning" for w in result.warnings)
                                    reason = result.reason if not pairing_error else "exception:trace_pairing"
                                    scores[i, rep, ki, bi, ei] = result.score
                                    eligible[i, rep, ki, bi, ei] = usable
                                    reasons[i, rep, ki, bi, ei] = reason
                                    writer.writerow({
                                        "input_id": i, "source_test_row_id": int(prepared.row_ids[i]),
                                        "repetition": rep, "k": k, "budget": budget, "explanation": name,
                                        "score": result.score, "population_score": population[i, ei],
                                        "error": result.score - population[i, ei], "valid": finite,
                                        "analysis_eligible": usable, "numerical_instability": unstable, "reason": reason,
                                        "exception": exception or "", "exception_stage": "trace_pairing" if pairing_error else (result.exception_stage or ""),
                                        "score_source": "quantus.FaithfulnessCorrelation.__call__",
                                    })
                                    counts["n_official_finite_scores"] += int(finite)
                                    counts["n_analysis_eligible_scores"] += int(usable)
                                    counts["n_numerically_unstable_scores"] += int(unstable)
                                    counts["n_undefined_scores"] += int(not finite and exception is None)
                                    counts["n_exception_scores"] += int(exception is not None)
                                    if exception and (pairing_error or not _public_validation_rejection(result)):
                                        stream.flush()
                                        raise RuntimeError(f"FC/adapter failure retained before stopping: {exception}")
                finally:
                    np.savez_compressed(output / "traces" / f"input_{i:04d}.npz",
                                        U=u, V=v, perturbed_outputs=perturbed, original_output=original,
                                        mask_coverage=mask_coverage, U_coverage=u_coverage,
                                        **{f"subsets_k{k}": values for k, values in masks.items()},
                                        subset_sizes=np.array(sizes), budgets=np.array(budgets),
                                        explanation_names=np.array(REAL_EXPLANATION_NAMES))

        completed_fc = manifest["counts"]["n_public_FC_calls"] - manifest["counts"]["n_exception_scores"]
        completed_public = completed_fc + manifest["counts"]["n_public_FE_calls"] - manifest["counts"]["n_FE_exceptions"]
        manifest["validation"].update({
            "n_public_calls_without_exception": completed_public,
            "all_nonexception_FC_calls_paired": manifest["counts"]["n_pairing_checks_passed"] == completed_fc,
            "all_nonexception_public_calls_trace_checked": manifest["counts"]["n_trace_checks_passed"] == completed_public,
            "all_nonexception_public_calls_score_parity_classified": manifest["counts"]["n_score_parity_passed"] + manifest["counts"]["n_score_parity_deferred_near_constant"] == completed_public,
        })
        analyzed = analyze_real_scores(
            scores, reasons, population, sizes, budgets, row_ids=prepared.row_ids,
            tie_tolerance=config["analysis"]["practical_tie_tolerance"],
            bootstrap_resamples=config["analysis"]["bootstrap_resamples"],
            bootstrap_confidence=config["analysis"]["confidence"], analysis_seed=config["seeds"]["analysis"],
            analysis_eligible=eligible,
        )
        for name, rows in analyzed.items():
            write_csv(output / f"{name}.csv", rows)
        answer, explanation = make_rq_answer(manifest, population, enumeration_rows, analyzed)
        _write_json(output / "rq_answer.json", answer)
        (output / "rq_answer.txt").write_text(explanation, encoding="utf-8")
        plot_real_results(output, scores, eligible, population, prepared.row_ids, enumeration_rows, analyzed, config)
        if implementation_hash() != manifest["implementation_sha256"] or canonical_hash(load_exp01r_config(config_path)) != manifest["config_sha256"]:
            raise RuntimeError("source or configuration changed during the run; saved outputs are not a stable-protocol run")
        if snapshot.exists() and file_hash(snapshot) != manifest["protocol_source"]["sha256"]:
            raise RuntimeError("protocol snapshot changed during the run")
        manifest["status"] = "completed"
        manifest["rq_answer"] = answer
        manifest["evidence_scope"] = "Observed implementation smoke, not development or confirmation." if config["phase"] == "smoke" else "Observed development; reserved inputs/settings have not been evaluated."
        print(explanation, flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["counts"]["n_fatal_errors"] += 1
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        failures.append({"stage": "run", "category": "fatal_error", "reason": f"{type(error).__name__}: {error}"})
        raise
    finally:
        manifest["validation"].update({
            "max_public_trace_replay_absolute_error_all_diagnostics": max_trace_error,
            "max_stable_public_trace_replay_absolute_error": max_stable_error,
        })
        manifest["counts"]["n_failures_logged"] = len(failures)
        write_csv(output / "failures.csv", failures, FAILURE_FIELDS)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - timer
        manifest["artifacts_sha256"] = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
                                          if p.is_file() and p.name != "manifest.json"}
        _write_json(output / "manifest.json", manifest)
    return manifest
