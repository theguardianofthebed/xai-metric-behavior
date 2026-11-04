"""EXP-01 execution, evidence records, and a prospective confirmation lock."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .analysis import analyze_scores
from .core import (EXPLANATION_NAMES, LinearOracle, checked_pearson,
                   contribution_vectors, enumerate_fc, singleton_fe)
from .quantus_backend import (METRIC_VARIANT, PublicScore, backend_provenance,
                             evaluator_seed_words, public_score, verify_pairing)


DEFAULT_WEIGHTS = [2.5, -2.0, 1.5, 1.0, -0.5] + [0.0] * 15
PHASE_IDS = {"smoke": 0, "development": 1, "confirmation": 2}
FAILURE_FIELDS = ["stage", "input_id", "repetition", "k", "budget", "explanation",
                  "category", "reason"]


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash() -> str:
    """Hash source modules only; reports/results/tests cannot alter the lock."""
    package = Path(__file__).resolve().parent
    files = {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))}
    return canonical_hash(files)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("experiment_id") != "EXP-01":
        raise ValueError("configuration must identify experiment_id: EXP-01")
    allowed = {"experiment_id", "phase", "dimension", "weights", "baseline", "n_inputs", "repetitions",
               "subset_sizes", "budgets", "seeds", "analysis", "enumeration", "write_traces"}
    if set(config) != allowed:
        raise ValueError(f"unexpected/missing configuration fields: {sorted(set(config) ^ allowed)}")
    nested = {
        "seeds": {"inputs", "evaluator", "analysis"},
        "analysis": {"numerical_tolerance", "practical_tie_tolerance", "bootstrap_resamples", "confidence"},
        "enumeration": {"enabled", "n_inputs", "max_subsets"},
    }
    for key, fields in nested.items():
        if not isinstance(config.get(key), dict) or set(config[key]) != fields:
            raise ValueError(f"{key} must contain exactly {sorted(fields)}; unsupported settings are never ignored")
    if config.get("phase") not in PHASE_IDS:
        raise ValueError("phase must be smoke, development, or confirmation")
    if config.get("dimension") != 20 or config.get("weights") != DEFAULT_WEIGHTS:
        raise ValueError("EXP-01 freezes the Atlas SET-01 20-dimensional sparse oracle")
    if config.get("baseline") != 0.0:
        raise ValueError("EXP-01 uses literal numeric baseline zero")
    for key in ("n_inputs", "repetitions"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("subset_sizes", "budgets"):
        vals = config.get(key)
        if not isinstance(vals, list) or not vals or any(type(v) is not int for v in vals):
            raise ValueError(f"{key} must be a nonempty integer list")
        if sorted(set(vals)) != vals:
            raise ValueError(f"{key} must be strictly increasing without duplicates")
    if not all(1 <= k < config["dimension"] for k in config["subset_sizes"]):
        raise ValueError("primary subset sizes must satisfy 1 <= k < d")
    if min(config["budgets"]) < 2:
        raise ValueError("Pearson budgets must be at least two draws")
    if config["phase"] == "confirmation" and config["subset_sizes"] != [2, 7, 13]:
        raise ValueError("the reserved confirmation subset sizes are [2, 7, 13]")
    if config["phase"] != "confirmation" and any(k in [2, 7, 13] for k in config["subset_sizes"]):
        raise ValueError("do not consume reserved confirmation sizes during development/smoke")
    seeds = config.get("seeds", {})
    if any(type(seeds.get(k)) is not int or seeds[k] < 0 for k in ("inputs", "evaluator", "analysis")):
        raise ValueError("declare nonnegative integer input, evaluator and analysis seeds")
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("use distinct input, evaluator and analysis seed roots")
    analysis = config.get("analysis", {})
    if not (0 < analysis.get("numerical_tolerance", 0) < 0.001):
        raise ValueError("declare a small positive numerical tolerance")
    if not (0 < analysis.get("practical_tie_tolerance", 0) < 1):
        raise ValueError("declare a practical tie tolerance between zero and one")
    if type(analysis.get("bootstrap_resamples")) is not int or analysis["bootstrap_resamples"] < 2:
        raise ValueError("declare at least two bootstrap resamples")
    if not (0 < analysis.get("confidence", 0) < 1):
        raise ValueError("bootstrap confidence must be between zero and one")
    if config.get("write_traces") is not True:
        raise ValueError("EXP-01 requires computation traces; write_traces must be true")
    enumeration = config.get("enumeration", {})
    if enumeration.get("enabled") is not True or enumeration.get("n_inputs") != 1:
        raise ValueError("the protocol enumerates the first input plus the worked toy fixture")
    if type(enumeration.get("max_subsets")) is not int or enumeration["max_subsets"] < 184756:
        raise ValueError("the exact enumeration cap must permit C(20,10)=184756 subsets")
    return config


def write_csv(path: Path, rows: list[dict[str, Any]], default_fields: list[str] | None = None) -> None:
    fields = list(dict.fromkeys([*(default_fields or []), *(key for row in rows for key in row)]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    def safe(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: safe(val) for key, val in item.items()}
        if isinstance(item, list):
            return [safe(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    status = command("status", "--porcelain", "--untracked-files=no")
    return {"commit": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"),
            "tracked_worktree_dirty": None if status is None else bool(status)}


def _runtime() -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "scipy", "PyYAML", "matplotlib", "quantus", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def _checked_confirmation(config: dict[str, Any], freeze_path: str | Path | None) -> dict[str, Any] | None:
    if config["phase"] != "confirmation":
        if freeze_path is not None:
            raise ValueError("a confirmation freeze is only applicable to the confirmation phase")
        return None
    if freeze_path is None:
        raise ValueError("confirmation requires --freeze from the freeze command after development")
    freeze_file = Path(freeze_path)
    frozen = json.loads(freeze_file.read_text(encoding="utf-8"))
    if frozen.get("config_sha256") != canonical_hash(config):
        raise ValueError("confirmation configuration differs from its frozen specification")
    if frozen.get("implementation_sha256") != implementation_hash():
        raise ValueError("source code differs from the freeze; redevelop and freeze a new protocol")
    if frozen.get("backend_execution_sha256") != backend_provenance()["execution_sha256"]:
        raise ValueError("Quantus backend differs from the freeze; redevelop and freeze a new protocol")
    if frozen.get("experiment_id") != "EXP-01" or frozen.get("status") != "frozen_before_confirmation":
        raise ValueError("invalid confirmation freeze record")
    return {"file": str(freeze_file.resolve()), "sha256": file_hash(freeze_file), "record": frozen}


def freeze_confirmation(
    config_path: str | Path, development_run: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Freeze reserved conditions after completed development; never run holdouts.

    A local content lock is not external preregistration or proof of a timestamp.
    The operator must preserve the record before examining confirmation outcomes.
    """
    config = load_config(config_path)
    if config["phase"] != "confirmation":
        raise ValueError("freeze requires a confirmation configuration")
    run_dir = Path(development_run)
    manifest_file = run_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("phase") != "development":
        raise ValueError("freeze requires a completed development run, not smoke or partial outputs")
    if manifest.get("implementation_sha256") != implementation_hash():
        raise ValueError("source changed after development; repeat development before freezing")
    backend = backend_provenance()
    if manifest.get("backend_execution_sha256") != backend["execution_sha256"]:
        raise ValueError("Quantus backend changed after development; repeat development before freezing")
    dev_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if set(dev_config["subset_sizes"]) & set(config["subset_sizes"]):
        raise ValueError("confirmation sizes must be reserved from development")
    if any(dev_config["seeds"][name] == config["seeds"][name] for name in ("inputs", "evaluator", "analysis")):
        raise ValueError("confirmation uses fresh data, evaluator and analysis seed roots")
    for key in ("weights", "baseline", "dimension", "analysis"):
        if dev_config[key] != config[key]:
            raise ValueError(f"confirmation changed the developed {key} contract")
    expected = manifest.get("artifacts_sha256", {})
    for filename in ("resolved_config.yaml", "population_scores.csv"):
        if expected.get(filename) != file_hash(run_dir / filename):
            raise ValueError(f"development {filename} was changed after the run")
    margins = []
    with (run_dir / "population_scores.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["explanation"] == "A" and int(row["k"]) == dev_config["subset_sizes"][0]:
                margins.append(float(row["population_margin_A_minus_B"]))
    if not margins or not np.isfinite(margins).all():
        raise ValueError("development population margins are unavailable or nonfinite")
    record = {
        "experiment_id": "EXP-01", "status": "frozen_before_confirmation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_type": "local_content_lock_not_external_preregistration",
        "config_sha256": canonical_hash(config), "configuration": config,
        "implementation_sha256": implementation_hash(), "git": _git_provenance(),
        "backend_execution_sha256": backend["execution_sha256"], "quantus_backend": backend,
        "development_manifest_sha256": file_hash(manifest_file),
        "development_run_id": manifest["run_id"],
        "fixed_pair": {"A": [[0, 5]], "B": [[0, 5], [1, 6]], "indices": "zero_based"},
        "development_population_margin": {"min": float(np.min(margins)), "median": float(np.median(margins)),
                                          "mean": float(np.mean(margins)), "max": float(np.max(margins))},
        "prediction": "FC_population equals coordinate Pearson for every admissible fixed subset size; singleton FE agrees.",
        "challenge_rule": "Exact-enumeration discrepancy exceeding numerical_tolerance with verified premises requires diagnosis.",
        "finite_estimator_prediction": "Bias, variance and undefined probability may differ with k and budget; no forced monotonicity or reversal.",
        "uncertainty": "Pointwise input-cluster bootstrap of within-input jointly-valid paired summaries; conditional finite-score estimand.",
        "failure_handling": "Retain undefined/exception outcomes and eligibility; do not assign zero or claim unconditional equivalence.",
        "precision_status": "Fixed computational budgets; no power or coverage guarantee. Inconclusive intervals remain inconclusive.",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return record


def _enumeration_checks(
    config: dict[str, Any], model: LinearOracle, inputs: np.ndarray,
    baseline: np.ndarray, explanations: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    from math import comb
    toy_model = LinearOracle(np.array([2.0, 1.0, 0.0]))
    toy_x = np.array([3.0, 4.0, 5.0])
    toy_v = np.array([6.0, 4.0, 0.0])
    toy_a = {"exact": toy_v, "toy_relocated": np.array([0.0, 4.0, 6.0])}
    fixtures = [("worked_d3", toy_model, toy_x, np.zeros(3), toy_a, [1, 2]),
                ("first_current_input_d20", model, inputs[0], baseline,
                 {name: values[0] for name, values in explanations.items()}, config["subset_sizes"])]
    rows = []
    tolerance = config["analysis"]["numerical_tolerance"]
    for name, current_model, x, base, a, sizes in fixtures:
        reference = (x - base) * current_model.weights
        for k in sizes:
            exact = enumerate_fc(current_model, x, base, a, k,
                                 max_subsets=config["enumeration"]["max_subsets"])
            for explanation, score in exact.items():
                coordinate = checked_pearson(a[explanation], reference)
                fe = singleton_fe(current_model, x, base, a[explanation])
                error = abs(score.score - coordinate.score)
                passed = score.valid and coordinate.valid and fe.valid and error <= tolerance
                passed = bool(passed and abs(fe.score - coordinate.score) <= tolerance)
                rows.append({
                    "fixture": name, "input_id": 0 if name.startswith("first") else "toy",
                    "dimension": len(x), "k": k, "n_subsets": comb(len(x), k),
                    "explanation": explanation, "coordinate_prediction": coordinate.score,
                    "enumerated_FC": score.score, "reference_singleton_FE": fe.score,
                    "reference_engine": "independent_numpy_model_deletion_check",
                    "absolute_error": error, "numerical_tolerance": tolerance,
                    "valid": score.valid, "reason": score.reason, "passed": passed,
                    "scope": "one_exact_fixture_not_all_inputs",
                })
    return rows


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a declared phase and return its manifest. Existing outputs are protected."""
    config = load_config(config_path)
    freeze = _checked_confirmation(config, freeze_path)
    backend = backend_provenance()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    started = datetime.now(timezone.utc).isoformat()
    timer = time.monotonic()
    snapshot = Path(__file__).resolve().parents[2] / "docs" / "atlas_quantus_contract_2026-11-04.json"
    manifest: dict[str, Any] = {
        "experiment_id": "EXP-01", "phase": config["phase"], "status": "running",
        "run_id": canonical_hash({"config": config, "started": started})[:16],
        "started_at_utc": started, "config_sha256": canonical_hash(config),
        "implementation_sha256": implementation_hash(), "git": _git_provenance(),
        "runtime": _runtime(), "metric_variant": METRIC_VARIANT,
        "library_execution": "Official Quantus 0.6.0 public FC/FE returns; NumPy is reference/trace verification only.",
        "backend_execution_sha256": backend["execution_sha256"], "quantus_backend": backend,
        "atlas": {"id": "3973d962-747e-45ef-b05c-18dd541358d9", "source_revision": 9,
                  "snapshot_sha256": file_hash(snapshot) if snapshot.exists() else None},
        "rng": {"input_generator": "PCG64", "evaluator_generator": "Quantus native NumPy legacy MT19937",
                "seed_roots": config["seeds"],
                "evaluator_seed_sequence": "SeedSequence([evaluator_seed, phase_id, input_id, repetition]).generate_state(4, uint32)",
                "phase_id": PHASE_IDS[config["phase"]],
                "coupling": "Reset identical legacy state per condition/candidate; verify actual native k/budget/explanation prefixes",
                "global_state": "Serial scoped calls restore NumPy state and Torch default dtype"},
        "contract": {"target": "scalar g(x)=sum_j w_j*x_j; no softmax/class selection",
                     "baseline": 0.0, "signed": True, "abs": False, "normalise": False,
                     "selection": "uniform size-k without replacement within draw; independent draws",
                     "comparator": "Pearson", "direction": "higher", "aggregation": "equal-weight inputs after within-input repetitions",
                     "undefined": "official nonfinite returns retained with reasons and all-attempt denominators",
                     "analysis_eligibility": "finite official return, no exception, and no upstream NearConstantInputWarning",
                     "numerical_instability": "preserve official score/error; exclude warned calls from conditional estimation/ranking; retain counts",
                     "counts": "n_valid_scores aliases official finiteness; undefined and numerical-instability reason counts may overlap",
                     "pair_A_swaps": [[0, 5]], "pair_B_swaps": [[0, 5], [1, 6]], "indexing": "zero_based"},
        "confirmation_freeze": freeze,
        "counts": {"n_inputs": config["n_inputs"], "repetitions_per_input": config["repetitions"],
                   "n_planned_scores": config["n_inputs"] * config["repetitions"] * len(config["subset_sizes"]) * len(config["budgets"]) * 3,
                   "n_attempted_scores": 0, "n_valid_scores": 0, "n_nonfinite_scores": 0,
                   "n_official_finite_scores": 0, "n_analysis_eligible_scores": 0,
                   "n_numerically_unstable_scores": 0, "n_undefined_scores": 0,
                   "n_exception_scores": 0, "n_exceptions": 0, "n_failures_logged": 0,
                   "n_public_FC_calls": 0, "n_public_FE_calls": 0,
                   "n_FE_official_finite": 0, "n_FE_analysis_eligible": 0,
                   "n_FE_numerically_unstable": 0, "n_FE_undefined": 0,
                   "n_trace_checks_passed": 0, "n_pairing_checks_passed": 0,
                   "n_score_parity_passed": 0, "n_score_parity_deferred_near_constant": 0,
                   "n_warning_records": 0},
        "interpretation": "Population theorem under declared premises; finite estimates and illustrative budget choices do not establish universal metric reliability.",
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_json(output / "manifest.json", manifest)
    failures: list[dict[str, Any]] = []
    call_index = 0
    max_trace_error = 0.0
    max_stable_trace_error = 0.0

    def invoke(kind: str, input_id: int, explanation: str, x_value: np.ndarray,
               attribution: np.ndarray, seed_words: np.ndarray, *,
               k: int | None = None, budget: int | None = None) -> PublicScore:
        manifest["counts"][f"n_public_{kind}_calls"] += 1
        try:
            return public_score(kind, x_value, weights, attribution, seed_words,
                                subset_size=k, budget=budget,
                                tolerance=config["analysis"]["numerical_tolerance"])
        except Exception as error:
            # There is no official return in this case. NaN records absence;
            # an oracle score is never used to fill it.
            return PublicScore(float("nan"), False, f"exception:{type(error).__name__}",
                               exception=f"{type(error).__name__}: {error}")

    def retain_call(kind: str, input_id: int, explanation: str, result: PublicScore,
                    seed_words: np.ndarray, *, repetition: int | None = None,
                    k: int | None = None, budget: int | None = None,
                    guard_error: str | None = None, pairing_verified: bool = False) -> str | None:
        nonlocal call_index, max_trace_error, max_stable_trace_error
        call_index += 1
        exception = result.exception or guard_error
        trace_path = None
        if kind == "FE" or exception:
            name = (f"FE_input_{input_id:04d}_{explanation}.npz" if kind == "FE" else
                    f"failed_call_{call_index:07d}.npz")
            trace_path = f"traces/{name}"
            np.savez_compressed(output / trace_path,
                                **{key: np.asarray(value) for key, value in result.trace.items()
                                   if value is not None})
        if result.checks:
            manifest["counts"]["n_trace_checks_passed"] += 1
            parity_status = result.checks.get("score_parity_status")
            if parity_status == "passed":
                manifest["counts"]["n_score_parity_passed"] += 1
            elif parity_status == "not_asserted_near_constant":
                manifest["counts"]["n_score_parity_deferred_near_constant"] += 1
            discrepancy = result.checks.get("trace_replay_absolute_error")
            if discrepancy is not None:
                max_trace_error = max(max_trace_error, float(discrepancy))
                if parity_status == "passed":
                    max_stable_trace_error = max(max_stable_trace_error, float(discrepancy))
        if pairing_verified:
            manifest["counts"]["n_pairing_checks_passed"] += 1
        manifest["counts"]["n_warning_records"] += len(result.warnings)
        record = {
            "call_id": call_index, "metric": kind,
            "public_api": f"quantus.Faithfulness{'Correlation' if kind == 'FC' else 'Estimate'}.__call__",
            "input_id": input_id, "repetition": repetition, "explanation": explanation,
            "k": k, "budget": budget, "seed_words": seed_words.tolist(),
            "score": float(result.score) if np.isfinite(result.score) else None,
            "raw_score_repr": repr(float(result.score)),
            "valid": bool(np.isfinite(result.score)),
            "analysis_eligible": bool(result.analysis_eligible and exception is None),
            "numerical_instability": any(item["category"] == "NearConstantInputWarning" for item in result.warnings),
            "reason": result.reason if guard_error is None else "exception:trace_pairing",
            "exception": exception, "checks": result.checks,
            "pairing_verified": pairing_verified,
            "mask_sha256": result.trace.get("mask_sha256"),
            "uv_sha256": result.trace.get("uv_sha256"),
            "model_input_sha256": result.trace.get("model_input_sha256"),
            "trace_file": trace_path, "warnings": result.warnings,
        }
        with (output / "metric_calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return exception
    try:
        n, repetitions, dimension = config["n_inputs"], config["repetitions"], config["dimension"]
        sizes, budgets = config["subset_sizes"], config["budgets"]
        weights, baseline = np.array(config["weights"], dtype=float), np.zeros(dimension)
        model = LinearOracle(weights)
        data_rng = np.random.Generator(np.random.PCG64(config["seeds"]["inputs"]))
        x = data_rng.standard_normal((n, dimension))
        candidates = contribution_vectors(x, weights, baseline)
        stacked = np.stack([candidates[name] for name in EXPLANATION_NAMES], axis=1)
        np.savez_compressed(output / "inputs.npz", x=x, weights=weights, baseline=baseline,
                            explanations=stacked, explanation_names=np.array(EXPLANATION_NAMES))
        population = np.full((n, len(EXPLANATION_NAMES)), np.nan)
        population_rows, max_fe_error, max_fe_error_all_finite, max_margin_error = [], 0.0, 0.0, 0.0
        tolerance = config["analysis"]["numerical_tolerance"]
        for i in range(n):
            v = candidates["exact"][i]
            d_v = float(np.sum((v - v.mean()) ** 2))
            theoretical_margin = float(v[1] ** 2 / d_v) if d_v > 0 else float("nan")
            coordinate = [checked_pearson(stacked[i, ei], v) for ei in range(len(EXPLANATION_NAMES))]
            for ei, reference in enumerate(coordinate):
                population[i, ei] = reference.score
            actual_margin = float(population[i, 1] - population[i, 2])
            if np.isfinite(actual_margin) and np.isfinite(theoretical_margin):
                max_margin_error = max(max_margin_error, abs(actual_margin - theoretical_margin))
            for ei, explanation in enumerate(EXPLANATION_NAMES):
                reference = coordinate[ei]
                fe_seed = evaluator_seed_words(config["seeds"]["evaluator"], PHASE_IDS[config["phase"]], i, 0)
                fe = invoke("FE", i, explanation, x[i], stacked[i, ei], fe_seed)
                fe_exception = retain_call("FE", i, explanation, fe, fe_seed, k=1, budget=dimension)
                fe_finite = bool(np.isfinite(fe.score))
                fe_unstable = any(item["category"] == "NearConstantInputWarning" for item in fe.warnings)
                fe_eligible = bool(fe.analysis_eligible and fe_exception is None)
                manifest["counts"]["n_FE_official_finite"] += int(fe_finite)
                manifest["counts"]["n_FE_analysis_eligible"] += int(fe_eligible)
                manifest["counts"]["n_FE_numerically_unstable"] += int(fe_unstable)
                manifest["counts"]["n_FE_undefined"] += int(not fe_finite and fe_exception is None)
                if fe_exception:
                    failures.append({"stage": "public_singleton_FE", "input_id": i,
                                     "explanation": explanation, "category": "exception", "reason": fe_exception})
                    raise RuntimeError(f"Quantus FE/adapter failed; official return and trace retained: {fe_exception}")
                if reference.valid and fe_finite:
                    max_fe_error_all_finite = max(max_fe_error_all_finite, abs(reference.score - fe.score))
                    if fe_eligible:
                        max_fe_error = max(max_fe_error, abs(reference.score - fe.score))
                if fe_unstable:
                    failures.append({"stage": "public_singleton_FE", "input_id": i, "explanation": explanation,
                                     "category": "numerical_instability", "reason": fe.reason})
                if not reference.valid or not fe_finite:
                    failures.append({"stage": "population", "input_id": i, "explanation": explanation,
                                     "category": "undefined", "reason": reference.reason if not reference.valid else fe.reason})
                denominator = float(np.abs(v).sum())
                reference_error = float(np.abs(stacked[i, ei] - v).sum() / denominator) if denominator else float("nan")
                for k in sizes:
                    population_rows.append({
                        "input_id": i, "k": k, "explanation": explanation,
                        "population_FC": reference.score, "reference_type": "HYP-01_coordinate_formula",
                        "singleton_FE": fe.score, "singleton_FE_source": "quantus.FaithfulnessEstimate.__call__",
                        "valid": reference.valid and fe_finite,
                        "singleton_FE_valid": fe_finite, "singleton_FE_analysis_eligible": fe_eligible,
                        "singleton_FE_numerical_instability": fe_unstable,
                        "reference_error_L1": reference_error, "reference_error_denominator": denominator,
                        "reference_error_reason": "valid" if denominator else "zero_reference_L1",
                        "population_margin_A_minus_B": actual_margin,
                        "derived_margin_feature2_squared_over_centered_ss": theoretical_margin,
                        "reason": reference.reason if not reference.valid else fe.reason,
                    })
        write_csv(output / "population_scores.csv", population_rows)
        enumeration_rows = _enumeration_checks(config, model, x, baseline, candidates)
        write_csv(output / "exact_enumeration.csv", enumeration_rows)
        enumerations_passed = all(row["passed"] for row in enumeration_rows)
        manifest["validation"] = {
            "enumeration_comparisons": len(enumeration_rows), "enumeration_all_passed": enumerations_passed,
            "max_enumeration_absolute_error": float(max(row["absolute_error"] for row in enumeration_rows)),
            "max_singleton_FE_coordinate_error_eligible_inputs": max_fe_error,
            "max_singleton_FE_coordinate_error_all_finite": max_fe_error_all_finite,
            "max_fixed_pair_margin_identity_error": max_margin_error,
            "numerical_tolerance": tolerance,
            "d20_enumeration_scope": "first input at configured subset sizes only",
        }
        if not enumerations_passed or max_fe_error > tolerance or max_margin_error > tolerance:
            raise RuntimeError("population/reference validation failed; inspect exact_enumeration.csv and manifest")

        shape = (n, repetitions, len(sizes), len(budgets), len(EXPLANATION_NAMES))
        scores = np.full(shape, np.nan)
        analysis_eligible = np.zeros(shape, dtype=bool)
        reasons = np.full(shape, "not_evaluated", dtype="U64")
        raw_fields = ["input_id", "repetition", "draw_plan_id", "k", "budget", "explanation",
                      "score", "population_score", "error", "valid", "analysis_eligible", "numerical_instability",
                      "reason", "exception", "score_source"]
        max_budget = max(budgets)
        with (output / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=raw_fields)
            writer.writeheader()
            for i in range(n):
                observed = {k: np.full((repetitions, max_budget, k), -1, dtype=np.int16) for k in sizes}
                us = np.full((repetitions, len(sizes), max_budget, len(EXPLANATION_NAMES)), np.nan)
                vs = np.full((repetitions, len(sizes), max_budget), np.nan)
                outputs = np.full_like(vs, np.nan)
                original_output = float("nan")
                try:
                    for rep in range(repetitions):
                        seed_words = evaluator_seed_words(config["seeds"]["evaluator"], PHASE_IDS[config["phase"]], i, rep)
                        draw_id = f"{config['phase']}:input-{i}:rep-{rep}"
                        reference_trace = None
                        # The first real call is the largest k/B exact candidate.
                        # Its observed masks are the pairing reference; no extra
                        # uncounted metric call or injected permutation is needed.
                        for ki in reversed(range(len(sizes))):
                            k = sizes[ki]
                            for bi in reversed(range(len(budgets))):
                                budget = budgets[bi]
                                for ei, explanation in enumerate(EXPLANATION_NAMES):
                                    manifest["counts"]["n_attempted_scores"] += 1
                                    result = invoke("FC", i, explanation, x[i], stacked[i, ei], seed_words,
                                                    k=k, budget=budget)
                                    guard_error, pairing_verified = None, False
                                    if result.exception is None:
                                        try:
                                            trace = result.trace
                                            if reference_trace is None:
                                                reference_trace = trace
                                                original_output = float(trace["original_output"])
                                            verify_pairing(reference_trace, trace, budget=budget, subset_size=k)
                                            if float(trace["original_output"]) != original_output:
                                                raise RuntimeError("original prediction changed between paired public calls")
                                            if budget == max_budget:
                                                us[rep, ki, :, ei] = trace["U"]
                                                if ei == 0:
                                                    observed[k][rep] = trace["subsets"]
                                                    vs[rep, ki], outputs[rep, ki] = trace["V"], trace["perturbed_outputs"]
                                                elif not (np.array_equal(vs[rep, ki], trace["V"]) and
                                                          np.array_equal(outputs[rep, ki], trace["perturbed_outputs"])):
                                                    raise RuntimeError("model responses differ across paired explanations")
                                            elif not (np.array_equal(us[rep, ki, :budget, ei], trace["U"]) and
                                                      np.array_equal(vs[rep, ki, :budget], trace["V"]) and
                                                      np.array_equal(outputs[rep, ki, :budget], trace["perturbed_outputs"])):
                                                raise RuntimeError("actual U/V/model outputs violate budget-prefix replay")
                                            pairing_verified = True
                                        except Exception as error:
                                            guard_error = f"{type(error).__name__}: {error}"
                                    exception = retain_call("FC", i, explanation, result, seed_words,
                                                            repetition=rep, k=k, budget=budget,
                                                            guard_error=guard_error, pairing_verified=pairing_verified)
                                    valid = bool(np.isfinite(result.score))
                                    eligible = bool(result.analysis_eligible and exception is None)
                                    unstable = any(item["category"] == "NearConstantInputWarning" for item in result.warnings)
                                    reason = result.reason if guard_error is None else "exception:trace_pairing"
                                    scores[i, rep, ki, bi, ei], reasons[i, rep, ki, bi, ei] = result.score, reason
                                    analysis_eligible[i, rep, ki, bi, ei] = eligible
                                    writer.writerow({
                                        "input_id": i, "repetition": rep, "draw_plan_id": draw_id,
                                        "k": k, "budget": budget, "explanation": explanation,
                                        "score": result.score, "population_score": population[i, ei],
                                        "error": result.score - population[i, ei], "valid": valid, "reason": reason,
                                        "analysis_eligible": eligible, "numerical_instability": unstable,
                                        "exception": exception or "", "score_source": "quantus.FaithfulnessCorrelation.__call__",
                                    })
                                    # Finiteness describes the unmodified library return, even
                                    # if a later structural guard aborts this attempted score.
                                    manifest["counts"]["n_valid_scores" if valid else "n_nonfinite_scores"] += 1
                                    manifest["counts"]["n_official_finite_scores"] += int(valid)
                                    manifest["counts"]["n_analysis_eligible_scores"] += int(eligible)
                                    manifest["counts"]["n_numerically_unstable_scores"] += int(unstable)
                                    manifest["counts"]["n_undefined_scores"] += int(not valid and exception is None)
                                    if exception:
                                        manifest["counts"]["n_exception_scores"] += 1
                                        failures.append({"stage": "public_FC_or_adapter", "input_id": i,
                                                         "repetition": rep, "k": k, "budget": budget,
                                                         "explanation": explanation, "category": "exception", "reason": exception})
                                        stream.flush()
                                        raise RuntimeError(f"Quantus FC/adapter failed; official return retained: {exception}")
                                    if unstable:
                                        failures.append({"stage": "finite_correlation", "input_id": i, "repetition": rep,
                                                         "k": k, "budget": budget, "explanation": explanation,
                                                         "category": "numerical_instability", "reason": reason})
                                    if not valid:
                                        failures.append({"stage": "finite_correlation", "input_id": i, "repetition": rep,
                                                         "k": k, "budget": budget, "explanation": explanation,
                                                         "category": "undefined", "reason": reason})
                finally:
                    np.savez_compressed(
                        output / "traces" / f"input_{i:04d}.npz", U=us, V=vs,
                        perturbed_outputs=outputs, original_output=original_output,
                        **{f"subsets_k{k}": values for k, values in observed.items()},
                        subset_sizes=np.array(sizes), budgets=np.array(budgets),
                        explanation_names=np.array(EXPLANATION_NAMES),
                    )
        anchor_values = np.where(analysis_eligible[..., 0], scores[..., 0], np.nan)
        finite_anchor = anchor_values[np.isfinite(anchor_values)]
        anchor_error = float(np.max(np.abs(finite_anchor - 1.0))) if len(finite_anchor) else None
        manifest["validation"]["max_eligible_exact_anchor_error"] = anchor_error
        manifest["validation"]["max_public_trace_replay_absolute_error_all_diagnostics"] = max_trace_error
        manifest["validation"]["max_stable_public_trace_replay_absolute_error"] = max_stable_trace_error
        manifest["validation"]["actual_pairing_checks"] = manifest["counts"]["n_pairing_checks_passed"]
        manifest["validation"]["all_FC_calls_paired"] = (
            manifest["counts"]["n_pairing_checks_passed"] == manifest["counts"]["n_attempted_scores"])
        manifest["validation"]["public_FE_calls"] = manifest["counts"]["n_public_FE_calls"]
        manifest["validation"]["all_public_calls_trace_checked"] = (
            manifest["counts"]["n_trace_checks_passed"] ==
            manifest["counts"]["n_public_FC_calls"] + manifest["counts"]["n_public_FE_calls"])
        manifest["validation"]["all_public_calls_score_parity_classified"] = (
            manifest["counts"]["n_score_parity_passed"] + manifest["counts"]["n_score_parity_deferred_near_constant"] ==
            manifest["counts"]["n_public_FC_calls"] + manifest["counts"]["n_public_FE_calls"])
        if anchor_error is not None and anchor_error > tolerance:
            raise RuntimeError("an analysis-eligible exact-anchor correlation differed from one beyond tolerance")
        analyzed = analyze_scores(
            scores, reasons, population, sizes, budgets,
            tie_tolerance=config["analysis"]["practical_tie_tolerance"],
            bootstrap_resamples=config["analysis"]["bootstrap_resamples"],
            bootstrap_confidence=config["analysis"]["confidence"], analysis_seed=config["seeds"]["analysis"],
            analysis_eligible=analysis_eligible,
        )
        for filename, rows in analyzed.items():
            write_csv(output / f"{filename}.csv", rows)
        from .plotting import plot_results
        plot_results(output, population_rows, enumeration_rows, analyzed, config)
        manifest["status"] = "completed"
        manifest["evidence_scope"] = (
            "Implementation/sanity evidence only; smoke is not full development or confirmation."
            if config["phase"] == "smoke" else
            "Observed development outputs; reserved confirmation conditions have not been consumed by this run."
            if config["phase"] == "development" else
            "Observed reserved-condition outputs under the linked frozen protocol; inspect failures and precision before a claim."
        )
    except Exception as error:
        manifest["counts"]["n_exceptions"] += 1
        failures.append({"stage": "run", "category": "exception", "reason": f"{type(error).__name__}: {error}"})
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        # Preserve the distinction even if a later guard aborts before analysis.
        manifest.setdefault("validation", {}).update({
            "max_public_trace_replay_absolute_error_all_diagnostics": max_trace_error,
            "max_stable_public_trace_replay_absolute_error": max_stable_trace_error,
        })
        manifest["counts"]["n_failures_logged"] = len(failures)
        write_csv(output / "failures.csv", failures, FAILURE_FIELDS)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - timer
        manifest["artifacts_sha256"] = {
            str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        _write_json(output / "manifest.json", manifest)
    return manifest
